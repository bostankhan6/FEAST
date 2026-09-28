import torch
import torch.nn as nn
from collections import OrderedDict
from contextlib import contextmanager
import numpy as np

from ofa.utils.layers import (
    IdentityLayer,
    ResidualBlock,
    ConvLayer,
    LinearLayer,
    MyModule,
    set_layer_from_config,  # for building a static subnet from a config dict
)
from ofa.utils import get_same_padding, make_divisible, MyNetwork, val2list, get_net_device, build_activation, MyGlobalAvgPool2d
from ofa.imagenet_classification.elastic_nn.modules.dynamic_layers import (
    DynamicConvLayer,
    DynamicLinearLayer
)

from ofa.imagenet_classification.elastic_nn.modules.dynamic_op import (
    DynamicSeparableConv2d,
    DynamicConv2d,
    DynamicBatchNorm2d,
    DynamicSE,
    DynamicGroupNorm,
    DynamicLinear
)
from ofa.imagenet_classification.elastic_nn.modules.dynamic_layers import copy_bn

class NewResidualBlock(MyModule):
    """
    Residual block derived from DynamicResidualBlock.
    This block is a part of subnetwork extracted from supernetwork
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        expand_ratio=0.25,
        mid_channels=None,
        act_func="relu",
        groups=1,
        downsample_mode="avgpool_conv",
        force_downsample=False,
    ):
        super(NewResidualBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.force_downsample = force_downsample

        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels
        self.act_func = act_func
        self.groups = groups

        self.downsample_mode = downsample_mode

        if self.mid_channels is None:
            feature_dim = round(self.out_channels * self.expand_ratio)
        else:
            feature_dim = self.mid_channels

        feature_dim = make_divisible(feature_dim, MyNetwork.CHANNEL_DIVISIBLE)
        self.mid_channels = feature_dim

        pad = get_same_padding(self.kernel_size)
        self.conv1 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv2d(
                            self.in_channels,
                            feature_dim,
                            kernel_size,
                            stride,
                            groups=groups,
                            padding=pad,
                            bias=False,
                        ),
                    ),
                    ("bn", nn.BatchNorm2d(feature_dim)),
                    ("act", build_activation(self.act_func, inplace=True)),
                ]
            )
        )
        self.conv2 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv2d(
                            feature_dim,
                            self.out_channels,
                            kernel_size,
                            padding=pad,
                            bias=False,
                        ),
                    ),
                    ("bn", nn.BatchNorm2d(self.out_channels)),
                    ("act", build_activation(self.act_func, inplace=True)),
                ]
            )
        )

        if stride == 1 and in_channels == out_channels and not force_downsample:
            self.downsample = IdentityLayer(in_channels, out_channels)
        elif self.downsample_mode == "avgpool_conv":
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "avg_pool",
                            nn.AvgPool2d(
                                kernel_size=stride,
                                stride=stride,
                                padding=0,
                                ceil_mode=True,
                            ),
                        ),
                        (
                            "conv",
                            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                        ),
                        ("bn", nn.BatchNorm2d(out_channels)),
                    ]
                )
            )
        else:
            raise NotImplementedError

        self.final_act = build_activation(self.act_func, inplace=True)

    def forward(self, x):
        residual = self.downsample(x)

        x = self.conv1(x)
        x = self.conv2(x)

        x = x + residual
        x = self.final_act(x)
        return x

    @property
    def module_str(self):
        return "(%s, %s)" % (
            "%dx%d_ResidualBlock_%d->%d->%d_S%d_G%d"
            % (
                self.kernel_size,
                self.kernel_size,
                self.in_channels,
                self.mid_channels,
                self.out_channels,
                self.stride,
                self.groups,
            ),
            "Identity"
            if isinstance(self.downsample, IdentityLayer)
            else self.downsample_mode,
        )

    @property
    def config(self):
        return {
            "name": NewResidualBlock.__name__,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "expand_ratio": self.expand_ratio,
            "mid_channels": self.mid_channels,
            "act_func": self.act_func,
            "groups": self.groups,
            "downsample_mode": self.downsample_mode,
        }

    @staticmethod
    def build_from_config(config):
        return NewResidualBlock(**config)


class DynamicResidualBlock(MyModule):
    """
    Dynamic Residual Block part of supernet ranging from Resnets
    from 10-26 layers.
    """

    def __init__(
        self,
        in_channel_list,
        out_channel_list,
        expand_ratio_list=0.25,
        kernel_size=3,
        stride=1,
        act_func="relu",
        downsample_mode="avgpool_conv",
        bn_gamma_zero_init=False,
    ):
        super(DynamicResidualBlock, self).__init__()

        self.in_channel_list = in_channel_list
        self.out_channel_list = out_channel_list
        self.expand_ratio_list = val2list(expand_ratio_list)
        self.bn_gamma_zero_init = bn_gamma_zero_init

        self.kernel_size = kernel_size
        self.stride = stride
        self.act_func = act_func
        self.downsample_mode = downsample_mode

        # build modules
        max_middle_channel = make_divisible(
            round(max(self.out_channel_list) * max(self.expand_ratio_list)),
            MyNetwork.CHANNEL_DIVISIBLE,
        )

        self.conv1 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        DynamicConv2d(
                            max(self.in_channel_list),
                            max_middle_channel,
                            kernel_size,
                            stride,
                        ),
                    ),
                    ("bn", DynamicBatchNorm2d(max_middle_channel)),
                    ("act", build_activation(self.act_func, inplace=True)),
                ]
            )
        )
        self.conv2 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        DynamicConv2d(
                            max_middle_channel, max(self.out_channel_list), kernel_size,
                        ),
                    ),
                    ("bn", DynamicBatchNorm2d(max(self.out_channel_list))),
                    ("act", build_activation(self.act_func, inplace=True)),
                ]
            )
        )

        # For stride=1 and same input/output channels, use IdentityLayer to match static subnet behavior
        # This is critical for ensuring supernet forward matches extracted static subnet forward
        if self.stride == 1 and self.in_channel_list == self.out_channel_list:
            self.downsample = IdentityLayer(
                max(self.in_channel_list), max(self.out_channel_list)
            )
        elif self.downsample_mode == "avgpool_conv":
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "avg_pool",
                            nn.AvgPool2d(
                                kernel_size=stride,
                                stride=stride,
                                padding=0,
                                ceil_mode=True,
                            ),
                        ),
                        (
                            "conv",
                            DynamicConv2d(
                                max(self.in_channel_list), max(self.out_channel_list), kernel_size=1
                            ),
                        ),
                        ("bn", DynamicBatchNorm2d(max(self.out_channel_list))),
                    ]
                )
            )
        else:
            raise NotImplementedError


        self.final_act = build_activation(self.act_func, inplace=True)
        self.active_expand_ratio = max(self.expand_ratio_list)
        self.active_out_channel = max(self.out_channel_list)

        if self.bn_gamma_zero_init:
            nn.init.constant_(self.conv2.bn.bn.weight, 0)

    def forward(self, x):
        feature_dim = self.active_middle_channels
        self.conv1.conv.active_out_channel = feature_dim
        self.conv2.conv.active_out_channel = self.active_out_channel
        if not isinstance(self.downsample, IdentityLayer):
            self.downsample.conv.active_out_channel = self.active_out_channel

        residual = self.downsample(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + residual
        x = self.final_act(x)
        return x

    @property
    def module_str(self):
        return "(%s, %s)" % (
            "%dx%d_Residual_in->%d->%d_S%d"
            % (
                self.kernel_size,
                self.kernel_size,
                self.active_middle_channels,
                self.active_out_channel,
                self.stride,
            ),
            "Identity"
            if isinstance(self.downsample, IdentityLayer)
            else self.downsample_mode,
        )

    @property
    def config(self):
        return {
            "name": DynamicResidualBlock.__name__,
            "in_channel_list": self.in_channel_list,
            "out_channel_list": self.out_channel_list,
            "expand_ratio_list": self.expand_ratio_list,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "act_func": self.act_func,
            "downsample_mode": self.downsample_mode,
        }

    @staticmethod
    def build_from_config(config):
        return DynamicResidualBlock(**config)

    @property
    def in_channels(self):
        return max(self.in_channel_list)

    @property
    def out_channels(self):
        return max(self.out_channel_list)

    @property
    def active_middle_channels(self):
        feature_dim = round(self.active_out_channel * self.active_expand_ratio)
        feature_dim = make_divisible(feature_dim, MyNetwork.CHANNEL_DIVISIBLE)
        return feature_dim

    def get_active_subnet(self, in_channel, preserve_weight=True):
        # build the new layer
        sub_layer = NewResidualBlock.build_from_config(
            self.get_active_subnet_config(in_channel)
        )
        sub_layer = sub_layer.to(get_net_device(self))
        if not preserve_weight:
            return sub_layer

        # copy weight from current layer
        sub_layer.conv1.conv.weight.data.copy_(
            self.conv1.conv.get_active_filter(
                self.active_middle_channels, in_channel
            ).data
        )
        copy_bn(sub_layer.conv1.bn, self.conv1.bn.bn)

        sub_layer.conv2.conv.weight.data.copy_(
            self.conv2.conv.get_active_filter(
                self.active_out_channel, self.active_middle_channels
            ).data
        )
        copy_bn(sub_layer.conv2.bn, self.conv2.bn.bn)

        if not isinstance(self.downsample, IdentityLayer) and not isinstance(sub_layer.downsample, IdentityLayer):
            sub_layer.downsample.conv.weight.data.copy_(
                self.downsample.conv.get_active_filter(
                    self.active_out_channel, in_channel
                ).data
            )
            copy_bn(sub_layer.downsample.bn, self.downsample.bn.bn)

        return sub_layer

    def get_active_subnet_config(self, in_channel):
        # If the supernet block has a real downsample (not IdentityLayer),
        # force the static subnet to also have a real downsample even when
        # in_channels == out_channels. This preserves the learned projection.
        has_real_downsample = not isinstance(self.downsample, IdentityLayer)
        return {
            "in_channels": in_channel,
            "out_channels": self.active_out_channel,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "expand_ratio": self.active_expand_ratio,
            "mid_channels": self.active_middle_channels,
            "act_func": self.act_func,
            "groups": 1,
            "downsample_mode": self.downsample_mode,
            "force_downsample": has_real_downsample,
        }

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        raise NotImplementedError

class GenericStaticResNetSubnet(MyNetwork): # Analogous to ResNets32x32_10_26
    def __init__(self, input_stem, blocks, classifier):
        super(GenericStaticResNetSubnet, self).__init__()
        self.input_stem = nn.ModuleList(input_stem)
        self.blocks = nn.ModuleList(blocks)
        self.global_avg_pool = MyGlobalAvgPool2d(keep_dim=False)
        self.classifier = classifier

    def forward(self, x):
        for layer in self.input_stem:
            x = layer(x)
        for block in self.blocks:
            x = block(x)
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return x

    @property
    def config(self):
        # TODO: Implement this to reflect the specific static configuration
        return {
            'name': GenericStaticResNetSubnet.__name__,
            'bn': self.get_bn_param(),
            'input_stem': [layer.config for layer in self.input_stem],
            'blocks': [block.config for block in self.blocks],
            'classifier': self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        # TODO: Implement this thoroughly
        input_stem = [set_layer_from_config(cfg) for cfg in config.get('input_stem', [])]
        blocks = [NewResidualBlock.build_from_config(cfg) for cfg in config.get('blocks', [])] # Assuming NewResidualBlock is for static
        classifier = set_layer_from_config(config['classifier'])
        
        net = GenericStaticResNetSubnet(input_stem, blocks, classifier)
        if 'bn' in config:
            net.set_bn_param(**config['bn'])
        return net


class GenericOFAResNet(MyNetwork): # Analogous to OFAResNets32x32_10_26
    def __init__(self,
                 num_stages: int,
                 initial_input_hw: int, # Not directly used in layer construction, but for info
                 initial_input_channels: int,
                 stem_stride: int,
                 original_stem_out_channels: int,
                 original_stage_base_channels: list, # list of ints
                 stage_downsample_factors: list, # list of ints (strides)
                 max_extra_blocks_per_stage: int,
                 channel_divisible_by: int,
                 width_multiplier_choices: list, # list of floats (global reference list)
                 expansion_ratio_choices: list, # list of floats
                 n_classes=1000,
                 bn_param=(0.1, 1e-5), # (momentum, eps)
                 dropout_rate=0,
                 bn_gamma_zero_init=False,
                 act_func='relu', # Default activation
                 per_position_max_w_indices: list = None, # Optional: [stem_max, s0_max, s1_max, ...] for sub-supernets
                 per_position_max_d: list = None,         # Optional: [max_d_s0, max_d_s1, ...] for sub-supernets
                ):
        super(GenericOFAResNet, self).__init__()

        # Store architecture definition parameters
        self.num_stages = num_stages
        self.initial_input_hw = initial_input_hw
        self.initial_input_channels = initial_input_channels
        self.stem_stride = stem_stride
        self.original_stem_out_channels = original_stem_out_channels
        self.original_stage_base_channels = val2list(original_stage_base_channels, self.num_stages)
        self.stage_downsample_factors = val2list(stage_downsample_factors, self.num_stages)
        self.max_extra_blocks_per_stage = max_extra_blocks_per_stage  # Global max (e_indices stride reference)
        self.channel_divisible_by = channel_divisible_by
        self.width_multiplier_choices = sorted(list(set(val2list(width_multiplier_choices))))
        self.expansion_ratio_choices = sorted(list(set(val2list(expansion_ratio_choices))))
        self.n_classes = n_classes
        self.bn_gamma_zero_init = bn_gamma_zero_init
        self.act_func = act_func
        self.activation_checkpointing = False

        # Per-stage depth limits -- the per-position depth/width bounds d*_i and
        # w*_i of Supplementary Eq. sub_supernet_bounds. For the full supernet
        # all stages share max_extra_blocks_per_stage. For a sub-supernet, each
        # stage may have a smaller limit so that we only allocate weight tensors
        # for blocks that budget-constrained clients can ever use.
        if per_position_max_d is not None:
            assert len(per_position_max_d) == self.num_stages, \
                f"per_position_max_d must have {self.num_stages} entries (one per stage)"
            for si, md in enumerate(per_position_max_d):
                assert md <= max_extra_blocks_per_stage, \
                    f"per_position_max_d[{si}]={md} exceeds global max_extra_blocks_per_stage={max_extra_blocks_per_stage}"
            self.max_extra_blocks_per_stage_list = list(per_position_max_d)
        else:
            self.max_extra_blocks_per_stage_list = [max_extra_blocks_per_stage] * self.num_stages

        # Compute per-position channel lists.
        # When per_position_max_w_indices is provided (sub-supernet mode), each position
        # only allocates channels up to its own max index — not the global max.
        # This is the key fix: the stem needing index 9 no longer forces Stage 3 to
        # allocate 512x1.0 channels when it only needs up to 512x0.3.
        if per_position_max_w_indices is not None:
            assert len(per_position_max_w_indices) == self.num_stages + 1, \
                f"per_position_max_w_indices must have {self.num_stages + 1} entries (stem + {self.num_stages} stages)"
            stem_width_choices = self.width_multiplier_choices[:per_position_max_w_indices[0] + 1]
            stage_width_choices_list = [
                self.width_multiplier_choices[:per_position_max_w_indices[si + 1] + 1]
                for si in range(self.num_stages)
            ]
        else:
            stem_width_choices = self.width_multiplier_choices
            stage_width_choices_list = [self.width_multiplier_choices] * self.num_stages

        # Stem channel list: channels for each valid stem width choice
        self.stem_out_channel_list = [
            make_divisible(self.original_stem_out_channels * w, self.channel_divisible_by)
            for w in stem_width_choices
        ]
        if not self.stem_out_channel_list:
            self.stem_out_channel_list = [self.original_stem_out_channels]

        # Per-stage channel lists: each stage independently sized to its own max width
        self.stage_out_channel_lists = [
            [
                make_divisible(self.original_stage_base_channels[si] * w, self.channel_divisible_by)
                for w in stage_width_choices_list[si]
            ]
            for si in range(self.num_stages)
        ]
        for si in range(self.num_stages):
            if not self.stage_out_channel_lists[si]:
                self.stage_out_channel_lists[si] = [self.original_stage_base_channels[si]]

        # Input Stem
        self.input_stem = nn.ModuleList([
            DynamicConvLayer(
                in_channel_list=val2list(self.initial_input_channels),
                out_channel_list=self.stem_out_channel_list,
                kernel_size=3, # Assuming 3x3 stem, make configurable if needed
                stride=self.stem_stride,
                act_func=self.act_func,
                use_bn=True
            )
        ])

        # Blocks — each stage uses its own per-position channel and depth lists.
        # self.max_extra_blocks_per_stage_list[i] controls how many blocks are allocated
        # for stage i; this is <= max_extra_blocks_per_stage for sub-supernets.
        self.blocks = nn.ModuleList()
        current_max_in_channels_list = self.stem_out_channel_list # Input to the first stage

        for i in range(self.num_stages):
            stage_max_out_channels_list = self.stage_out_channel_lists[i]

            for block_idx in range(self.max_extra_blocks_per_stage_list[i] + 1):
                stride = self.stage_downsample_factors[i] if block_idx == 0 else 1

                # For the first block of a stage, in_channel_list is from the previous stage/stem.
                # For subsequent blocks within the same stage, in_channel_list is the current stage's out_channel_list.
                block_in_channels_list = current_max_in_channels_list if block_idx == 0 else stage_max_out_channels_list

                residual_block = DynamicResidualBlock( # Using the block from ofa_resnets_32x32_10_26
                    in_channel_list=block_in_channels_list,
                    out_channel_list=stage_max_out_channels_list,
                    expand_ratio_list=self.expansion_ratio_choices,
                    kernel_size=3, # Assuming 3x3, make configurable if needed
                    stride=stride,
                    act_func=self.act_func,
                    bn_gamma_zero_init=self.bn_gamma_zero_init,
                    # downsample_mode can be 'avgpool_conv' or 'conv'
                )
                self.blocks.append(residual_block)
            current_max_in_channels_list = stage_max_out_channels_list # Output of this stage is input to next

        # Global Average Pool
        self.global_avg_pool = MyGlobalAvgPool2d(keep_dim=False)

        # Classifier
        self.classifier = DynamicLinearLayer(
            in_features_list=current_max_in_channels_list, # From the last stage
            out_features=self.n_classes,
            dropout_rate=dropout_rate
        )

        # Runtime depth: stores how many blocks to *skip* from the end of each stage's definition.
        # Initialised to the per-stage max (will be overwritten by set_max_net below).
        self.runtime_depth = list(self.max_extra_blocks_per_stage_list)
        self.set_bn_param(*bn_param)
        self.set_max_net() # Initialize to max network

    def set_activation_checkpointing(self, enabled: bool = True):
        self.activation_checkpointing = bool(enabled)

    @staticmethod
    def _bn_running_buffers(module):
        for m in module.modules():
            buffers = {}
            for name in ("running_mean", "running_var", "num_batches_tracked"):
                value = getattr(m, name, None)
                if value is not None:
                    buffers[name] = value.clone()
            if buffers:
                yield m, buffers

    @classmethod
    def _checkpoint_contexts(cls, module):
        saved_bn_state = {}

        @contextmanager
        def forward_context():
            try:
                yield
            finally:
                saved_bn_state.clear()
                for bn_module, buffers in cls._bn_running_buffers(module):
                    saved_bn_state[bn_module] = buffers

        @contextmanager
        def recompute_context():
            try:
                yield
            finally:
                for bn_module, buffers in saved_bn_state.items():
                    for name, value in buffers.items():
                        getattr(bn_module, name).copy_(value)

        return forward_context(), recompute_context()

    @property
    def grouped_block_index(self):
        # Returns per-stage lists of block indices within self.blocks.
        # Each stage i has (max_extra_blocks_per_stage_list[i] + 1) block slots,
        # which may be less than the global max for sub-supernets with per_position_max_d.
        grouped_indexes = []
        current_idx = 0
        for si in range(self.num_stages):
            n = self.max_extra_blocks_per_stage_list[si] + 1
            grouped_indexes.append(list(range(current_idx, current_idx + n)))
            current_idx += n
        return grouped_indexes

    def set_active_subnet(self, d: list, e_indices: list, w_indices: list, **kwargs):
        """
        Sets (d, e_indices, w_indices) as defined by the elastic search space of
        Supplementary Eq. elastic_search_space (Sec. B): d_l in {0..max_extra_blocks},
        e_l in expansion_ratio_choices per stage, w_j in width_multiplier_choices
        per stem/stage position.

        d: list of extra block counts for each stage. len(d) == num_stages. d[i] in [0, max_extra_blocks_per_stage]
        e_indices: list of indices into self.expansion_ratio_choices, one per stage.
                   Expected length: num_stages. All blocks within a stage share the same expansion ratio.
        w_indices: list of indices into the per-position channel lists (stem + stages).
                   len(w_indices) == num_stages + 1.
                   w_indices[0] indexes self.stem_out_channel_list.
                   w_indices[i+1] indexes self.stage_out_channel_lists[i].
                   For the full supernet all lists are the same (global width_multiplier_choices).
                   For sub-supernets each list is truncated to that position's max index.
        """
        if len(d) != self.num_stages:
            raise ValueError(f"Depth vector 'd' length {len(d)} != num_stages {self.num_stages}")
        if len(w_indices) != self.num_stages + 1:
            raise ValueError(f"Width_indices vector 'w_indices' length {len(w_indices)} != num_stages+1 {self.num_stages + 1}")

        # e_indices: one per stage (stage-level expansion ratios)
        if len(e_indices) != self.num_stages:
            raise ValueError(f"Expansion_indices vector 'e_indices' length {len(e_indices)} != num_stages {self.num_stages}")

        # Validate per-position width bounds
        if not (0 <= w_indices[0] < len(self.stem_out_channel_list)):
            raise ValueError(f"w_indices[0]={w_indices[0]} out of range for stem (max={len(self.stem_out_channel_list)-1})")
        for si in range(self.num_stages):
            if not (0 <= w_indices[si + 1] < len(self.stage_out_channel_lists[si])):
                raise ValueError(f"w_indices[{si+1}]={w_indices[si+1]} out of range for stage {si} (max={len(self.stage_out_channel_lists[si])-1})")

        # Set active stem output channel (indexed into stem's own channel list)
        stem_w_idx = w_indices[0]
        self.input_stem[0].active_out_channel = self.stem_out_channel_list[stem_w_idx]

        for stage_id in range(self.num_stages):
            # Per-stage depth limit for this supernet (< global max for sub-supernets)
            stage_max_d = self.max_extra_blocks_per_stage_list[stage_id]
            num_extra_blocks_active = d[stage_id]

            if not (0 <= num_extra_blocks_active <= stage_max_d):
                raise ValueError(
                    f"Depth choice d[{stage_id}]={num_extra_blocks_active} out of range "
                    f"[0, {stage_max_d}] for this (sub-)supernet"
                )

            # Blocks to skip = (stage's allocated max) - (active count)
            self.runtime_depth[stage_id] = stage_max_d - num_extra_blocks_active

            # Look up channel count from this stage's own channel list
            stage_w_idx = w_indices[stage_id + 1]
            stage_active_out_channel = self.stage_out_channel_lists[stage_id][stage_w_idx]
            stage_active_out_channel = max(stage_active_out_channel, self.channel_divisible_by)

            stage_block_indices = self.grouped_block_index[stage_id]  # indices into self.blocks
            num_blocks_allocated_this_stage = stage_max_d + 1  # blocks actually in self.blocks

            # Stage-level expansion ratio: one index per stage, shared by all blocks
            expansion_idx_for_stage = e_indices[stage_id]
            if not (0 <= expansion_idx_for_stage < len(self.expansion_ratio_choices)):
                raise ValueError(
                    f"e_indices[{stage_id}]={expansion_idx_for_stage} "
                    f"out of range for expansion_ratio_choices (len {len(self.expansion_ratio_choices)})"
                )
            stage_expand_ratio = self.expansion_ratio_choices[expansion_idx_for_stage]

            for i in range(num_blocks_allocated_this_stage):
                block_supernet_idx = stage_block_indices[i]
                current_block = self.blocks[block_supernet_idx]
                current_block.active_out_channel = stage_active_out_channel
                # All blocks in this stage share the same expansion ratio
                current_block.active_expand_ratio = stage_expand_ratio


        # Set active features for the classifier using the last stage's own channel list
        final_stage_w_idx = w_indices[self.num_stages]  # w_indices[-1]
        final_stage_active_out_channel = self.stage_out_channel_lists[-1][final_stage_w_idx]
        final_stage_active_out_channel = max(final_stage_active_out_channel, self.channel_divisible_by)

        self.classifier.active_in_features = final_stage_active_out_channel

    def set_max_net(self):
        # d_max respects per-stage limits (full supernet: all equal global max)
        d_max = list(self.max_extra_blocks_per_stage_list)
        # e_indices: one per stage, all at max expansion index
        e_indices_max = [len(self.expansion_ratio_choices) - 1] * self.num_stages
        # Per-position max width: each position uses its own list's last index
        w_indices_max = (
            [len(self.stem_out_channel_list) - 1] +
            [len(self.stage_out_channel_lists[si]) - 1 for si in range(self.num_stages)]
        )
        self.set_active_subnet(d=d_max, e_indices=e_indices_max, w_indices=w_indices_max)

    def forward(self, x):
        # Stem
        for layer in self.input_stem:
            x = layer(x)

        use_checkpointing = (
            self.activation_checkpointing
            and self.training
            and torch.is_grad_enabled()
        )
        if use_checkpointing:
            from torch.utils.checkpoint import checkpoint

        # Blocks — use per-stage allocated max to compute active count
        all_stage_indices = self.grouped_block_index
        for stage_id in range(self.num_stages):
            stage_max_d = self.max_extra_blocks_per_stage_list[stage_id]
            num_active_blocks_this_stage = (stage_max_d + 1) - self.runtime_depth[stage_id]
            active_block_indices_this_stage = all_stage_indices[stage_id][:num_active_blocks_this_stage]
            
            for block_idx_in_supernet in active_block_indices_this_stage:
                block = self.blocks[block_idx_in_supernet]
                if use_checkpointing:
                    x = checkpoint(
                        block,
                        x,
                        use_reentrant=False,
                        context_fn=lambda block=block: self._checkpoint_contexts(block),
                    )
                else:
                    x = block(x)
        
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return x

    def get_active_subnet(self, preserve_weight=True):
        """Extracts the currently active configuration as a standalone static
        GenericStaticResNetSubnet, optionally copying the corresponding weight
        slices (Supplementary Eq. conv_slice: leading-dimension channel slicing)."""
        # 1. Get active stem
        active_stem_layers = [self.input_stem[0].get_active_subnet(self.initial_input_channels, preserve_weight)]
        current_active_channels = self.input_stem[0].active_out_channel

        active_blocks = []
        all_supernet_stage_indices = self.grouped_block_index

        for stage_id in range(self.num_stages):
            stage_max_d = self.max_extra_blocks_per_stage_list[stage_id]
            num_active_blocks_this_stage = (stage_max_d + 1) - self.runtime_depth[stage_id]
            supernet_indices_for_active_blocks = all_supernet_stage_indices[stage_id][:num_active_blocks_this_stage]

            for block_supernet_idx in supernet_indices_for_active_blocks:
                supernet_block = self.blocks[block_supernet_idx]
                # The get_active_subnet of DynamicResidualBlock needs the *actual* active input channel count
                static_block = supernet_block.get_active_subnet(current_active_channels, preserve_weight)
                active_blocks.append(static_block)
                current_active_channels = supernet_block.active_out_channel # This should be the out_channel for this stage

        active_classifier = self.classifier.get_active_subnet(current_active_channels, preserve_weight)
        
        subnet = GenericStaticResNetSubnet(active_stem_layers, active_blocks, active_classifier)
        subnet.set_bn_param(**self.get_bn_param())
        return subnet

    @property
    def config(self):
        # TODO: Implement this to represent the supernetwork's full potential configuration
        return {
            'name': GenericOFAResNet.__name__,
            'bn': self.get_bn_param(),
            'num_stages': self.num_stages,
            'original_stage_base_channels': self.original_stage_base_channels,
            'input_stem': [layer.config for layer in self.input_stem],
            'blocks': [block.config for block in self.blocks],
            'classifier': self.classifier.config,
        }

    def sample_active_subnet(self):
        """ Samples a random configuration for d, e_indices, w_indices """
        
        # Sample d: per-stage, each within its own allocated max depth
        d_setting = [
            np.random.randint(0, self.max_extra_blocks_per_stage_list[si] + 1)
            for si in range(self.num_stages)
        ]

        # Sample e_indices: one per stage (stage-level expansion ratios)
        e_indices_setting = [
            np.random.randint(0, len(self.expansion_ratio_choices))
            for _ in range(self.num_stages)
        ]
        
        # Sample w_indices: per-position, each within its own channel list's range
        w_indices_setting = (
            [np.random.randint(0, len(self.stem_out_channel_list))] +
            [np.random.randint(0, len(self.stage_out_channel_lists[si])) for si in range(self.num_stages)]
        )
        
        arch_config = {
            "d": d_setting,
            "e_indices": e_indices_setting,
            "w_indices": w_indices_setting,
        }
        self.set_active_subnet(**arch_config)
        return arch_config # Return the configuration dict

    def get_active_net_config(self):
        # This should return the config of the currently *active* subnet,
        # in a format that GenericStaticResNetSubnet.build_from_config can understand.

        # 1. Get active stem config
        # DynamicConvLayer.get_active_subnet_config(in_channel)
        active_stem_configs = [self.input_stem[0].get_active_subnet_config(self.initial_input_channels)]
        current_active_channels = self.input_stem[0].active_out_channel

        active_block_configs = []
        all_supernet_stage_indices = self.grouped_block_index

        for stage_id in range(self.num_stages):
            stage_max_d = self.max_extra_blocks_per_stage_list[stage_id]
            num_active_blocks_this_stage = (stage_max_d + 1) - self.runtime_depth[stage_id]
            supernet_indices_for_active_blocks = all_supernet_stage_indices[stage_id][:num_active_blocks_this_stage]

            for block_supernet_idx in supernet_indices_for_active_blocks:
                supernet_block = self.blocks[block_supernet_idx]
                # DynamicResidualBlock.get_active_subnet_config(in_channel)
                block_config = supernet_block.get_active_subnet_config(current_active_channels)
                active_block_configs.append(block_config)
                current_active_channels = supernet_block.active_out_channel

        # DynamicLinearLayer.get_active_subnet_config(in_features)
        active_classifier_config = self.classifier.get_active_subnet_config(current_active_channels)
        
        return {
            'name': GenericStaticResNetSubnet.__name__, # Target static class
            'bn': self.get_bn_param(),
            'input_stem': active_stem_configs,
            'blocks': active_block_configs,
            'classifier': active_classifier_config,
        }
    
    def set_min_net(self):
        d_min = [0] * self.num_stages  # Smallest depth: 0 extra blocks
        e_indices_min = [0] * self.num_stages  # one per stage, all at min expansion index
        w_indices_min = [0] * (self.num_stages + 1)
        self.set_active_subnet(d=d_min, e_indices=e_indices_min, w_indices=w_indices_min)
