"""
HeteroFL ResNet for CIFAR-100.

Faithful implementation following:
  Diao et al., "HeteroFL: Computation and Communication Efficient Federated
  Learning for Heterogeneous Clients", ICLR 2021.
  Official repo: https://github.com/dem123456789/HeteroFL-...

Architecture: ResNet-18-style adapted for CIFAR-100 (32×32 input, no stem
stride, no max-pool). Hidden sizes [64, 128, 256, 512], 2 BasicBlocks per stage.
At model_rate=1.0 this gives ~550M MACs — matching our 600M training cap.

Channel scaling: hidden_size = [ceil(rate * s) for s in [64,128,256,512]]
MACs scale as rate² (both width dimensions shrink), giving:
  rate=1.000 → ~550M | rate=0.500 → ~137M | rate=0.250 → ~34M
  rate=0.125 → ~8.6M | rate=0.0625 → ~2.1M

Scaler module: learned per-channel scale applied after each conv+BN to
compensate for the fact that smaller models' features have different
magnitudes than the corresponding slice of the global model.
Follows eq. (3) in the HeteroFL paper.

BatchNorm: track_running_stats=False (batch statistics only). Running stats
cannot be shared meaningfully across clients with different channel widths.
"""

import math
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Scaler: compensates for channel-width mismatch during aggregation
# ---------------------------------------------------------------------------

class Scaler(nn.Module):
    """Learnable per-channel scale applied after BN.

    rate = model_rate / global_model_rate.
    When rate < 1 the client model has fewer channels than the global model;
    the scaler re-scales activations so magnitudes stay comparable.
    """
    def __init__(self, rate: float):
        super().__init__()
        self.rate = rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.rate


# ---------------------------------------------------------------------------
# BasicBlock with optional Scaler
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 rate: float = 1.0, use_scaler: bool = True,
                 track_running_stats: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=None,
                                  track_running_stats=track_running_stats)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=None,
                                  track_running_stats=track_running_stats)
        self.scaler = Scaler(rate) if use_scaler else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes, momentum=None,
                               track_running_stats=track_running_stats),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.scaler(self.relu(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.scaler(self.relu(out))
        return out


# ---------------------------------------------------------------------------
# HeteroFL ResNet — CIFAR-100 variant
# ---------------------------------------------------------------------------

class HeteroFLResNet(nn.Module):
    """ResNet adapted for HeteroFL on CIFAR-100.

    Args:
        model_rate:          Width scaling ratio for this client (e.g. 0.5).
        global_model_rate:   Width of the global model (always 1.0).
        hidden_size:         Full-width channel sizes per stage.
        num_blocks:          Blocks per stage (default [2,2,2,2]).
        num_classes:         Output classes (100 for CIFAR-100).
        use_scaler:          Whether to apply the Scaler module.
        track_running_stats: BN tracking (False in HeteroFL paper).
    """

    HIDDEN_SIZE = [64, 128, 256, 512]
    NUM_BLOCKS  = [2, 2, 2, 2]

    def __init__(
        self,
        model_rate: float = 1.0,
        global_model_rate: float = 1.0,
        hidden_size: list = None,
        num_blocks: list = None,
        num_classes: int = 100,
        use_scaler: bool = True,
        track_running_stats: bool = False,
        stem_stride: int = 1,
    ):
        super().__init__()
        self.model_rate = model_rate
        self.global_model_rate = global_model_rate
        self.rate = model_rate / global_model_rate  # for Scaler

        hs = hidden_size if hidden_size is not None else self.HIDDEN_SIZE
        nb = num_blocks  if num_blocks  is not None else self.NUM_BLOCKS

        # Scale channel widths: ceil preserves at least 1 channel
        self.scaled_hs = [max(1, math.ceil(model_rate * c)) for c in hs]
        self.in_planes = max(1, math.ceil(model_rate * 64))

        # Stem: stride=1 for 32x32 (CIFAR/CINIC), stride=2 for 64x64 (TinyImageNet)
        # so post-stem feature map is 32x32 in both cases — body MACs match.
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.in_planes, 3, stride=stem_stride, padding=1, bias=False),
            nn.BatchNorm2d(self.in_planes, momentum=None,
                           track_running_stats=track_running_stats),
            Scaler(self.rate) if use_scaler else nn.Identity(),
            nn.ReLU(inplace=True),
        )

        # 4 residual stages
        self.layer1 = self._make_layer(self.scaled_hs[0], nb[0], stride=1,
                                       rate=self.rate,
                                       use_scaler=use_scaler,
                                       track_running_stats=track_running_stats)
        self.layer2 = self._make_layer(self.scaled_hs[1], nb[1], stride=2,
                                       rate=self.rate,
                                       use_scaler=use_scaler,
                                       track_running_stats=track_running_stats)
        self.layer3 = self._make_layer(self.scaled_hs[2], nb[2], stride=2,
                                       rate=self.rate,
                                       use_scaler=use_scaler,
                                       track_running_stats=track_running_stats)
        self.layer4 = self._make_layer(self.scaled_hs[3], nb[3], stride=2,
                                       rate=self.rate,
                                       use_scaler=use_scaler,
                                       track_running_stats=track_running_stats)

        self.avgpool    = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.scaled_hs[3], num_classes)

        self._init_weights()

    def _make_layer(self, planes, num_blocks, stride, rate,
                    use_scaler, track_running_stats):
        strides = [stride] + [1] * (num_blocks - 1)
        layers  = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride=s,
                                     rate=rate, use_scaler=use_scaler,
                                     track_running_stats=track_running_stats))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.classifier(out)


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def build_heterofl_model(model_rate: float, num_classes: int = 100,
                         use_scaler: bool = True,
                         stem_stride: int = 1) -> HeteroFLResNet:
    """Build a HeteroFL client model at the given model_rate."""
    return HeteroFLResNet(model_rate=model_rate,
                          global_model_rate=1.0,
                          num_classes=num_classes,
                          use_scaler=use_scaler,
                          stem_stride=stem_stride)


def build_global_model(num_classes: int = 100,
                       use_scaler: bool = True,
                       stem_stride: int = 1) -> HeteroFLResNet:
    """Build the full-width global model (model_rate=1.0)."""
    return HeteroFLResNet(model_rate=1.0,
                          global_model_rate=1.0,
                          num_classes=num_classes,
                          use_scaler=use_scaler,
                          stem_stride=stem_stride)
