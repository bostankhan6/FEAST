"""
FIARSE ResNet for CIFAR-100.

Faithful implementation following:
  Wu et al., "FIARSE: Model-Heterogeneous Federated Learning via
  Importance-Aware Submodel Extraction", NeurIPS 2024.
  Official repo: https://github.com/HarliWu/FIARSE

Architecture: ResNet-18-style adapted for CIFAR-100 (32×32 input, no stem
stride, no max-pool). Hidden sizes [64, 128, 256, 512], 2 BasicBlocks per stage.
At model_size=1.0 this gives ~555M MACs — matching our 600M training cap.

Core mechanism: Unstructured magnitude-based pruning with TCB-GD.
  - model_size = fraction of total parameters to keep (e.g., 0.25 = 25%)
  - generate_mask(): TopK over |θ| across ALL masked layers → threshold
  - Forward pass: weight * Bern(|weight|, threshold) — masked params are zeroed
  - Bern (TCB-GD): custom autograd function with biased gradient 2θ/(|x|+θ)²
    that pushes parameters near the threshold decisively above or below it
  - BN parameters are NOT masked (always fully active, shared across all sizes)

Budget mapping: model_size = budget / FULL_MODEL_MACS (linear, since FLOPs
scale linearly with parameter fraction under unstructured sparsity).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# TCB-GD: Threshold-Controlled Biased Gradient Descent
# ---------------------------------------------------------------------------

class Bern(torch.autograd.Function):
    """Binary mask with biased backward gradient (TCB-GD).

    Forward: hard threshold — returns (|scores| >= threshold).
    Backward: biased gradient 2θ/(|x|+θ)² × mask, which pushes params
    near the threshold decisively above or below it.
    """
    @staticmethod
    def forward(ctx, scores, threshold):
        ctx.save_for_backward(scores, threshold)
        return (scores >= threshold)

    @staticmethod
    def backward(ctx, grad_output):
        scores, threshold = ctx.saved_tensors
        grad = 2 * threshold / torch.pow(scores + threshold, 2)
        grad = torch.nan_to_num(grad, nan=0., posinf=0., neginf=0.)
        mask = (scores >= threshold)
        return grad_output * grad * mask, None


# ---------------------------------------------------------------------------
# Masked layers — forward applies Bern masking, BN excluded
# ---------------------------------------------------------------------------

class MaskedConv2d(nn.Conv2d):
    """Conv2d with element-wise magnitude masking via Bern TCB-GD."""
    def __init__(self, in_features, out_features, kernel_size, **kwargs):
        super().__init__(in_features, out_features, kernel_size=kernel_size,
                         **kwargs)
        self.threshold = torch.tensor(0.)
        self.bern = True

    def forward(self, x):
        threshold = self.threshold.to(self.weight.device)
        if self.bern:
            weight_mask = Bern.apply(torch.abs(self.weight), threshold)
        else:
            weight_mask = (torch.abs(self.weight) >= threshold)
        effective_weight = self.weight * weight_mask

        if self.bias is not None:
            if self.bern:
                bias_mask = Bern.apply(torch.abs(self.bias), threshold)
            else:
                bias_mask = (torch.abs(self.bias) >= threshold)
            effective_bias = self.bias * bias_mask
        else:
            effective_bias = None

        return self._conv_forward(x, effective_weight, effective_bias)

    def set_threshold(self, value, bern=True):
        self.threshold, self.bern = value, bern

    @property
    def size(self):
        return (np.prod(self.weight.shape) if self.weight is not None else 0) \
               + (np.prod(self.bias.shape) if self.bias is not None else 0)


class MaskedLinear(nn.Linear):
    """Linear layer with element-wise magnitude masking via Bern TCB-GD."""
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__(in_features, out_features, **kwargs)
        self.threshold = torch.tensor(0.)
        self.bern = True

    def forward(self, x):
        threshold = self.threshold.to(self.weight.device)
        if self.bern:
            weight_mask = Bern.apply(torch.abs(self.weight), threshold)
            bias_mask   = Bern.apply(torch.abs(self.bias), threshold)
        else:
            weight_mask = (torch.abs(self.weight) >= threshold)
            bias_mask   = (torch.abs(self.bias) >= threshold)
        effective_weight = self.weight * weight_mask
        effective_bias   = self.bias * bias_mask
        return F.linear(x, effective_weight, effective_bias)

    def set_threshold(self, value, bern=True):
        self.threshold, self.bern = value, bern

    @property
    def size(self):
        return (np.prod(self.weight.shape) if self.weight is not None else 0) \
               + (np.prod(self.bias.shape) if self.bias is not None else 0)


class MaskedBatchNorm2d(nn.BatchNorm2d):
    """BN excluded from masking (size=0). FIARSE paper: BN parameters are
    always fully active and shared across all model sizes."""
    size = 0


class MaskedSequential(nn.Sequential):
    """Sequential container that propagates set_threshold to masked children."""
    def set_threshold(self, value, bern=True):
        for module in self:
            if hasattr(module, 'set_threshold') and getattr(module, 'size', 0) != 0:
                module.set_threshold(value, bern=bern)

    @property
    def size(self):
        return sum(getattr(m, 'size', 0) for m in self)


# ---------------------------------------------------------------------------
# BasicBlock with masked layers
# ---------------------------------------------------------------------------

class MaskedBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, track_running_stats=False):
        super().__init__()
        self.conv1 = MaskedConv2d(in_planes, planes, 3, stride=stride,
                                   padding=1, bias=False)
        self.bn1 = MaskedBatchNorm2d(planes, momentum=None,
                                      track_running_stats=track_running_stats)
        self.conv2 = MaskedConv2d(planes, planes, 3, stride=1,
                                   padding=1, bias=False)
        self.bn2 = MaskedBatchNorm2d(planes, momentum=None,
                                      track_running_stats=track_running_stats)

        self.shortcut = MaskedSequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = MaskedSequential(
                MaskedConv2d(in_planes, planes, 1, stride=stride, bias=False),
                MaskedBatchNorm2d(planes, momentum=None,
                                   track_running_stats=track_running_stats),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def set_threshold(self, value, bern=True):
        self.conv1.set_threshold(value, bern)
        self.conv2.set_threshold(value, bern)
        self.shortcut.set_threshold(value, bern)

    @property
    def size(self):
        return self.conv1.size + self.conv2.size + self.shortcut.size


# ---------------------------------------------------------------------------
# FIARSEResNet — ResNet-18 with unstructured magnitude masking
# ---------------------------------------------------------------------------

HIDDEN_SIZE = [64, 128, 256, 512]
NUM_BLOCKS  = [2, 2, 2, 2]
FULL_MODEL_MACS = 555_000_000  # ~555M MACs at model_size=1.0


class FIARSEResNet(nn.Module):
    """ResNet-18 with unstructured FIARSE masking for CIFAR-100.

    All conv/linear layers use MaskedConv2d/MaskedLinear with Bern TCB-GD.
    BN layers are excluded from masking (always fully active).

    generate_mask(model_size) computes a global TopK threshold across all
    masked parameters, keeping the top `model_size` fraction active.
    """

    def __init__(self, num_classes=100, hidden_size=None, num_blocks=None,
                 track_running_stats=False, stem_stride: int = 1):
        super().__init__()
        hs = hidden_size if hidden_size is not None else HIDDEN_SIZE
        nb = num_blocks  if num_blocks  is not None else NUM_BLOCKS
        self.in_planes = hs[0]

        self.conv1 = MaskedConv2d(3, hs[0], 3, stride=stem_stride, padding=1, bias=False)
        self.bn1 = MaskedBatchNorm2d(hs[0], momentum=None,
                                      track_running_stats=track_running_stats)

        self.layer1 = self._make_layer(hs[0], nb[0], stride=1,
                                        track_running_stats=track_running_stats)
        self.layer2 = self._make_layer(hs[1], nb[1], stride=2,
                                        track_running_stats=track_running_stats)
        self.layer3 = self._make_layer(hs[2], nb[2], stride=2,
                                        track_running_stats=track_running_stats)
        self.layer4 = self._make_layer(hs[3], nb[3], stride=2,
                                        track_running_stats=track_running_stats)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = MaskedLinear(hs[3], num_classes)

        self._init_weights()

    def _make_layer(self, planes, num_blocks, stride, track_running_stats):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(MaskedBasicBlock(self.in_planes, planes, stride=s,
                                           track_running_stats=track_running_stats))
            self.in_planes = planes
        return MaskedSequential(*layers)

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

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    # ---------------------------------------------------------------
    # Mask generation and threshold management
    # ---------------------------------------------------------------

    def _get_all_masked_params(self):
        """Collect all parameters from masked layers (excluding BN) as flat vector."""
        vectors = []
        for layer in [self.conv1, self.layer1, self.layer2,
                      self.layer3, self.layer4, self.linear]:
            self._collect_param_vectors(layer, vectors)
        return torch.cat(vectors) if vectors else torch.tensor([])

    def _collect_param_vectors(self, module, vectors):
        """Recursively collect weight vectors from masked layers."""
        if isinstance(module, (MaskedConv2d, MaskedLinear)):
            vectors.append(module.weight.view(-1))
            if module.bias is not None:
                vectors.append(module.bias.view(-1))
        elif isinstance(module, (MaskedSequential, MaskedBasicBlock)):
            for child in module.children():
                self._collect_param_vectors(child, vectors)

    def generate_mask(self, model_size=1.0, topk=True, bern=True):
        """Generate unstructured mask keeping top model_size fraction of params.

        Args:
            model_size: fraction of parameters to keep (0.0 to 1.0)
            topk: if True, use TopK selection; if False, keep all (threshold=0)
            bern: if True, use Bern TCB-GD in forward; if False, hard threshold

        Returns:
            Actual fraction of parameters kept.
        """
        scores = torch.abs(self._get_all_masked_params())

        if topk and model_size < 1.0:
            numel = len(scores)
            k = max(1, int(numel * model_size))
            topk_result = torch.topk(scores, k=k)
            self.threshold = topk_result.values[-1]
            self.set_threshold(self.threshold, bern=bern)
        else:
            self.threshold = torch.tensor(0., device=scores.device)
            self.set_threshold(self.threshold, bern=bern)

        actual_size = torch.sum(scores >= self.threshold).float() / len(scores)
        return actual_size.item()

    def set_threshold(self, value, bern=True):
        """Propagate threshold to all masked layers."""
        self.conv1.set_threshold(value, bern)
        self.layer1.set_threshold(value, bern)
        self.layer2.set_threshold(value, bern)
        self.layer3.set_threshold(value, bern)
        self.layer4.set_threshold(value, bern)
        self.linear.set_threshold(value, bern)


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def build_global_model(num_classes=100, stem_stride: int = 1) -> FIARSEResNet:
    """Build the full FIARSE global model."""
    return FIARSEResNet(num_classes=num_classes, stem_stride=stem_stride)


def budget_to_model_size(budget_macs: float,
                         full_macs: float = FULL_MODEL_MACS) -> float:
    """Map a MAC budget to a model_size (parameter fraction).

    Under unstructured sparsity, FLOPs scale linearly with the fraction
    of active parameters. So model_size = budget / full_model_macs.

    Returns a float in [1/256, 1.0].
    """
    return max(1.0 / 256, min(1.0, budget_macs / full_macs))
