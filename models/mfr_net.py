"""Proposed Architecture: Multi-Scale Feature Rectification Network (MFR-Net).
Includes the Multi-Scale Feature Rectification (MFR) module and Uncertainty Calibration components.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .baselines import resnet18, resnet50, mobilenet_v2


class MultiScaleFeatureRectification(nn.Module):
    """Multi-Scale Feature Rectification (MFR) Module.
    
    Decomposes representations into multi-receptive context and fine spatial detail,
    employing a dynamic rectification gate to suppress high-frequency noise and
    enhance semantic boundary awareness.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid_channels = max(16, channels // reduction)

        # Contextual multi-scale branch (Dilation 1, 2, 3)
        self.branch_d1 = nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, dilation=1, groups=channels // 2, bias=False)
        self.branch_d2 = nn.Conv2d(channels, channels // 2, kernel_size=3, padding=2, dilation=2, groups=channels // 2, bias=False)
        self.fusion_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.fusion_bn = nn.BatchNorm2d(channels)

        # Dynamic channel-spatial rectification gate
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.gate_fc = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels, bias=False),
            nn.Sigmoid()
        )

        # Spatial refinement
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        # Multi-scale contextual extraction
        out_d1 = self.branch_d1(x)
        out_d2 = self.branch_d2(x)
        ctx = torch.cat([out_d1, out_d2], dim=1)
        ctx = self.fusion_bn(self.fusion_conv(ctx))

        # Channel dynamic calibration gate
        channel_weights = self.global_pool(ctx).view(b, c)
        gate = self.gate_fc(channel_weights).view(b, c, 1, 1)

        # Rectify features
        rectified = ctx * gate
        out = self.spatial_refine(rectified)
        return x + out


class MFRBlock(nn.Module):
    """Residual block integrated with Multi-Scale Feature Rectification."""
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, use_mfr: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.use_mfr = use_mfr
        if use_mfr:
            self.mfr = MultiScaleFeatureRectification(planes)
        else:
            self.mfr = nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.mfr(out)
        out += residual
        out = F.relu(out)
        return out


class MFRNet(nn.Module):
    """Multi-Scale Feature Rectification Network (MFR-Net).
    
    Equipped with MFR blocks across residual stages and an optional temperature
    scaling parameter for post-hoc/in-situ uncertainty calibration.
    """
    def __init__(
        self,
        num_classes: int = 10,
        layers: list = [2, 2, 2, 2],
        use_mfr: bool = True,
        use_temp_scale: bool = True,
        base_channels: int = 64
    ):
        super().__init__()
        self.in_planes = base_channels
        self.use_temp_scale = use_temp_scale

        # Stem convolution adapted for 32x32 image sizes
        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)

        # 4 hierarchical stages with MFR blocks
        self.layer1 = self._make_layer(base_channels, layers[0], stride=1, use_mfr=use_mfr)
        self.layer2 = self._make_layer(base_channels * 2, layers[1], stride=2, use_mfr=use_mfr)
        self.layer3 = self._make_layer(base_channels * 4, layers[2], stride=2, use_mfr=use_mfr)
        self.layer4 = self._make_layer(base_channels * 8, layers[3], stride=2, use_mfr=use_mfr)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 8, num_classes)

        # Learnable temperature scaling parameter (log_temp initialized to 0 -> T=1.0)
        if self.use_temp_scale:
            self.temperature = nn.Parameter(torch.ones(1) * 1.2)
        else:
            self.register_buffer("temperature", torch.ones(1))

        self._initialize_weights()

    def _make_layer(self, planes: int, num_blocks: int, stride: int, use_mfr: bool) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(MFRBlock(self.in_planes, planes, s, use_mfr=use_mfr))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor, apply_temperature: bool = True) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.global_pool(out)
        feat = torch.flatten(out, 1)
        logits = self.fc(feat)

        if apply_temperature and self.use_temp_scale:
            temp = torch.clamp(self.temperature, min=0.1, max=5.0)
            return logits / temp
        return logits


def build_model(
    model_name: str,
    num_classes: int,
    use_mfr: bool = True,
    use_temp_scale: bool = True,
) -> nn.Module:
    """Model factory supporting MFR-Net, ResNet baselines, and MobileNetV2."""
    model_name = model_name.lower().replace("-", "_")

    if model_name in ["mfr_net", "mfrnet", "ours"]:
        return MFRNet(
            num_classes=num_classes,
            use_mfr=use_mfr,
            use_temp_scale=use_temp_scale
        )
    elif model_name == "resnet18":
        return resnet18(num_classes=num_classes)
    elif model_name == "resnet50":
        return resnet50(num_classes=num_classes)
    elif model_name in ["mobilenetv2", "mobilenet_v2"]:
        return mobilenet_v2(num_classes=num_classes)
    else:
        raise ValueError(
            f"Unknown model name '{model_name}'. "
            f"Available: ['mfr_net', 'resnet18', 'resnet50', 'mobilenet_v2']"
        )
