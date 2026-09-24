"""
Dual-stream deepfake detector:
  Stream A: pretrained EfficientNet-B0 / ResNet50 on the RGB image (spatial artifacts:
            blending boundaries, inconsistent lighting, warped geometry).
  Stream B: small CNN on the FFT log-magnitude spectrum (spectral artifacts:
            periodic peaks left by GAN/diffusion upsampling).
The two feature vectors are concatenated and passed to a classifier head.
"""
import torch
import torch.nn as nn
import torchvision.models as tvm


class FrequencyBranch(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):
        x = self.net(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def _build_backbone(name: str, pretrained: bool):
    if name == "efficientnet_b0":
        weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = tvm.efficientnet_b0(weights=weights)
        feature_extractor = net.features
        out_dim = 1280
    elif name == "resnet50":
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = tvm.resnet50(weights=weights)
        feature_extractor = nn.Sequential(*list(net.children())[:-2])
        out_dim = 2048
    else:
        raise ValueError(f"Unknown backbone: {name}")
    return feature_extractor, out_dim


class DualStreamDeepfakeDetector(nn.Module):
    def __init__(self, backbone="efficientnet_b0", pretrained=True,
                 freeze_early_layers=True, freq_out_dim=128, num_classes=2):
        super().__init__()
        self.backbone_name = backbone
        self.backbone_features, backbone_out_dim = _build_backbone(backbone, pretrained)
        self.backbone_pool = nn.AdaptiveAvgPool2d(1)

        if freeze_early_layers:
            children = list(self.backbone_features.children())
            cutoff = max(1, len(children) // 2)
            for child in children[:cutoff]:
                for p in child.parameters():
                    p.requires_grad = False

        self.freq_branch = FrequencyBranch(out_dim=freq_out_dim)

        self.classifier = nn.Sequential(
            nn.Linear(backbone_out_dim + freq_out_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, rgb, freq):
        feat = self.backbone_features(rgb)
        feat = self.backbone_pool(feat)
        feat = torch.flatten(feat, 1)

        freq_feat = self.freq_branch(freq)

        combined = torch.cat([feat, freq_feat], dim=1)
        return self.classifier(combined)

    def get_gradcam_target_layer(self):
        """Returns the last conv layer of the backbone, used by Grad-CAM."""
        if self.backbone_name == "efficientnet_b0":
            return self.backbone_features[-1]
        return list(self.backbone_features.children())[-1]


def build_model(cfg):
    m = cfg["model"]
    return DualStreamDeepfakeDetector(
        backbone=m["backbone"],
        pretrained=m["pretrained"],
        freeze_early_layers=m["freeze_early_layers"],
        freq_out_dim=m["freq_branch_out_dim"],
        num_classes=m["num_classes"],
    )
