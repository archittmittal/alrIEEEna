"""
model.py — Build a timm-based classifier for 397-class image classification.
"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import config


# ─────────────────────────────────────────────────────────────
# 1. Main model class
# ─────────────────────────────────────────────────────────────

class HackathonClassifier(nn.Module):
    """
    Wraps any timm backbone with a custom classification head.

    Architecture:
        backbone  →  Global Avg Pool (done inside timm)
                  →  Dropout(0.3)
                  →  Linear(feat_dim → 512)  →  GELU  →  Dropout(0.2)
                  →  Linear(512 → num_classes)

    The two-layer head gives the model a learnable bottleneck and helps
    avoid the common pitfall of a single linear layer snapping to the
    pretrained distribution too quickly.
    """

    def __init__(
        self,
        backbone_name: str,
        num_classes: int = config.NUM_CLASSES,
        pretrained: bool = True,
        drop_rate: float = 0.3,
    ):
        super().__init__()
        self.backbone_name = backbone_name

        # Create backbone WITHOUT its original classification head
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,              # remove pretrained head
            global_pool="avg",          # global average pooling
        )
        feat_dim = self.backbone.num_features

        # Custom classification head
        self.head = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )

        # Initialize the custom head weights properly
        self._init_head()

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)     # (B, feat_dim)
        logits   = self.head(features)  # (B, num_classes)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Useful for feature extraction / visualisation."""
        return self.backbone(x)


# ─────────────────────────────────────────────────────────────
# 2. Label-smoothed cross-entropy (standalone)
# ─────────────────────────────────────────────────────────────

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy with label smoothing.
    Prevents the model from becoming over-confident.
    smoothing=0 → standard CE.
    """

    def __init__(self, smoothing: float = 0.1, weight: torch.Tensor = None):
        super().__init__()
        self.smoothing = smoothing
        self.weight    = weight          # optional class weights for imbalance

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Smooth targets
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = (-smooth_targets * log_probs).sum(dim=-1)   # (B,)

        if self.weight is not None:
            # Apply per-sample class weight
            w = self.weight[targets]
            loss = loss * w

        return loss.mean()


# ─────────────────────────────────────────────────────────────
# 3. Factory helper
# ─────────────────────────────────────────────────────────────

def build_model(backbone_name: str, pretrained: bool = True) -> HackathonClassifier:
    """Convenience wrapper used by train.py."""
    model = HackathonClassifier(
        backbone_name=backbone_name,
        num_classes=config.NUM_CLASSES,
        pretrained=pretrained,
    )
    total_params   = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {backbone_name}")
    print(f"        Total params:     {total_params:,}")
    print(f"        Trainable params: {trainable_params:,}")
    return model
