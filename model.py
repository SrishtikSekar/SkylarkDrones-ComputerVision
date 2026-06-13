"""
model.py — Dual-head GCP network.

Architecture:
  Backbone : EfficientNet-B2 (pretrained ImageNet) — good accuracy/speed trade-off
             on T4.  Swap to B0 for faster iteration, B3/B4 for more accuracy.
  Neck     : Global Average Pool → shared FC(512) → BN → ReLU → Dropout
  Head 1   : Regression  — FC(2)   predicts (dx, dy) normalised to [-1, 1]
  Head 2   : Classification — FC(3)  predicts shape logits

Loss:
  total = λ_reg * WingLoss(coords) + λ_cls * CrossEntropyLoss(shape)

Wing Loss is better than L1/L2 for keypoint regression because it gives larger
gradient for small errors (where precision matters) while being robust to outliers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ── Wing Loss ─────────────────────────────────────────────────────────────────
class WingLoss(nn.Module):
    """
    Wing loss for robust keypoint regression.
    Paper: https://arxiv.org/abs/1711.06753
    """
    def __init__(self, w: float = 10.0, epsilon: float = 2.0):
        super().__init__()
        self.w = w
        self.epsilon = epsilon
        self.C = w - w * (1.0 + w / epsilon).log()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = (pred - target).abs()
        loss = torch.where(
            diff < self.w,
            self.w * torch.log1p(diff / self.epsilon),
            diff - self.C
        )
        return loss.mean()


# ── Main model ────────────────────────────────────────────────────────────────
class GCPNet(nn.Module):
    """
    Args:
        backbone_name : timm model name, e.g. 'efficientnet_b2', 'efficientnet_b0',
                        'mobilenetv3_large_100', 'resnet34'
        num_classes   : number of shape classes (3)
        pretrained    : load ImageNet weights
        dropout       : dropout before heads
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b2",
        num_classes: int = 3,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,      # remove classifier
            global_pool="avg",  # global average pooling
        )
        feat_dim = self.backbone.num_features

        # Shared neck
        self.neck = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Regression head: predict (x, y) normalised offsets in [-1, 1]
        self.reg_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 2),
            nn.Tanh(),    # bounds output to (-1, 1)
        )

        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.SiLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor):
        feat   = self.backbone(x)          # [B, feat_dim]
        neck   = self.neck(feat)           # [B, 512]
        coords = self.reg_head(neck)       # [B, 2]  in (-1, 1)
        logits = self.cls_head(neck)       # [B, num_classes]
        return coords, logits


# ── Combined loss ─────────────────────────────────────────────────────────────
class GCPLoss(nn.Module):
    def __init__(
        self,
        class_weights: torch.Tensor = None,
        lambda_reg: float = 1.0,
        lambda_cls: float = 1.0,
    ):
        super().__init__()
        self.wing     = WingLoss()
        self.ce       = nn.CrossEntropyLoss(weight=class_weights)
        self.lam_reg  = lambda_reg
        self.lam_cls  = lambda_cls

    def forward(
        self,
        pred_coords: torch.Tensor,  # [B, 2]
        pred_logits: torch.Tensor,  # [B, C]
        gt_coords:   torch.Tensor,  # [B, 2]
        gt_labels:   torch.Tensor,  # [B]
    ):
        loss_reg = self.wing(pred_coords, gt_coords)
        loss_cls = self.ce(pred_logits, gt_labels)
        total    = self.lam_reg * loss_reg + self.lam_cls * loss_cls
        return total, loss_reg, loss_cls


# ── Coordinate de-normalisation helpers ──────────────────────────────────────
def denorm_to_absolute(
    norm_coords: torch.Tensor,   # [B, 2] in [-1, 1]
    crop_offsets: torch.Tensor,  # [B, 2] = (crop_x1, crop_y1) in absolute px
    crop_size: int = 512,
) -> torch.Tensor:
    """Convert normalised crop-relative coords back to absolute image coords."""
    # norm in [-1,1] → pixel in [0, crop_size]
    px = (norm_coords + 1.0) / 2.0 * crop_size    # [B, 2]
    return px + crop_offsets                        # [B, 2]


# ── PCK evaluation metric ─────────────────────────────────────────────────────
def compute_pck(
    pred_abs: torch.Tensor,   # [N, 2]
    gt_abs:   torch.Tensor,   # [N, 2]
    thresholds=(10, 25, 50),
) -> dict:
    dists = (pred_abs - gt_abs).norm(dim=1)   # [N]
    results = {}
    for t in thresholds:
        results[f"PCK@{t}px"] = (dists <= t).float().mean().item()
    results["mean_dist_px"] = dists.mean().item()
    return results


if __name__ == "__main__":
    # Quick sanity check
    model = GCPNet("efficientnet_b2", pretrained=False)
    x = torch.randn(2, 3, 512, 512)
    coords, logits = model(x)
    print(f"coords: {coords.shape}  logits: {logits.shape}")
    # Should print: coords: torch.Size([2, 2])  logits: torch.Size([2, 3])
