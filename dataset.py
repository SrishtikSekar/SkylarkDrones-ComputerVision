"""
dataset.py — GCP dataset with patch-based sampling.

Strategy:
  - Images are high-res (up to ~4000×3000). Loading full images per step is slow.
  - We crop a 512×512 patch centred on the GCP (with random jitter during training).
  - Regress the (Δx, Δy) offset of the GCP centre within the 512×512 patch
    (normalised to [-1, 1] so the regression head has a bounded target).
  - Classify shape from the same patch.
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Label helpers ─────────────────────────────────────────────────────────────
SHAPE_CLASSES = ["Cross", "Square", "L-Shape"]   # canonical names
SHAPE_TO_IDX  = {s: i for i, s in enumerate(SHAPE_CLASSES)}

def normalise_shape(raw: str) -> str:
    """Unify label variants (e.g. 'L-Shaped' → 'L-Shape')."""
    raw = raw.strip()
    if "l-shape" in raw.lower():
        return "L-Shape"
    if "square" in raw.lower():
        return "Square"
    if "cross" in raw.lower():
        return "Cross"
    return raw  # fallback


# ── Transforms ───────────────────────────────────────────────────────────────
CROP_SIZE = 512   # patch size fed to the model

def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.6),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))

def get_val_transforms():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))


# ── Core dataset ──────────────────────────────────────────────────────────────
class GCPDataset(Dataset):
    """
    Returns:
      image  : FloatTensor [3, CROP_SIZE, CROP_SIZE]
      coords : FloatTensor [2]   — (dx, dy) normalised to [-1, 1] within crop
      label  : LongTensor  scalar — shape class index
      meta   : dict        — raw path, absolute centre for evaluation
    """

    def __init__(
        self,
        data_root: str,
        annotations: Dict,
        transform=None,
        jitter: int = 100,      # random offset of crop centre during training
        crop_size: int = CROP_SIZE,
        is_train: bool = True,
    ):
        self.data_root  = Path(data_root)
        self.transform  = transform
        self.jitter     = jitter if is_train else 0
        self.crop_size  = crop_size
        self.is_train   = is_train

        # Filter out bad entries
        self.samples: List[Tuple[str, float, float, int]] = []
        for rel_path, ann in annotations.items():
            shape = ann.get("verified_shape")
            if not shape:
                continue
            shape = normalise_shape(shape)
            if shape not in SHAPE_TO_IDX:
                continue
            x, y = ann["mark"]["x"], ann["mark"]["y"]
            self.samples.append((rel_path, x, y, SHAPE_TO_IDX[shape]))

    def __len__(self):
        return len(self.samples)

    def _load_crop(self, img_path: Path, cx: float, cy: float):
        """Load image and extract a crop centred near (cx, cy)."""
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]

        # Random jitter of crop centre
        jx = random.randint(-self.jitter, self.jitter) if self.jitter else 0
        jy = random.randint(-self.jitter, self.jitter) if self.jitter else 0
        crop_cx = cx + jx
        crop_cy = cy + jy

        half = self.crop_size // 2
        x1 = int(round(crop_cx)) - half
        y1 = int(round(crop_cy)) - half
        x2 = x1 + self.crop_size
        y2 = y1 + self.crop_size

        # Clamp to image boundaries and pad if needed
        pad_left   = max(0, -x1)
        pad_top    = max(0, -y1)
        pad_right  = max(0, x2 - W)
        pad_bottom = max(0, y2 - H)

        x1c = max(0, x1); x2c = min(W, x2)
        y1c = max(0, y1); y2c = min(H, y2)

        crop = img[y1c:y2c, x1c:x2c]
        if any([pad_left, pad_top, pad_right, pad_bottom]):
            crop = cv2.copyMakeBorder(
                crop, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_REFLECT_101
            )

        # GCP position within the crop (before augment)
        kp_x = cx - x1 + pad_left   # may be outside [0, crop_size] if jitter large
        kp_y = cy - y1 + pad_top

        return crop, kp_x, kp_y

    def __getitem__(self, idx):
        rel_path, cx, cy, label = self.samples[idx]
        img_path = self.data_root / rel_path

        crop, kp_x, kp_y = self._load_crop(img_path, cx, cy)

        if self.transform:
            transformed = self.transform(
                image=crop,
                keypoints=[(kp_x, kp_y)]
            )
            crop   = transformed["image"]          # [3, H, W] tensor
            kps    = transformed["keypoints"]
            kp_x, kp_y = kps[0] if kps else (kp_x, kp_y)

        # Normalise keypoint to [-1, 1] within crop
        norm_x = (kp_x / self.crop_size) * 2.0 - 1.0
        norm_y = (kp_y / self.crop_size) * 2.0 - 1.0

        coords = torch.tensor([norm_x, norm_y], dtype=torch.float32)
        label  = torch.tensor(label, dtype=torch.long)

        meta = {
            "rel_path": rel_path,
            "abs_cx": cx,
            "abs_cy": cy,
        }
        return crop, coords, label, meta


# ── Test / inference dataset (no labels) ─────────────────────────────────────
class GCPTestDataset(Dataset):
    """
    Slides a grid of overlapping crops over each test image and aggregates
    predictions. But for simplicity we also support single-centre-crop inference
    (pass use_grid=False) — faster and often good enough once the model is robust.
    """

    def __init__(
        self,
        data_root: str,
        image_paths: List[str],   # relative paths
        transform=None,
        crop_size: int = CROP_SIZE,
        stride: int = 256,        # used only in grid mode
        use_grid: bool = False,
    ):
        self.data_root  = Path(data_root)
        self.transform  = transform
        self.crop_size  = crop_size
        self.stride     = stride
        self.use_grid   = use_grid

        # Flatten: each item = (rel_path, crop_x1, crop_y1)
        self.items: List[Tuple[str, int, int]] = []
        for rel_path in image_paths:
            if use_grid:
                img_path = self.data_root / rel_path
                img = cv2.imread(str(img_path))
                if img is None:
                    # fallback: single centre crop
                    self.items.append((rel_path, -1, -1))
                    continue
                H, W = img.shape[:2]
                for y in range(0, H, stride):
                    for x in range(0, W, stride):
                        self.items.append((rel_path, x, y))
            else:
                self.items.append((rel_path, -1, -1))   # -1 = centre crop

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rel_path, crop_x, crop_y = self.items[idx]
        img_path = self.data_root / rel_path

        img = cv2.imread(str(img_path))
        if img is None:
            # Return black image with sentinel
            crop = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
            offset_x, offset_y = 0, 0
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H, W = img.shape[:2]

            if crop_x == -1:   # centre crop
                crop_x = max(0, W // 2 - self.crop_size // 2)
                crop_y = max(0, H // 2 - self.crop_size // 2)

            x2 = crop_x + self.crop_size
            y2 = crop_y + self.crop_size
            pad_r = max(0, x2 - W)
            pad_b = max(0, y2 - H)
            crop = img[crop_y:min(H, y2), crop_x:min(W, x2)]
            if pad_r or pad_b:
                crop = cv2.copyMakeBorder(crop, 0, pad_b, 0, pad_r,
                                          cv2.BORDER_REFLECT_101)
            offset_x, offset_y = crop_x, crop_y

        if self.transform:
            crop = self.transform(image=crop)["image"]

        return crop, rel_path, offset_x, offset_y


# ── Factory helpers ───────────────────────────────────────────────────────────
def build_dataloaders(
    data_root: str,
    json_path: str,
    val_fraction: float = 0.15,
    batch_size: int = 16,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """Split annotations, build train/val DataLoaders."""
    with open(json_path) as f:
        annotations = json.load(f)

    # Stratified split by shape
    from collections import defaultdict
    by_shape = defaultdict(list)
    for k, v in annotations.items():
        shape = v.get("verified_shape")
        if not shape:
            continue
        by_shape[normalise_shape(shape)].append(k)

    rng = random.Random(seed)
    train_keys, val_keys = [], []
    for shape, keys in by_shape.items():
        rng.shuffle(keys)
        n_val = max(1, int(len(keys) * val_fraction))
        val_keys.extend(keys[:n_val])
        train_keys.extend(keys[n_val:])

    train_ann = {k: annotations[k] for k in train_keys}
    val_ann   = {k: annotations[k] for k in val_keys}

    print(f"Train: {len(train_ann)}  Val: {len(val_ann)}")

    train_ds = GCPDataset(data_root, train_ann,
                          transform=get_train_transforms(), is_train=True)
    val_ds   = GCPDataset(data_root, val_ann,
                          transform=get_val_transforms(), is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    stats = {
        "train_size": len(train_ds),
        "val_size":   len(val_ds),
        "class_weights": _compute_class_weights(train_ann),
    }
    return train_loader, val_loader, stats


def _compute_class_weights(annotations: Dict) -> torch.Tensor:
    counts = [0] * len(SHAPE_CLASSES)
    for v in annotations.values():
        s = v.get("verified_shape")
        if not s:
            continue
        s = normalise_shape(s)
        if s in SHAPE_TO_IDX:
            counts[SHAPE_TO_IDX[s]] += 1
    total = sum(counts)
    weights = [total / (len(SHAPE_CLASSES) * c) if c > 0 else 1.0 for c in counts]
    return torch.tensor(weights, dtype=torch.float32)
