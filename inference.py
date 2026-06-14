"""
inference.py — Run a trained GCPNet on the test_dataset and generate predictions.json.

Strategy: Test-Time Augmentation (TTA) with multiple crops.
  For each test image we sample N crops from a regular grid and average the
  coordinate predictions. We take the majority-vote class. This is much more
  robust than a single centre-crop because the GCP can appear anywhere in the
  high-res image.

Usage:
  python inference.py \
    --test_root  /path/to/test_dataset \
    --weights    ./runs/exp1/best_model.pth \
    --output     predictions.json \
    --backbone   efficientnet_b2 \
    --stride     300
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from dataset import SHAPE_CLASSES, CROP_SIZE, normalise_shape
from model import GCPNet


# ── Sliding-window inference ──────────────────────────────────────────────────
def get_test_transform():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def predict_image(
    model: torch.nn.Module,
    img_path: str,
    device: torch.device,
    transform,
    crop_size: int = CROP_SIZE,
    stride: int = 300,
    tta_flips: bool = True,
    batch_size: int = 8,
):
    """
    Returns:
        pred_x, pred_y : float — absolute pixel coords in original image
        shape_class    : str   — one of SHAPE_CLASSES
        confidence     : float — softmax confidence for shape
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]

    # ── Build list of crop offsets (sliding window) ───────────────────────────
    xs = list(range(0, max(1, W - crop_size + 1), stride))
    ys = list(range(0, max(1, H - crop_size + 1), stride))
    # Always include centre
    cx0 = max(0, W // 2 - crop_size // 2)
    cy0 = max(0, H // 2 - crop_size // 2)
    offsets = list(set([(x, y) for x in xs for y in ys] + [(cx0, cy0)]))

    def extract_crop(x1, y1):
        x2, y2 = x1 + crop_size, y1 + crop_size
        pad_r = max(0, x2 - W)
        pad_b = max(0, y2 - H)
        crop = img[y1:min(H, y2), x1:min(W, x2)]
        if pad_r or pad_b:
            crop = cv2.copyMakeBorder(crop, 0, pad_b, 0, pad_r,
                                      cv2.BORDER_REFLECT_101)
        return crop

    # ── Batch prediction ──────────────────────────────────────────────────────
    all_abs_x, all_abs_y, all_cls_logits = [], [], []

    def run_crops(crops_with_offsets, flip_h=False, flip_v=False):
        tensors = []
        offs = []
        for (x1, y1), crop in crops_with_offsets:
            c = crop.copy()
            if flip_h: c = c[:, ::-1, :].copy()
            if flip_v: c = c[::-1, :, :].copy()
            t = transform(image=c)["image"]
            tensors.append(t)
            offs.append((x1, y1, flip_h, flip_v))

        for i in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[i:i+batch_size]).to(device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                pred_norm, logits = model(batch)
            pred_norm = pred_norm.cpu().float()
            logits    = logits.cpu().float()

            for j, (x1, y1, fh, fv) in enumerate(offs[i:i+batch_size]):
                nx, ny = pred_norm[j].tolist()
                # un-flip
                if fh: nx = -nx
                if fv: ny = -ny
                # [-1,1] → pixel in crop → absolute
                px = (nx + 1.0) / 2.0 * crop_size + x1
                py = (ny + 1.0) / 2.0 * crop_size + y1
                all_abs_x.append(px)
                all_abs_y.append(py)
                all_cls_logits.append(logits[j])

    base_crops = [((x1, y1), extract_crop(x1, y1)) for x1, y1 in offsets]
    run_crops(base_crops, flip_h=False, flip_v=False)
    if tta_flips:
        run_crops(base_crops, flip_h=True,  flip_v=False)
        run_crops(base_crops, flip_h=False, flip_v=True)

    # ── Aggregate predictions ─────────────────────────────────────────────────
    # Use median for robustness (outlier crops won't corrupt)
    pred_x = float(np.median(all_abs_x))
    pred_y = float(np.median(all_abs_y))

    # Shape: average logits → softmax
    avg_logits = torch.stack(all_cls_logits).mean(dim=0)
    probs = F.softmax(avg_logits, dim=0)
    cls_idx = probs.argmax().item()
    shape_class = SHAPE_CLASSES[cls_idx]
    confidence  = probs[cls_idx].item()

    return pred_x, pred_y, shape_class, confidence


# ── Collect test image paths ──────────────────────────────────────────────────
def collect_test_images(test_root: str) -> List[str]:
    """Return relative paths of all JPG/jpeg images under test_root."""
    root = Path(test_root)
    rel_paths = []
    for ext in ("*.JPG", "*.jpg", "*.jpeg", "*.JPEG"):
        for p in root.rglob(ext):
            rel_paths.append(str(p.relative_to(root)))
    return sorted(rel_paths)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_root", required=True, help="Path to test_dataset folder")
    parser.add_argument("--weights",   required=True, help="Path to best_model.pth")
    parser.add_argument("--output",    default="predictions.json")
    parser.add_argument("--backbone",  default="efficientnet_b2")
    parser.add_argument("--stride",    type=int, default=300,
                        help="Sliding window stride in px. Smaller = slower but more crops")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--no_tta",    action="store_true",
                        help="Disable test-time augmentation (faster but less accurate)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    model = GCPNet(backbone_name=args.backbone, num_classes=len(SHAPE_CLASSES),
                   pretrained=False)
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)   # handle both raw and wrapped saves
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"Loaded weights from {args.weights}")

    transform = get_test_transform()

    # ── Collect images ────────────────────────────────────────────────────────
    test_images = collect_test_images(args.test_root)
    print(f"Found {len(test_images)} test images")

    # ── Predict ───────────────────────────────────────────────────────────────
    predictions = {}
    errors = []

    for rel_path in tqdm(test_images, desc="Inference"):
        img_path = os.path.join(args.test_root, rel_path)
        try:
            pred_x, pred_y, shape, conf = predict_image(
                model, img_path, device, transform,
                crop_size=CROP_SIZE,
                stride=args.stride,
                tta_flips=not args.no_tta,
                batch_size=args.batch_size,
            )
            predictions[rel_path] = {
                "mark": {"x": pred_x, "y": pred_y},
                "verified_shape": shape,
                "_confidence": round(conf, 4),   # extra debug field
            }
        except Exception as e:
            print(f"  ERROR on {rel_path}: {e}")
            errors.append(rel_path)

    print(f"\nPredicted: {len(predictions)}  Errors: {len(errors)}")

    # ── Shape distribution summary ────────────────────────────────────────────
    shape_counts = Counter(v["verified_shape"] for v in predictions.values())
    print(f"Shape distribution: {dict(shape_counts)}")

    # ── Save predictions.json (without confidence field to match required format)
    clean_preds = {}
    for k, v in predictions.items():
        clean_preds[k] = {
            "mark": v["mark"],
            "verified_shape": v["verified_shape"],
        }

    with open(args.output, "w") as f:
        json.dump(clean_preds, f, indent=2)

    # Also save debug version with confidence
    debug_path = args.output.replace(".json", "_debug.json")
    with open(debug_path, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"\nSaved predictions → {args.output}")
    print(f"Debug predictions → {debug_path}")

    if errors:
        print(f"Failed images ({len(errors)}):")
        for e in errors:
            print(f"  {e}")


if __name__ == "__main__":
    main()
