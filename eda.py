"""
EDA for GCP dataset.
Run this first to understand your data before training.
Usage: python eda.py --data_root /path/to/train_dataset --json curated_gcp_marks.json
"""

import json
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
from pathlib import Path


def run_eda(data_root: str, json_path: str, output_dir: str = "eda_outputs"):
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path) as f:
        annotations = json.load(f)

    # ── 1. Basic stats ────────────────────────────────────────────────────────
    print(f"Total annotated images : {len(annotations)}")

    # Normalize shape labels (assignment says "L-Shaped", JSON has "L-Shape")
    shape_counter = Counter()
    missing_shape = []
    for k, v in annotations.items():
        s = v.get("verified_shape")
        if not s:
            missing_shape.append(k)
        else:
            shape_counter[s] += 1

    print(f"Missing shape label    : {len(missing_shape)}")
    print(f"Shape distribution     : {dict(shape_counter)}")

    # ── 2. Coordinate stats ───────────────────────────────────────────────────
    xs = [v["mark"]["x"] for v in annotations.values()]
    ys = [v["mark"]["y"] for v in annotations.values()]
    print(f"\nX  → min={min(xs):.1f}  max={max(xs):.1f}  mean={np.mean(xs):.1f}")
    print(f"Y  → min={min(ys):.1f}  max={max(ys):.1f}  mean={np.mean(ys):.1f}")

    # ── 3. Check which files actually exist on disk ───────────────────────────
    found, missing_file = 0, []
    image_sizes = []
    for rel_path in list(annotations.keys())[:200]:   # sample 200 to be fast
        full = os.path.join(data_root, rel_path)
        if os.path.exists(full):
            found += 1
            try:
                with Image.open(full) as img:
                    image_sizes.append(img.size)   # (W, H)
            except Exception:
                pass
        else:
            missing_file.append(rel_path)

    print(f"\nFile existence check (first 200): found={found}, missing={len(missing_file)}")
    if image_sizes:
        ws = [s[0] for s in image_sizes]
        hs = [s[1] for s in image_sizes]
        print(f"Image W → {min(ws)} – {max(ws)}   H → {min(hs)} – {max(hs)}")
        size_counter = Counter(image_sizes)
        print("Most common sizes:", size_counter.most_common(5))

    # ── 4. Plots ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Class distribution
    labels, counts = zip(*shape_counter.items())
    axes[0].bar(labels, counts, color=["steelblue", "coral", "green"])
    axes[0].set_title("Shape Distribution")
    axes[0].set_ylabel("Count")

    # Keypoint heatmap (normalised)
    axes[1].scatter(xs, ys, alpha=0.1, s=1)
    axes[1].set_title("GCP Center Distribution")
    axes[1].set_xlabel("x (px)"); axes[1].set_ylabel("y (px)")
    axes[1].invert_yaxis()

    # Per-class coord spread
    for shape in shape_counter:
        sx = [v["mark"]["x"] for v in annotations.values()
              if v.get("verified_shape") == shape]
        sy = [v["mark"]["y"] for v in annotations.values()
              if v.get("verified_shape") == shape]
        axes[2].scatter(sx, sy, alpha=0.15, s=2, label=shape)
    axes[2].set_title("GCP Centers by Shape")
    axes[2].legend(); axes[2].invert_yaxis()

    plt.tight_layout()
    out = os.path.join(output_dir, "eda_summary.png")
    plt.savefig(out, dpi=120)
    print(f"\nEDA plot saved → {out}")

    # ── 5. Sample image viewer ────────────────────────────────────────────────
    sample_keys = [k for k in annotations if annotations[k].get("verified_shape")][:6]
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
    for ax, key in zip(axes2.flat, sample_keys):
        full = os.path.join(data_root, key)
        if not os.path.exists(full):
            ax.set_title("FILE NOT FOUND"); continue
        img = np.array(Image.open(full).convert("RGB"))
        cx, cy = annotations[key]["mark"]["x"], annotations[key]["mark"]["y"]
        # Show 512×512 crop centred on GCP
        H, W = img.shape[:2]
        x1 = max(0, int(cx) - 256); x2 = min(W, x1 + 512)
        y1 = max(0, int(cy) - 256); y2 = min(H, y1 + 512)
        crop = img[y1:y2, x1:x2]
        ax.imshow(crop)
        ax.plot(cx - x1, cy - y1, "r+", markersize=20, markeredgewidth=2)
        ax.set_title(f"{annotations[key]['verified_shape']}\n{os.path.basename(key)}", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    out2 = os.path.join(output_dir, "sample_crops.png")
    plt.savefig(out2, dpi=100)
    print(f"Sample crops saved → {out2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="Path to train_dataset folder")
    parser.add_argument("--json", required=True, help="Path to curated_gcp_marks.json")
    parser.add_argument("--output_dir", default="eda_outputs")
    args = parser.parse_args()
    run_eda(args.data_root, args.json, args.output_dir)
