# Skylark Drones — GCP Pose Estimation

A production-grade pipeline for detecting Ground Control Point (GCP) markers in high-resolution aerial drone imagery. Simultaneously performs keypoint localisation (predicting the pixel-centre of the marker) and shape classification (Cross / Square / L-Shape).

---

## Architecture

```
Input image (high-res, e.g. 4000×3000)
         │
         ▼
  Sliding-window crop (512×512, stride 300)
         │
         ▼
  EfficientNet-B2 backbone (ImageNet pretrained)
         │  Global Average Pool
         ▼
  Shared Neck: FC(512) → BN → SiLU → Dropout(0.3)
        / \
       /   \
      ▼     ▼
Reg Head   Cls Head
FC(256)→   FC(128)→
SiLU→      SiLU→
FC(2)→     FC(3)
Tanh       (logits)

Output: (Δx, Δy) normalised to [-1,1]  +  shape class
```

**Why this design?**

- **Patch-based**: Avoids loading full 12 MP images per step; 512×512 crops are processed in batches on T4 GPU with fp16.
- **Tanh output + Wing Loss**: Tanh bounds regression to (−1, 1) preventing coordinate explosions; Wing Loss applies stronger gradient for sub-pixel errors (better than MSE/L1 for localisation).
- **EfficientNet-B2**: Best accuracy/FLOPs trade-off in the B-series. Pretrained backbone provides strong texture/edge features critical for GCP marker detection.
- **Sliding-window inference + median aggregation**: Multiple overlapping crops cover the whole image; median is robust to crops where the GCP is near the border.
- **Class-weighted CE**: Handles class imbalance (Cross 177 / Square 328 / L-Shape 491) without oversampling.

---

## Loss Function

```
total = λ_reg × WingLoss(pred_xy, gt_xy) + λ_cls × CrossEntropy(pred_shape, gt_shape)
```

Wing Loss: `L(x) = w·ln(1 + |x|/ε)` for `|x| < w`, else `|x| − C`

Default: `λ_reg = λ_cls = 1.0`, `w = 10`, `ε = 2`.

---

## Dataset Notes (EDA findings)

| Stat | Value |
|------|-------|
| Total annotated | 1000 images |
| Classes | Cross: 177, Square: 328, L-Shape: 491 |
| Missing shape label | 4 (dropped) |
| Image resolution | Up to ~4000×3000 |
| GCP x range | ~10 – 4100 px |
| GCP y range | ~60 – 2950 px |

**Label inconsistency**: Some entries use `"L-Shaped"` while others use `"L-Shape"`. The pipeline normalises all variants to `"L-Shape"`.

**Real-world noise**: Images come from multiple survey sites with different lighting, altitude, marker degradation. Augmentation (flips, rotations, colour jitter, Gaussian noise) addresses this.

---

## Quickstart (Google Colab / Kaggle T4)

### 1. Install

```bash
pip install timm albumentations scikit-learn tqdm
```

### 2. EDA

```bash
python eda.py \
  --data_root /path/to/train_dataset \
  --json /path/to/train_dataset/curated_gcp_marks.json
```

### 3. Train

```bash
python train.py \
  --data_root /path/to/train_dataset \
  --json      /path/to/train_dataset/curated_gcp_marks.json \
  --output_dir ./runs/exp1 \
  --backbone  efficientnet_b2 \
  --epochs    50 \
  --batch_size 16 \
  --lr         3e-4 \
  --fp16
```

**T4 memory guide:**

| Backbone | Batch Size | VRAM | Speed |
|----------|-----------|------|-------|
| efficientnet_b0 | 32 | ~8 GB | ~3 min/epoch |
| efficientnet_b2 | 16 | ~11 GB | ~5 min/epoch |
| efficientnet_b3 | 12 | ~13 GB | ~7 min/epoch |

### 4. Inference

```bash
python inference.py \
  --test_root /path/to/test_dataset \
  --weights   ./runs/exp1/best_model.pth \
  --output    predictions.json \
  --backbone  efficientnet_b2 \
  --stride    300
```

`--stride 300` → ~110 crops per 4K image. Use `--stride 512` for faster inference (fewer crops, slight accuracy drop).

---

## Output Format

`predictions.json` matches the training label format exactly:

```json
{
  "project/survey/GCP-01/DJI_0001.JPG": {
    "mark": {
      "x": 2134.7,
      "y": 891.2
    },
    "verified_shape": "Cross"
  }
}
```

---

## Evaluation Metrics

- **PCK@10px / @25px / @50px** — Percentage of Correct Keypoints within threshold
- **Macro F1-Score** — Balanced across 3 shape classes
- Best model saved by **PCK@25px**

---

## Assumptions

1. The GCP marker is always visible and within the image bounds.
2. Images without a `verified_shape` label (~4 entries) are excluded from training.
3. Label variant `"L-Shape"` and `"L-Shaped"` are treated as the same class.
4. Jitter of ±100px during training crop sampling (equivalent to simulating the GCP being off-centre in the crop) improves generalisation.
5. Sliding-window with stride=300 on inference achieves sufficient coverage of 4K images.

---

## File Structure

```
gcp_pipeline/
├── eda.py           # Exploratory data analysis
├── dataset.py       # Dataset, transforms, dataloader factory
├── model.py         # GCPNet, WingLoss, GCPLoss, PCK metric
├── train.py         # Training loop with OneCycleLR, AMP, checkpointing
├── inference.py     # Sliding-window TTA inference → predictions.json
├── GCP_Pipeline.ipynb  # Colab notebook tying everything together
└── README.md
```
