# Decision Log — Aerial GCP Pose Estimation

This document records the key design decisions, assumptions, and trade-offs made while building the GCP keypoint localization + shape classification pipeline.

---

## 1. Architecture — EfficientNet-B2 Dual-Head Network

**Decision**: Single shared EfficientNet-B2 backbone (ImageNet pretrained), feeding two heads:
- Regression head (Tanh-bounded) → normalized (Δx, Δy) keypoint offset
- Classification head → 3-way shape logits (Cross / Square / L-Shape)

**Rationale**: A shared backbone is lightweight and lets both tasks benefit from the same learned visual features — marker shape and local appearance are correlated, so joint learning is a natural fit. EfficientNet-B2 offers a good accuracy/compute trade-off for a T4 GPU.

---

## 2. Patch-Based Training (512×512 Crops)

**Decision**: Train on 512×512 crops centered on the GCP, with ±100px random jitter, rather than full-resolution images.

**Rationale**: Source images are up to ~4000×3000px. Training on full images would be too slow and memory-intensive on a T4. Cropping preserves fine marker detail (critical for sub-pixel localization) while keeping batch sizes and training time practical. Jitter simulates the GCP appearing off-center, improving robustness for inference where the exact location is unknown.

---

## 3. Wing Loss for Keypoint Regression

**Decision**: Used Wing Loss instead of standard MSE/L1 for the coordinate regression term.

**Rationale**: Wing Loss applies a stronger gradient for small errors (where precision matters most for PCK@10/25px) while remaining robust to large outlier errors from poorly-cropped patches. This better matches the evaluation metric (PCK at tight pixel thresholds) than uniform L1/L2 loss.

---

## 4. Class-Weighted Cross-Entropy for Shape Classification

**Decision**: Applied inverse-frequency class weights to the classification loss rather than oversampling/undersampling.

**Rationale**: The dataset is imbalanced — Cross: 177, Square: 328, L-Shape: 491 (out of 1000 labeled samples). Class weighting corrects for this without duplicating or discarding training data, preserving the natural distribution of crop appearances.

---

## 5. Label Normalization

**Assumption**: The provided JSON contains inconsistent shape labels — both `"L-Shape"` and `"L-Shaped"` variants appear, while the assignment brief specifies the three classes as Cross, Square, L-Shaped.

**Decision**: Normalized all variants to a single canonical class name (`"L-Shape"`) during data loading, treating them as identical.

---

## 6. Missing Image Files

**Issue**: A small number of image paths referenced in `curated_gcp_marks.json` (≈3-4 entries) do not exist in the downloaded dataset, likely due to incomplete sync from the source Google Drive folder.

**Decision**: Rather than crashing the training run, the dataset loader catches missing-file errors and resamples a different valid training example in its place. This keeps the pipeline robust to real-world data gaps without manual cleanup, consistent with the assignment's note that the dataset "reflects real-world production conditions."

---

## 7. Sliding-Window TTA at Inference

**Decision**: For test images (full resolution, unknown GCP location), inference uses overlapping 512×512 sliding-window crops combined with horizontal/vertical flip test-time augmentation. Coordinate predictions across all crops/flips are aggregated via **median** (robust to noisy border crops), and shape predictions via **averaged softmax** across crops.

**Rationale**: At training time the model only sees GCP-centered crops, but at test time the GCP location is unknown. Sliding-window inference ensures the marker is captured in at least one crop near its center, and median aggregation reduces the impact of crops where the marker is partially visible or absent.

---

## 8. Training Duration — 15 Epochs

**Decision**: Trained for 15 epochs with OneCycleLR scheduling, given ~850 training images after the train/val split.

**Rationale**: Given the dataset size and T4 time constraints, 15 epochs was sufficient for convergence — validation PCK@25px reached ~1.0 and mean pixel distance dropped to ~12px by the final epochs, with diminishing returns observed beyond this point.

---

## 9. Train/Val Split

**Decision**: Stratified 85/15 split by shape class (848 train / 148 val), seeded for reproducibility.

**Rationale**: Stratification ensures all three shape classes are represented proportionally in both splits, given the class imbalance noted in (4).


## 10. Model Output

You can download the best model from:
https://drive.google.com/file/d/1mJOTiRR3bXqc9p_P1pfmmwK-WNRSD_7G/view?usp=sharing


and also the final model from :
https://drive.google.com/file/d/1SBm70Op88fJ4utlIYXEtKxz_tmqOJw9i/view?usp=sharing