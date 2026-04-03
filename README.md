# 🧠 3D Attention U-Net for Brain Tumor Segmentation
### BraTS 2021 · PyTorch · Monte Carlo Dropout · FastAPI + React

> Multi-class volumetric brain tumor segmentation with uncertainty estimation — designed for clinical decision support.

---

## What This Does

This project trains a **3D Attention U-Net** on the [BraTS 2021](https://www.synapse.org/#!Synapse:syn27046444/wiki/616571) dataset to segment brain tumors from multi-modal MRI scans into three subregions:

| Class | Region | Clinical Meaning |
|-------|--------|-----------------|
| 1 | Tumor Core | Necrotic / non-enhancing center |
| 2 | Whole Tumor | All subregions including edema |
| 3 | Enhancing Tumor | Actively proliferating tissue |

It also surfaces **uncertainty maps** via Monte Carlo Dropout — flagging low-confidence predictions before they silently fail in a clinical setting.

---

## Results

Evaluated on 20 validation patients using sliding window inference:

| Class | Dice Score |
|-------|-----------|
| Whole Tumor | **0.6686** |
| Enhancing Tumor | **0.6709** |
| Tumor Core | 0.4214 |

Three patch strategies were systematically compared:

| Strategy | Core | Whole | Enhancing | Verdict |
|----------|------|-------|-----------|---------|
| Random Crop (baseline) | ~0.58 | ~0.61 | ~0.56 | Balanced, decent everywhere |
| Tumor-Aware Crop | ~0.40 | ~0.49 | ~0.65 | Biased — boosts Class 3 at the cost of others |
| **Sliding Window** | 0.42 | **0.67** | **0.67** | ✅ Most stable, best for evaluation |

---

## Architecture

```
Input: 4-channel MRI volume (T1 · T1CE · T2 · FLAIR) — shape: 4 × 128 × 128 × 128
       │
  ┌────▼──────────────────────────────────┐
  │  Encoder                              │
  │  Block 1 → Conv3D → InstanceNorm → ReLU (×2) → F1   │
  │  ↓ MaxPool3D                          │
  │  Block 2 → Conv3D → InstanceNorm → ReLU (×2) → F2   │
  │  ↓ MaxPool3D                          │
  │  Block 3 → Conv3D → InstanceNorm → ReLU (×2) → F3   │
  │  ↓ MaxPool3D                          │
  │  Bottleneck → Conv3D × 2             │
  └───────────────────────────────────────┘
       │
  ┌────▼──────────────────────────────────┐
  │  Decoder                              │
  │  TransposedConv + Concat(F3) → Conv×2 │
  │  TransposedConv + Concat(F2) → Conv×2 │
  │  TransposedConv + Concat(F1) → Conv×2 │
  │  1×1×1 Conv → Softmax                │
  └───────────────────────────────────────┘
       │
Output: Voxel-wise class probabilities → argmax → Segmentation Mask
```

**Key design choices:**
- **Instance Normalisation** over Batch Norm — works correctly at batch size 1
- **Skip connections** preserve spatial detail lost during pooling
- **Transposed Conv3D** for learnable upsampling
- **Monte Carlo Dropout** at inference time for uncertainty estimation

---

## Loss Function

Combined **Dice + Cross Entropy** loss, chosen after ablation:

| Configuration | Stability | Notes |
|--------------|-----------|-------|
| Dice alone | ❌ Unstable | Erratic early convergence |
| Dice + Focal | ⚠️ Inconsistent | Focal weighting amplified noise |
| **Dice + CE** | ✅ Stable | Consistent loss reduction — chosen |

---

## Training Setup

| Parameter | Value |
|-----------|-------|
| Framework | PyTorch |
| Optimizer | Adam |
| Learning Rate | 1e-4 → 5e-5 → 1e-5 (staged decay) |
| Batch Size | 1 (GPU memory constraint) |
| Patch Size | 128 × 128 × 128 |
| Mixed Precision | AMP enabled (Kaggle) |
| Platform | Google Colab → Kaggle |
| Checkpointing | Every 3 epochs |

Training ran in 3-epoch blocks with checkpoint recovery — necessary given free-tier GPU instability.

---

## Uncertainty Estimation

Monte Carlo Dropout is enabled at inference time. The model runs **N stochastic forward passes** with dropout active, then:
- Averages predictions → final segmentation
- Computes variance across passes → **uncertainty map**

High-variance regions are overlaid on the segmentation output, giving clinicians a visual signal of where the model is least confident — reducing silent failure risk.

---

## System Architecture

```
┌─────────────────┐     REST API      ┌──────────────────┐
│   React Frontend │ ◄──────────────► │  FastAPI Backend  │
│  (Upload MRI)   │                   │  (3D U-Net + MC  │
│  (View reports) │                   │   Dropout)        │
└─────────────────┘                   └──────────────────┘
                                              │
                                   Structured Report Output:
                                   • Segmentation overlay
                                   • Uncertainty map
                                   • Per-class Dice / HD95
```

---

## Dataset

**BraTS 2021** — 1,251 multi-institutional MRI cases, each containing:
- `T1` — anatomical baseline
- `T1CE` — contrast-enhanced (highlights active tumor)
- `T2` — sensitive to edema and tumor borders
- `FLAIR` — suppresses CSF, reveals peri-tumoral regions

All volumes are co-registered, skull-stripped, and resampled to 1mm³ isotropic resolution.

**Preprocessing pipeline:**
1. Load `.nii.gz` volumes with NiBabel
2. Z-score normalise each modality independently
3. Remap BraTS labels (original 4 → sequential 0–3)
4. Extract 128³ patches via chosen strategy
5. Cache patches to mitigate I/O bottleneck

---

## Installation

```bash
git clone https://github.com/pritha-cmd/brats-3d-unet
cd brats-3d-unet
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, PyTorch ≥ 1.13, nibabel, numpy, FastAPI, uvicorn

---

## Usage

**Training:**
```bash
python train.py --patch_strategy sliding_window --lr 1e-4 --epochs 30
```

**Inference with uncertainty:**
```bash
python inference.py --checkpoint checkpoints/best.pth --input path/to/patient/ --mc_samples 10
```

**Launch API:**
```bash
uvicorn app.main:app --reload
```

---

## Supervision & Context

Developed during a research internship at **IIT (ISM) Dhanbad** under **Dr. Scindhiya Laxmi** (Jan 2026 – ongoing). The ML pipeline and system architecture were designed independently; the FastAPI backend and React frontend were developed collaboratively.

---

## Future Work

- [ ] Attention gates on skip connections (Attention U-Net)
- [ ] Deep supervision with auxiliary decoder heads
- [ ] HD95 as an additional loss term
- [ ] Test-time augmentation (TTA)
- [ ] Self-supervised encoder pretraining on unlabelled MRI
- [ ] Ensemble with Swin-UNETR for improved robustness
