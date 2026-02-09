# Experimental Study Failure - Root Cause & Fix Summary

## Problem: All 4 Experiments Failed (R² < 0)

The experimental study (calibration_experimental_study_20260204_150508) showed **catastrophic failure** with all R² scores negative (-0.38 to -0.23), while the architecture study on the **same dataset** achieved R²=0.9951.

---

## Root Causes Identified

### 🔴 CRITICAL Issue #1: Image Resizing to 224×224

**Failed Study:**
```python
# Default preprocessing in train_densityCNN_HPC.py
transforms.Resize((224, 224))  # ❌ Destroys 81% of pixel information
```

**Successful Study:**
```python
# train_calibration_architecture_study.py
# No resize - keeps 512×512  # ✅ Preserves all information
```

**Impact:**
- Original: 512×512 = 262,144 pixels
- Resized: 224×224 = 50,176 pixels (**81% loss**)
- Beads at 512×512: 3-10 pixels diameter (clearly visible)
- Beads at 224×224: 1-4 pixels (**indistinguishable from noise**)
- **Result**: Model cannot count beads → predicts constant ~300 beads/mm² for all inputs

---

### 🟡 MODERATE Issue #2: Wrong Model Selected

**PBS script did not include:**
```bash
--use_enhanced_model         # ❌ Missing flag
--enhanced_preprocessing     # ❌ Missing flag
```

**Result:**
- Used simple `DensityRegressionCNN` (3 conv layers, ~1.7M params)
- Instead of `EnhancedDensityRegressionCNN` (6 conv layers, ~6.8M params)
- Insufficient capacity for calibration task

---

## Image Preprocessing Pipeline

### Current Flow (Failed)

```
512×512 PNG
    ↓
Load as grayscale (PIL)
    ↓
Resize((224, 224))  ← ❌ DESTROYS INFORMATION
    ↓
ToTensor() [0, 1]
    ↓
Normalize(mean=0.5, std=0.5) → [-1, 1]
    ↓
Model → Prediction (beads/mm²)
```

### Fixed Flow (Debug Version)

```
512×512 PNG
    ↓
Load as grayscale (PIL)
    ↓
(NO RESIZE - keep 512×512)  ← ✅ PRESERVES INFORMATION
    ↓
ToTensor() [0, 1]
    ↓
Normalize(mean=0.5, std=0.5) → [-1, 1]
    ↓
Enhanced Model → Prediction (beads/mm²)
```

**Note: Targets (density values) are NOT normalized**
- Predictions are in raw beads/mm² units (15-7,964 range)
- No denormalization needed
- Loss/metrics computed on original scale

---

## Fixes Applied in Debug Version

### 1. Training Script (`train_densityCNN_HPC_DEBUG.py`)

```python
# Line 50: Default to enhanced model
parser.add_argument('--use_enhanced_model', action='store_true',
                    default=True,  # ✅ CHANGED
                    help='Use enhanced CNN architecture')

# Line 52: Default to enhanced preprocessing (no resize)
parser.add_argument('--enhanced_preprocessing', action='store_true',
                    default=True,  # ✅ CHANGED
                    help='Keep images at 512x512')

# Line 264: Fixed pattern matching
pattern = '|'.join([f'dilution_{factor}_' for factor in dilution_factors])  # ✅ FIXED

# Lines 377-407: Added diagnostics
# - Epoch 0: Print prediction range
# - Epoch 10: Check R², abort if negative
# - Validation: Detailed evaluation stats
```

### 2. PBS Script (`pbs_calibration_experimental_study_DEBUG.sh`)

```bash
# Lines 106-109: Added missing flags
PYTHON_ARGS="--input_dir \"$INPUT_DIR\" \
  ...
  --use_enhanced_model \         # ✅ ADDED
  --enhanced_preprocessing \     # ✅ ADDED
  --seed 42"
```

---

## Expected Results

Based on **Architecture Study** (same dataset, successful training):

| Experiment | Filters | Failed Study R² | Expected Debug R² | Expected MAE |
|------------|---------|-----------------|-------------------|--------------|
| C01 (Minimal) | [16,32,64] | **-0.38** | 0.85-0.92 | ~350 |
| C02 (Standard) | [32,64,128] | **-0.36** | 0.92-0.96 | ~200 |
| C03 (Enhanced) | [64,128,256] | **-0.28** | 0.96-0.98 | ~120 |
| C04 (Deep) | [128,256,512] | **-0.23** | 0.97-0.99 | ~100 |

**Reference from Architecture Study:**
- Baseline_Shallow: R²=0.9854, MAE=85 beads/mm²
- Baseline_Deep: R²=0.9951, MAE=106 beads/mm²

---

## How to Run

### Quick Start

```bash
# On HPC
cd ~/scratch/densityCNN-HPC

# Submit debug job
qsub pbs_calibration_experimental_study_DEBUG.sh

# Monitor
tail -f Calibration_Experimental_Study_DEBUG.o*
```

### Verify Success

Look for these in logs:

✅ **Epoch 0 - Wide prediction range:**
```
🐛 EPOCH 0 DIAGNOSTICS:
   Prediction range: [-250.12, 3845.67]  # Good
   Target range: [15.00, 7964.00]
```

✅ **Epoch 10 - Positive R²:**
```
🐛 EPOCH 10 R² CHECK: 0.8234  # Good
```

✅ **Final evaluation:**
```
   R² Score: 0.9712  # Excellent!
   MAE: 134.56
```

❌ **Failure (like original):**
```
   Prediction range: [0.00, 482.34]  # Bad - collapsed
❌ CRITICAL: R² is negative after 10 epochs!
```

---

## Files Created

1. **`train_densityCNN_HPC_DEBUG.py`**
   - Fixed training script with diagnostics
   - Defaults to enhanced preprocessing + enhanced model

2. **`pbs_calibration_experimental_study_DEBUG.sh`**
   - Fixed PBS script with correct flags
   - Verbose debug output

3. **`EXPERIMENTAL_STUDY_DEBUG_GUIDE.md`**
   - Detailed technical guide (this document's companion)
   - Complete preprocessing explanation
   - Architecture comparisons

4. **`DEBUG_SUMMARY.md`**
   - This quick reference summary

---

## Key Takeaways

### ✅ What Worked (Architecture Study)
- Keep images at **512×512** resolution
- Use **Enhanced model architecture** (6 conv layers)
- **No target normalization** (predict raw density)
- Normalize images to **[-1, 1]** only

### ❌ What Failed (Original Experimental Study)
- Resized to **224×224** (lost critical detail)
- Used **simple model** (insufficient capacity)
- Missing **enhanced_preprocessing flag** in PBS

### 🔧 Debug Version Fixes
- **Default enhanced_preprocessing=True** (keep 512×512)
- **Default use_enhanced_model=True** (better architecture)
- **Early termination** if R²<0 at epoch 10 (save time)
- **Verbose diagnostics** (catch issues immediately)

---

## Preprocessing: Common Questions

**Q: Are targets (density values) normalized?**
- **No**. Targets remain in beads/mm² (15-7,964 range).

**Q: Do we denormalize predictions?**
- **No**. Model learns to output raw density directly.

**Q: What does image normalization do?**
- Scales pixel values from [0,1] to [-1,1] for stable training.
- Formula: `(pixel - 0.5) / 0.5`

**Q: Why did resizing fail?**
- Beads are 3-10 pixels at 512×512
- At 224×224, they shrink to 1-4 pixels
- CNN cannot distinguish beads from noise

**Q: Is there augmentation?**
- **Minimal**: RandomAdjustSharpness (30% chance)
- **No rotation/flip** (preserves density)

---

## Next Actions

1. ✅ **Created debug files** (done)
2. 🔄 **Upload to HPC** (if editing locally)
3. 🚀 **Run debug job** on HPC
4. 📊 **Compare results** to architecture study
5. ✅ **Expect R² > 0.95** for C03/C04

---

**Contact**: phyzxi@nus.edu.sg
**Date**: February 4, 2026
**Status**: Ready for HPC testing
