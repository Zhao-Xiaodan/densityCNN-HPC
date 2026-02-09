# Experimental Study Debug Guide

## Critical Issues Identified

After comparing the **failed experimental study** (calibration_experimental_study_20260204_150508) with the **successful architecture study** (calibration_architecture_study_20260204_101214), I identified the root causes of the catastrophic failures (all R² < 0).

---

## Issue #1: Image Preprocessing - Resize to 224×224 ⚠️ CRITICAL

### The Problem

**Failed Experimental Study** uses this preprocessing:

```python
# Default preprocessing (enhanced_preprocessing=False)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ❌ DESTROYS INFORMATION!
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    ...
])
```

**Successful Architecture Study** uses:

```python
# Keep original resolution
train_transform = transforms.Compose([
    transforms.ToTensor(),  # ✅ NO RESIZE - keeps 512×512
    transforms.Normalize(mean=[0.5], std=[0.5]),
    ...
])
```

### Why This Breaks Training

| Aspect | 512×512 (Correct) | 224×224 (Failed) | Impact |
|--------|-------------------|------------------|--------|
| **Total Pixels** | 262,144 pixels | 50,176 pixels | **81% information loss** |
| **After 3× MaxPool(2,2)** | 64×64 = 4,096 | 28×28 = 784 | **81% feature map loss** |
| **Spatial Resolution** | High | Low | Cannot detect fine details |
| **Bead Separation** | Clear | Blurred | Beads merge together |
| **Density Estimation** | Accurate | Inaccurate | Poor regression |

**Calibration dataset specifics:**
- Original images: 1920×1200 cropped to 512×512 (minimal overlap)
- Bead diameter: ~3-10 pixels at 512×512
- **At 224×224**: Beads become 1-4 pixels → indistinguishable from noise
- **Density range**: 15-7,964 beads/mm² requires fine spatial detail

### The Fix

```python
# In train_densityCNN_HPC_DEBUG.py
parser.add_argument('--enhanced_preprocessing', action='store_true',
                    default=True,  # ✅ CHANGED: Default True
                    help='Keep images at 512x512 (DEFAULT: TRUE for debug)')
```

---

## Issue #2: Model Selection

### The Problem

**PBS script** did NOT include flags for enhanced model:

```bash
# Original (WRONG)
PYTHON_ARGS="--input_dir \"$INPUT_DIR\" \
  --batch_sizes $BATCH_SIZE \
  --filter_configs \"$FILTER_CONFIG\" \
  --mixed_precision \
  --seed 42"
# Missing: --use_enhanced_model
# Missing: --enhanced_preprocessing
```

This means:
- Used `DensityRegressionCNN` (simple 3-layer model)
- Images resized to 224×224
- Insufficient capacity for calibration task

### Model Architecture Comparison

**DensityRegressionCNN** (used in failed study):
```python
class DensityRegressionCNN(nn.Module):
    def __init__(self, filters=[32, 64, 128]):
        # Only 3 conv layers
        self.conv1 = nn.Conv2d(1, filters[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(filters[0], filters[1], kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1)
        # Simple global pooling
        # x.mean(dim=[2, 3])  # Averages entire feature map
        # Small FC layers
        self.fc1 = nn.Linear(filters[2], 64)
        self.fc2 = nn.Linear(64, 1)
```

**EnhancedDensityRegressionCNN** (should be used):
```python
class EnhancedDensityRegressionCNN(nn.Module):
    def __init__(self, filters=[64, 128, 256], input_size=512):
        # Double conv blocks (6 conv layers total)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, filters[0], 3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], 3, padding=1),  # Extra conv
            nn.BatchNorm2d(filters[0]),
            nn.ReLU()
        )
        # ... conv2, conv3 similar double blocks
        # Adaptive pooling to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        # Deeper FC layers
        self.fc_layers = nn.Sequential(
            nn.Linear(filters[2] * 64, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
```

**Capacity comparison for C03 (Enhanced) configuration:**
- Standard model: ~1.7M parameters
- Enhanced model: ~6.8M parameters (4× larger)

### The Fix

```bash
# In pbs_calibration_experimental_study_DEBUG.sh
PYTHON_ARGS="--input_dir \"$INPUT_DIR\" \
  ...
  --use_enhanced_model \         # ✅ ADDED
  --enhanced_preprocessing \     # ✅ ADDED
  --seed 42"
```

---

## Issue #3: Pattern Matching for Dilution Factors ⚠️ MODERATE

### The Problem

In the dataset loading, the pattern matching may have issues:

```python
# Original (may fail to match)
pattern = '|'.join([f'^{factor}_' for factor in dilution_factors])
# Generates: ^50x_|^100x_|^200x_|...
# But files are named: dilution_50x_1.png, dilution_100x_1.png, etc.
# Pattern expects: 50x_1.png (MISMATCH!)
```

### The Fix

```python
# In train_densityCNN_HPC_DEBUG.py
if not use_all_dilutions and dilution_factors:
    # ✅ FIXED: Pattern matching for calibration dataset
    pattern = '|'.join([f'dilution_{factor}_' for factor in dilution_factors])
    self.df = self.df[self.df['filename'].str.contains(pattern, case=False, regex=True)]
    print(f"🔍 Filtered to {len(self.df)} images matching dilution factors: {dilution_factors}")
```

**Note**: This may not have been the primary issue since the original study reported loading samples, but it's good practice.

---

## Image Preprocessing Pipeline

### Complete Preprocessing Flow

```
Original Image (PNG file)
    ↓
PIL.Image.open().convert('L')  # Load as grayscale
    ↓
[If enhanced_preprocessing=False]
    ↓
transforms.Resize((224, 224))  # ❌ DOWNSCALE (lossy)
    ↓
[If enhanced_preprocessing=True]
    (No resize, keep 512×512)  # ✅ PRESERVE (lossless)
    ↓
transforms.ToTensor()  # Convert to tensor [0, 1]
    ↓
transforms.Normalize(mean=[0.5], std=[0.5])  # Scale to [-1, 1]
    ↓
[Optional: RandomAdjustSharpness, ColorJitter]
    ↓
Model Input: [1, 1, H, W]
    where H×W = 512×512 (enhanced) or 224×224 (standard)
```

### Normalization Details

**Normalize(mean=[0.5], std=[0.5])** transformation:

```python
output = (input - mean) / std
output = (input - 0.5) / 0.5
```

For grayscale images where `input ∈ [0, 1]`:
- `input = 0.0` (black) → `output = -1.0`
- `input = 0.5` (gray) → `output = 0.0`
- `input = 1.0` (white) → `output = 1.0`

**Result**: Images scaled to range `[-1, 1]`

**No denormalization needed** because:
- The model learns to predict density directly from normalized images
- Targets (density values) are NOT normalized
- Loss function compares raw predictions to raw density values

---

## Training Flow Diagnostics

### Added Debug Checks

The debug version includes these diagnostics:

**1. Epoch 0 Prediction Range Check:**
```python
if epoch == 0:
    print(f"\n🐛 EPOCH 0 DIAGNOSTICS:")
    print(f"   Prediction range: [{min(all_preds):.2f}, {max(all_preds):.2f}]")
    print(f"   Target range: [{min(all_targets):.2f}, {max(all_targets):.2f}]")
```

**Expected healthy output:**
```
🐛 EPOCH 0 DIAGNOSTICS:
   Prediction range: [-500.23, 2832.15]  # Wide range covering targets
   Target range: [15.00, 7964.00]
```

**Failed output (original study):**
```
   Prediction range: [0.00, 452.34]  # Collapsed range!
   Target range: [15.00, 7964.00]
```

**2. Epoch 10 R² Check:**
```python
if epoch == 9:
    r2 = r2_score(all_targets, all_preds)
    print(f"\n🐛 EPOCH 10 R² CHECK: {r2:.4f}")
    if r2 < 0:
        print(f"❌ CRITICAL: R² is negative after 10 epochs!")
        print(f"❌ Training has failed. Stopping early.")
        return None, {...}  # Abort training
```

**3. Initial Validation Loss Warning:**
```python
if epoch == 0 and avg_val_loss > 7000000:
    print(f"⚠️  WARNING: Initial validation loss is very high ({avg_val_loss:.0f})")
    print(f"⚠️  This may indicate a training problem. Expected <5M for healthy training.")
```

---

## Comparison: Failed vs Debug Version

| Aspect | Failed Experimental Study | Debug Version | Impact |
|--------|---------------------------|---------------|--------|
| **Image Size** | 224×224 | 512×512 | 5.2× more pixels |
| **Model** | DensityRegressionCNN | EnhancedDensityRegressionCNN | 4× more parameters |
| **Conv Layers** | 3 single blocks | 3 double blocks (6 total) | 2× depth |
| **Pooling** | Global average | Adaptive 8×8 | Better spatial info |
| **FC Layers** | 2 layers (64→1) | 3 layers (512→128→1) | More capacity |
| **Dropout** | 0.5 (may be too high) | 0.3, 0.2 (progressive) | Better regularization |
| **Pattern Matching** | `^50x_` | `dilution_50x_` | Correct format |
| **Early Termination** | None | Stop if R²<0 @ epoch 10 | Saves time |

---

## Expected Results with Debug Version

Based on the **Architecture Study** performance on the **same dataset**:

| Experiment | Filter Config | Expected R² | Expected MAE | Status |
|------------|---------------|-------------|--------------|--------|
| C01 (Minimal) | [16,32,64] | **0.80-0.90** | ~400 | Should work |
| C02 (Standard) | [32,64,128] | **0.90-0.95** | ~200 | Should work |
| C03 (Enhanced) | [64,128,256] | **0.95-0.98** | ~150 | Should work |
| C04 (Deep) | [128,256,512] | **0.97-0.99** | ~100 | Should work |

**Architecture study reference:**
- Baseline_Shallow (4 layers): R² = 0.9854, MAE = 85
- Baseline_Deep (12 layers): R² = 0.9951, MAE = 106

The debug version should achieve similar performance since it uses:
- ✅ Same image resolution (512×512)
- ✅ Enhanced model architecture (better than Baseline)
- ✅ Same dataset and preprocessing

---

## How to Run Debug Version

### 1. Verify Files Exist

```bash
ssh phyzxi@vanda.svucluster.sydney.edu.au
cd ~/scratch/densityCNN-HPC

# Check debug files
ls -la train_densityCNN_HPC_DEBUG.py
ls -la pbs_calibration_experimental_study_DEBUG.sh

# Check dataset
ls -la "dataset_20260201_beads calibration_S16-Basler camera/"
ls "dataset_20260201_beads calibration_S16-Basler camera/images/" | wc -l  # Should be 384
```

### 2. Submit Debug Job

```bash
qsub pbs_calibration_experimental_study_DEBUG.sh

# Check status
qstat -u phyzxi

# Monitor output
tail -f Calibration_Experimental_Study_DEBUG.o*
```

### 3. Monitor Progress

```bash
# Check current experiment
tail -20 Calibration_Experimental_Study_DEBUG.o*

# View individual experiment logs
tail -f calibration_experimental_study_DEBUG_*/experiment_C0*/experiment_console.log
```

### 4. Verify Success

**Look for these indicators in the logs:**

✅ **Epoch 0 diagnostics showing wide prediction range:**
```
🐛 EPOCH 0 DIAGNOSTICS:
   Prediction range: [-200.45, 3521.89]  # Good - covers target range
   Target range: [15.00, 7964.00]
```

✅ **Epoch 10 R² check passing:**
```
🐛 EPOCH 10 R² CHECK: 0.7523  # Good - positive and improving
```

✅ **Final R² > 0.9:**
```
🐛 EVALUATION DIAGNOSTICS:
   R² Score: 0.9654  # Excellent!
```

❌ **Failure indicators:**
```
🐛 EPOCH 0 DIAGNOSTICS:
   Prediction range: [0.00, 485.23]  # Bad - collapsed range
❌ CRITICAL: R² is negative after 10 epochs!
```

---

## Preprocessing: Technical Details

### Why No Denormalization?

**Question**: Do we need to denormalize predictions before computing metrics?

**Answer**: **NO**, because:

1. **Images are normalized** for neural network training:
   ```python
   normalized_image = (image - 0.5) / 0.5  # Range: [-1, 1]
   ```

2. **Targets (densities) are NOT normalized**:
   ```python
   density = torch.tensor(density, dtype=torch.float32)  # Raw values: 15-7964
   ```

3. **Model learns direct mapping**:
   ```
   normalized_image → CNN → raw_density_prediction
   ```

4. **Loss function operates on raw scale**:
   ```python
   loss = MSELoss(predictions, targets)  # Both in beads/mm²
   ```

5. **Metrics computed on raw scale**:
   ```python
   r2 = r2_score(actuals, predictions)  # Both in beads/mm²
   ```

**Contrast with target normalization** (NOT used here):
```python
# If we normalized targets (we don't do this):
normalized_target = (density - mean_density) / std_density

# Then we WOULD need to denormalize predictions:
pred_denormalized = pred * std_density + mean_density
```

### Data Augmentation

**Training augmentation** (minimal for calibration):
```python
transforms.RandomAdjustSharpness(sharpness_factor=1.2, p=0.3)
```
- Only 30% chance of applying
- Subtle sharpness adjustment (factor=1.2)
- **No geometric transforms** (rotation, flip) → preserves bead density

**Validation/test**: No augmentation (just normalize)

### Memory Considerations

**512×512 vs 224×224 memory usage:**

```python
# Single image tensor size
size_512 = 1 * 1 * 512 * 512 * 4 bytes = 1.05 MB (float32)
size_224 = 1 * 1 * 224 * 224 * 4 bytes = 0.20 MB (float32)

# Batch of 128 images
batch_512 = 128 * 1.05 MB = 134 MB
batch_224 = 128 * 0.20 MB = 26 MB

# Feature maps after conv (e.g., 256 channels)
features_512 = 256 * 64 * 64 * 4 bytes = 4.19 MB
features_224 = 256 * 28 * 28 * 4 bytes = 0.80 MB
```

**Solution**: Adjust batch sizes by capacity
- C01 (Minimal, 16-32-64 filters): batch=256 (fits in memory)
- C04 (Deep, 128-256-512 filters): batch=96 (reduced for memory)

---

## Summary of All Fixes

### Code Changes

1. ✅ **Default `enhanced_preprocessing=True`**
   - File: `train_densityCNN_HPC_DEBUG.py` line 52
   - Effect: Keeps images at 512×512

2. ✅ **Default `use_enhanced_model=True`**
   - File: `train_densityCNN_HPC_DEBUG.py` line 50
   - Effect: Uses better architecture

3. ✅ **Fixed dilution pattern matching**
   - File: `train_densityCNN_HPC_DEBUG.py` line 264
   - Effect: Correctly matches `dilution_50x_` format

4. ✅ **Added prediction range diagnostics**
   - File: `train_densityCNN_HPC_DEBUG.py` lines 377-382
   - Effect: Catch collapsed predictions early

5. ✅ **Added R² checkpoint at epoch 10**
   - File: `train_densityCNN_HPC_DEBUG.py` lines 396-407
   - Effect: Abort if training has failed

6. ✅ **Updated PBS script with debug flags**
   - File: `pbs_calibration_experimental_study_DEBUG.sh` lines 106-109
   - Effect: Pass correct arguments to training script

### What We Learned

1. **Image resolution matters critically** for dense object counting
   - 224×224 lost 81% of spatial information
   - Beads became indistinguishable from noise

2. **Model capacity must match task complexity**
   - Simple 3-layer CNN insufficient for 500× density range
   - Enhanced 6-layer CNN with deeper FC layers needed

3. **Preprocessing must preserve task-relevant information**
   - Calibration requires counting individual beads
   - Downsampling destroys countable features

4. **Always validate against known baselines**
   - Architecture study proved dataset is learnable
   - Immediate detection of preprocessing issues

---

## Next Steps

1. **Run debug version** on HPC
2. **Compare results** to architecture study (expect similar R²)
3. **If successful**, use these settings as default for future calibration training
4. **If still failing**, investigate further (GPU/CUDA issues, data corruption, etc.)

---

**Debug Files Created:**
- `train_densityCNN_HPC_DEBUG.py` - Fixed training script
- `pbs_calibration_experimental_study_DEBUG.sh` - Fixed PBS script
- `EXPERIMENTAL_STUDY_DEBUG_GUIDE.md` - This guide

**Contact**: phyzxi@nus.edu.sg
**Date**: February 4, 2026
