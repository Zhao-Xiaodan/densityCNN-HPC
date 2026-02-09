# Debug Run Update - Job 555205

## Status: ❌ Failed with Syntax Error (FIXED)

**Job ID**: 555205.stdct-mgmt-02
**Date**: February 9, 2026
**Duration**: 2 seconds (immediate failure)
**Exit Status**: 2 (syntax error)

---

## Issue: Bash Syntax Error

### Error Message

```bash
/var/spool/pbs/mom_priv/jobs/555205.stdct-mgmt-02.SC: line 62: syntax error near unexpected token `('
/var/spool/pbs/mom_priv/jobs/555205.stdct-mgmt-02.SC: line 62: `echo "GPU: NVIDIA A40 (44GB memory)"'
```

### Root Cause

**Line 61** had an **escaped closing quote** instead of a regular closing quote:

```bash
# WRONG (caused syntax error)
echo "=== CALIBRATION EXPERIMENTAL STUDY CONFIGURATION ===\"
                                                          ^^^
                                                      Escaped quote!
```

This left the string **unclosed**, causing the next line (line 62) to be interpreted as part of the string. When bash encountered the parentheses in `(44GB memory)`, it failed because parentheses have special meaning in bash.

---

## Fix Applied

### Changed Lines

**Line 61:**
```bash
# BEFORE (WRONG)
echo "=== CALIBRATION EXPERIMENTAL STUDY CONFIGURATION ===\"

# AFTER (FIXED)
echo "=== CALIBRATION EXPERIMENTAL STUDY CONFIGURATION ==="
```

**Line 132:**
```bash
# BEFORE (WRONG)
echo "=== GPU MEMORY USAGE AFTER $EXP_NAME ===\"

# AFTER (FIXED)
echo "=== GPU MEMORY USAGE AFTER $EXP_NAME ==="
```

### Files Updated

- ✅ `/Users/xiaodan/densityCNN/densityCNN-HPC/pbs_calibration_experimental_study_DEBUG.sh`

---

## Next Steps

### 1. Verify Fix Locally

```bash
cd /Users/xiaodan/densityCNN/densityCNN-HPC

# Syntax check
bash -n pbs_calibration_experimental_study_DEBUG.sh

# Expected output: (nothing = success)
```

### 2. Re-upload to HPC

If you edited locally, upload the fixed file:

```bash
# From local machine
scp pbs_calibration_experimental_study_DEBUG.sh \
    phyzxi@vanda:~/scratch/densityCNN-HPC/
```

### 3. Resubmit Job

```bash
# On HPC
ssh phyzxi@vanda.svucluster.sydney.edu.au
cd ~/scratch/densityCNN-HPC

# Verify fix
bash -n pbs_calibration_experimental_study_DEBUG.sh

# Resubmit
qsub pbs_calibration_experimental_study_DEBUG.sh

# Monitor
tail -f Calibration_Experimental_Study_DEBUG.o*
```

---

## Expected Output After Fix

### Startup Messages

```
=======================================================================
CALIBRATION DATASET EXPERIMENTAL STUDY - 🐛 DEBUG VERSION
=======================================================================
Job started on Mon Feb  9 XX:XX:XX PM +08 2026
Running on node: GN-A40-XXX
Job ID: XXXXXX.stdct-mgmt-02
Dataset: 20260201 Beads Calibration (50x-51200x, 384 images)

🐛 DEBUG FIXES APPLIED:
  ✅ Force enhanced_preprocessing (keep 512x512, no resize)
  ✅ Force use_enhanced_model (better architecture)
  ✅ Fixed dilution factor pattern matching
  ✅ Added prediction range diagnostics
  ✅ Early termination if R² < 0 after 10 epochs
=======================================================================
=== CALIBRATION EXPERIMENTAL STUDY CONFIGURATION ===
GPU: NVIDIA A40 (44GB memory)
Dataset: ./dataset_20260201_beads calibration_S16-Basler camera
...
```

### First Experiment Start

```
####################################################################
STARTING EXPERIMENT C01: Minimal (DEBUG MODE)
####################################################################
Configuration: Batch=256, Filters=16,32,64
Started at: Mon Feb  9 XX:XX:XX PM +08 2026

=== EXPERIMENT C01: Minimal (DEBUG) ===
Python: Python 3.X.X
PyTorch: 2.4.0
CUDA available: True
GPU: NVIDIA A40
...
🐛 DEBUG MODE: Enhanced preprocessing + Enhanced model enabled
```

### Training Output (Expected)

```
🐛 DEBUG MODE - CALIBRATION EXPERIMENTAL STUDY
============================================================
✅ Enhanced Preprocessing: True (keeps 512x512)
✅ Enhanced Model: True
...
✅ Dataset loaded: 384 samples
   Density range: [15.00, 7964.00] beads/mm²
...
🚀 Training Configuration:
   Batch Size: 256
   Filters: [16, 32, 64]
   Device: cuda
...
🐛 EPOCH 0 DIAGNOSTICS:
   Prediction range: [-250.45, 3521.89]  ← Should be wide!
   Target range: [15.00, 7964.00]
...
Epoch [1/50] - Train Loss: 4523456.23, Val Loss: 3245678.90
Epoch [2/50] - Train Loss: 2345678.12, Val Loss: 1876543.21
...
🐛 EPOCH 10 R² CHECK: 0.7834  ← Should be positive!
...
✅ Training completed in 2.45 minutes
✅ Best validation loss: 234567.89

🐛 EVALUATION DIAGNOSTICS:
   Predictions range: [18.23, 7856.45]
   Actuals range: [15.00, 7964.00]
   R² Score: 0.9234  ← Should be > 0.8!
   MAE: 287.56
```

---

## What to Watch For

### ✅ Success Indicators

1. **Script starts** without syntax errors
2. **Dataset loads** 384 samples
3. **Epoch 0**: Prediction range is **wide** (e.g., -200 to 3800)
4. **Epoch 10**: R² is **positive** (e.g., > 0.7)
5. **Final R²**: **> 0.85** (preferably > 0.9)

### ❌ Failure Indicators

1. **Syntax errors** (check script)
2. **Dataset loads 0 samples** (pattern matching issue)
3. **Epoch 0**: Prediction range is **narrow** (e.g., 0 to 500)
4. **Epoch 10**: R² is **negative**
5. **Validation loss stays flat** at ~7,848,000

---

## Debugging Commands

### If Job Fails Again

```bash
# Check PBS output
cat Calibration_Experimental_Study_DEBUG.oXXXXXX

# Check experiment logs
tail -100 calibration_experimental_study_DEBUG_*/experiment_C01_*/experiment_console.log

# Check if dataset loaded
grep "Dataset loaded" calibration_experimental_study_DEBUG_*/experiment_C01_*/experiment_console.log

# Check epoch 0 diagnostics
grep -A 5 "EPOCH 0 DIAGNOSTICS" calibration_experimental_study_DEBUG_*/experiment_C01_*/experiment_console.log

# Check R² at epoch 10
grep "EPOCH 10 R² CHECK" calibration_experimental_study_DEBUG_*/experiment_C01_*/experiment_console.log
```

---

## Comparison: Syntax Error vs Expected Execution

| Aspect | Job 555205 (Failed) | Expected (After Fix) |
|--------|---------------------|----------------------|
| **Exit Status** | 2 (syntax error) | 0 (success) |
| **Walltime Used** | 00:00:02 | ~30-40 minutes |
| **CPU Time** | 00:00:00 | ~5-10 hours |
| **Memory Used** | 19 MB | ~2-8 GB |
| **Output** | Syntax error only | Full training logs |
| **Experiments Completed** | 0/4 | 4/4 (expected) |

---

## Root Cause Analysis

### Why Did This Happen?

The escaped quote (`\"`) was likely:
1. **Copy-paste error** from a different shell context
2. **Text editor auto-formatting** that escaped special characters
3. **Carried over from original script** that had different quoting

### How to Prevent

1. **Use plain text editors** for bash scripts (avoid Word, rich text editors)
2. **Syntax check locally** before uploading:
   ```bash
   bash -n script.sh
   ```
3. **Test on HPC** with minimal resource allocation first:
   ```bash
   #PBS -l walltime=00:10:00  # 10 minutes for syntax test
   ```

---

## Timeline

| Time | Event |
|------|-------|
| **Feb 4, 15:05** | Original experimental study submitted (failed, all R² < 0) |
| **Feb 4, 15:53** | Original study completed (catastrophic failure) |
| **Feb 9, ~15:30** | Debug version created with fixes |
| **Feb 9, 15:40:19** | Debug job 555205 submitted |
| **Feb 9, 15:40:20** | Job failed with syntax error (2 seconds) |
| **Feb 9, ~16:00** | Syntax error identified and fixed |
| **Next** | **Resubmit with fixed script** |

---

## Summary

### What Went Wrong
- ❌ Bash syntax error due to escaped quote on line 61
- ❌ Job failed immediately (2 seconds)
- ❌ No training occurred

### What Was Fixed
- ✅ Removed escaped quotes on lines 61 and 132
- ✅ Script now passes syntax check (`bash -n`)
- ✅ Ready for resubmission

### Next Action
- 🚀 **Resubmit the fixed PBS script**
- 📊 **Expect training to run for ~30-40 minutes**
- ✅ **Expect R² > 0.9 for C03/C04 experiments**

---

## Files Status

| File | Status | Notes |
|------|--------|-------|
| `pbs_calibration_experimental_study_DEBUG.sh` | ✅ FIXED | Syntax error corrected |
| `train_densityCNN_HPC_DEBUG.py` | ✅ OK | No changes needed |
| `EXPERIMENTAL_STUDY_DEBUG_GUIDE.md` | ✅ OK | Documentation |
| `DEBUG_SUMMARY.md` | ✅ OK | Quick reference |

---

**Status**: Ready for resubmission
**Expected Duration**: 30-40 minutes for all 4 experiments
**Expected Result**: R² > 0.9 (vs original R² < 0)

**Contact**: phyzxi@nus.edu.sg
**Date**: February 9, 2026
