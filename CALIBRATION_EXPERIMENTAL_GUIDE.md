# Calibration Experimental Study Guide

## Overview

This experimental study tests **4 different CNN architectures** on the calibration dataset to identify the optimal model configuration for the 50x-51200x dilution range.

---

## Dataset

**Dataset**: `dataset_20260201_beads calibration_S16-Basler camera`
- **Location**: `~/scratch/densityCNN-HPC/dataset_20260201_beads calibration_S16-Basler camera/`
- **Images**: 384 (512×512 crops)
- **Dilution Series**: 50x, 100x, 200x, 400x, 800x, 1600x, 3200x, 6400x, 12800x, 25600x, 51200x
- **Density Method**: Blob DoG detection + 50x extrapolated (R²=0.9956)

---

## Experimental Design

### **4 Architecture Configurations**

| Exp ID | Name | Filter Config | Batch Size | Model Capacity |
|--------|------|---------------|------------|----------------|
| **C01** | Minimal | [16,32,64] | 256 | ~120K params |
| **C02** | Standard | [32,64,128] | 192 | ~420K params |
| **C03** | Enhanced | [64,128,256] | 128 | ~1.7M params |
| **C04** | Deep | [128,256,512] | 96 | ~6.8M params |

### **Fixed Parameters (All Experiments)**
- Learning Rate: 3e-4
- Max Epochs: 50
- Early Stopping Patience: 15
- Data: 100% (all 384 images)
- Dilution Factors: All 11 (50x-51200x)
- Mixed Precision: Enabled
- Random Seed: 42

---

## Files

### **PBS Script**
`pbs_calibration_experimental_study.sh`
- Runs 4 experiments sequentially
- Total runtime: ~4-8 hours (estimated)
- Automatic result collection and analysis

### **Training Script**
`train_densityCNN_HPC.py`
- Standard CNN training script
- Supports multiple filter configurations
- Automatic experiment comparison CSV generation

### **Output Structure**
```
calibration_experimental_study_YYYYMMDD_HHMMSS/
├── experiment_C01_Minimal/
│   ├── run_*/
│   │   ├── best_model_*.pth
│   │   ├── training_history.csv
│   │   └── experiment_comparison.csv
│   └── experiment_console.log
├── experiment_C02_Standard/
├── experiment_C03_Enhanced/
├── experiment_C04_Deep/
├── best_model_C01_Minimal.pth
├── best_model_C02_Standard.pth
├── best_model_C03_Enhanced.pth
├── best_model_C04_Deep.pth
└── calibration_experimental_report.txt
```

---

## Usage

### **Step 1: Verify Files on HPC**

```bash
ssh phyzxi@vanda.svucluster.sydney.edu.au
cd ~/scratch/densityCNN-HPC

# Check scripts
ls -la pbs_calibration_experimental_study.sh
ls -la train_densityCNN_HPC.py

# Check dataset
ls -la "dataset_20260201_beads calibration_S16-Basler camera/"
ls "dataset_20260201_beads calibration_S16-Basler camera/images/" | wc -l  # Should show 384
```

### **Step 2: Submit Job**

```bash
qsub pbs_calibration_experimental_study.sh

# Check status
qstat -u phyzxi

# Monitor output
tail -f Calibration_Experimental_Study.o*
```

### **Step 3: Monitor Progress**

```bash
# Check which experiment is running
tail -20 Calibration_Experimental_Study.o*

# View individual experiment logs
tail -f calibration_experimental_study_*/experiment_C0*_*/experiment_console.log
```

---

## Expected Output

### **Console Output Summary**
```
🚀 Starting Calibration Experimental Study: 4 Architecture Variants

####################################################################
STARTING EXPERIMENT C01: Minimal
Configuration: Batch=256, Filters=[16,32,64]
✅ Experiment C01 (Minimal) completed successfully
📊 Quick Performance Summary:
   R² Score: 0.XXXX
   MSE: XX.XX
   Training Time: XX min
####################################################################

[... C02, C03, C04 ...]

======================================================================
CALIBRATION EXPERIMENTAL STUDY COMPLETED - 4 EXPERIMENTS
Successful experiments: 4/4
Total execution time: XXX minutes
======================================================================
```

### **Results Files**
1. **`calibration_experimental_report.txt`** - Execution summary
2. **`best_model_C0X_*.pth`** - Top model checkpoints
3. **Individual experiment CSVs** - Detailed metrics per experiment

---

## Research Questions

This experiment will answer:

1. **Which architecture handles the wide calibration density range (50x-51200x) best?**
   - Compare C01 (minimal) vs C04 (deep)

2. **How does model capacity affect extrapolated 50x density prediction?**
   - Critical since 50x uses extrapolated value (7,964 beads/mm²)

3. **Trade-off between model complexity and training time for 384 images?**
   - Is deeper network worth the training time for this dataset size?

4. **Which configuration balances accuracy and efficiency?**
   - Identify sweet spot for production deployment

---

## Comparison with Full Architecture Study

| Aspect | Architecture Study | Experimental Study |
|--------|-------------------|-------------------|
| **Script** | `pbs_calibration_architecture_study.sh` | `pbs_calibration_experimental_study.sh` |
| **# Experiments** | 10 (Baseline, ResNet, UNet, DenseNet) | 4 (Architecture variants) |
| **Focus** | Architecture types (skip connections, encoder-decoder) | Filter configurations (capacity scaling) |
| **Runtime** | ~8-10 hours | ~4-6 hours |
| **Output** | Comprehensive comparison | Quick architecture selection |

**Recommendation**:
- Run **Experimental Study first** (faster, identifies best capacity)
- Then run **Architecture Study** with best capacity to test skip connections/UNet

---

## Troubleshooting

### **Issue: "train_densityCNN_HPC.py not found"**
```bash
# Copy from parent directory
cp /home/svu/phyzxi/densityCNN/train_densityCNN_HPC.py \
   ~/scratch/densityCNN-HPC/
```

### **Issue: Dataset not found**
```bash
# Verify path with spaces is correct
ls "dataset_20260201_beads calibration_S16-Basler camera/"
```

### **Issue: Experiment failed**
```bash
# Check experiment log
tail -100 calibration_experimental_study_*/experiment_C0X_*/experiment_console.log
```

---

## Next Steps After Completion

1. **Review Results**
   ```bash
   cat calibration_experimental_study_*/calibration_experimental_report.txt
   ```

2. **Identify Best Architecture**
   - Look for highest R² score
   - Consider training time vs performance trade-off

3. **Download Best Models**
   ```bash
   # From local machine
   rsync -avz phyzxi@vanda:~/scratch/densityCNN-HPC/calibration_experimental_study_*/best_model_*.pth \
     ./calibration_models/
   ```

4. **Use Best Model for Production**
   - Deploy best checkpoint for calibration curve predictions
   - Use identified architecture for future training

---

## Differences from Original Experimental Study

| Parameter | Original Study | **Calibration Study** |
|-----------|---------------|----------------------|
| Dataset | `dataset_preprocessed` | `dataset_20260201_beads calibration_S16-Basler camera` |
| Images | Variable | **384** |
| Dilution Factors | 80x-10240x (8 factors) | **50x-51200x (11 factors)** |
| Data Usage | 50% | **100%** |
| # Experiments | 12 (arch+prep+training combos) | **4 (arch variants only)** |
| Preprocessing | Tested (Basic/Enhanced/Hybrid) | **Standard (not tested)** |
| Training Strategy | Tested (Standard/Enhanced) | **Standard only** |

---

## Estimated Timeline

| Experiment | Est. Training Time | Est. Total Time |
|------------|-------------------|-----------------|
| C01 (Minimal) | 30-45 min | ~1 hour |
| C02 (Standard) | 45-60 min | ~1 hour |
| C03 (Enhanced) | 60-90 min | ~1.5 hours |
| C04 (Deep) | 90-120 min | ~2 hours |
| **Total** | - | **~5-6 hours** |

*Actual times may vary based on GPU availability and convergence speed*

---

## Quick Reference Commands

```bash
# Submit job
qsub pbs_calibration_experimental_study.sh

# Check status
qstat -u phyzxi

# View output
tail -f Calibration_Experimental_Study.o*

# Quick results check
tail -50 calibration_experimental_study_*/calibration_experimental_report.txt

# Download results
rsync -avz phyzxi@vanda:~/scratch/densityCNN-HPC/calibration_experimental_study_* ./
```
