# Calibration Dataset Training Guide

## Dataset Information

**Dataset**: `dataset_20260201_beads calibration_S16-Basler camera`
- **Location on HPC**: `~/scratch/densityCNN-HPC/dataset_20260201_beads calibration_S16-Basler camera/`
- **Structure**:
  ```
  dataset_20260201_beads calibration_S16-Basler camera/
  ├── images/              # 384 cropped 512x512 images
  └── density.csv          # Density labels (filename, density)
  ```

### Dilution Series
- **Factors**: 50x, 100x, 200x, 400x, 800x, 1600x, 3200x, 6400x, 12800x, 25600x, 51200x
- **Protocol**: Double dilution series (each step = 2x dilution)
- **Total Images**: 384
  - 32 images per dilution (50x-51200x except 25600x)
  - 64 images for 25600x
- **Original Size**: 1920×1200 → Cropped to 512×512 (4×2 grid = 8 crops per original)

### Density Calculation Method
1. **Blob DoG Detection**: Used for 100x-51200x dilutions
2. **Linear Extrapolation**: 50x density extrapolated from log-log linear fit
   - Observed 50x: 1,325 beads/mm² (undercount due to connected beads)
   - **Extrapolated 50x: 7,964 beads/mm²** (all 32 images use this value)
   - R² = 0.9956 (excellent fit quality)
   - Slope = -0.9285 (close to ideal -1.0 for perfect double dilution)

---

## Modified Files

### 1. PBS Script: `pbs_calibration_architecture_study.sh`

**Key Changes from Original**:
- **Dataset Path**: `./20260201_beads calibration_S16-Basler camera`
- **Dilution Factors**: Updated to 50x-51200x (11 factors)
- **Data Percentage**: 100% (use all 384 images)
- **Training Script**: Calls `train_calibration_architecture_study.py`
- **Job Name**: `Calibration_CNN_Architecture_Study`
- **Output Naming**: `calibration_architecture_study_YYYYMMDD_HHMMSS`

**Resources**:
- Walltime: 48 hours
- GPUs: 1
- CPUs: 36
- Memory: 240GB

### 2. Python Training Script: `train_calibration_architecture_study.py`

**Key Changes from Original**:
- **Default Input Dir**: `20260201_beads calibration_S16-Basler camera`
- **Default Dilution Factors**: `['50x', '100x', ..., '51200x']`
- **Default Data Percentage**: 100% (was 50%)
- **Updated Header**: Documentation reflects calibration dataset details
- **Path Structure**: Already compatible (looks for `images/` and `density.csv`)

**No changes needed in**:
- Dataset class (already handles the structure correctly)
- Model architectures
- Training/evaluation logic
- Visualization code

---

## Usage Instructions

### Step 1: Verify Dataset on HPC

```bash
ssh phyzxi@vanda.svucluster.sydney.edu.au
cd ~/scratch/densityCNN-HPC

# Check dataset structure
ls -la "dataset_20260201_beads calibration_S16-Basler camera/"
# Should show: density.csv  images/

# Check images count
ls "dataset_20260201_beads calibration_S16-Basler camera/images/" | wc -l
# Should show: 384

# Check density.csv format
head "dataset_20260201_beads calibration_S16-Basler camera/density.csv"
# Should show: image_name,density format
```

### Step 2: Commit and Push Modified Scripts (Local)

```bash
cd /Users/xiaodan/densityCNN/densityCNN-HPC

# Check what changed
git status

# Add new files
git add pbs_calibration_architecture_study.sh
git add train_calibration_architecture_study.py
git add CALIBRATION_TRAINING_GUIDE.md

# Commit
git commit -m "Add calibration dataset training scripts for 50x-51200x dilution series"

# Push to GitHub
git push origin main
```

### Step 3: Pull on HPC and Submit Job

```bash
# On HPC
cd ~/scratch/densityCNN-HPC

# Pull latest code
git pull origin main

# Verify new scripts exist
ls -la pbs_calibration_architecture_study.sh
ls -la train_calibration_architecture_study.py

# Make PBS script executable
chmod +x pbs_calibration_architecture_study.sh

# Submit job
qsub pbs_calibration_architecture_study.sh

# Check job status
qstat -u phyzxi

# Monitor output (once job starts)
tail -f calibration_study_console_*.log
```

---

## Expected Output

After successful completion, you'll find:

```
calibration_architecture_study_YYYYMMDD_HHMMSS/
├── comprehensive_architecture_comparison.csv    # Main results table
├── complete_comprehensive_study.json           # Full experimental data
├── experiment_*_results.json                   # Individual experiment results
├── best_model_*.pth                           # Model checkpoints
├── training_analysis_*.png                    # Training curves
├── evaluation_*.png                           # Evaluation plots
├── gradient_analysis_*.png                    # Gradient flow analysis
└── statistical_analysis.json                  # Hypothesis testing results
```

### Key Results Files

1. **`comprehensive_architecture_comparison.csv`**
   - Rankings by R² score
   - Parameters count
   - Training time
   - Architecture type

2. **Console Log**: `calibration_study_console_*.log`
   - Full training output
   - Top 3 performers
   - Architecture-specific results

---

## Expected Training Time

Based on 384 images with 11 dilution factors:

| Architecture Type | Count | Est. Time per Model | Total Time |
|-------------------|-------|---------------------|------------|
| Baseline | 2 | ~30 min | ~1 hour |
| ResNet | 2 | ~45 min | ~1.5 hours |
| UNet | 4 | ~60 min | ~4 hours |
| DenseNet | 1 | ~60 min | ~1 hour |
| **Total** | **~9 models** | - | **~8-10 hours** |

Actual time may vary based on GPU availability and early stopping.

---

## Troubleshooting

### Issue: "Input directory does not exist"
```bash
# Check path (note the space in folder name)
ls "dataset_20260201_beads calibration_S16-Basler camera/"
```

### Issue: "Density file does not exist"
```bash
# Verify density.csv is in the correct location
ls "dataset_20260201_beads calibration_S16-Basler camera/density.csv"
```

### Issue: "No images found"
```bash
# Check images folder
ls "dataset_20260201_beads calibration_S16-Basler camera/images/" | head
```

### Issue: Job fails with memory error
- Reduce `--base_batch_size` from 64 to 32
- Reduce `--base_num_workers` from 8 to 4

---

## Comparison with Previous Dataset

| Aspect | Previous Dataset | Calibration Dataset |
|--------|------------------|---------------------|
| Dilution Factors | 8 (80x-10240x) | 11 (50x-51200x) |
| Images per Factor | Variable | 32-64 |
| Total Images | Variable | 384 |
| Data Usage | 50% | 100% |
| Density Method | Standard | Blob DoG + 50x extrapolation |
| Image Size | 512×512 | 512×512 |
| Grid Layout | 8×4 | 4×2 |

---

## Next Steps After Training

1. **Review Results**:
   ```bash
   # View top performers
   head -20 calibration_architecture_study_*/comprehensive_architecture_comparison.csv
   ```

2. **Download Results**:
   ```bash
   # From local machine
   rsync -avz phyzxi@vanda:~/scratch/densityCNN-HPC/calibration_architecture_study_*/ \
     ./local_results/
   ```

3. **Analyze Performance**:
   - Check R² scores across dilution ranges
   - Verify model performs well on high-density (50x) and low-density (51200x)
   - Compare parameter efficiency

4. **Deploy Best Model**:
   - Use best checkpoint for production predictions
   - Test on new calibration data

---

## Notes

- **50x Density**: All 50x images use the same extrapolated density (7,964 beads/mm²) because blob detection cannot resolve individual beads at this density
- **Linear Fit Quality**: R² = 0.9956 confirms the dilution series follows theoretical expectations
- **Data Completeness**: Using 100% of data (384 images) for calibration is appropriate for this controlled dataset
