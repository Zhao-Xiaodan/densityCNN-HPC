# HPC Synchronization Guide - Reduced-Layer Study

## 📋 Required Files for HPC Deployment

### Essential Python Files (Must Sync)
```bash
# Core training and architecture files
train_reduced_layer_study.py              # Main training script
reduced_layer_study_architectures.py      # Architecture definitions
degradation_analysis.py                   # Analysis framework
comprehensive_comparison_suite.py         # Master comparison tool

# Supporting analysis files
simple_mse_analysis.py                     # Dependency-free MSE analysis
mse_analysis_and_comparison.py            # Advanced MSE visualization
analyze_existing_results.py               # Results comparison

# Enhanced training script (if not already on HPC)
train_comprehensive_architecture_study.py # Updated with dynamic density ranges
```

### HPC Job Scripts (Must Sync)
```bash
pbs_reduced_layer_study.sh                # Main reduced-layer study job
pbs_degradation_analysis.sh               # Degradation analysis job
```

### Optional Documentation
```bash
REDUCED_LAYER_STUDY_GUIDE.md              # Usage instructions
HPC_SYNC_GUIDE.md                         # This file
conversation_summary_reduced_layer_study.txt  # Research context
```

## 🔄 Synchronization Commands

### Option A: Using rsync (Recommended)
```bash
# Sync to HPC (from local machine)
rsync -avz --progress \
  train_reduced_layer_study.py \
  reduced_layer_study_architectures.py \
  degradation_analysis.py \
  comprehensive_comparison_suite.py \
  simple_mse_analysis.py \
  pbs_reduced_layer_study.sh \
  pbs_degradation_analysis.sh \
  train_comprehensive_architecture_study.py \
  your_username@hpc_host:/home/svu/phyzxi/scratch/densityCNN-HPC/

# Sync results back (to local machine)
rsync -avz --progress \
  your_username@hpc_host:/home/svu/phyzxi/scratch/densityCNN-HPC/reduced_layer_study_* \
  your_username@hpc_host:/home/svu/phyzxi/scratch/densityCNN-HPC/degradation_analysis_output \
  ./
```

### Option B: Using scp
```bash
# Upload files
scp train_reduced_layer_study.py \
    reduced_layer_study_architectures.py \
    degradation_analysis.py \
    pbs_reduced_layer_study.sh \
    pbs_degradation_analysis.sh \
    your_username@hpc_host:/home/svu/phyzxi/scratch/densityCNN-HPC/

# Download results
scp -r your_username@hpc_host:/home/xiaodan/densityCNN/Claude/skip_connections_study/reduced_layer_study_* ./
scp -r your_username@hpc_host:/home/xiaodan/densityCNN/Claude/skip_connections_study/degradation_analysis_output ./
```

## 🚀 HPC Execution Workflow

### Step 1: Submit Reduced-Layer Study
```bash
# On HPC cluster
cd /home/xiaodan/densityCNN/Claude/skip_connections_study
qsub pbs_reduced_layer_study.sh

# Monitor job
qstat -u xiaodan
tail -f reduced_layer_study.o<JOBID>
```

### Step 2: Submit Degradation Analysis (After Step 1 Completes)
```bash
# Wait for reduced-layer study to complete, then:
qsub pbs_degradation_analysis.sh

# Monitor analysis job
qstat -u xiaodan
tail -f degradation_analysis.o<JOBID>
```

### Step 3: Download Results
```bash
# From local machine
rsync -avz --progress \
  your_username@hpc_host:/home/svu/phyzxi/scratch/densityCNN-HPC/reduced_layer_study_* \
  your_username@hpc_host:/home/svu/phyzxi/scratch/densityCNN-HPC/degradation_analysis_output \
  ./
```

## 📂 Expected Directory Structure After Sync

```
/home/xiaodan/densityCNN/Claude/skip_connections_study/
├── train_reduced_layer_study.py           # ✅ REQUIRED
├── reduced_layer_study_architectures.py   # ✅ REQUIRED  
├── degradation_analysis.py                # ✅ REQUIRED
├── comprehensive_comparison_suite.py      # ✅ REQUIRED
├── simple_mse_analysis.py                 # ✅ REQUIRED
├── pbs_reduced_layer_study.sh             # ✅ REQUIRED
├── pbs_degradation_analysis.sh            # ✅ REQUIRED
├── train_comprehensive_architecture_study.py  # If needed
├── REDUCED_LAYER_STUDY_GUIDE.md          # Optional
└── existing_study_results/               # Pre-existing
    ├── comprehensive_architecture_study_20250911_173012/
    └── skip_connections_study_20250903_151020/
```

## 🔍 Verification Commands

### On HPC, verify files are present:
```bash
cd /home/svu/phyzxi/scratch/densityCNN-HPC
echo "📋 Checking required files..."
ls -la train_reduced_layer_study.py reduced_layer_study_architectures.py degradation_analysis.py pbs_reduced_layer_study.sh

echo "🧪 Testing architecture loading..."
singularity exec --nv /app1/common/singularity-img/hopper/pytorch/pytorch_2.4.0a0-cuda_12.5.0_ngc_24.06.sif \
  python3 -c "
from reduced_layer_study_architectures import get_reduced_layer_architectures
archs = get_reduced_layer_architectures()
print(f'✅ Loaded {len(archs)} architectures successfully')
for arch in archs[:3]:
    print(f'   - {arch.name}: {arch.depth} layers')
"

echo "📊 Testing degradation analyzer..."
singularity exec /app1/common/singularity-img/hopper/pytorch/pytorch_2.4.0a0-cuda_12.5.0_ngc_24.06.sif \
  python3 -c "
from degradation_analysis import DegradationAnalyzer
print('✅ DegradationAnalyzer imports successfully')
"
```

## ⚠️ Important Notes

### Dependencies
- **PyTorch Container**: Uses existing `pytorch_2.4.0a0-cuda_12.5.0_ngc_24.06.sif`
- **Dataset Path**: Assumes `./dataset_preprocessed` exists in `/home/svu/phyzxi/scratch/densityCNN-HPC/`
- **Results Dependency**: Degradation analysis requires existing study results

### Resource Requirements
- **Reduced-Layer Study**: GPU, 36 CPUs, 240GB RAM, 24h walltime
- **Degradation Analysis**: CPU-only, 18 CPUs, 64GB RAM, 4h walltime
- **PBS Format**: Compatible with standard PBS Pro directives (no queue specification)

### File Size Considerations
- Training script outputs can be 100MB+ per model (checkpoints)
- Complete study may generate 1-2GB of results
- Consider disk space on HPC scratch filesystem

### Troubleshooting
1. **Import errors**: Ensure PYTHONPATH includes proper paths
2. **Dataset not found**: Verify dataset path in HPC environment
3. **CUDA memory errors**: Reduce batch_size in PBS script if needed
4. **Permission errors**: Ensure execute permissions on PBS scripts: `chmod +x *.sh`
5. **Argument parsing errors**: Fixed in v4 - standardized all argument names across scripts and moved argument parsing to main() functions to prevent module-level execution conflicts
6. **Wrong script execution**: Ensure you're calling `train_reduced_layer_study.py` not `train_comprehensive_architecture_study.py`

## 🎯 Success Criteria

After successful execution:
- ✅ 7 trained reduced-layer models with checkpoints
- ✅ `reduced_layer_study_YYYYMMDD_HHMMSS/` directory with results
- ✅ `degradation_analysis_output/` with pattern analysis
- ✅ CSV files with performance comparisons
- ✅ PNG visualizations of degradation patterns
- ✅ Statistical report on architectural robustness

This provides definitive answers to research questions about skip connection effectiveness under depth constraints.