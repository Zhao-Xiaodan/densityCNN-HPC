#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -k oed
#PBS -N Fair_Architecture_Comparison
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

# Fair Architecture Comparison Study PBS Script
# =============================================
# Systematic comparison of CNN architectures with individual hyperparameter optimization

echo "======================================================================="
echo "FAIR ARCHITECTURE COMPARISON STUDY"
echo "======================================================================="
echo "Study: Systematic CNN architecture comparison with individual optimization"
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $PBS_JOBID"
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"
echo "Memory: $(free -h | grep Mem | awk '{print $2}'), CPUs: $(nproc)"
echo "======================================================================="
echo ""

# Memory optimization settings - Fixed for HPC CUDA allocator compatibility
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
export CUDA_LAUNCH_BLOCKING=0
export PYTHONUNBUFFERED=1
export PYTORCH_NO_CUDA_MEMORY_CACHING=0

# Load required modules
module load singularity

# Define singularity container
CONTAINER=/app1/common/singularity-img/hopper/pytorch/pytorch_2.4.0a0-cuda_12.5.0_ngc_24.06.sif

if [ ! -f "$CONTAINER" ]; then
    echo "❌ ERROR: Container not found at $CONTAINER"
    echo "Please check container path"
    exit 1
fi

echo "✅ Container found: $CONTAINER"

# Set working directory
cd $PBS_O_WORKDIR
echo "Working directory: $(pwd)"
STUDY_DIR="./fair_architecture_comparison_$(date +%Y%m%d_%H%M%S)"

echo "Creating study directory: $STUDY_DIR"
mkdir -p "$STUDY_DIR"

# Copy training script to study directory
cp train_fair_architecture_comparison.py "$STUDY_DIR/"
cp pbs_fair_architecture_comparison.sh "$STUDY_DIR/"

echo "✅ Training script copied (no external dependencies required)"

# Dataset path (adjust as needed)
DATASET_PATH="./dataset_preprocessed"

echo "=== FAIR COMPARISON STUDY CONFIGURATION ==="
echo "Dataset: $DATASET_PATH"
echo "Study Directory: $STUDY_DIR"
echo "Optimization Trials: 75 per architecture"
echo "Evaluation Runs: 5 per architecture"
echo "Container: $CONTAINER"
echo "=============================================="
echo ""

# Start fair comparison study
echo "======================================================================="
echo "STARTING FAIR ARCHITECTURE COMPARISON"
echo "======================================================================="

singularity exec --nv "$CONTAINER" python "$STUDY_DIR/train_fair_architecture_comparison.py" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$STUDY_DIR" \
    --optimization_trials 75 \
    --evaluation_runs 5 \
    2>&1 | tee "$STUDY_DIR/fair_comparison_console.log"

EXIT_CODE=${PIPESTATUS[0]}

echo "======================================================================="
echo "FAIR ARCHITECTURE COMPARISON COMPLETE"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "======================================================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Study completed successfully!"
    echo "📊 Results available in: $STUDY_DIR"

    # Display quick summary if results exist
    if [ -f "$STUDY_DIR/fair_comparison_results.csv" ]; then
        echo ""
        echo "📈 Quick Results Summary:"
        echo "------------------------"
        head -10 "$STUDY_DIR/fair_comparison_results.csv"
    fi

    echo ""
    echo "📁 Generated files:"
    ls -la "$STUDY_DIR"/*.{csv,json,png,log} 2>/dev/null || echo "   (Files may still be generating)"

else
    echo "❌ Study failed with exit code: $EXIT_CODE"
    echo "📝 Check the log file for details: $STUDY_DIR/fair_comparison_console.log"
fi

echo "Study directory: $STUDY_DIR"