#!/bin/bash
#PBS -N fair_architecture_comparison
#PBS -l select=1:ncpus=36:ngpus=1:gpu_model=a40:mem=240gb
#PBS -l walltime=24:00:00
#PBS -q gpu
#PBS -j oe
#PBS -M xiaodan@vt.edu
#PBS -m abe

# Fair Architecture Comparison Study PBS Script
# =============================================
# Systematic comparison of CNN architectures with individual hyperparameter optimization

echo "=========================================="
echo "Fair Architecture Comparison Study"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

# Environment setup
module purge
module load site/tinkercliffs/easybuild/setup
module load Singularity/3.8.7-GCC-10.3.0

# Set working directory
cd $PBS_O_WORKDIR
echo "Working directory: $(pwd)"

# Container and paths
CONTAINER="/projects/wacc_containers/pytorch_2.4.0a0-cuda_12.5.0_ngc_24.06.sif"
STUDY_DIR="./fair_architecture_comparison_$(date +%Y%m%d_%H%M%S)"

echo "Creating study directory: $STUDY_DIR"
mkdir -p "$STUDY_DIR"

# Copy training script to study directory
cp train_fair_architecture_comparison.py "$STUDY_DIR/"
cp pbs_fair_architecture_comparison.sh "$STUDY_DIR/"

# Environment variables for optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Dataset path (adjust as needed)
DATASET_PATH="../dataset_preprocessed"

echo "Environment setup complete"
echo "Container: $CONTAINER"
echo "Dataset path: $DATASET_PATH"
echo "Study directory: $STUDY_DIR"

# Start fair comparison study
echo "=========================================="
echo "Starting Fair Architecture Comparison"
echo "=========================================="

singularity exec --nv \
    --bind /projects:/projects \
    --bind $(pwd):$(pwd) \
    "$CONTAINER" \
    python "$STUDY_DIR/train_fair_architecture_comparison.py" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$STUDY_DIR" \
    --optimization_trials 75 \
    --evaluation_runs 5 \
    2>&1 | tee "$STUDY_DIR/fair_comparison_console.log"

EXIT_CODE=${PIPESTATUS[0]}

echo "=========================================="
echo "Fair Architecture Comparison Complete"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

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