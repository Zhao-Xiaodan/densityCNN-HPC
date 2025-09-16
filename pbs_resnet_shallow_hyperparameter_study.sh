#!/bin/bash
#PBS -N ResNet_Shallow_Hyperparameter_Study
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -k oed
#PBS -m abe
#PBS -M phyzxi@nus.edu.sg


# ============================================================================
# ResNet_Shallow Hyperparameter Bottleneck Study - HPC Execution Script
# ============================================================================
#
# Research Focus: Systematic investigation of what caused ResNet_Shallow's
# dramatic performance improvement (R² = 0.983 → 0.995)
#
# Key Research Questions:
# 1. Is it gradient noise from smaller batch sizes?
# 2. Is it memory pressure relief?
# 3. Is it learning rate interaction with batch size?
# 4. Is it data loading efficiency?
# 5. Is it gradient accumulation vs. true batch size effects?
#
# Experimental Design:
# - ~40+ systematic experiments across 5 categories
# - Batch size variations (16, 32, 64, 96, 128, 192, 256)
# - Learning rate scaling experiments
# - Gradient accumulation vs true batch size
# - Memory and data loading optimization
# - Learning rate scheduler comparisons
# ============================================================================

echo "🔬 Starting ResNet_Shallow Hyperparameter Bottleneck Study"
echo "============================================================================"
echo "Job ID: $PBS_JOBID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Working Directory: $PBS_O_WORKDIR"
echo "============================================================================"

# Change to working directory
cd $PBS_O_WORKDIR

# Load environment
echo "🔧 Loading HPC environment..."

# Load container environment
module load singularity
export SINGULARITY_CACHEDIR=$PBS_O_WORKDIR/singularity_cache
export SINGULARITY_TMPDIR=$PBS_O_WORKDIR/singularity_tmp
mkdir -p $SINGULARITY_CACHEDIR $SINGULARITY_TMPDIR

# HPC Environment Optimizations
echo "⚙️  Setting HPC optimizations..."

# CUDA and PyTorch optimizations
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=18
export CUDA_DEVICE_MAX_CONNECTIONS=32

# Memory optimizations
export MALLOC_MMAP_THRESHOLD_=131072
export MALLOC_TRIM_THRESHOLD_=262144

# Python optimizations
export PYTHONPATH=$PBS_O_WORKDIR:$PYTHONPATH
export PYTHONUNBUFFERED=1

# Display environment info
echo "🖥️  Environment Information:"
echo "   CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "   OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "   PBS_NCPUS: $PBS_NCPUS"
echo "   PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"

# Check GPU availability
echo "🔍 GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits

# Define container and dataset paths
CONTAINER_PATH="/opt/ohpc/pub/containers/pytorch_2.4.0a0-cuda_12.5.0_ngc_24.06.sif"
DATASET_PATH="$PBS_O_WORKDIR/../dataset_preprocessed"

echo "📁 Paths:"
echo "   Container: $CONTAINER_PATH"
echo "   Dataset: $DATASET_PATH"
echo "   Working Directory: $PBS_O_WORKDIR"

# Verify paths exist
if [ ! -f "$CONTAINER_PATH" ]; then
    echo "❌ Container not found: $CONTAINER_PATH"
    exit 1
fi

if [ ! -d "$DATASET_PATH" ]; then
    echo "❌ Dataset not found: $DATASET_PATH"
    exit 1
fi

if [ ! -f "train_resnet_shallow_hyperparameter_study.py" ]; then
    echo "❌ Training script not found: train_resnet_shallow_hyperparameter_study.py"
    exit 1
fi

echo "✅ All required files and paths verified"

# Create output directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="resnet_shallow_hyperparameter_study_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR

echo "📊 Output directory: $OUTPUT_DIR"

# HPC resource monitoring setup
echo "📈 Setting up resource monitoring..."

# Function to monitor resources
monitor_resources() {
    while true; do
        echo "$(date): GPU Memory:" >> ${OUTPUT_DIR}/resource_monitor.log
        nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits >> ${OUTPUT_DIR}/resource_monitor.log
        echo "$(date): System Memory:" >> ${OUTPUT_DIR}/resource_monitor.log
        free -h >> ${OUTPUT_DIR}/resource_monitor.log
        echo "---" >> ${OUTPUT_DIR}/resource_monitor.log
        sleep 300  # Monitor every 5 minutes
    done &
}

# Start monitoring
monitor_resources
MONITOR_PID=$!

# Trap to kill monitoring when script exits
trap "kill $MONITOR_PID 2>/dev/null" EXIT

# ============================================================================
# MAIN EXECUTION
# ============================================================================

echo ""
echo "🚀 Starting ResNet_Shallow Hyperparameter Study..."
echo "============================================================================"

# Record start time
START_TIME=$(date)
START_TIMESTAMP=$(date +%s)

echo "Start Time: $START_TIME"

# Execute the hyperparameter study
singularity exec \
    --nv \
    --bind $PBS_O_WORKDIR:/workspace \
    --bind $DATASET_PATH:/workspace/dataset_preprocessed \
    --pwd /workspace \
    $CONTAINER_PATH \
    python train_resnet_shallow_hyperparameter_study.py \
        --input_dir /workspace/dataset_preprocessed \
        --output_dir $OUTPUT_DIR \
        --epochs 50 \
        --patience 15 \
        --seed 42 \
        --data_percentage 50 \
        --mixed_precision \
        --dilution_factors 80x 160x 320x 640x 1280x 2560x 5120x 10240x 2>&1 | tee ${OUTPUT_DIR}/study_console_output.log

# Record execution status
EXECUTION_STATUS=$?
END_TIME=$(date)
END_TIMESTAMP=$(date +%s)
EXECUTION_DURATION=$((END_TIMESTAMP - START_TIMESTAMP))

echo ""
echo "============================================================================"
echo "🎯 ResNet_Shallow Hyperparameter Study Completed"
echo "============================================================================"
echo "End Time: $END_TIME"
echo "Total Duration: $((EXECUTION_DURATION / 3600)) hours $((EXECUTION_DURATION % 3600 / 60)) minutes $((EXECUTION_DURATION % 60)) seconds"
echo "Execution Status: $EXECUTION_STATUS"

# Stop resource monitoring
kill $MONITOR_PID 2>/dev/null

# Create execution summary
cat > ${OUTPUT_DIR}/execution_summary.txt << EOF
ResNet_Shallow Hyperparameter Bottleneck Study Execution Summary
=============================================================

Job Information:
- Job ID: $PBS_JOBID
- Node: $(hostname)
- Start Time: $START_TIME
- End Time: $END_TIME
- Duration: $((EXECUTION_DURATION / 3600))h $((EXECUTION_DURATION % 3600 / 60))m $((EXECUTION_DURATION % 60))s
- Exit Status: $EXECUTION_STATUS

Environment:
- Container: $CONTAINER_PATH
- Dataset: $DATASET_PATH
- Output Directory: $OUTPUT_DIR
- CUDA Device: $CUDA_VISIBLE_DEVICES
- OMP Threads: $OMP_NUM_THREADS

Configuration:
- Epochs: 50
- Data Percentage: 50%
- Mixed Precision: Enabled
- Patience: 15
- Seed: 42

Research Focus:
This study systematically investigates what caused ResNet_Shallow's dramatic
performance improvement from R² = 0.983 to R² = 0.995 in recent experiments.

Key Research Questions:
1. Is it gradient noise from smaller batch sizes?
2. Is it memory pressure relief?
3. Is it learning rate interaction with batch size?
4. Is it data loading efficiency?
5. Is it gradient accumulation vs. true batch size effects?

Experimental Categories:
- Batch Size Study: Testing 16, 32, 64, 96, 128, 192, 256
- Learning Rate Scaling: Batch-dependent LR scaling
- Gradient Accumulation: True vs effective batch size
- Memory/Data Loading: Worker and cleanup optimization
- LR Scheduler: Different scheduler strategies

Expected Outputs:
- Individual experiment results (JSON)
- Complete study results (JSON)
- Summary analysis (CSV)
- Analysis report (Markdown)
- Training console logs
- Resource monitoring logs

GPU Information at Start:
$(nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv)

Resource Usage Summary:
- Peak GPU Memory: See resource_monitor.log
- CPU Utilization: See resource_monitor.log
- Training Time per Experiment: See individual results

EOF

# Final GPU status
echo ""
echo "🔍 Final GPU Status:"
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits

# List output files
echo ""
echo "📁 Generated Files:"
find $OUTPUT_DIR -type f -name "*.json" -o -name "*.csv" -o -name "*.md" -o -name "*.log" | head -20

# Check for critical output files
echo ""
echo "🔍 Critical Output Verification:"
if [ -f "${OUTPUT_DIR}/complete_hyperparameter_study.json" ]; then
    echo "✅ Complete study results found"
    RESULT_SIZE=$(stat -c%s "${OUTPUT_DIR}/complete_hyperparameter_study.json")
    echo "   File size: $((RESULT_SIZE / 1024)) KB"
else
    echo "❌ Complete study results missing"
fi

if [ -f "${OUTPUT_DIR}/hyperparameter_study_summary.csv" ]; then
    echo "✅ Summary analysis found"
    SUMMARY_LINES=$(wc -l < "${OUTPUT_DIR}/hyperparameter_study_summary.csv")
    echo "   Experiments: $((SUMMARY_LINES - 1))"
else
    echo "❌ Summary analysis missing"
fi

if [ -f "${OUTPUT_DIR}/analysis_report.md" ]; then
    echo "✅ Analysis report found"
else
    echo "❌ Analysis report missing"
fi

# Disk usage summary
echo ""
echo "💾 Disk Usage:"
du -sh $OUTPUT_DIR
df -h . | tail -1

# Final status message
echo ""
if [ $EXECUTION_STATUS -eq 0 ]; then
    echo "🎉 SUCCESS: ResNet_Shallow Hyperparameter Study completed successfully!"
    echo ""
    echo "📊 Next Steps:"
    echo "   1. Review ${OUTPUT_DIR}/analysis_report.md for key findings"
    echo "   2. Examine ${OUTPUT_DIR}/hyperparameter_study_summary.csv for detailed results"
    echo "   3. Check individual experiment files for training details"
    echo "   4. Analyze gradient and memory patterns in training histories"
    echo ""
    echo "🔬 Expected Insights:"
    echo "   - Optimal batch size for ResNet_Shallow"
    echo "   - Role of gradient noise in performance"
    echo "   - Memory pressure vs. performance relationship"
    echo "   - Learning rate scaling effects"
    echo "   - True vs. effective batch size impact"
else
    echo "❌ FAILURE: ResNet_Shallow Hyperparameter Study failed with exit code $EXECUTION_STATUS"
    echo ""
    echo "🔍 Troubleshooting:"
    echo "   1. Check ${OUTPUT_DIR}/study_console_output.log for errors"
    echo "   2. Review ${OUTPUT_DIR}/resource_monitor.log for resource issues"
    echo "   3. Verify dataset accessibility and container functionality"
    echo "   4. Check individual experiment logs for specific failures"
fi

echo "============================================================================"
echo "Job completed at $(date)"
echo "============================================================================"

# Clean up temporary files if successful
if [ $EXECUTION_STATUS -eq 0 ]; then
    echo "🧹 Cleaning up temporary files..."
    rm -rf $SINGULARITY_TMPDIR
fi

exit $EXECUTION_STATUS
