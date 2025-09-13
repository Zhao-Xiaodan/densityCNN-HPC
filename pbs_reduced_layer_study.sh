#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -k oed
#PBS -N Reduced_Layer_Study
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

# Memory optimization settings - Fixed for HPC CUDA allocator compatibility
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
export CUDA_LAUNCH_BLOCKING=0
export PYTHONUNBUFFERED=1
export PYTORCH_NO_CUDA_MEMORY_CACHING=0
export OMP_NUM_THREADS=36

# Load required modules
module load singularity

# Define singularity container - CORRECT HPC PATH
image=/app1/common/singularity-img/hopper/pytorch/pytorch_2.4.0a0-cuda_12.5.0_ngc_24.06.sif

if [ ! -f "$image" ]; then
    echo "❌ ERROR: Container not found at $image"
    echo "Please check container path"
    exit 1
fi

echo "✅ Container found: $image"

# Navigate to the correct HPC project directory
cd /home/svu/phyzxi/scratch/densityCNN-HPC

echo "🔬 STARTING REDUCED-LAYER CNN ARCHITECTURE STUDY"
echo "================================================="
echo "Job ID: ${PBS_JOBID}"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"
echo "CUDA device: $CUDA_VISIBLE_DEVICES"
echo ""

# Check GPU availability
singularity exec --nv "$image" python3 -c "import torch; print(f'🎮 GPU Available: {torch.cuda.is_available()}'); print(f'🎮 GPU Device: {torch.cuda.get_device_name()}' if torch.cuda.is_available() else '❌ No GPU detected')"

echo ""
echo "🚀 Launching reduced-layer architecture training..."
echo ""

# Run the reduced-layer study with optimized parameters for HPC - CONSISTENT ARGUMENTS
singularity exec --nv "$image" python3 train_reduced_layer_study.py \
  --input_dir ./dataset_preprocessed \
  --output_dir reduced_layer_study \
  --epochs 40 \
  --patience 12 \
  --learning_rate 3e-4 \
  --base_batch_size 128 \
  --base_num_workers 18 \
  --data_percentage 50 \
  --mixed_precision \
  --seed 42

exit_code=$?

echo ""
echo "================================================="
echo "🎉 REDUCED-LAYER STUDY COMPLETED"
echo "Exit code: $exit_code"
echo "End time: $(date)"
echo "Job ID: ${PBS_JOBID}"
echo ""

if [ $exit_code -eq 0 ]; then
    echo "✅ Study completed successfully!"
    echo "📊 Results should be available in the output directory"
    echo ""
    echo "🔍 Quick results summary:"
    # Display the latest results directory
    latest_dir=$(ls -td reduced_layer_study_* 2>/dev/null | head -1)
    if [ -n "$latest_dir" ]; then
        echo "   📁 Results directory: $latest_dir"
        if [ -f "$latest_dir/reduced_layer_comparison.csv" ]; then
            echo "   📋 Top 3 performing models:"
            head -4 "$latest_dir/reduced_layer_comparison.csv" | tail -3
        fi
    fi
else
    echo "❌ Study failed with exit code: $exit_code"
    echo "💡 Check the log files for detailed error information"
fi

echo ""
echo "📧 Email notification sent to xiaodan.liang@unimelb.edu.au"
echo "================================================="