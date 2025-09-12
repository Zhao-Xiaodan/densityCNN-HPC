#!/bin/bash
#PBS -N Microbead_CNN_Reduced_Layer_Study
#PBS -q gpu_q
#PBS -l select=1:ncpus=36:mem=240GB:ngpus=1:gpu_model=a40
#PBS -l walltime=24:00:00
#PBS -M xiaodan.liang@unimelb.edu.au
#PBS -m abe
#PBS -j oe
#PBS -o /home/xiaodan/densityCNN/reduced_layer_study.o${PBS_JOBID}

# Set up environment variables for optimal PyTorch performance
export PYTHONPATH="/home/xiaodan/densityCNN:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
export OMP_NUM_THREADS=36
export CUDA_LAUNCH_BLOCKING=0
export CUBLAS_WORKSPACE_CONFIG=:16:8

# Navigate to the project directory
cd /home/xiaodan/densityCNN/Claude/skip_connections_study

# Load Singularity container with PyTorch 2.4.0
module load singularity
export SINGULARITY_IMAGE="/home/xiaodan/containers/pytorch_2.4.0a0-cuda_12.5.0_ngc_24.06.sif"

echo "🔬 STARTING REDUCED-LAYER CNN ARCHITECTURE STUDY"
echo "================================================="
echo "Job ID: ${PBS_JOBID}"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"
echo "CUDA device: $CUDA_VISIBLE_DEVICES"
echo ""

# Check GPU availability
singularity exec --nv $SINGULARITY_IMAGE python3 -c "import torch; print(f'🎮 GPU Available: {torch.cuda.is_available()}'); print(f'🎮 GPU Device: {torch.cuda.get_device_name()}' if torch.cuda.is_available() else '❌ No GPU detected')"

echo ""
echo "🚀 Launching reduced-layer architecture training..."
echo ""

# Run the reduced-layer study with optimized parameters for HPC
singularity exec --nv $SINGULARITY_IMAGE python3 train_reduced_layer_study.py \
  --input_dir /home/xiaodan/densityCNN/dataset/dataset_preprocessed \
  --output_dir reduced_layer_study \
  --epochs 40 \
  --patience 12 \
  --learning_rate 3e-4 \
  --batch_size 128 \
  --num_workers 18 \
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