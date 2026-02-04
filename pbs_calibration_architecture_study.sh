#!/bin/bash
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -k oed
#PBS -N Calibration_CNN_Architecture_Study
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

# =======================================================================
# CALIBRATION DATASET CNN ARCHITECTURE STUDY
# =======================================================================
# Dataset: 20260201_beads calibration_S16-Basler camera
# Dilution Series: 50x - 51200x (11 dilution factors, double dilution)
# Images: 512x512 crops from 1920x1200 original images
# Density: Calculated by blob_dog with 50x extrapolated from linear fit
# =======================================================================

echo "======================================================================="
echo "CALIBRATION DATASET CNN ARCHITECTURE STUDY"
echo "======================================================================="
echo "Dataset: 20260201 Beads Calibration - Basler Camera S16"
echo "Dilution Series: 50x, 100x, 200x, 400x, 800x, 1600x, 3200x,"
echo "                 6400x, 12800x, 25600x, 51200x"
echo "Total Images: 384 (32 per dilution for 50x-51200x, 64 for 25600x)"
echo "Image Size: 512x512 pixels"
echo "Density Method: Blob DoG detection + 50x linear extrapolation"
echo "======================================================================="

# Job information
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $PBS_JOBID"
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"
echo "Memory: $(free -h | grep Mem | awk '{print $2}'), CPUs: $(nproc)"
echo ""

# =======================================================================
# CALIBRATION STUDY CONFIGURATION
# =======================================================================

echo "=== CALIBRATION STUDY CONFIGURATION ==="
echo "Dataset: ./dataset_20260201_beads calibration_S16-Basler camera"
echo "Images Folder: ./dataset_20260201_beads calibration_S16-Basler camera/images"
echo "Density CSV: ./dataset_20260201_beads calibration_S16-Basler camera/density.csv"
echo "Data Percentage: 100% (all 384 images)"
echo "Learning Rate: 3e-4"
echo "Max Epochs: 50 (with early stopping)"
echo "Patience: 15"
echo "Base Workers: 8 (adaptive per architecture)"
echo "Base Batch Size: 64 (adaptive per architecture)"
echo "Mixed Precision: Enabled"
echo "Gradient Tracking: Enabled with detailed analysis"
echo "Memory Management: Enhanced with architecture-specific optimization"
echo "Conservative Mode: Enabled for HPC stability"
echo "Cleanup Frequency: Every 5 batches"
echo "Dilution Factors: 50x 100x 200x 400x 800x 1600x 3200x 6400x 12800x 25600x 51200x"
echo "=============================================="
echo ""

# Memory optimization settings - Fixed for HPC CUDA allocator compatibility
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
export CUDA_LAUNCH_BLOCKING=0
export PYTHONUNBUFFERED=1
export PYTORCH_NO_CUDA_MEMORY_CACHING=0

# Load required modules
module load singularity

# Define singularity container
image=/app1/common/singularity-img/hopper/pytorch/pytorch_2.4.0a0-cuda_12.5.0_ngc_24.06.sif

if [ ! -f "$image" ]; then
    echo "❌ ERROR: Container not found at $image"
    echo "Please check container path"
    exit 1
fi

echo "✅ Container found: $image"

# =======================================================================
# GPU MEMORY STATUS CHECK
# =======================================================================

echo "=== GPU MEMORY STATUS BEFORE CALIBRATION STUDY ==="
singularity exec --nv "$image" python3 -c "
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('GPU Memory: {:.1f} GB'.format(torch.cuda.get_device_properties(0).total_memory / 1024**3))
    print('GPU Memory - Total, Used, Free (MB):')
    import subprocess
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,nounits,noheader'], capture_output=True, text=True)
        print(result.stdout)
    except:
        print('nvidia-smi not available')
else:
    print('CUDA not available')
"
echo "================================================="
echo ""

# =======================================================================
# CALIBRATION ARCHITECTURE STUDY EXECUTION
# =======================================================================

echo "🚀 Starting Calibration CNN Architecture Study..."
echo "Testing ALL architectures on calibration dataset"
echo "Enhanced with detailed analysis and visualization"
echo "=================================================="
echo ""

# Create timestamped output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="calibration_architecture_study_${TIMESTAMP}"

echo "📁 Output directory: $OUTPUT_DIR"
echo ""

# Execute the comprehensive study with calibration dataset parameters
cd /home/svu/phyzxi/scratch/densityCNN-HPC
singularity exec --nv "$image" python3 train_calibration_architecture_study.py \
    --input_dir "./dataset_20260201_beads calibration_S16-Basler camera" \
    --output_dir ./$OUTPUT_DIR \
    --epochs 50 \
    --patience 15 \
    --learning_rate 3e-4 \
    --base_batch_size 64 \
    --base_num_workers 8 \
    --data_percentage 100 \
    --dilution_factors 50x 100x 200x 400x 800x 1600x 3200x 6400x 12800x 25600x 51200x \
    --mixed_precision \
    --track_gradients \
    --run_baselines \
    --run_resnets \
    --run_unets \
    --run_densenet \
    --memory_efficient \
    --conservative_mode \
    --cleanup_frequency 5 \
    --seed 42 2>&1 | tee calibration_study_console_${TIMESTAMP}.log

# Capture exit code
EXIT_CODE=$?

echo ""
echo "======================================================================="
echo "CALIBRATION CNN ARCHITECTURE STUDY COMPLETED"
echo "======================================================================="
echo "Job finished on $(date)"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Exit code: $EXIT_CODE ✅ SUCCESS"
else
    echo "Exit code: $EXIT_CODE ❌ ERROR"
fi
echo "Study output directory: ./$OUTPUT_DIR"
echo ""

# =======================================================================
# POST-EXECUTION ANALYSIS AND REPORTING
# =======================================================================

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Calibration Architecture Study completed successfully!"
    echo ""

    echo "📊 Generated Analysis Files:"
    echo "📈 Performance Overview:"
    if [ -f "./$OUTPUT_DIR/comprehensive_architecture_comparison.csv" ]; then
        echo "✅ comprehensive_architecture_comparison.csv - Complete performance comparison"
        echo "   Top 3 performers by R² score:"
        singularity exec --nv "$image" python3 -c "
import pandas as pd
import sys
try:
    df = pd.read_csv('./$OUTPUT_DIR/comprehensive_architecture_comparison.csv')
    top3 = df.nlargest(3, 'R2_Score')[['Model', 'Architecture_Type', 'R2_Score', 'Parameters', 'Training_Time_Min']]
    print(top3.to_string(index=False))
except Exception as e:
    print('Error reading results:', e)
"
    else
        echo "❌ comprehensive_architecture_comparison.csv not found"
    fi
    echo ""

    echo "📂 Study Results Summary:"
    if [ -f "./$OUTPUT_DIR/complete_comprehensive_study.json" ]; then
        echo "✅ complete_comprehensive_study.json - Full experimental results"
        echo "   Study summary:"
        singularity exec --nv "$image" python3 -c "
import json
import sys
try:
    with open('./$OUTPUT_DIR/complete_comprehensive_study.json', 'r') as f:
        data = json.load(f)
    info = data['study_info']
    print(f\"   Total time: {info['total_time_minutes']:.2f} minutes\")
    print(f\"   Successful experiments: {info['successful_experiments']}\")
    print(f\"   Total architectures tested: {info['total_architectures_tested']}\")
except Exception as e:
    print('Error reading study info:', e)
"
    else
        echo "❌ complete_comprehensive_study.json not found"
    fi
    echo ""

else
    echo "❌ Calibration Architecture Study failed!"
    echo "Check the console log for detailed error information:"
    echo "   calibration_study_console_${TIMESTAMP}.log"
    echo ""
fi

echo ""
echo "=== CALIBRATION STUDY RESULTS SUMMARY ==="
echo "=============================================="

# Architecture-specific performance summary
if [ -f "./$OUTPUT_DIR/comprehensive_architecture_comparison.csv" ] && [ $EXIT_CODE -eq 0 ]; then
    echo "🏆 PERFORMANCE RESULTS BY ARCHITECTURE TYPE:"
    echo "============================================"

    singularity exec --nv "$image" python3 -c "
import pandas as pd
import numpy as np
try:
    df = pd.read_csv('./$OUTPUT_DIR/comprehensive_architecture_comparison.csv')

    print('🥇 BASELINE ARCHITECTURES (No Skip Connections):')
    baseline = df[df['Architecture_Type'] == 'Baseline'].sort_values('R2_Score', ascending=False)
    for _, row in baseline.iterrows():
        print(f'   {row[\"Model\"]:20} | R² = {row[\"R2_Score\"]:.4f} | Params: {row[\"Parameters\"]:7,} | Time: {row[\"Training_Time_Min\"]:5.1f}min')

    print()
    print('🥈 RESNET ARCHITECTURES (With Skip Connections):')
    resnet = df[df['Architecture_Type'] == 'ResNet'].sort_values('R2_Score', ascending=False)
    for _, row in resnet.iterrows():
        print(f'   {row[\"Model\"]:20} | R² = {row[\"R2_Score\"]:.4f} | Params: {row[\"Parameters\"]:7,} | Time: {row[\"Training_Time_Min\"]:5.1f}min')

    print()
    print('🥉 UNET ARCHITECTURES (Encoder-Decoder + Skip):')
    unet = df[df['Architecture_Type'] == 'UNet'].sort_values('R2_Score', ascending=False)
    for _, row in unet.iterrows():
        print(f'   {row[\"Model\"]:20} | R² = {row[\"R2_Score\"]:.4f} | Params: {row[\"Parameters\"]:7,} | Time: {row[\"Training_Time_Min\"]:5.1f}min')

    print()
    print('🏅 DENSENET ARCHITECTURES (Dense Connections):')
    densenet = df[df['Architecture_Type'] == 'DenseNet'].sort_values('R2_Score', ascending=False)
    for _, row in densenet.iterrows():
        print(f'   {row[\"Model\"]:20} | R² = {row[\"R2_Score\"]:.4f} | Params: {row[\"Parameters\"]:7,} | Time: {row[\"Training_Time_Min\"]:5.1f}min')

    print()
    print('📊 ARCHITECTURE TYPE AVERAGES:')
    print('==============================')
    for arch_type in df['Architecture_Type'].unique():
        subset = df[df['Architecture_Type'] == arch_type]
        avg_r2 = subset['R2_Score'].mean()
        avg_params = subset['Parameters'].mean()
        avg_time = subset['Training_Time_Min'].mean()
        count = len(subset)
        print(f'   {arch_type:12} | Avg R² = {avg_r2:.4f} | Avg Params: {avg_params:7,.0f} | Avg Time: {avg_time:5.1f}min | Count: {count}')

except Exception as e:
    print('Error analyzing results:', e)
"

else
    echo "❌ Unable to generate performance summary - results file not available"
fi

echo ""
echo "=== FINAL STUDY STATUS ==="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ CALIBRATION CNN ARCHITECTURE STUDY SUCCESSFUL!"
    echo "📁 All results saved in: ./$OUTPUT_DIR"
    echo "📊 Comprehensive comparison on calibration dataset completed"
    echo "🎯 Dataset: 384 images, 11 dilution factors (50x-51200x)"
    echo ""
    echo "🔍 Next Steps:"
    echo "1. Review comprehensive_architecture_comparison.csv for performance rankings"
    echo "2. Examine individual evaluation plots for detailed model analysis"
    echo "3. Check statistical_analysis.json for hypothesis testing results"
    echo "4. Use best performing models for production deployment"
else
    echo "❌ CALIBRATION CNN ARCHITECTURE STUDY FAILED"
    echo "📋 Check error logs and debug information above"
    echo "🔧 Common issues: Memory allocation, path problems, container issues"
fi

echo ""
echo "======================================================================="
echo "CALIBRATION CNN ARCHITECTURE STUDY - EXECUTION COMPLETE"
echo "Dataset: 20260201 Beads Calibration (50x-51200x, 384 images)"
echo "Enhanced: Memory optimization, comprehensive analysis, detailed plots"
echo "======================================================================="
