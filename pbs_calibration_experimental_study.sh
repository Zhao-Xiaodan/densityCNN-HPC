#!/bin/bash
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -k oed
#PBS -N Calibration_Experimental_Study
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe
#PBS -q batch_gpu

cd $PBS_O_WORKDIR

# Load required modules
module load singularity

# Define singularity container to use
image=/app1/common/singularity-img/hopper/pytorch/pytorch_2.4.0a0-cuda_12.5.0_ngc_24.06.sif

# Set environment variables for optimal GPU performance
export OMP_NUM_THREADS=18
export MKL_NUM_THREADS=18
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Additional GPU optimizations
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDNN_BENCHMARK=1

echo "======================================================================="
echo "CALIBRATION DATASET EXPERIMENTAL STUDY"
echo "======================================================================="
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $PBS_JOBID"
echo "Dataset: 20260201 Beads Calibration (50x-51200x, 384 images)"
echo "======================================================================="

# Create main timestamped output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_OUTPUT_DIR="./calibration_experimental_study_${TIMESTAMP}"
mkdir -p $MAIN_OUTPUT_DIR

# Updated dataset directory for calibration
INPUT_DIR="./dataset_20260201_beads calibration_S16-Basler camera"
LEARNING_RATE=3e-4
MAX_EPOCHS=50
PATIENCE=15
NUM_WORKERS=18
DATA_PERCENTAGE=100

# Calibration dilution factors
DILUTION_FACTORS="50x 100x 200x 400x 800x 1600x 3200x 6400x 12800x 25600x 51200x"

echo "=== CALIBRATION EXPERIMENTAL STUDY CONFIGURATION ==="
echo "GPU: NVIDIA A40 (44GB memory)"
echo "Dataset: $INPUT_DIR"
echo "Data Percentage: $DATA_PERCENTAGE% (all calibration images)"
echo "Learning Rate: $LEARNING_RATE"
echo "Max Epochs: $MAX_EPOCHS"
echo "Workers: $NUM_WORKERS"
echo "Dilution Factors: $DILUTION_FACTORS"
echo "Total Experiments: 4 key configurations"
echo "======================================================="

# Function to run a specific experiment configuration
run_experiment() {
    local EXP_ID=$1
    local EXP_NAME=$2
    local BATCH_SIZE=$3
    local FILTER_CONFIG=$4

    echo ""
    echo "####################################################################"
    echo "STARTING EXPERIMENT $EXP_ID: $EXP_NAME"
    echo "####################################################################"
    echo "Configuration: Batch=$BATCH_SIZE, Filters=$FILTER_CONFIG"
    echo "Started at: $(date)"

    # Create specific output directory for this experiment
    EXP_OUTPUT_DIR="$MAIN_OUTPUT_DIR/experiment_${EXP_ID}_${EXP_NAME}"
    mkdir -p $EXP_OUTPUT_DIR

    # Build command line arguments
    PYTHON_ARGS="--input_dir \"$INPUT_DIR\" \
  --output_dir \"$EXP_OUTPUT_DIR\" \
  --batch_sizes $BATCH_SIZE \
  --filter_configs \"$FILTER_CONFIG\" \
  --epochs $MAX_EPOCHS \
  --patience $PATIENCE \
  --learning_rate $LEARNING_RATE \
  --data_percentage $DATA_PERCENTAGE \
  --dilution_factors $DILUTION_FACTORS \
  --use_all_dilutions \
  --num_workers $NUM_WORKERS \
  --mixed_precision \
  --use_enhanced_model \
  --enhanced_preprocessing \
  --seed 42"

    # Run experiment inside singularity container
    singularity exec --nv --bind /tmp:/tmp $image bash << EOFCONTAINER > $EXP_OUTPUT_DIR/experiment_console.log 2>&1

echo "=== EXPERIMENT $EXP_ID: $EXP_NAME ==="
echo "Python: \$(python --version)"
echo "PyTorch: \$(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: \$(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: \$(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\")')"

echo "Dataset: $INPUT_DIR"
echo "Output: $EXP_OUTPUT_DIR"
echo "==============================="

# Set TMPDIR for faster I/O operations
export TMPDIR=/tmp
export TORCH_HOME=/tmp/torch_cache_exp${EXP_ID}
mkdir -p \$TORCH_HOME

echo "Starting $EXP_NAME experiment..."

# Run the training script with configuration
python -u train_densityCNN_HPC.py $PYTHON_ARGS

echo "$EXP_NAME experiment completed on \$(date)"
echo "=== GPU MEMORY USAGE AFTER $EXP_NAME ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo "======================================="

EOFCONTAINER

    local EXIT_CODE=$?

    echo "Experiment $EXP_ID ($EXP_NAME) finished at: $(date)"
    echo "Exit code: $EXIT_CODE"

    # Copy best model with descriptive name
    for model_file in "$EXP_OUTPUT_DIR"/run_*/best_model_*.pth; do
        if [ -f "$model_file" ]; then
            cp "$model_file" "$MAIN_OUTPUT_DIR/best_model_${EXP_ID}_${EXP_NAME}.pth"
            echo "✅ Best model copied: best_model_${EXP_ID}_${EXP_NAME}.pth"
            break
        fi
    done

    # Check for results and provide summary
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Experiment $EXP_ID ($EXP_NAME) completed successfully"

        # Look for experiment results
        RESULTS_FILE=$(find "$EXP_OUTPUT_DIR" -name "experiment_comparison.csv" | head -1)
        if [ -f "$RESULTS_FILE" ]; then
            echo "📊 Quick Performance Summary:"
            tail -n +2 "$RESULTS_FILE" | head -1 | \
            awk -F',' '{
                if (NF >= 6) print "   R² Score: " $6
                if (NF >= 7) print "   MSE: " $7
                if (NF >= 8) print "   MAE: " $8
                if (NF >= 12) print "   Training Time: " $12 " min"
            }'
        fi
    else
        echo "❌ WARNING: Experiment $EXP_ID ($EXP_NAME) failed with exit code $EXIT_CODE"
        echo "📋 Check logs: $EXP_OUTPUT_DIR/experiment_console.log"
    fi

    echo "####################################################################"
    echo "COMPLETED EXPERIMENT $EXP_ID: $EXP_NAME"
    echo "####################################################################"
    echo ""

    # Brief GPU cleanup
    singularity exec --nv $image python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
" 2>/dev/null || true

    sleep 5  # Brief pause between experiments

    return $EXIT_CODE
}

# ======================================================================
# EXECUTE 4 KEY EXPERIMENTS SEQUENTIALLY
# ======================================================================

echo ""
echo "🚀 Starting Calibration Experimental Study: 4 Architecture Variants"
echo ""

# Initialize tracking
declare -a EXP_EXITS
declare -a EXP_TIMES
TOTAL_START_TIME=$(date +%s)
SUCCESSFUL_EXPERIMENTS=0

echo "📊 ARCHITECTURE VARIANTS (Calibration Dataset)"

# C01: Minimal Architecture
EXP_START=$(date +%s)
run_experiment "C01" "Minimal" 256 "16,32,64"
EXP_EXITS[1]=$?
EXP_END=$(date +%s)
EXP_TIMES[1]=$((EXP_END - EXP_START))
[ ${EXP_EXITS[1]} -eq 0 ] && ((SUCCESSFUL_EXPERIMENTS++))

# C02: Standard Architecture
EXP_START=$(date +%s)
run_experiment "C02" "Standard" 192 "32,64,128"
EXP_EXITS[2]=$?
EXP_END=$(date +%s)
EXP_TIMES[2]=$((EXP_END - EXP_START))
[ ${EXP_EXITS[2]} -eq 0 ] && ((SUCCESSFUL_EXPERIMENTS++))

# C03: Enhanced Architecture
EXP_START=$(date +%s)
run_experiment "C03" "Enhanced" 128 "64,128,256"
EXP_EXITS[3]=$?
EXP_END=$(date +%s)
EXP_TIMES[3]=$((EXP_END - EXP_START))
[ ${EXP_EXITS[3]} -eq 0 ] && ((SUCCESSFUL_EXPERIMENTS++))

# C04: Deep Architecture
EXP_START=$(date +%s)
run_experiment "C04" "Deep" 96 "128,256,512"
EXP_EXITS[4]=$?
EXP_END=$(date +%s)
EXP_TIMES[4]=$((EXP_END - EXP_START))
[ ${EXP_EXITS[4]} -eq 0 ] && ((SUCCESSFUL_EXPERIMENTS++))

TOTAL_END_TIME=$(date +%s)
TOTAL_TIME=$((TOTAL_END_TIME - TOTAL_START_TIME))

# ======================================================================
# EXPERIMENTAL ANALYSIS
# ======================================================================

echo ""
echo "======================================================================="
echo "CALIBRATION EXPERIMENTAL STUDY COMPLETED - 4 EXPERIMENTS"
echo "======================================================================="
echo "Job finished on $(date)"
echo "Total execution time: $((TOTAL_TIME / 60)) minutes"
echo "Successful experiments: $SUCCESSFUL_EXPERIMENTS/4"
echo "Main output directory: $MAIN_OUTPUT_DIR"

# Create detailed experimental report
EXPERIMENTAL_REPORT="$MAIN_OUTPUT_DIR/calibration_experimental_report.txt"

cat > $EXPERIMENTAL_REPORT << EOF
CALIBRATION DATASET EXPERIMENTAL STUDY REPORT
==============================================
Generated: $(date)
Job ID: $PBS_JOBID
Node: $(hostname)
Total Execution Time: $((TOTAL_TIME / 60)) minutes
Successful Experiments: $SUCCESSFUL_EXPERIMENTS/4

DATASET INFORMATION:
====================
Dataset: $INPUT_DIR
Images: 384 (512x512 cropped)
Dilution Series: 50x - 51200x (11 factors, double dilution)
Density Method: Blob DoG + 50x extrapolated (R²=0.9956)
Data Percentage: $DATA_PERCENTAGE%

EXPERIMENTAL CONFIGURATIONS:
============================
C01: Minimal Architecture  - Filters: [16,32,64]   - Batch: 256
C02: Standard Architecture - Filters: [32,64,128]  - Batch: 192
C03: Enhanced Architecture - Filters: [64,128,256] - Batch: 128
C04: Deep Architecture     - Filters: [128,256,512] - Batch: 96

EXPERIMENT EXECUTION SUMMARY:
============================
EOF

# Add execution results for each experiment
EXPERIMENT_NAMES=("" "Minimal" "Standard" "Enhanced" "Deep")

for i in {1..4}; do
    EXP_ID="C$(printf "%02d" $i)"
    EXP_NAME="${EXPERIMENT_NAMES[$i]}"

    echo "" >> $EXPERIMENTAL_REPORT
    if [ ${EXP_EXITS[$i]} -eq 0 ]; then
        echo "$EXP_ID ($EXP_NAME): ✅ SUCCESS" >> $EXPERIMENTAL_REPORT
        echo "  Execution Time: $((${EXP_TIMES[$i]} / 60)) minutes" >> $EXPERIMENTAL_REPORT

        # Try to extract performance metrics
        RESULTS_FILE=$(find "$MAIN_OUTPUT_DIR/experiment_${EXP_ID}_${EXP_NAME}" -name "experiment_comparison.csv" | head -1)
        if [ -f "$RESULTS_FILE" ] && [ $(wc -l < "$RESULTS_FILE") -gt 1 ]; then
            echo "  Performance Metrics:" >> $EXPERIMENTAL_REPORT
            tail -n +2 "$RESULTS_FILE" | head -1 | \
            awk -F',' '{
                if (NF >= 6) print "    R² Score: " $6
                if (NF >= 7) print "    MSE: " $7
                if (NF >= 8) print "    MAE: " $8
                if (NF >= 12) print "    Training Time: " $12 " min"
            }' >> $EXPERIMENTAL_REPORT
        else
            echo "    Performance data not available" >> $EXPERIMENTAL_REPORT
        fi
    else
        echo "$EXP_ID ($EXP_NAME): ❌ FAILED (Exit Code: ${EXP_EXITS[$i]})" >> $EXPERIMENTAL_REPORT
        echo "  Execution Time: $((${EXP_TIMES[$i]} / 60)) minutes" >> $EXPERIMENTAL_REPORT
    fi
done

echo "" >> $EXPERIMENTAL_REPORT
echo "RESEARCH QUESTIONS:" >> $EXPERIMENTAL_REPORT
echo "===================" >> $EXPERIMENTAL_REPORT
echo "1. Which architecture handles calibration density range (50x-51200x) best?" >> $EXPERIMENTAL_REPORT
echo "2. How does model capacity affect extrapolated 50x density prediction?" >> $EXPERIMENTAL_REPORT
echo "3. Trade-off between model complexity and training time for 384 images?" >> $EXPERIMENTAL_REPORT
echo "4. Which configuration balances accuracy and efficiency?" >> $EXPERIMENTAL_REPORT

echo "📋 Experimental analysis report saved: $EXPERIMENTAL_REPORT"

echo ""
echo "🎉 CALIBRATION EXPERIMENTAL STUDY COMPLETED!"
echo "============================================"
echo "📁 Main Results Directory: $MAIN_OUTPUT_DIR"
echo "📋 Experimental Report: $EXPERIMENTAL_REPORT"
echo "📊 Successful Experiments: $SUCCESSFUL_EXPERIMENTS/4"
echo "⏱️ Total Execution Time: $((TOTAL_TIME / 60)) minutes"
echo ""
echo "✅ Ready for analysis and model selection for calibration dataset"
echo "============================================"
