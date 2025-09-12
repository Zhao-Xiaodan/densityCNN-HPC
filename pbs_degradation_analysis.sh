#!/bin/bash
#PBS -l walltime=4:00:00
#PBS -j oe
#PBS -k oed
#PBS -N Degradation_Analysis
#PBS -l select=1:ncpus=18:mpiprocs=1:ompthreads=18:mem=64gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

# Set up environment variables
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=18

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

echo "🔍 STARTING DEGRADATION PATTERN ANALYSIS"
echo "======================================="
echo "Job ID: ${PBS_JOBID}"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"
echo ""

# Find the most recent study directories
echo "📂 Locating study directories..."
original_study=$(ls -td comprehensive_architecture_study_* 2>/dev/null | head -1)
reduced_study=$(ls -td reduced_layer_study_* 2>/dev/null | head -1)

if [ -z "$original_study" ] || [ -z "$reduced_study" ]; then
    echo "❌ Error: Could not find both study directories"
    echo "   Original study: $original_study"
    echo "   Reduced study: $reduced_study"
    echo "   Please ensure both studies have been completed"
    exit 1
fi

echo "✅ Found study directories:"
echo "   📊 Original: $original_study"
echo "   📊 Reduced:  $reduced_study"
echo ""

echo "🚀 Running degradation analysis..."
echo ""

# Run degradation analysis
singularity exec "$image" python3 -c "
from degradation_analysis import DegradationAnalyzer

print('🔧 Initializing degradation analyzer...')
analyzer = DegradationAnalyzer('$original_study', '$reduced_study')

print('📊 Calculating degradation metrics...')
degradation_data = analyzer.calculate_degradation_metrics()

print('📈 Generating analysis plots...')
analyzer.create_degradation_analysis_plots('degradation_analysis_output')

print('📋 Creating degradation report...')
analyzer.generate_degradation_report('degradation_analysis_output')

print('✅ Degradation analysis complete!')
"

exit_code=$?

echo ""
echo "======================================="
echo "🎉 DEGRADATION ANALYSIS COMPLETED"
echo "Exit code: $exit_code"
echo "End time: $(date)"
echo "Job ID: ${PBS_JOBID}"
echo ""

if [ $exit_code -eq 0 ]; then
    echo "✅ Analysis completed successfully!"
    echo "📁 Results available in: degradation_analysis_output/"
    echo ""
    echo "📊 Generated outputs:"
    echo "   🖼️  degradation_pattern_analysis.png - 8-panel visualization"
    echo "   📋 degradation_analysis_report.txt - Statistical analysis"
    echo "   📊 detailed_degradation_analysis.csv - Raw comparison data"
else
    echo "❌ Analysis failed with exit code: $exit_code"
    echo "💡 Check the log files for detailed error information"
fi

echo ""
echo "📧 Email notification sent to xiaodan.liang@unimelb.edu.au"
echo "======================================="