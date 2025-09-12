#!/usr/bin/env python3
"""
Comprehensive Comparison Analysis Suite
Integrates all analysis tools for complete architectural comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import argparse
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import our analysis modules
from degradation_analysis import DegradationAnalyzer
from simple_mse_analysis import analyze_existing_study_results  # If available

plt.style.use('default')
sns.set_palette("husl")

class ComprehensiveComparisonSuite:
    """Complete suite for CNN architecture comparison analysis"""
    
    def __init__(self, study_directories):
        self.studies = {}
        self.load_all_studies(study_directories)
        self.comparison_data = None
        
    def load_all_studies(self, directories):
        """Load results from multiple study directories"""
        
        for study_name, study_dir in directories.items():
            try:
                study_path = Path(study_dir)
                
                # Try different file patterns
                result_files = (
                    list(study_path.glob("complete_*_study.json")) +
                    list(study_path.glob("*comparison*.csv")) +
                    list(study_path.glob("experiment_*_results.json"))
                )
                
                if result_files:
                    if result_files[0].suffix == '.json':
                        if 'experiment_' in result_files[0].name:
                            # Load individual experiment files
                            experiments = []
                            for exp_file in result_files:
                                with open(exp_file) as f:
                                    experiments.append(json.load(f))
                            self.studies[study_name] = experiments
                        else:
                            # Load complete study file
                            with open(result_files[0]) as f:
                                data = json.load(f)
                                self.studies[study_name] = data.get('experimental_results', data)
                    else:
                        # Load CSV
                        df = pd.read_csv(result_files[0])
                        self.studies[study_name] = df.to_dict('records')
                    
                    print(f"✅ Loaded {study_name}: {len(self.studies[study_name])} results")
                else:
                    print(f"⚠️  No results found for {study_name} in {study_dir}")
                    
            except Exception as e:
                print(f"❌ Error loading {study_name}: {e}")
    
    def create_unified_comparison_dataset(self):
        """Create a unified dataset for cross-study comparison"""
        
        unified_data = []
        
        for study_name, results in self.studies.items():
            for result in results:
                try:
                    # Extract model information
                    model_info = self.extract_model_info(result, study_name)
                    if model_info:
                        unified_data.append(model_info)
                except Exception as e:
                    print(f"Error processing result in {study_name}: {e}")
                    continue
        
        self.comparison_data = pd.DataFrame(unified_data)
        return self.comparison_data
    
    def extract_model_info(self, result, study_name):
        """Extract standardized model information from different result formats"""
        
        try:
            # Handle different result formats
            if isinstance(result, dict):
                if 'evaluation' in result:
                    # Experiment result format
                    model_name = result['evaluation']['model_name']
                    metrics = result['evaluation']['performance_metrics']
                    training_info = result.get('training', {}).get('training_performance', {})
                    architecture_info = result.get('training', {}).get('architecture_info', {})
                    
                    return {
                        'study_name': study_name,
                        'model_name': model_name,
                        'architecture_family': model_name.split('_')[0],
                        'architecture_type': self.classify_architecture_type(model_name),
                        'has_skip_connections': architecture_info.get('has_skip_connections', self.infer_skip_connections(model_name)),
                        'depth': architecture_info.get('depth', self.infer_depth(model_name)),
                        'parameters': architecture_info.get('parameters', 0),
                        
                        # Performance metrics
                        'r2_score': float(metrics.get('r2_score', 0)),
                        'mse': float(metrics.get('mse', 0)),
                        'mae': float(metrics.get('mae', 0)),
                        'rmse': float(metrics.get('rmse', 0)),
                        'mape': float(metrics.get('mape', 0)),
                        
                        # Training metrics
                        'training_time_min': float(training_info.get('training_minutes', 0)),
                        'convergence_epoch': int(training_info.get('convergence_epoch', 0)),
                        'epochs_completed': int(training_info.get('epochs_completed', 0)),
                        'best_val_loss': float(training_info.get('best_val_loss', 0)),
                        
                        # Derived metrics
                        'parameter_efficiency': float(metrics.get('r2_score', 0)) / (architecture_info.get('parameters', 1) / 1_000_000),
                        'time_efficiency': float(metrics.get('r2_score', 0)) / max(float(training_info.get('training_minutes', 1)), 0.1)
                    }
                    
                elif 'model_name' in result:
                    # Direct format
                    model_name = result['model_name']
                    
                    return {
                        'study_name': study_name,
                        'model_name': model_name,
                        'architecture_family': model_name.split('_')[0],
                        'architecture_type': self.classify_architecture_type(model_name),
                        'has_skip_connections': result.get('has_skip_connections', self.infer_skip_connections(model_name)),
                        'depth': result.get('depth', self.infer_depth(model_name)),
                        'parameters': int(result.get('parameters', 0)),
                        
                        'r2_score': float(result.get('r2_score', 0)),
                        'mse': float(result.get('mse', 0)),
                        'mae': float(result.get('mae', 0)),
                        'rmse': float(result.get('rmse', 0)),
                        'mape': float(result.get('mape', 0)),
                        
                        'training_time_min': float(result.get('training_time_min', 0)),
                        'convergence_epoch': int(result.get('convergence_epoch', 0)),
                        'best_val_loss': float(result.get('best_val_loss', 0)),
                        
                        'parameter_efficiency': float(result.get('r2_score', 0)) / (int(result.get('parameters', 1)) / 1_000_000),
                        'time_efficiency': float(result.get('r2_score', 0)) / max(float(result.get('training_time_min', 1)), 0.1)
                    }
            
            return None
            
        except Exception as e:
            print(f"Error extracting model info: {e}")
            return None
    
    def classify_architecture_type(self, model_name):
        """Classify architecture into broad categories"""
        name_lower = model_name.lower()
        
        if 'baseline' in name_lower:
            return 'Baseline'
        elif 'resnet' in name_lower:
            return 'ResNet'
        elif 'unet' in name_lower:
            return 'U-Net'
        elif 'densenet' in name_lower:
            return 'DenseNet'
        elif 'reduced' in name_lower:
            return 'Reduced'
        else:
            return 'Other'
    
    def infer_skip_connections(self, model_name):
        """Infer if model has skip connections from name"""
        name_lower = model_name.lower()
        return any(keyword in name_lower for keyword in ['resnet', 'unet', 'densenet'])
    
    def infer_depth(self, model_name):
        """Infer model depth from name"""
        if 'shallow' in model_name.lower():
            return 4
        elif 'deep' in model_name.lower():
            return 12
        else:
            return 6  # Default
    
    def create_master_comparison_visualization(self, output_dir):
        """Create comprehensive master comparison visualization"""
        
        if self.comparison_data is None:
            self.create_unified_comparison_dataset()
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        df = self.comparison_data
        
        # Create master figure
        fig = plt.figure(figsize=(24, 20))
        gs = fig.add_gridspec(5, 4, hspace=0.4, wspace=0.3)
        
        fig.suptitle('Master CNN Architecture Comparison Analysis', fontsize=20, fontweight='bold')
        
        # 1. Overall Performance Ranking
        ax1 = fig.add_subplot(gs[0, :2])
        
        # Sort by R² score
        top_models = df.nlargest(15, 'r2_score')
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_models)))
        
        bars = ax1.barh(range(len(top_models)), top_models['r2_score'], color=colors)
        ax1.set_yticks(range(len(top_models)))
        ax1.set_yticklabels(top_models['model_name'], fontsize=10)
        ax1.set_xlabel('R² Score')
        ax1.set_title('Top 15 Models by R² Score')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, r2) in enumerate(zip(bars, top_models['r2_score'])):
            ax1.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{r2:.4f}', va='center', fontsize=8)
        
        # 2. MSE vs R² Performance Map
        ax2 = fig.add_subplot(gs[0, 2:])
        
        scatter = ax2.scatter(df['r2_score'], df['mse'], 
                             s=df['parameters']/5000,
                             c=df['training_time_min'], 
                             cmap='plasma', alpha=0.7)
        ax2.set_xlabel('R² Score')
        ax2.set_ylabel('MSE')
        ax2.set_title('Performance Landscape\\n(Size=Parameters, Color=Training Time)')
        ax2.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Training Time (min)')
        
        # 3. Architecture Family Performance
        ax3 = fig.add_subplot(gs[1, :2])
        
        family_stats = df.groupby('architecture_family').agg({
            'r2_score': ['mean', 'std', 'count'],
            'mse': 'mean',
            'parameter_efficiency': 'mean'
        }).round(4)
        
        families = family_stats.index
        mean_r2 = family_stats[('r2_score', 'mean')]
        std_r2 = family_stats[('r2_score', 'std')]
        
        x_pos = np.arange(len(families))
        bars = ax3.bar(x_pos, mean_r2, yerr=std_r2, capsize=5, alpha=0.7, color='skyblue')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(families, rotation=45)
        ax3.set_ylabel('Average R² Score')
        ax3.set_title('Performance by Architecture Family\\n(Error bars = ±1 std)')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, r2 in zip(bars, mean_r2):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{r2:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Skip Connections Analysis
        ax4 = fig.add_subplot(gs[1, 2:])
        
        skip_data = df.groupby('has_skip_connections').agg({
            'r2_score': ['mean', 'std'],
            'mse': ['mean', 'std'],
            'parameter_efficiency': 'mean',
            'time_efficiency': 'mean'
        }).round(4)
        
        categories = ['R² Score', 'MSE', 'Param Efficiency', 'Time Efficiency']
        skip_true = [skip_data[('r2_score', 'mean')][True],
                    skip_data[('mse', 'mean')][True],
                    skip_data[('parameter_efficiency', 'mean')][True],
                    skip_data[('time_efficiency', 'mean')][True]]
        skip_false = [skip_data[('r2_score', 'mean')][False],
                     skip_data[('mse', 'mean')][False],
                     skip_data[('parameter_efficiency', 'mean')][False],
                     skip_data[('time_efficiency', 'mean')][False]]
        
        # Normalize for comparison (higher is better)
        skip_true_norm = np.array(skip_true) / np.array([1, max(skip_data[('mse', 'mean')]), 1, 1])
        skip_false_norm = np.array(skip_false) / np.array([1, max(skip_data[('mse', 'mean')]), 1, 1])
        skip_true_norm[1] = 1 - skip_true_norm[1]  # Invert MSE (lower is better)
        skip_false_norm[1] = 1 - skip_false_norm[1]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax4.bar(x - width/2, skip_true_norm, width, label='Skip Connections', alpha=0.7)
        ax4.bar(x + width/2, skip_false_norm, width, label='No Skip Connections', alpha=0.7)
        
        ax4.set_xlabel('Metrics')
        ax4.set_ylabel('Normalized Performance')
        ax4.set_title('Skip Connections vs No Skip Connections\\n(Normalized, Higher = Better)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(categories, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Parameter Efficiency Landscape
        ax5 = fig.add_subplot(gs[2, :2])
        
        scatter = ax5.scatter(df['parameters']/1_000_000, df['parameter_efficiency'],
                             c=df['has_skip_connections'], cmap='RdBu', s=60, alpha=0.7)
        ax5.set_xlabel('Parameters (Millions)')
        ax5.set_ylabel('R² per Million Parameters')
        ax5.set_title('Parameter Efficiency Landscape\\n(Color: Red=Skip, Blue=No Skip)')
        ax5.grid(True, alpha=0.3)
        
        # Add best efficiency annotations
        best_efficiency = df.loc[df['parameter_efficiency'].idxmax()]
        ax5.annotate(f"Best: {best_efficiency['model_name'][:15]}...",
                    xy=(best_efficiency['parameters']/1_000_000, best_efficiency['parameter_efficiency']),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # 6. Training Dynamics Analysis
        ax6 = fig.add_subplot(gs[2, 2:])
        
        # Training time vs performance colored by convergence epoch
        scatter = ax6.scatter(df['training_time_min'], df['r2_score'],
                             c=df['convergence_epoch'], cmap='coolwarm', s=60, alpha=0.7)
        ax6.set_xlabel('Training Time (minutes)')
        ax6.set_ylabel('R² Score')
        ax6.set_title('Training Dynamics\\n(Color = Convergence Epoch)')
        ax6.grid(True, alpha=0.3)
        
        cbar2 = plt.colorbar(scatter, ax=ax6)
        cbar2.set_label('Convergence Epoch')
        
        # 7. Study Comparison
        ax7 = fig.add_subplot(gs[3, :2])
        
        if len(df['study_name'].unique()) > 1:
            study_comparison = df.groupby('study_name').agg({
                'r2_score': 'mean',
                'mse': 'mean',
                'training_time_min': 'mean',
                'parameter_efficiency': 'mean'
            }).round(4)
            
            studies = study_comparison.index
            x_pos = np.arange(len(studies))
            
            bars = ax7.bar(x_pos, study_comparison['r2_score'], alpha=0.7)
            ax7.set_xticks(x_pos)
            ax7.set_xticklabels(studies, rotation=45, ha='right')
            ax7.set_ylabel('Average R² Score')
            ax7.set_title('Performance Comparison Across Studies')
            ax7.grid(True, alpha=0.3)
            
            for bar, r2 in zip(bars, study_comparison['r2_score']):
                ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{r2:.3f}', ha='center', va='bottom', fontsize=9)
        else:
            ax7.text(0.5, 0.5, 'Single Study Analysis', ha='center', va='center', fontsize=14)
            ax7.set_xlim(0, 1)
            ax7.set_ylim(0, 1)
        
        # 8. Performance Distribution Analysis
        ax8 = fig.add_subplot(gs[3, 2:])
        
        # Box plots by architecture type
        arch_types = df['architecture_type'].unique()
        data_for_boxplot = [df[df['architecture_type'] == arch]['r2_score'].values 
                           for arch in arch_types]
        
        bp = ax8.boxplot(data_for_boxplot, labels=arch_types, patch_artist=True)
        
        # Color the boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax8.set_xticklabels(arch_types, rotation=45)
        ax8.set_ylabel('R² Score')
        ax8.set_title('Performance Distribution by Architecture Type')
        ax8.grid(True, alpha=0.3)
        
        # 9. Comprehensive Rankings Table
        ax9 = fig.add_subplot(gs[4, :])
        ax9.axis('off')
        
        # Create comprehensive ranking
        df_ranked = df.copy()
        
        # Calculate composite scores
        df_ranked['performance_rank'] = df_ranked['r2_score'].rank(ascending=False)
        df_ranked['efficiency_rank'] = df_ranked['parameter_efficiency'].rank(ascending=False)
        df_ranked['time_rank'] = df_ranked['time_efficiency'].rank(ascending=False)
        df_ranked['composite_score'] = (df_ranked['performance_rank'] + 
                                       df_ranked['efficiency_rank'] + 
                                       df_ranked['time_rank']) / 3
        
        # Get top 8 models
        top_composite = df_ranked.nsmallest(8, 'composite_score')
        
        table_data = []
        for i, (_, row) in enumerate(top_composite.iterrows(), 1):
            table_data.append([
                f"{i}",
                row['model_name'][:20] + "..." if len(row['model_name']) > 20 else row['model_name'],
                f"{row['r2_score']:.4f}",
                f"{row['mse']:.0f}",
                f"{row['parameter_efficiency']:.3f}",
                f"{row['training_time_min']:.1f}",
                "✓" if row['has_skip_connections'] else "✗",
                row['study_name'][:10] + "..." if len(row['study_name']) > 10 else row['study_name']
            ])
        
        table = ax9.table(cellText=table_data,
                         colLabels=['Rank', 'Model', 'R²', 'MSE', 'Efficiency', 'Time(min)', 'Skip', 'Study'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Color code the table
        for i in range(1, 9):  # Skip header row
            for j in range(len(table_data[0])):
                if i <= 3:  # Top 3
                    color = 'lightgreen'
                elif i <= 5:  # Top 5
                    color = 'lightyellow'
                else:
                    color = 'lightgray'
                table[(i, j)].set_facecolor(color)
        
        ax9.set_title('Top Models - Comprehensive Ranking (Performance + Efficiency + Speed)', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Save the master visualization
        output_file = output_path / 'master_comparison_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\\n🎨 Master comparison visualization saved: {output_file}")
        
        return output_file
    
    def generate_executive_summary(self, output_dir):
        """Generate executive summary of all analyses"""
        
        if self.comparison_data is None:
            self.create_unified_comparison_dataset()
        
        df = self.comparison_data
        
        # Calculate key statistics
        best_model = df.loc[df['r2_score'].idxmax()]
        most_efficient = df.loc[df['parameter_efficiency'].idxmax()]
        fastest_training = df.loc[df['time_efficiency'].idxmax()]
        
        skip_conn_performance = df[df['has_skip_connections']]['r2_score'].mean()
        no_skip_performance = df[~df['has_skip_connections']]['r2_score'].mean()
        
        report_lines = [
            "=" * 80,
            "EXECUTIVE SUMMARY - CNN ARCHITECTURE COMPARISON STUDY",
            "=" * 80,
            f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Models Analyzed: {len(df)}",
            f"Studies Included: {', '.join(df['study_name'].unique())}",
            f"Architecture Families: {', '.join(df['architecture_family'].unique())}",
            "",
            "🏆 KEY FINDINGS:",
            "",
            f"BEST OVERALL PERFORMANCE:",
            f"   Model: {best_model['model_name']}",
            f"   R² Score: {best_model['r2_score']:.4f}",
            f"   MSE: {best_model['mse']:.0f}",
            f"   Architecture: {best_model['architecture_family']} ({'with' if best_model['has_skip_connections'] else 'without'} skip connections)",
            "",
            f"MOST PARAMETER EFFICIENT:",
            f"   Model: {most_efficient['model_name']}",
            f"   Efficiency: {most_efficient['parameter_efficiency']:.3f} R²/M parameters",
            f"   Parameters: {most_efficient['parameters']:,}",
            "",
            f"FASTEST TRAINING:",
            f"   Model: {fastest_training['model_name']}",
            f"   Time Efficiency: {fastest_training['time_efficiency']:.3f} R²/minute",
            f"   Training Time: {fastest_training['training_time_min']:.1f} minutes",
            "",
            f"SKIP CONNECTIONS ANALYSIS:",
            f"   With Skip Connections:    Average R² = {skip_conn_performance:.4f}",
            f"   Without Skip Connections: Average R² = {no_skip_performance:.4f}",
            f"   Advantage: {'Skip Connections' if skip_conn_performance > no_skip_performance else 'No Skip Connections'} "
            f"by {abs(skip_conn_performance - no_skip_performance):.4f}",
            "",
            "📊 STATISTICAL OVERVIEW:",
            f"   R² Score Range: {df['r2_score'].min():.4f} - {df['r2_score'].max():.4f}",
            f"   MSE Range: {df['mse'].min():.0f} - {df['mse'].max():.0f}",
            f"   Parameter Range: {df['parameters'].min():,} - {df['parameters'].max():,}",
            f"   Training Time Range: {df['training_time_min'].min():.1f} - {df['training_time_min'].max():.1f} minutes",
            "",
            "🎯 RECOMMENDATIONS:",
            f"   • For maximum accuracy: Use {best_model['model_name']}",
            f"   • For resource constraints: Use {most_efficient['model_name']}",
            f"   • For fast deployment: Use {fastest_training['model_name']}",
            f"   • Skip connections: {'Recommended' if skip_conn_performance > no_skip_performance else 'Not recommended'} for this task",
            "",
            "=" * 80
        ]
        
        # Save executive summary
        output_path = Path(output_dir)
        summary_file = output_path / 'executive_summary.txt'
        with open(summary_file, 'w') as f:
            f.write('\\n'.join(report_lines))
        
        # Save detailed CSV
        detailed_csv = output_path / 'comprehensive_comparison_data.csv'
        df.to_csv(detailed_csv, index=False)
        
        print(f"📋 Executive summary saved: {summary_file}")
        print(f"📊 Detailed data saved: {detailed_csv}")
        
        return summary_file
    
    def run_complete_analysis(self, output_dir):
        """Run the complete analysis suite"""
        
        print("🔬 COMPREHENSIVE COMPARISON ANALYSIS SUITE")
        print("=" * 60)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create unified dataset
        print("📊 Creating unified comparison dataset...")
        self.create_unified_comparison_dataset()
        
        if self.comparison_data is None or len(self.comparison_data) == 0:
            print("❌ No data available for analysis")
            return
        
        print(f"✅ Unified dataset created: {len(self.comparison_data)} models")
        
        # Generate all analyses
        print("🎨 Creating master visualization...")
        viz_file = self.create_master_comparison_visualization(output_dir)
        
        print("📋 Generating executive summary...")
        summary_file = self.generate_executive_summary(output_dir)
        
        print("\\n🎉 COMPREHENSIVE ANALYSIS COMPLETE!")
        print(f"📁 Output directory: {output_path}")
        print(f"📊 Master visualization: {viz_file.name}")
        print(f"📋 Executive summary: {summary_file.name}")
        
        return {
            'output_dir': output_path,
            'visualization': viz_file,
            'summary': summary_file,
            'data': output_path / 'comprehensive_comparison_data.csv'
        }

def main():
    """Main function for running comprehensive comparison"""
    
    parser = argparse.ArgumentParser(description='Comprehensive CNN Architecture Comparison Suite')
    parser.add_argument('--studies', nargs='+', required=True,
                       help='Study directories to compare (format: study_name:directory_path)')
    parser.add_argument('--output', type=str, default='comprehensive_analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # Parse study directories
    study_dirs = {}
    for study_spec in args.studies:
        if ':' in study_spec:
            name, path = study_spec.split(':', 1)
            study_dirs[name] = path
        else:
            # Use directory name as study name
            study_dirs[Path(study_spec).name] = study_spec
    
    print(f"Studies to compare: {list(study_dirs.keys())}")
    
    # Create comparison suite
    suite = ComprehensiveComparisonSuite(study_dirs)
    
    # Run complete analysis
    results = suite.run_complete_analysis(args.output)
    
    return results

if __name__ == "__main__":
    # Example usage for testing
    print("🔬 Comprehensive Comparison Analysis Suite")
    print("This suite integrates all analysis tools for complete architectural comparison")
    print("\\nExample usage:")
    print("python comprehensive_comparison_suite.py \\\\")
    print("  --studies original:study1_dir reduced:study2_dir \\\\")
    print("  --output comprehensive_analysis")
    
    # You can also run it directly with hardcoded paths for testing
    try:
        study_directories = {
            'Original': '../skip_connections_study/comprehensive_architecture_study_20250911_173012',
            # Add more studies here when available
        }
        
        suite = ComprehensiveComparisonSuite(study_directories)
        if hasattr(suite, 'studies') and suite.studies:
            print(f"\\n✅ Loaded {len(suite.studies)} studies for analysis")
        else:
            print("\\n⚠️  No studies loaded - this is expected for testing")
            
    except Exception as e:
        print(f"\\n⚠️  Testing mode - {e}")
        print("Use with actual study directories for full analysis")