#!/usr/bin/env python3
"""
Degradation Pattern Analysis Framework
Compares how different architectures degrade when layer depth is reduced
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

class DegradationAnalyzer:
    """Analyzes performance degradation patterns between original and reduced architectures"""
    
    def __init__(self, original_results_dir, reduced_results_dir):
        self.original_results = self.load_results(original_results_dir)
        self.reduced_results = self.load_results(reduced_results_dir)
        self.degradation_data = None
        
    def load_results(self, results_dir):
        """Load results from study directory"""
        results_path = Path(results_dir)
        
        # Try to find the complete results JSON file
        json_files = list(results_path.glob("complete_*_study.json"))
        if not json_files:
            json_files = list(results_path.glob("*comparison*.csv"))
        
        if not json_files:
            raise FileNotFoundError(f"No results found in {results_dir}")
        
        if json_files[0].suffix == '.json':
            with open(json_files[0]) as f:
                data = json.load(f)
                return data.get('experimental_results', data)
        else:
            # Load from CSV
            df = pd.read_csv(json_files[0])
            return df.to_dict('records')
    
    def match_architectures(self):
        """Match original and reduced architectures for comparison"""
        
        architecture_pairs = []
        
        # Define expected mappings
        mappings = {
            'Baseline_Shallow': 'ReducedBaseline_Shallow',
            'Baseline_Deep': 'ReducedBaseline_Deep',
            'ResNet_Shallow': 'ReducedResNet_Shallow', 
            'ResNet_Deep': 'ReducedResNet_Deep',
            'UNet_channel_reduced_32filters': 'ReducedUNet_channel_reduced_32filters',
            'UNet_channel_reduced_36filters': 'ReducedUNet_channel_reduced_36filters',
            'DenseNet_Style': 'ReducedDenseNet_Style'
        }
        
        for original_name, reduced_name in mappings.items():
            # Find original result
            original_result = None
            for result in self.original_results:
                if self.get_model_name(result) == original_name:
                    original_result = result
                    break
            
            # Find reduced result
            reduced_result = None
            for result in self.reduced_results:
                if self.get_model_name(result) == reduced_name:
                    reduced_result = result
                    break
            
            if original_result and reduced_result:
                architecture_pairs.append({
                    'architecture_family': original_name.split('_')[0],
                    'architecture_type': original_name,
                    'original': original_result,
                    'reduced': reduced_result
                })
        
        return architecture_pairs
    
    def get_model_name(self, result):
        """Extract model name from result, handling different formats"""
        if isinstance(result, dict):
            return result.get('model_name', result.get('evaluation', {}).get('model_name', 'Unknown'))
        return getattr(result, 'model_name', 'Unknown')
    
    def calculate_degradation_metrics(self):
        """Calculate comprehensive degradation metrics"""
        
        architecture_pairs = self.match_architectures()
        degradation_data = []
        
        print("🔍 DEGRADATION ANALYSIS")
        print("=" * 60)
        print(f"{'Architecture':<25} {'Original R²':<12} {'Reduced R²':<12} {'Degradation':<12}")
        print("-" * 60)
        
        for pair in architecture_pairs:
            original = pair['original']
            reduced = pair['reduced']
            
            # Extract performance metrics
            orig_perf = self.extract_performance_metrics(original)
            red_perf = self.extract_performance_metrics(reduced)
            
            if orig_perf and red_perf:
                # Calculate degradation metrics
                r2_degradation = orig_perf['r2_score'] - red_perf['r2_score']
                r2_degradation_pct = (r2_degradation / orig_perf['r2_score']) * 100 if orig_perf['r2_score'] != 0 else 0
                
                mse_degradation = red_perf['mse'] - orig_perf['mse']
                mse_degradation_pct = (mse_degradation / orig_perf['mse']) * 100 if orig_perf['mse'] != 0 else 0
                
                # Calculate robustness score (lower degradation = higher robustness)
                robustness_score = 1 - abs(r2_degradation_pct) / 100
                
                # Extract architectural info
                orig_params = self.extract_parameters(original)
                red_params = self.extract_parameters(reduced)
                
                degradation_entry = {
                    'architecture_family': pair['architecture_family'],
                    'architecture_type': pair['architecture_type'],
                    'has_skip_connections': self.has_skip_connections(original),
                    
                    # Original metrics
                    'original_r2': orig_perf['r2_score'],
                    'original_mse': orig_perf['mse'],
                    'original_parameters': orig_params,
                    'original_training_time': orig_perf.get('training_time', 0),
                    
                    # Reduced metrics
                    'reduced_r2': red_perf['r2_score'],
                    'reduced_mse': red_perf['mse'], 
                    'reduced_parameters': red_params,
                    'reduced_training_time': red_perf.get('training_time', 0),
                    
                    # Degradation metrics
                    'r2_degradation': r2_degradation,
                    'r2_degradation_pct': r2_degradation_pct,
                    'mse_degradation': mse_degradation,
                    'mse_degradation_pct': mse_degradation_pct,
                    'robustness_score': robustness_score,
                    
                    # Efficiency metrics
                    'parameter_reduction': orig_params - red_params if orig_params and red_params else 0,
                    'parameter_reduction_pct': ((orig_params - red_params) / orig_params * 100) if orig_params and red_params and orig_params > 0 else 0,
                    'performance_per_param_original': orig_perf['r2_score'] / (orig_params / 1_000_000) if orig_params else 0,
                    'performance_per_param_reduced': red_perf['r2_score'] / (red_params / 1_000_000) if red_params else 0
                }
                
                degradation_data.append(degradation_entry)
                
                # Print comparison
                print(f"{pair['architecture_type']:<25} {orig_perf['r2_score']:<12.4f} {red_perf['r2_score']:<12.4f} {r2_degradation_pct:<12.1f}%")
        
        self.degradation_data = pd.DataFrame(degradation_data)
        return self.degradation_data
    
    def extract_performance_metrics(self, result):
        """Extract performance metrics from result, handling different formats"""
        try:
            if isinstance(result, dict):
                # Handle experiment result format
                if 'evaluation' in result:
                    metrics = result['evaluation']['performance_metrics']
                    training_time = result.get('training', {}).get('training_performance', {}).get('training_minutes', 0)
                elif 'r2_score' in result:
                    # Handle direct metrics format
                    metrics = result
                    training_time = result.get('training_time_min', 0)
                else:
                    return None
                
                return {
                    'r2_score': float(metrics.get('r2_score', 0)),
                    'mse': float(metrics.get('mse', 0)),
                    'mae': float(metrics.get('mae', 0)),
                    'rmse': float(metrics.get('rmse', 0)),
                    'training_time': training_time
                }
            return None
        except Exception as e:
            print(f"Error extracting performance metrics: {e}")
            return None
    
    def extract_parameters(self, result):
        """Extract parameter count from result"""
        try:
            if isinstance(result, dict):
                if 'evaluation' in result:
                    return result['evaluation']['efficiency_metrics']['parameters']
                elif 'parameters' in result:
                    return result['parameters']
            return None
        except:
            return None
    
    def has_skip_connections(self, result):
        """Determine if architecture has skip connections"""
        try:
            if isinstance(result, dict):
                if 'training' in result and 'architecture_info' in result['training']:
                    return result['training']['architecture_info'].get('has_skip_connections', False)
                return result.get('has_skip_connections', False)
            return False
        except:
            return False
    
    def create_degradation_analysis_plots(self, output_dir):
        """Create comprehensive degradation analysis visualizations"""
        
        if self.degradation_data is None:
            self.calculate_degradation_metrics()
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create main figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle('CNN Architecture Degradation Pattern Analysis', fontsize=18, fontweight='bold')
        
        df = self.degradation_data
        
        # 1. R² Degradation by Architecture Family
        ax1 = fig.add_subplot(gs[0, :2])
        
        families = df['architecture_family'].unique()
        skip_conn_data = []
        no_skip_data = []
        
        for family in families:
            family_data = df[df['architecture_family'] == family]
            if family_data['has_skip_connections'].iloc[0]:
                skip_conn_data.extend(family_data['r2_degradation_pct'].values)
            else:
                no_skip_data.extend(family_data['r2_degradation_pct'].values)
        
        x_pos = np.arange(len(families))
        degradation_by_family = [df[df['architecture_family'] == family]['r2_degradation_pct'].mean() 
                                for family in families]
        colors = ['red' if df[df['architecture_family'] == family]['has_skip_connections'].iloc[0] 
                 else 'blue' for family in families]
        
        bars = ax1.bar(x_pos, degradation_by_family, color=colors, alpha=0.7)
        ax1.set_xlabel('Architecture Family')
        ax1.set_ylabel('Average R² Degradation (%)')
        ax1.set_title('Performance Degradation by Architecture Family\\n(Red=Skip Connections, Blue=No Skip)')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(families, rotation=45)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, degradation_by_family):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # 2. Robustness Score Comparison
        ax2 = fig.add_subplot(gs[0, 2])
        skip_robustness = df[df['has_skip_connections']]['robustness_score']
        no_skip_robustness = df[~df['has_skip_connections']]['robustness_score']
        
        ax2.boxplot([skip_robustness, no_skip_robustness], labels=['Skip Connections', 'No Skip'])
        ax2.set_ylabel('Robustness Score')
        ax2.set_title('Robustness Score Distribution')
        ax2.grid(True, alpha=0.3)
        
        # 3. Original vs Reduced Performance Scatter
        ax3 = fig.add_subplot(gs[1, :2])
        scatter = ax3.scatter(df['original_r2'], df['reduced_r2'], 
                             c=df['has_skip_connections'], cmap='RdBu', 
                             s=df['original_parameters']/10000, alpha=0.7)
        ax3.plot([0.9, 1.0], [0.9, 1.0], 'k--', alpha=0.5, label='Perfect retention')
        ax3.set_xlabel('Original R² Score')
        ax3.set_ylabel('Reduced R² Score')
        ax3.set_title('Original vs Reduced Performance\\n(Size=Parameters, Color=Skip Connections)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Add architecture labels
        for idx, row in df.iterrows():
            ax3.annotate(row['architecture_family'], 
                        (row['original_r2'], row['reduced_r2']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 4. Parameter Efficiency vs Robustness
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.scatter(df['performance_per_param_original'], df['robustness_score'],
                   c=df['has_skip_connections'], cmap='RdBu', s=100, alpha=0.7)
        ax4.set_xlabel('Original Performance per Million Parameters')
        ax4.set_ylabel('Robustness Score')
        ax4.set_title('Parameter Efficiency vs Robustness')
        ax4.grid(True, alpha=0.3)
        
        # 5. Degradation Pattern Heatmap
        ax5 = fig.add_subplot(gs[2, :2])
        
        # Prepare heatmap data
        heatmap_data = df.pivot_table(values='r2_degradation_pct', 
                                     index='architecture_type', 
                                     columns='has_skip_connections',
                                     aggfunc='mean')
        
        if heatmap_data.shape[1] > 1:
            sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlBu_r', 
                       center=0, ax=ax5, cbar_kws={'label': 'R² Degradation (%)'})
            ax5.set_title('Degradation Pattern Heatmap')
            ax5.set_xlabel('Skip Connections')
            ax5.set_ylabel('Architecture Type')
        
        # 6. Training Time vs Performance Trade-off
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.scatter(df['reduced_training_time'], df['reduced_r2'],
                   c=df['r2_degradation_pct'], cmap='RdYlGn_r', s=100, alpha=0.7)
        ax6.set_xlabel('Reduced Model Training Time (min)')
        ax6.set_ylabel('Reduced Model R² Score')
        ax6.set_title('Training Efficiency vs Performance\\n(Color=Degradation %)')
        ax6.grid(True, alpha=0.3)
        
        # 7. Architecture Comparison Table
        ax7 = fig.add_subplot(gs[3, :])
        ax7.axis('off')
        
        # Prepare summary statistics
        summary_stats = []
        for arch_family in df['architecture_family'].unique():
            family_data = df[df['architecture_family'] == arch_family]
            summary_stats.append([
                arch_family,
                f"{family_data['original_r2'].iloc[0]:.3f}",
                f"{family_data['reduced_r2'].iloc[0]:.3f}",
                f"{family_data['r2_degradation_pct'].iloc[0]:.1f}%",
                f"{family_data['robustness_score'].iloc[0]:.3f}",
                "✓" if family_data['has_skip_connections'].iloc[0] else "✗"
            ])
        
        # Sort by robustness score (descending)
        summary_stats.sort(key=lambda x: float(x[4]), reverse=True)
        
        table = ax7.table(cellText=summary_stats,
                         colLabels=['Architecture', 'Original R²', 'Reduced R²', 'Degradation', 'Robustness', 'Skip Conn.'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color code by robustness
        for i in range(1, len(summary_stats) + 1):
            robustness = float(summary_stats[i-1][4])
            if robustness > 0.95:
                color = 'lightgreen'
            elif robustness > 0.90:
                color = 'lightyellow' 
            else:
                color = 'lightcoral'
            
            for j in range(len(summary_stats[0])):
                table[(i, j)].set_facecolor(color)
        
        ax7.set_title('Architecture Robustness Rankings', fontsize=14, pad=20)
        
        # Save the comprehensive plot
        output_file = output_path / 'degradation_pattern_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\\n📊 Degradation analysis plot saved: {output_file}")
        
        return output_file
    
    def generate_degradation_report(self, output_dir):
        """Generate comprehensive degradation analysis report"""
        
        if self.degradation_data is None:
            self.calculate_degradation_metrics()
        
        df = self.degradation_data
        
        report_lines = [
            "=" * 80,
            "CNN ARCHITECTURE DEGRADATION PATTERN ANALYSIS REPORT", 
            "=" * 80,
            f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Architectures Analyzed: {len(df)}",
            "",
            "RESEARCH QUESTION:",
            "How do different CNN architectures degrade when layer depth is reduced?",
            "Do skip connections provide more graceful degradation under constraints?",
            "",
            "=" * 40 + " EXECUTIVE SUMMARY " + "=" * 40,
            ""
        ]
        
        # Overall statistics
        avg_degradation_skip = df[df['has_skip_connections']]['r2_degradation_pct'].mean()
        avg_degradation_no_skip = df[~df['has_skip_connections']]['r2_degradation_pct'].mean()
        avg_robustness_skip = df[df['has_skip_connections']]['robustness_score'].mean()
        avg_robustness_no_skip = df[~df['has_skip_connections']]['robustness_score'].mean()
        
        report_lines.extend([
            f"SKIP CONNECTIONS vs NO SKIP CONNECTIONS:",
            f"   Average Degradation (Skip Connections):    {avg_degradation_skip:6.1f}%",
            f"   Average Degradation (No Skip Connections): {avg_degradation_no_skip:6.1f}%",
            f"   Average Robustness (Skip Connections):     {avg_robustness_skip:6.3f}",
            f"   Average Robustness (No Skip Connections):  {avg_robustness_no_skip:6.3f}",
            "",
        ])
        
        # Determine winner
        if avg_robustness_skip > avg_robustness_no_skip:
            winner = "SKIP CONNECTIONS"
            advantage = avg_robustness_skip - avg_robustness_no_skip
        else:
            winner = "NO SKIP CONNECTIONS" 
            advantage = avg_robustness_no_skip - avg_robustness_skip
        
        report_lines.extend([
            f"🏆 WINNER: {winner}",
            f"   Robustness Advantage: {advantage:.3f}",
            "",
            "=" * 35 + " DETAILED RANKINGS " + "=" * 35,
            "",
            "ROBUSTNESS RANKING (Best to Worst):",
            "Rank | Architecture              | Original R² | Reduced R² | Degradation | Robustness | Skip",
            "-" * 100,
        ])
        
        # Sort by robustness score
        ranked_df = df.sort_values('robustness_score', ascending=False)
        
        for i, (_, row) in enumerate(ranked_df.iterrows(), 1):
            skip_indicator = "✓" if row['has_skip_connections'] else "✗"
            report_lines.append(
                f"{i:4d} | {row['architecture_type']:25} | {row['original_r2']:11.4f} | "
                f"{row['reduced_r2']:10.4f} | {row['r2_degradation_pct']:10.1f}% | "
                f"{row['robustness_score']:10.3f} | {skip_indicator:4s}"
            )
        
        report_lines.extend([
            "",
            "=" * 30 + " ARCHITECTURE FAMILY ANALYSIS " + "=" * 30,
            "",
        ])
        
        # Analysis by architecture family
        for family in df['architecture_family'].unique():
            family_data = df[df['architecture_family'] == family].iloc[0]
            
            status = "ROBUST" if family_data['robustness_score'] > 0.95 else \
                    "MODERATE" if family_data['robustness_score'] > 0.90 else "FRAGILE"
            
            report_lines.extend([
                f"🏗️  {family.upper()} ARCHITECTURE:",
                f"   Skip Connections: {'Yes' if family_data['has_skip_connections'] else 'No'}",
                f"   Original Performance: R² = {family_data['original_r2']:.4f}",
                f"   Reduced Performance:  R² = {family_data['reduced_r2']:.4f}",
                f"   Performance Degradation: {family_data['r2_degradation_pct']:.1f}%",
                f"   Robustness Classification: {status}",
                f"   Parameter Efficiency: {family_data['performance_per_param_reduced']:.3f} R²/M params",
                "",
            ])
        
        report_lines.extend([
            "=" * 35 + " KEY INSIGHTS " + "=" * 35,
            "",
            f"💡 ARCHITECTURAL INSIGHTS:",
        ])
        
        # Generate insights based on data
        best_arch = ranked_df.iloc[0]
        worst_arch = ranked_df.iloc[-1]
        
        insights = [
            f"   • Most Robust Architecture: {best_arch['architecture_type']} (Robustness: {best_arch['robustness_score']:.3f})",
            f"   • Least Robust Architecture: {worst_arch['architecture_type']} (Robustness: {worst_arch['robustness_score']:.3f})",
        ]
        
        if avg_robustness_skip > avg_robustness_no_skip:
            insights.append(f"   • Skip connections provide {advantage:.1%} better robustness on average")
        else:
            insights.append(f"   • Simple architectures are {advantage:.1%} more robust than skip connections")
        
        # Parameter efficiency insights
        most_efficient = df.loc[df['performance_per_param_reduced'].idxmax()]
        insights.append(f"   • Most Parameter Efficient: {most_efficient['architecture_type']} ({most_efficient['performance_per_param_reduced']:.3f} R²/M params)")
        
        report_lines.extend(insights)
        
        report_lines.extend([
            "",
            f"📊 STATISTICAL SIGNIFICANCE:",
            f"   Sample Size: {len(df)} architecture pairs",
            f"   Degradation Range: {df['r2_degradation_pct'].min():.1f}% to {df['r2_degradation_pct'].max():.1f}%",
            f"   Robustness Range: {df['robustness_score'].min():.3f} to {df['robustness_score'].max():.3f}",
            "",
            "=" * 40 + " RECOMMENDATIONS " + "=" * 40,
            "",
            f"🎯 FOR RESOURCE-CONSTRAINED DEPLOYMENT:",
            f"   1. Use {best_arch['architecture_type']} for best robustness",
            f"   2. Consider parameter efficiency vs robustness trade-offs",
            f"   3. {'Prefer' if avg_robustness_skip > avg_robustness_no_skip else 'Avoid'} skip connections for depth-limited scenarios",
            "",
            f"⚖️  PERFORMANCE vs EFFICIENCY TRADE-OFFS:",
            f"   • High robustness architectures maintain performance under constraints",
            f"   • Consider training time vs robustness for deployment decisions",
            f"   • Parameter reduction opportunities vary by architecture family",
            "",
            "=" * 80,
            "END OF DEGRADATION ANALYSIS REPORT",
            "=" * 80
        ])
        
        # Save report
        output_path = Path(output_dir)
        report_file = output_path / 'degradation_analysis_report.txt'
        with open(report_file, 'w') as f:
            f.write('\\n'.join(report_lines))
        
        # Save detailed CSV
        csv_file = output_path / 'detailed_degradation_analysis.csv'
        df.to_csv(csv_file, index=False)
        
        print(f"\\n📋 Degradation analysis report saved: {report_file}")
        print(f"📊 Detailed data saved: {csv_file}")
        
        return report_file

def main():
    """Main function for testing"""
    print("🔍 Degradation Pattern Analysis Framework")
    print("This framework compares original vs reduced-layer architectures")
    print("\\nTo use:")
    print("1. Run your original comprehensive study")
    print("2. Run the reduced-layer study") 
    print("3. Use this analyzer to compare results")
    
    # Example usage (if data is available)
    try:
        analyzer = DegradationAnalyzer(
            "../skip_connections_study/comprehensive_architecture_study_20250911_173012",
            "reduced_layer_study_20250912_000000"  # Placeholder
        )
        
        # This would run the full analysis
        # degradation_data = analyzer.calculate_degradation_metrics()
        # analyzer.create_degradation_analysis_plots("degradation_analysis_output")
        # analyzer.generate_degradation_report("degradation_analysis_output")
        
    except FileNotFoundError:
        print("\\n⚠️  No study results found - this is expected for testing")
        print("Run the studies first, then use this analyzer")

if __name__ == "__main__":
    main()