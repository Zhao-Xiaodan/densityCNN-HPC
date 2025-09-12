#!/usr/bin/env python3
"""
Simple MSE Analysis using only standard libraries and numpy
Works without pandas/matplotlib dependencies
"""

import json
import numpy as np
import glob
import os
from pathlib import Path

def load_experiment_results(study_dir):
    """Load all experimental results from JSON files"""
    
    study_path = Path(study_dir)
    experiment_files = list(study_path.glob("experiment_*_results.json"))
    
    print(f"Found {len(experiment_files)} experiment files in {study_dir}")
    
    results = []
    
    for exp_file in sorted(experiment_files):
        try:
            with open(exp_file, 'r') as f:
                exp_data = json.load(f)
            
            # Extract key metrics
            model_name = exp_data['evaluation']['model_name']
            metrics = exp_data['evaluation']['performance_metrics']
            training_info = exp_data['training']
            
            result = {
                'model_name': model_name,
                'r2_score': float(metrics['r2_score']),
                'mse': float(metrics['mse']),
                'rmse': float(metrics['rmse']),
                'mae': float(metrics['mae']),
                'parameters': int(exp_data['evaluation']['efficiency_metrics']['parameters']),
                'training_time_min': float(training_info['training_performance']['training_minutes']),
                'convergence_epoch': int(training_info['training_performance']['convergence_epoch']),
                'best_val_loss': float(training_info['training_performance']['best_val_loss']),
                'has_skip_connections': bool(exp_data['training']['architecture_info']['has_skip_connections'])
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {exp_file}: {e}")
    
    return results

def calculate_enhanced_metrics(results):
    """Calculate additional comparison metrics"""
    
    if not results:
        return results
    
    # Extract arrays for calculations
    mse_values = np.array([r['mse'] for r in results])
    r2_values = np.array([r['r2_score'] for r in results])
    param_values = np.array([r['parameters'] for r in results])
    time_values = np.array([r['training_time_min'] for r in results])
    
    # Calculate enhanced metrics
    for i, result in enumerate(results):
        # Normalized MSE (relative to worst model)
        result['normalized_mse'] = mse_values[i] / np.max(mse_values)
        
        # Parameter efficiency (R² per million parameters)
        result['r2_per_million_params'] = r2_values[i] / (param_values[i] / 1_000_000)
        
        # Training efficiency (R² per minute)
        result['r2_per_minute'] = r2_values[i] / time_values[i]
        
        # Error efficiency (lower is better)
        result['error_efficiency'] = mse_values[i] * (param_values[i] / 1_000_000)
        
        # MSE rank (1 = best MSE)
        result['mse_rank'] = int(np.sum(mse_values < mse_values[i]) + 1)
        
        # R² rank (1 = best R²)
        result['r2_rank'] = int(np.sum(r2_values > r2_values[i]) + 1)
    
    return results

def enhanced_density_range_analysis(actual_values, predictions, model_name):
    """Calculate performance across dynamic density ranges"""
    
    actual_values = np.array(actual_values)
    predictions = np.array(predictions)
    
    # Dynamic range detection
    max_density = int(np.max(actual_values))
    min_density = int(np.min(actual_values))
    
    print(f"\n=== {model_name} Enhanced Density Analysis ===")
    print(f"Data range: {min_density} - {max_density} density/μL")
    
    # Create comprehensive ranges
    if max_density <= 1000:
        density_ranges = [(0, 50), (50, 150), (150, 300), (300, max_density)]
    else:
        density_ranges = [
            (0, 50), (50, 150), (150, 300), (300, 600),
            (600, 1000), (1000, 2000), (2000, 3000), (3000, max_density)
        ]
        # Filter out empty ranges
        density_ranges = [(low, high) for low, high in density_ranges if high > low]
    
    range_results = []
    
    for min_d, max_d in density_ranges:
        mask = (actual_values >= min_d) & (actual_values <= max_d)
        n_samples = np.sum(mask)
        
        if n_samples > 5:  # Require minimum 5 samples
            actual_range = actual_values[mask]
            pred_range = predictions[mask]
            
            # Calculate R² manually
            ss_res = np.sum((actual_range - pred_range) ** 2)
            ss_tot = np.sum((actual_range - np.mean(actual_range)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Calculate other metrics
            mse = np.mean((actual_range - pred_range) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(actual_range - pred_range))
            
            range_results.append({
                'range': f"{min_d}-{max_d}",
                'min_density': min_d,
                'max_density': max_d,
                'n_samples': n_samples,
                'r2_score': r2,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'mean_actual': np.mean(actual_range),
                'mean_predicted': np.mean(pred_range)
            })
            
            print(f"Range {min_d:4d}-{max_d:4d}: n={n_samples:3d}, R²={r2:.4f}, MSE={mse:8.1f}, RMSE={rmse:6.1f}")
    
    return range_results

def create_analysis_report(results, output_file):
    """Generate a comprehensive text-based analysis report"""
    
    # Sort by different criteria
    mse_sorted = sorted(results, key=lambda x: x['mse'])
    efficiency_sorted = sorted(results, key=lambda x: x['r2_per_million_params'], reverse=True)
    composite_sorted = sorted(results, key=lambda x: (x['mse_rank'] + x['r2_rank'])/2)
    
    report_lines = [
        "=" * 80,
        "ENHANCED MSE ANALYSIS AND MODEL COMPARISON REPORT",
        "=" * 80,
        f"Total Models Analyzed: {len(results)}",
        f"R² Score Range: {min(r['r2_score'] for r in results):.4f} - {max(r['r2_score'] for r in results):.4f}",
        f"MSE Range: {min(r['mse'] for r in results):.0f} - {max(r['mse'] for r in results):.0f}",
        "",
        "=" * 30 + " MSE RANKING (BEST TO WORST) " + "=" * 30,
        "",
        "Rank | Model Name                | R²      | MSE      | RMSE    | Parameters | Time(min)",
        "-" * 95,
    ]
    
    for i, result in enumerate(mse_sorted, 1):
        report_lines.append(
            f"{i:4d} | {result['model_name']:25} | {result['r2_score']:7.4f} | "
            f"{result['mse']:8.0f} | {result['rmse']:7.1f} | {result['parameters']:10,d} | {result['training_time_min']:6.1f}"
        )
    
    report_lines.extend([
        "",
        "=" * 25 + " PARAMETER EFFICIENCY RANKING " + "=" * 25,
        "",
        "Rank | Model Name                | R²/M Params | Parameters | R² Score",
        "-" * 75,
    ])
    
    for i, result in enumerate(efficiency_sorted, 1):
        report_lines.append(
            f"{i:4d} | {result['model_name']:25} | {result['r2_per_million_params']:11.3f} | "
            f"{result['parameters']:10,d} | {result['r2_score']:8.4f}"
        )
    
    # Key insights
    best_mse = mse_sorted[0]
    best_efficiency = efficiency_sorted[0]
    
    report_lines.extend([
        "",
        "=" * 35 + " KEY INSIGHTS " + "=" * 35,
        "",
        f"🏆 BEST MSE PERFORMANCE:",
        f"   Model: {best_mse['model_name']}",
        f"   MSE: {best_mse['mse']:.0f}",
        f"   R²: {best_mse['r2_score']:.4f}",
        f"   Parameters: {best_mse['parameters']:,}",
        f"   Training Time: {best_mse['training_time_min']:.1f} min",
        "",
        f"🎯 MOST PARAMETER EFFICIENT:",
        f"   Model: {best_efficiency['model_name']}",
        f"   R² per Million Parameters: {best_efficiency['r2_per_million_params']:.3f}",
        f"   Total Parameters: {best_efficiency['parameters']:,}",
        f"   R²: {best_efficiency['r2_score']:.4f}",
        "",
        "📊 STATISTICAL SUMMARY:",
    ])
    
    # Calculate statistics
    mse_values = [r['mse'] for r in results]
    r2_values = [r['r2_score'] for r in results]
    param_values = [r['parameters'] for r in results]
    
    report_lines.extend([
        f"   Mean MSE: {np.mean(mse_values):.0f}",
        f"   MSE Standard Deviation: {np.std(mse_values):.0f}",
        f"   MSE Range: {min(mse_values):.0f} - {max(mse_values):.0f}",
        f"   Models with R² > 0.98: {len([r for r in results if r['r2_score'] > 0.98])}/{len(results)}",
        f"   Parameter Range: {min(param_values):,} - {max(param_values):,}",
        "",
        "💡 PERFORMANCE COMPARISON:",
        f"   Best vs Worst MSE: {max(mse_values)/min(mse_values):.1f}x difference",
        f"   R² Coefficient of Variation: {(np.std(r2_values)/np.mean(r2_values)*100):.2f}%",
        f"   Skip Connections Performance:"
    ])
    
    # Skip connections analysis
    skip_models = [r for r in results if r['has_skip_connections']]
    no_skip_models = [r for r in results if not r['has_skip_connections']]
    
    if skip_models and no_skip_models:
        skip_avg_mse = np.mean([r['mse'] for r in skip_models])
        no_skip_avg_mse = np.mean([r['mse'] for r in no_skip_models])
        
        report_lines.extend([
            f"      With Skip Connections: Avg MSE = {skip_avg_mse:.0f} ({len(skip_models)} models)",
            f"      Without Skip Connections: Avg MSE = {no_skip_avg_mse:.0f} ({len(no_skip_models)} models)",
        ])
    
    report_lines.extend([
        "",
        "🔄 MSE IMPROVEMENT OPPORTUNITIES:",
    ])
    
    # Identify improvement opportunities
    worst_mse = mse_sorted[-1]
    report_lines.extend([
        f"   Current worst MSE: {worst_mse['model_name']} ({worst_mse['mse']:.0f})",
        f"   Potential improvement: {((worst_mse['mse'] - best_mse['mse'])/worst_mse['mse']*100):.1f}% MSE reduction available",
        "",
        "=" * 80,
        "END OF REPORT",
        "=" * 80
    ])
    
    # Write report to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    return '\n'.join(report_lines)

def main():
    """Main analysis function"""
    print("=== Simple MSE Analysis (No External Dependencies) ===")
    
    # Study directory
    study_dir = "../skip_connections_study/comprehensive_architecture_study_20250911_173012"
    
    if not os.path.exists(study_dir):
        print(f"Study directory not found: {study_dir}")
        return
    
    # Load results
    results = load_experiment_results(study_dir)
    
    if not results:
        print("No results found!")
        return
    
    # Calculate enhanced metrics
    results = calculate_enhanced_metrics(results)
    
    # Generate report
    output_file = f"{study_dir}/simple_mse_analysis_report.txt"
    report_text = create_analysis_report(results, output_file)
    
    print(f"\nReport saved to: {output_file}")
    
    # Print quick summary
    print("\n" + "="*60)
    print("QUICK MSE RANKING:")
    print("="*60)
    
    mse_sorted = sorted(results, key=lambda x: x['mse'])
    for i, result in enumerate(mse_sorted, 1):
        print(f"{i}. {result['model_name']:25} MSE: {result['mse']:8.0f}  R²: {result['r2_score']:.4f}")
    
    print(f"\n📈 MSE Range: {mse_sorted[0]['mse']:.0f} (best) to {mse_sorted[-1]['mse']:.0f} (worst)")
    print(f"🎯 Best Model: {mse_sorted[0]['model_name']} (MSE: {mse_sorted[0]['mse']:.0f})")
    
    # Parameter efficiency summary
    efficiency_sorted = sorted(results, key=lambda x: x['r2_per_million_params'], reverse=True)
    print(f"💾 Most Efficient: {efficiency_sorted[0]['model_name']} ({efficiency_sorted[0]['r2_per_million_params']:.3f} R²/M params)")

if __name__ == "__main__":
    main()