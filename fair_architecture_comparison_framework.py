#!/usr/bin/env python3
"""
Fair Architecture Comparison Framework
=====================================

A systematic approach to compare CNN architectures with proper hyperparameter optimization
for each architecture to ensure fair evaluation.

Based on findings that different architectures (especially those with skip connections)
require different optimal hyperparameters for peak performance.
"""

import optuna
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import json
from pathlib import Path
import logging
from sklearn.model_selection import KFold
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class FairArchitectureComparison:
    """
    Framework for fair comparison of CNN architectures through systematic
    hyperparameter optimization and robust statistical evaluation.
    """

    def __init__(self,
                 dataset_path: str,
                 output_dir: str,
                 architectures: List[nn.Module],
                 optimization_trials: int = 100,
                 evaluation_runs: int = 5,
                 cv_folds: int = 3):
        """
        Initialize the fair comparison framework.

        Args:
            dataset_path: Path to preprocessed dataset
            output_dir: Directory for results
            architectures: List of architecture classes to compare
            optimization_trials: Number of hyperparameter optimization trials per architecture
            evaluation_runs: Number of evaluation runs with optimal hyperparameters
            cv_folds: Number of cross-validation folds
        """
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.architectures = architectures
        self.optimization_trials = optimization_trials
        self.evaluation_runs = evaluation_runs
        self.cv_folds = cv_folds

        # Results storage
        self.optimization_results = {}
        self.evaluation_results = {}
        self.statistical_results = {}

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def define_search_space(self, architecture_name: str) -> Dict[str, Any]:
        """
        Define hyperparameter search space, potentially architecture-specific.

        Args:
            architecture_name: Name of the architecture

        Returns:
            Dictionary defining the search space for Optuna
        """
        # Base search space
        search_space = {
            'learning_rate': ('log', 1e-5, 1e-2),
            'batch_size': ('categorical', [16, 32, 64, 128, 256]),
            'weight_decay': ('log', 1e-8, 1e-2),
            'optimizer': ('categorical', ['adam', 'adamw', 'sgd']),
            'scheduler': ('categorical', ['cosine', 'step', 'exponential', 'none']),
            'dropout_rate': ('uniform', 0.0, 0.5)
        }

        # Architecture-specific modifications
        if 'ResNet' in architecture_name or 'UNet' in architecture_name:
            # Skip connection architectures often need different learning rates
            search_space['learning_rate'] = ('log', 1e-6, 5e-3)
            # May benefit from different optimizers
            search_space['optimizer'] = ('categorical', ['adam', 'adamw'])

        elif 'Deep' in architecture_name:
            # Deeper networks may need smaller learning rates
            search_space['learning_rate'] = ('log', 1e-6, 1e-3)
            # Benefit from weight decay
            search_space['weight_decay'] = ('log', 1e-6, 1e-3)

        elif 'DenseNet' in architecture_name:
            # Dense connections often work well with smaller learning rates
            search_space['learning_rate'] = ('log', 1e-6, 1e-3)
            search_space['batch_size'] = ('categorical', [16, 32, 64])  # Smaller batches

        return search_space

    def objective_function(self, trial: optuna.Trial, architecture_class: nn.Module,
                          architecture_name: str) -> float:
        """
        Objective function for hyperparameter optimization.

        Args:
            trial: Optuna trial object
            architecture_class: Architecture class to instantiate
            architecture_name: Name of the architecture

        Returns:
            Validation R² score to maximize
        """
        # Sample hyperparameters
        search_space = self.define_search_space(architecture_name)
        params = {}

        for param_name, param_config in search_space.items():
            if param_config[0] == 'log':
                params[param_name] = trial.suggest_float(param_name, param_config[1],
                                                       param_config[2], log=True)
            elif param_config[0] == 'uniform':
                params[param_name] = trial.suggest_float(param_name, param_config[1],
                                                       param_config[2])
            elif param_config[0] == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, param_config[1])

        # Train and evaluate with these hyperparameters
        try:
            r2_scores = self._train_and_validate(architecture_class, params, architecture_name)
            mean_r2 = np.mean(r2_scores)

            # Log intermediate results
            self.logger.info(f"{architecture_name} Trial {trial.number}: R² = {mean_r2:.4f}")

            return mean_r2

        except Exception as e:
            self.logger.warning(f"Trial failed for {architecture_name}: {e}")
            return -1.0  # Return very poor score for failed trials

    def _train_and_validate(self, architecture_class: nn.Module, params: Dict[str, Any],
                          architecture_name: str) -> List[float]:
        """
        Train and validate model with given hyperparameters using cross-validation.

        Args:
            architecture_class: Architecture class to instantiate
            params: Hyperparameters to use
            architecture_name: Name of the architecture

        Returns:
            List of R² scores from cross-validation
        """
        # This is a placeholder - you would implement actual training logic here
        # based on your existing training infrastructure

        # Load dataset
        # dataset = self._load_dataset()

        # Perform cross-validation
        r2_scores = []
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        # For now, return simulated scores - replace with actual training
        # In real implementation, you would:
        # 1. Split data using kf.split()
        # 2. Train model with params on each fold
        # 3. Evaluate and collect R² scores

        # Placeholder simulation based on your existing results
        if 'ResNet_Shallow' in architecture_name:
            # Simulate the high variance you observed
            base_score = np.random.uniform(0.3, 0.99)
        elif 'Baseline' in architecture_name:
            # Simulate more stable performance
            base_score = np.random.uniform(0.98, 0.997)
        else:
            base_score = np.random.uniform(0.95, 0.99)

        r2_scores = [base_score + np.random.normal(0, 0.01) for _ in range(self.cv_folds)]

        return r2_scores

    def optimize_architecture(self, architecture_class: nn.Module,
                            architecture_name: str) -> Dict[str, Any]:
        """
        Perform hyperparameter optimization for a single architecture.

        Args:
            architecture_class: Architecture class to optimize
            architecture_name: Name of the architecture

        Returns:
            Dictionary containing optimization results
        """
        self.logger.info(f"Starting hyperparameter optimization for {architecture_name}")

        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',
            study_name=f"{architecture_name}_optimization",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        )

        # Optimize
        study.optimize(
            lambda trial: self.objective_function(trial, architecture_class, architecture_name),
            n_trials=self.optimization_trials,
            timeout=None
        )

        # Store results
        optimization_result = {
            'architecture_name': architecture_name,
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials),
            'optimization_history': [(trial.number, trial.value) for trial in study.trials if trial.value is not None]
        }

        self.optimization_results[architecture_name] = optimization_result

        self.logger.info(f"Optimization complete for {architecture_name}. Best R²: {study.best_value:.4f}")

        return optimization_result

    def evaluate_with_optimal_params(self, architecture_class: nn.Module,
                                   architecture_name: str) -> Dict[str, Any]:
        """
        Evaluate architecture multiple times with optimal hyperparameters.

        Args:
            architecture_class: Architecture class to evaluate
            architecture_name: Name of the architecture

        Returns:
            Dictionary containing evaluation results
        """
        if architecture_name not in self.optimization_results:
            raise ValueError(f"Must optimize {architecture_name} before evaluation")

        optimal_params = self.optimization_results[architecture_name]['best_params']

        self.logger.info(f"Evaluating {architecture_name} with optimal parameters")

        all_scores = []
        detailed_results = []

        for run in range(self.evaluation_runs):
            scores = self._train_and_validate(architecture_class, optimal_params, architecture_name)
            all_scores.extend(scores)

            detailed_results.append({
                'run': run,
                'cv_scores': scores,
                'mean_score': np.mean(scores),
                'std_score': np.std(scores)
            })

        evaluation_result = {
            'architecture_name': architecture_name,
            'optimal_params': optimal_params,
            'all_scores': all_scores,
            'mean_score': np.mean(all_scores),
            'std_score': np.std(all_scores),
            'confidence_interval_95': stats.t.interval(0.95, len(all_scores)-1,
                                                     loc=np.mean(all_scores),
                                                     scale=stats.sem(all_scores)),
            'detailed_runs': detailed_results
        }

        self.evaluation_results[architecture_name] = evaluation_result

        return evaluation_result

    def run_fair_comparison(self) -> Dict[str, Any]:
        """
        Run the complete fair comparison protocol.

        Returns:
            Dictionary containing all comparison results
        """
        self.logger.info("Starting fair architecture comparison")

        # Stage 1: Hyperparameter optimization for each architecture
        self.logger.info("Stage 1: Hyperparameter Optimization")
        for arch_class in self.architectures:
            arch_name = arch_class.__name__
            self.optimize_architecture(arch_class, arch_name)

        # Stage 2: Robust evaluation with optimal parameters
        self.logger.info("Stage 2: Robust Evaluation")
        for arch_class in self.architectures:
            arch_name = arch_class.__name__
            self.evaluate_with_optimal_params(arch_class, arch_name)

        # Stage 3: Statistical comparison
        self.logger.info("Stage 3: Statistical Analysis")
        self.statistical_results = self._perform_statistical_analysis()

        # Save all results
        self._save_results()

        # Generate comparison plots
        self._generate_comparison_plots()

        return {
            'optimization_results': self.optimization_results,
            'evaluation_results': self.evaluation_results,
            'statistical_results': self.statistical_results
        }

    def _perform_statistical_analysis(self) -> Dict[str, Any]:
        """
        Perform statistical analysis to compare architectures.

        Returns:
            Dictionary containing statistical analysis results
        """
        arch_names = list(self.evaluation_results.keys())
        statistical_results = {
            'pairwise_comparisons': {},
            'ranking': [],
            'significance_matrix': {}
        }

        # Pairwise statistical tests
        for i, arch1 in enumerate(arch_names):
            for j, arch2 in enumerate(arch_names):
                if i < j:  # Avoid duplicate comparisons
                    scores1 = self.evaluation_results[arch1]['all_scores']
                    scores2 = self.evaluation_results[arch2]['all_scores']

                    # Welch's t-test (doesn't assume equal variances)
                    t_stat, p_value = stats.ttest_ind(scores1, scores2, equal_var=False)

                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt((np.var(scores1) + np.var(scores2)) / 2)
                    cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std

                    comparison_key = f"{arch1}_vs_{arch2}"
                    statistical_results['pairwise_comparisons'][comparison_key] = {
                        'architecture_1': arch1,
                        'architecture_2': arch2,
                        'mean_diff': np.mean(scores1) - np.mean(scores2),
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'cohens_d': cohens_d,
                        'significant': p_value < 0.05,
                        'practically_significant': abs(cohens_d) > 0.2
                    }

        # Ranking by mean performance with confidence intervals
        ranking_data = []
        for arch_name, results in self.evaluation_results.items():
            ranking_data.append({
                'architecture': arch_name,
                'mean_score': results['mean_score'],
                'std_score': results['std_score'],
                'ci_lower': results['confidence_interval_95'][0],
                'ci_upper': results['confidence_interval_95'][1],
                'n_samples': len(results['all_scores'])
            })

        # Sort by mean score
        ranking_data.sort(key=lambda x: x['mean_score'], reverse=True)
        statistical_results['ranking'] = ranking_data

        return statistical_results

    def _save_results(self):
        """Save all results to files."""
        # Save optimization results
        with open(self.output_dir / 'hyperparameter_optimization_results.json', 'w') as f:
            json.dump(self.optimization_results, f, indent=2, default=str)

        # Save evaluation results
        with open(self.output_dir / 'robust_evaluation_results.json', 'w') as f:
            json.dump(self.evaluation_results, f, indent=2, default=str)

        # Save statistical results
        with open(self.output_dir / 'statistical_analysis_results.json', 'w') as f:
            json.dump(self.statistical_results, f, indent=2, default=str)

        # Create summary CSV
        summary_data = []
        for arch_name, results in self.evaluation_results.items():
            opt_results = self.optimization_results[arch_name]
            summary_data.append({
                'Architecture': arch_name,
                'Optimal_R2': results['mean_score'],
                'R2_Std': results['std_score'],
                'CI_Lower': results['confidence_interval_95'][0],
                'CI_Upper': results['confidence_interval_95'][1],
                'Best_LR': opt_results['best_params'].get('learning_rate', 'N/A'),
                'Best_Batch_Size': opt_results['best_params'].get('batch_size', 'N/A'),
                'Best_Optimizer': opt_results['best_params'].get('optimizer', 'N/A'),
                'Optimization_Trials': opt_results['n_trials']
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.output_dir / 'fair_comparison_summary.csv', index=False)

    def _generate_comparison_plots(self):
        """Generate visualization plots for the comparison results."""
        # Performance comparison with confidence intervals
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Performance ranking with confidence intervals
        ranking_data = self.statistical_results['ranking']
        arch_names = [r['architecture'] for r in ranking_data]
        mean_scores = [r['mean_score'] for r in ranking_data]
        ci_errors = [(r['mean_score'] - r['ci_lower'], r['ci_upper'] - r['mean_score'])
                     for r in ranking_data]

        ax1.errorbar(range(len(arch_names)), mean_scores,
                    yerr=np.array(ci_errors).T, fmt='o', capsize=5)
        ax1.set_xticks(range(len(arch_names)))
        ax1.set_xticklabels(arch_names, rotation=45, ha='right')
        ax1.set_ylabel('R² Score')
        ax1.set_title('Architecture Performance Comparison\n(with 95% Confidence Intervals)')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Hyperparameter optimization convergence
        for arch_name, opt_results in self.optimization_results.items():
            history = opt_results['optimization_history']
            trials, scores = zip(*history)
            # Running maximum (best score so far)
            running_max = np.maximum.accumulate(scores)
            ax2.plot(trials, running_max, label=arch_name, alpha=0.7)

        ax2.set_xlabel('Optimization Trial')
        ax2.set_ylabel('Best R² Score')
        ax2.set_title('Hyperparameter Optimization Convergence')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'fair_comparison_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()


# Example usage and configuration
if __name__ == "__main__":
    # This would be imported from your existing architecture definitions
    # from your_architectures import BaselineShallowCNN, ResNetShallowCNN, etc.

    # Example architecture list (replace with your actual architectures)
    architectures_to_compare = [
        # BaselineShallowCNN,
        # BaselineDeepCNN,
        # ResNetShallowCNN,
        # ResNetDeepCNN,
        # UNetShallowCNN,
        # DenseNetStyleCNN
    ]

    # Initialize comparison framework
    comparison = FairArchitectureComparison(
        dataset_path="./dataset_preprocessed",
        output_dir="./fair_architecture_comparison_study",
        architectures=architectures_to_compare,
        optimization_trials=100,  # Adjust based on computational budget
        evaluation_runs=5,
        cv_folds=3
    )

    # Run the complete fair comparison
    results = comparison.run_fair_comparison()

    print("Fair architecture comparison completed!")
    print(f"Results saved to: {comparison.output_dir}")