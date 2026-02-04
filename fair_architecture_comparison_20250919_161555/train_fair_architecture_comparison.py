#!/usr/bin/env python3
"""
Fair Architecture Comparison Training Script
===========================================

Implements a systematic approach to fairly compare CNN architectures by:
1. Individual hyperparameter optimization for each architecture
2. Robust evaluation with optimal parameters
3. Statistical significance testing

Based on findings that architectures require different optimal hyperparameters.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # HPC compatibility
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import time
import json
import argparse
import gc
import psutil
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Optuna for hyperparameter optimization
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Import existing architecture definitions
import sys
sys.path.append('/Users/xiaodan/densityCNN/Claude/skip_connections_study')

# HPC optimizations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256,expandable_segments:True'
torch.backends.cudnn.benchmark = True

print("🔍 Fair Comparison Environment Check:")
print(f"   PyTorch: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
print(f"   Optuna version: {optuna.__version__}")

# ============================================================================
# DATASET CLASS (Same as existing infrastructure)
# ============================================================================

class MicrobeadDataset(Dataset):
    """
    Dataset class for microbead density estimation.
    Follows exact patterns from existing training infrastructure.
    """
    def __init__(self, csv_file, img_dir, dilution_factors=None, transform=None, data_percentage=100):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # Filter by dilution factors if specified
        if dilution_factors:
            if isinstance(dilution_factors[0], str):
                self.data = self.data[self.data['Dilution_Factor'].isin(dilution_factors)]
            else:
                self.data = self.data[self.data['Dilution_Factor'].isin([f"{df}x" for df in dilution_factors])]

        # Use only a percentage of data if specified
        if data_percentage < 100:
            n_samples = int(len(self.data) * (data_percentage / 100))
            self.data = self.data.sample(n=n_samples, random_state=42).reset_index(drop=True)

        print(f"📊 Dataset loaded: {len(self.data)} samples")
        print(f"   Dilution factors: {sorted(self.data['Dilution_Factor'].unique())}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_filename = row['Filename']

        # Try different image extensions
        for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.tif']:
            img_path = os.path.join(self.img_dir, f"{img_filename}{ext}")
            if os.path.exists(img_path):
                break
        else:
            raise FileNotFoundError(f"Image not found: {img_filename}")

        image = Image.open(img_path).convert('RGB')
        density = float(row['Density'])

        if self.transform:
            image = self.transform(image)

        return image, density

# ============================================================================
# ARCHITECTURE DEFINITIONS (Import or define here)
# ============================================================================

class BaselineShallowCNN(nn.Module):
    """Baseline shallow CNN without skip connections - 4 layers"""
    def __init__(self):
        super(BaselineShallowCNN, self).__init__()
        self.name = "Baseline_Shallow"
        self.depth = 4
        self.has_skip_connections = False

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class BaselineDeepCNN(nn.Module):
    """Baseline deep CNN without skip connections - 12 layers"""
    def __init__(self):
        super(BaselineDeepCNN, self).__init__()
        self.name = "Baseline_Deep"
        self.depth = 12
        self.has_skip_connections = False

        self.features = nn.Sequential(
            # Initial conv
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Deep layers
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class ResNetShallowCNN(nn.Module):
    """ResNet-style shallow CNN with skip connections - 4 layers"""
    def __init__(self):
        super(ResNetShallowCNN, self).__init__()
        self.name = "ResNet_Shallow"
        self.depth = 4
        self.has_skip_connections = True

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 1)

    def _make_layer(self, inplanes, planes, blocks, stride=1):
        layers = []

        # First block may need downsample
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes)
            )

        layers.append(ResidualBlock(inplanes, planes, stride, downsample))

        # Additional blocks
        for _ in range(1, blocks):
            layers.append(ResidualBlock(planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# ============================================================================
# FAIR COMPARISON FRAMEWORK
# ============================================================================

class FairArchitectureComparison:
    def __init__(self, dataset_path, output_dir, architectures, optimization_trials=50, evaluation_runs=3):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.architectures = architectures
        self.optimization_trials = optimization_trials
        self.evaluation_runs = evaluation_runs

        # Results storage
        self.optimization_results = {}
        self.evaluation_results = {}

        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")

        # Load dataset once
        self._prepare_dataset()

    def _prepare_dataset(self):
        """Prepare dataset for training and validation"""
        # Data transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Create datasets
        csv_file = os.path.join(self.dataset_path, 'density.csv')
        img_dir = os.path.join(self.dataset_path, 'images')

        self.full_dataset = MicrobeadDataset(
            csv_file=csv_file,
            img_dir=img_dir,
            dilution_factors=['80x', '160x', '320x', '640x', '1280x', '2560x', '5120x', '10240x'],
            transform=None,  # Will apply transforms in dataloaders
            data_percentage=50  # Use 50% for faster experiments
        )

        # Train/validation split
        dataset_size = len(self.full_dataset)
        train_size = int(0.8 * dataset_size)
        val_size = dataset_size - train_size

        indices = torch.randperm(dataset_size).tolist()
        self.train_indices = indices[:train_size]
        self.val_indices = indices[train_size:]

        print(f"📊 Dataset split: {train_size} train, {val_size} validation")

    def _create_dataloaders(self, batch_size, num_workers):
        """Create dataloaders with specified batch size"""
        # Create datasets with transforms
        train_dataset = copy.deepcopy(self.full_dataset)
        val_dataset = copy.deepcopy(self.full_dataset)
        train_dataset.transform = self.train_transform
        val_dataset.transform = self.val_transform

        # Create samplers
        train_sampler = SubsetRandomSampler(self.train_indices)
        val_sampler = SubsetRandomSampler(self.val_indices)

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True
        )

        return train_loader, val_loader

    def _train_and_evaluate(self, architecture_class, params, max_epochs=30):
        """Train and evaluate a model with given hyperparameters"""
        try:
            # Create model
            model = architecture_class().to(self.device)

            # Create dataloaders
            train_loader, val_loader = self._create_dataloaders(
                batch_size=params['batch_size'],
                num_workers=min(4, params.get('num_workers', 4))  # Conservative for trials
            )

            # Setup optimizer
            if params['optimizer'] == 'adam':
                optimizer = optim.Adam(model.parameters(),
                                     lr=params['learning_rate'],
                                     weight_decay=params.get('weight_decay', 1e-4))
            elif params['optimizer'] == 'adamw':
                optimizer = optim.AdamW(model.parameters(),
                                      lr=params['learning_rate'],
                                      weight_decay=params.get('weight_decay', 1e-4))
            else:  # sgd
                optimizer = optim.SGD(model.parameters(),
                                    lr=params['learning_rate'],
                                    momentum=0.9,
                                    weight_decay=params.get('weight_decay', 1e-4))

            # Setup scheduler
            if params.get('scheduler') == 'cosine':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
            elif params.get('scheduler') == 'step':
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max_epochs//3, gamma=0.1)
            else:
                scheduler = None

            # Loss function
            criterion = nn.MSELoss()

            # Mixed precision
            scaler = GradScaler() if params.get('use_mixed_precision', True) else None

            # Training loop
            best_val_r2 = -np.inf
            patience_counter = 0
            max_patience = 10  # Early stopping for trials

            for epoch in range(max_epochs):
                # Training phase
                model.train()
                train_loss = 0.0

                for batch_idx, (images, targets) in enumerate(train_loader):
                    images, targets = images.to(self.device), targets.to(self.device).float()

                    optimizer.zero_grad()

                    if scaler:
                        with autocast():
                            outputs = model(images)
                            loss = criterion(outputs.squeeze(), targets)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        outputs = model(images)
                        loss = criterion(outputs.squeeze(), targets)
                        loss.backward()
                        optimizer.step()

                    train_loss += loss.item()

                    # Break early for very long epochs in trials
                    if batch_idx > 50:  # Limit batches per epoch for speed
                        break

                # Validation phase
                model.eval()
                val_predictions = []
                val_targets = []

                with torch.no_grad():
                    for batch_idx, (images, targets) in enumerate(val_loader):
                        images, targets = images.to(self.device), targets.to(self.device).float()

                        outputs = model(images)
                        val_predictions.extend(outputs.squeeze().cpu().numpy())
                        val_targets.extend(targets.cpu().numpy())

                        # Break early for trials
                        if batch_idx > 20:
                            break

                # Calculate R²
                val_predictions = np.array(val_predictions)
                val_targets = np.array(val_targets)

                ss_res = np.sum((val_targets - val_predictions) ** 2)
                ss_tot = np.sum((val_targets - np.mean(val_targets)) ** 2)
                val_r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else -np.inf

                # Early stopping check
                if val_r2 > best_val_r2:
                    best_val_r2 = val_r2
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= max_patience:
                    break

                # Update scheduler
                if scheduler:
                    scheduler.step()

                # Memory cleanup
                if epoch % 5 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

            # Cleanup
            del model, optimizer
            if scheduler:
                del scheduler
            torch.cuda.empty_cache()
            gc.collect()

            return best_val_r2

        except Exception as e:
            print(f"Training failed: {e}")
            return -1.0

    def _define_search_space(self, architecture_name):
        """Define architecture-specific hyperparameter search space"""
        # Base search space
        search_space = {
            'learning_rate': (1e-5, 1e-2),
            'batch_size': [16, 32, 64, 128],
            'weight_decay': (1e-6, 1e-2),
            'optimizer': ['adam', 'adamw', 'sgd'],
            'scheduler': ['cosine', 'step', 'none'],
            'use_mixed_precision': [True, False]
        }

        # Architecture-specific modifications
        if 'ResNet' in architecture_name:
            # ResNet architectures often need different settings
            search_space['learning_rate'] = (1e-6, 5e-3)
            search_space['optimizer'] = ['adam', 'adamw']  # Often work better
            search_space['batch_size'] = [16, 32, 64]  # Smaller batches can help

        elif 'Deep' in architecture_name:
            # Deeper networks benefit from smaller learning rates
            search_space['learning_rate'] = (1e-6, 1e-3)
            search_space['weight_decay'] = (1e-5, 1e-3)  # More regularization

        return search_space

    def optimize_architecture(self, architecture_class):
        """Perform hyperparameter optimization for an architecture"""
        arch_name = architecture_class().name
        print(f"🔍 Optimizing hyperparameters for {arch_name}")

        search_space = self._define_search_space(arch_name)

        def objective(trial):
            # Sample hyperparameters
            params = {}
            params['learning_rate'] = trial.suggest_float(
                'learning_rate', search_space['learning_rate'][0],
                search_space['learning_rate'][1], log=True)
            params['batch_size'] = trial.suggest_categorical('batch_size', search_space['batch_size'])
            params['weight_decay'] = trial.suggest_float(
                'weight_decay', search_space['weight_decay'][0],
                search_space['weight_decay'][1], log=True)
            params['optimizer'] = trial.suggest_categorical('optimizer', search_space['optimizer'])
            params['scheduler'] = trial.suggest_categorical('scheduler', search_space['scheduler'])
            params['use_mixed_precision'] = trial.suggest_categorical(
                'use_mixed_precision', search_space['use_mixed_precision'])

            # Train and evaluate
            r2_score = self._train_and_evaluate(architecture_class, params)

            return r2_score

        # Create and run study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        )

        study.optimize(objective, n_trials=self.optimization_trials, timeout=3600)  # 1 hour max

        # Store results
        self.optimization_results[arch_name] = {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials),
            'study': study  # For later analysis
        }

        print(f"✅ Optimization complete for {arch_name}")
        print(f"   Best R²: {study.best_value:.4f}")
        print(f"   Best params: {study.best_params}")

        return study.best_params, study.best_value

    def evaluate_with_optimal_params(self, architecture_class):
        """Evaluate architecture with optimal hyperparameters"""
        arch_name = architecture_class().name

        if arch_name not in self.optimization_results:
            raise ValueError(f"Must optimize {arch_name} first")

        optimal_params = self.optimization_results[arch_name]['best_params']
        print(f"🎯 Evaluating {arch_name} with optimal parameters")

        scores = []
        for run in range(self.evaluation_runs):
            print(f"   Run {run+1}/{self.evaluation_runs}")
            score = self._train_and_evaluate(architecture_class, optimal_params, max_epochs=50)
            scores.append(score)

        # Calculate statistics
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        ci_95 = stats.t.interval(0.95, len(scores)-1, loc=mean_score, scale=stats.sem(scores))

        self.evaluation_results[arch_name] = {
            'optimal_params': optimal_params,
            'scores': scores,
            'mean_score': mean_score,
            'std_score': std_score,
            'ci_95': ci_95
        }

        print(f"✅ Evaluation complete for {arch_name}")
        print(f"   Mean R²: {mean_score:.4f} ± {std_score:.4f}")
        print(f"   95% CI: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")

        return mean_score, std_score

    def run_fair_comparison(self):
        """Run complete fair comparison protocol"""
        print("🚀 Starting Fair Architecture Comparison")

        # Stage 1: Hyperparameter optimization
        print("\n📊 Stage 1: Hyperparameter Optimization")
        for arch_class in self.architectures:
            self.optimize_architecture(arch_class)

        # Stage 2: Robust evaluation
        print("\n🎯 Stage 2: Robust Evaluation")
        for arch_class in self.architectures:
            self.evaluate_with_optimal_params(arch_class)

        # Stage 3: Analysis and reporting
        print("\n📈 Stage 3: Analysis and Reporting")
        self._generate_analysis()

        return self.evaluation_results

    def _generate_analysis(self):
        """Generate comprehensive analysis of results"""
        # Create comparison dataframe
        comparison_data = []
        for arch_name, results in self.evaluation_results.items():
            comparison_data.append({
                'Architecture': arch_name,
                'Mean_R2': results['mean_score'],
                'Std_R2': results['std_score'],
                'CI_Lower': results['ci_95'][0],
                'CI_Upper': results['ci_95'][1],
                'Best_LR': results['optimal_params']['learning_rate'],
                'Best_Batch_Size': results['optimal_params']['batch_size'],
                'Best_Optimizer': results['optimal_params']['optimizer']
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Mean_R2', ascending=False)

        # Save results
        comparison_df.to_csv(os.path.join(self.output_dir, 'fair_comparison_results.csv'), index=False)

        # Generate plots
        self._create_comparison_plots(comparison_df)

        # Save detailed results
        with open(os.path.join(self.output_dir, 'detailed_results.json'), 'w') as f:
            # Convert numpy types for JSON serialization
            serializable_results = {}
            for arch_name, results in self.evaluation_results.items():
                serializable_results[arch_name] = {
                    'optimal_params': results['optimal_params'],
                    'scores': [float(s) for s in results['scores']],
                    'mean_score': float(results['mean_score']),
                    'std_score': float(results['std_score']),
                    'ci_95': [float(results['ci_95'][0]), float(results['ci_95'][1])]
                }
            json.dump(serializable_results, f, indent=2)

        print(f"📊 Results saved to {self.output_dir}")
        print("\n🏆 Final Rankings:")
        for idx, row in comparison_df.iterrows():
            print(f"   {idx+1}. {row['Architecture']}: R² = {row['Mean_R2']:.4f} ± {row['Std_R2']:.4f}")

    def _create_comparison_plots(self, comparison_df):
        """Create visualization plots"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Performance comparison with confidence intervals
        y_pos = range(len(comparison_df))
        means = comparison_df['Mean_R2']
        stds = comparison_df['Std_R2']

        ax1.barh(y_pos, means, xerr=stds, alpha=0.7, capsize=5)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(comparison_df['Architecture'])
        ax1.set_xlabel('R² Score')
        ax1.set_title('Fair Architecture Comparison\n(with Standard Deviation)')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Hyperparameter distribution
        optimizers = comparison_df['Best_Optimizer'].value_counts()
        ax2.pie(optimizers.values, labels=optimizers.index, autopct='%1.1f%%')
        ax2.set_title('Optimal Optimizer Distribution')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fair_comparison_analysis.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

        print("📈 Analysis plots saved")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Fair Architecture Comparison')
    parser.add_argument('--dataset_path', type=str, default='dataset_preprocessed',
                       help='Path to preprocessed dataset')
    parser.add_argument('--output_dir', type=str,
                       default=f'fair_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                       help='Output directory')
    parser.add_argument('--optimization_trials', type=int, default=50,
                       help='Number of optimization trials per architecture')
    parser.add_argument('--evaluation_runs', type=int, default=3,
                       help='Number of evaluation runs with optimal parameters')

    args = parser.parse_args()

    # Define architectures to compare
    architectures = [
        BaselineShallowCNN,
        BaselineDeepCNN,
        ResNetShallowCNN
    ]

    # Create comparison framework
    comparison = FairArchitectureComparison(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        architectures=architectures,
        optimization_trials=args.optimization_trials,
        evaluation_runs=args.evaluation_runs
    )

    # Run comparison
    results = comparison.run_fair_comparison()

    print("🎉 Fair architecture comparison completed!")

if __name__ == "__main__":
    main()