#!/usr/bin/env python3
"""
Reduced-Layer Study Training Script
Tests architectural robustness by comparing original vs reduced-layer architectures
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

# Import the reduced architectures
from reduced_layer_study_architectures import get_reduced_layer_architectures

# Import your existing components (assuming they're available)
try:
    from train_comprehensive_architecture_study import (
        MicrobeadDataset, get_transforms, evaluate_model_comprehensive,
        create_evaluation_plots, verify_hpc_environment
    )
except ImportError:
    print("Warning: Could not import from main training script. Some functions may need to be redefined.")

# Setup argument parser
parser = argparse.ArgumentParser(description='Reduced-Layer CNN Architecture Study')
parser.add_argument('--input_dir', type=str, default='../../dataset/dataset_preprocessed',
                   help='Path to preprocessed dataset')
parser.add_argument('--output_dir', type=str, default='reduced_layer_study',
                   help='Output directory name')
parser.add_argument('--epochs', type=int, default=40,
                   help='Number of training epochs (reduced for faster comparison)')
parser.add_argument('--patience', type=int, default=12,
                   help='Early stopping patience')
parser.add_argument('--learning_rate', type=float, default=3e-4,
                   help='Learning rate')
parser.add_argument('--batch_size', type=int, default=128,
                   help='Batch size')
parser.add_argument('--seed', type=int, default=42,
                   help='Random seed')
parser.add_argument('--num_workers', type=int, default=8,
                   help='Number of data loader workers')
parser.add_argument('--data_percentage', type=int, default=50,
                   help='Percentage of dataset to use')
parser.add_argument('--mixed_precision', action='store_true', default=True,
                   help='Use mixed precision training')
parser.add_argument('--track_gradients', action='store_true', default=False,
                   help='Track gradient norms (disabled by default for speed)')

args = parser.parse_args()

class ReducedLayerExperiment:
    """Manages individual reduced-layer architecture experiments"""
    
    def __init__(self, model, config, device, experiment_id):
        self.model = model
        self.config = config
        self.device = device
        self.experiment_id = experiment_id
        self.results = {}
        
        # Setup optimizer and scheduler
        self.optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config['epochs'])
        self.criterion = nn.MSELoss()
        
        # Mixed precision setup
        if config['mixed_precision']:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Training tracking
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (images, targets, _) in enumerate(train_loader):
            images, targets = images.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for images, targets, _ in val_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train_model(self, train_loader, val_loader):
        """Complete training loop"""
        print(f"\\n🚀 Training {self.model.name} (Experiment {self.experiment_id})")
        print(f"   Architecture: {self.model.depth} layers, Skip: {self.model.has_skip_connections}")
        print(f"   Reduction: {self.model.reduction_factor:.0%} of original depth")
        
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            # Training
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.learning_rates.append(current_lr)
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                
                # Save best model
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'train_loss': train_loss
                }, self.config['output_dir'] / f"best_model_{self.model.name}.pth")
                
            else:
                self.patience_counter += 1
            
            # Progress reporting
            if epoch % 5 == 0 or epoch == self.config['epochs'] - 1:
                print(f"   Epoch {epoch+1:3d}/{self.config['epochs']}: "
                      f"Train={train_loss:6.0f}, Val={val_loss:6.0f}, "
                      f"Best={self.best_val_loss:6.0f}@{self.best_epoch+1}, "
                      f"LR={current_lr:.2e}")
            
            # Early stopping
            if self.patience_counter >= self.config['patience']:
                print(f"   ⏹️  Early stopping at epoch {epoch+1} (patience {self.config['patience']})")
                break
        
        training_time = time.time() - start_time
        
        # Store training results
        self.results = {
            'model_name': self.model.name,
            'experiment_id': self.experiment_id,
            'architecture_info': {
                'depth': self.model.depth,
                'original_depth': self.model.original_depth,
                'has_skip_connections': self.model.has_skip_connections,
                'reduction_factor': self.model.reduction_factor,
                'parameters': sum(p.numel() for p in self.model.parameters())
            },
            'training_performance': {
                'epochs_completed': epoch + 1,
                'best_val_loss': self.best_val_loss,
                'training_minutes': training_time / 60,
                'final_train_loss': train_loss,
                'convergence_epoch': self.best_epoch + 1
            },
            'training_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'learning_rates': self.learning_rates
            }
        }
        
        print(f"   ✅ Completed in {training_time/60:.1f} min, Best Val Loss: {self.best_val_loss:.0f}")
        
        return self.results

def run_reduced_layer_study():
    """Main function to run the reduced-layer study"""
    
    print("🔬 REDUCED-LAYER CNN ARCHITECTURE STUDY")
    print("=" * 80)
    print(f"Objective: Test architectural robustness under depth constraints")
    print(f"Hypothesis: Skip connections provide more graceful degradation")
    
    # Setup environment
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")
    
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
        torch.cuda.empty_cache()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"{args.output_dir}_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    print(f"📁 Output: {output_dir}")
    
    # Setup dataset
    print("\\n📂 Loading dataset...")
    
    try:
        train_transform, eval_transform = get_transforms()
        
        full_dataset = MicrobeadDataset(
            os.path.join(args.input_dir, 'images'),
            os.path.join(args.input_dir, 'density.csv'),
            transform=train_transform,
            data_percentage=args.data_percentage
        )
        
        test_dataset = MicrobeadDataset(
            os.path.join(args.input_dir, 'images'),
            os.path.join(args.input_dir, 'density.csv'),
            transform=eval_transform,
            data_percentage=args.data_percentage
        )
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("Please ensure dataset is available and paths are correct")
        return
    
    # Create data splits
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    indices = list(range(total_size))
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    print(f"📊 Splits: {train_size} train, {val_size} val, {test_size} test")
    
    # Create data loaders
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    train_loader = DataLoader(full_dataset, batch_size=args.batch_size, sampler=train_sampler,
                             num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(full_dataset, batch_size=args.batch_size, sampler=val_sampler,
                           num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler,
                            num_workers=args.num_workers, pin_memory=True)
    
    # Get reduced-layer architectures
    print("\\n🏗️  Loading reduced-layer architectures...")
    architectures = get_reduced_layer_architectures()
    
    print(f"Found {len(architectures)} reduced-layer architectures:")
    for i, arch in enumerate(architectures, 1):
        params = sum(p.numel() for p in arch.parameters())
        print(f"  {i}. {arch.name:30} | {arch.depth} layers | {params:,} params | {arch.reduction_factor:.0%} reduction")
    
    # Training configuration
    config = {
        'epochs': args.epochs,
        'patience': args.patience,
        'learning_rate': args.learning_rate,
        'mixed_precision': args.mixed_precision,
        'output_dir': output_dir
    }
    
    # Run experiments
    print(f"\\n🚀 Starting {len(architectures)} reduced-layer experiments...")
    
    all_results = []
    study_start_time = time.time()
    
    for i, architecture in enumerate(architectures, 1):
        try:
            # Move model to device
            model = architecture.to(device)
            
            # Create experiment
            experiment = ReducedLayerExperiment(model, config, device, i)
            
            # Train model
            results = experiment.train_model(train_loader, val_loader)
            all_results.append(results)
            
            # Clean up GPU memory
            del model, experiment
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"❌ Error in experiment {i} ({architecture.name}): {e}")
            continue
    
    total_time = time.time() - study_start_time
    
    # Save complete results
    study_results = {
        'study_info': {
            'study_name': 'Reduced-Layer Architecture Study',
            'completion_time': datetime.now().isoformat(),
            'total_time_minutes': total_time / 60,
            'successful_experiments': len(all_results),
            'configuration': vars(args)
        },
        'experimental_results': all_results
    }
    
    results_file = output_dir / 'complete_reduced_layer_study.json'
    with open(results_file, 'w') as f:
        json.dump(study_results, f, indent=2, default=str)
    
    print(f"\\n🎉 REDUCED-LAYER STUDY COMPLETE!")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print(f"   Successful experiments: {len(all_results)}/{len(architectures)}")
    print(f"   Results saved: {results_file}")
    
    # Generate quick comparison
    create_quick_comparison(all_results, output_dir)
    
    return all_results

def create_quick_comparison(results, output_dir):
    """Create a quick performance comparison"""
    
    print("\\n📊 QUICK PERFORMANCE COMPARISON:")
    print("-" * 60)
    print(f"{'Model':<30} {'Layers':<8} {'Reduction':<10} {'Val Loss':<10} {'Time(min)':<10}")
    print("-" * 60)
    
    # Sort by validation loss
    sorted_results = sorted(results, key=lambda x: x['training_performance']['best_val_loss'])
    
    comparison_data = []
    
    for result in sorted_results:
        model_name = result['model_name']
        layers = result['architecture_info']['depth']
        reduction = f"{result['architecture_info']['reduction_factor']:.0%}"
        val_loss = result['training_performance']['best_val_loss']
        time_min = result['training_performance']['training_minutes']
        
        print(f"{model_name:<30} {layers:<8} {reduction:<10} {val_loss:<10.0f} {time_min:<10.1f}")
        
        comparison_data.append({
            'model_name': model_name,
            'layers': layers,
            'original_layers': result['architecture_info']['original_depth'],
            'reduction_factor': result['architecture_info']['reduction_factor'],
            'has_skip_connections': result['architecture_info']['has_skip_connections'],
            'parameters': result['architecture_info']['parameters'],
            'val_loss': val_loss,
            'training_time_min': time_min,
            'convergence_epoch': result['training_performance']['convergence_epoch']
        })
    
    # Save comparison CSV
    import pandas as pd
    df = pd.DataFrame(comparison_data)
    csv_file = output_dir / 'reduced_layer_comparison.csv'
    df.to_csv(csv_file, index=False)
    
    print(f"\\n📋 Detailed comparison saved: {csv_file}")

def main():
    """Main entry point"""
    
    # Verify environment (if function exists)
    try:
        if not verify_hpc_environment():
            print("⚠️  Environment verification failed, proceeding anyway...")
    except:
        print("⚠️  Could not verify environment, proceeding...")
    
    # Run the study
    try:
        results = run_reduced_layer_study()
        print("\\n✅ Reduced-layer study completed successfully!")
        return results
    except KeyboardInterrupt:
        print("\\n⏹️  Study interrupted by user")
        return None
    except Exception as e:
        print(f"\\n❌ Study failed with error: {e}")
        return None

if __name__ == "__main__":
    main()