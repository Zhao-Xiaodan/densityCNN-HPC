#!/usr/bin/env python
"""
ResNet_Shallow Hyperparameter Study: Bottleneck Analysis
========================================================
Systematic investigation of what factors caused ResNet_Shallow's dramatic performance improvement.

Research Questions:
1. Is it gradient noise from smaller batch sizes?
2. Is it memory pressure relief?
3. Is it learning rate interaction with batch size?
4. Is it data loading efficiency?
5. Is it gradient accumulation vs. true batch size effects?

Experimental Design:
- Systematic variation of batch_size, learning_rate, num_workers
- Gradient noise analysis with different batch sizes
- Memory pressure monitoring
- Gradient accumulation vs. true batch size comparison
- Learning rate scaling experiments
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

# HPC/Container optimizations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256,expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async for performance
torch.backends.cudnn.benchmark = True

print("🔬 ResNet_Shallow Hyperparameter Bottleneck Study")
print("=" * 80)

# ============================================================================
# ARGUMENT PARSER
# ============================================================================

parser = argparse.ArgumentParser(description='ResNet_Shallow Hyperparameter Study')
parser.add_argument('--input_dir', type=str, default='dataset_preprocessed',
                    help='Input directory containing images/ and density.csv')
parser.add_argument('--output_dir', type=str, default='resnet_shallow_hyperparameter_study',
                    help='Directory to save results')
parser.add_argument('--epochs', type=int, default=50,
                    help='Maximum number of epochs to train')
parser.add_argument('--patience', type=int, default=15,
                    help='Early stopping patience')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility')
parser.add_argument('--data_percentage', type=int, default=50,
                    help='Percentage of data to use')
parser.add_argument('--dilution_factors', nargs='+', type=str,
                    default=['80x', '160x', '320x', '640x', '1280x', '2560x', '5120x', '10240x'],
                    help='Dilution factors to include')
parser.add_argument('--mixed_precision', action='store_true', default=True,
                    help='Use mixed precision training')
parser.add_argument('--quick_study', action='store_true', default=False,
                    help='Run quick study with fewer epochs for testing')

args = parser.parse_args()

# Set seeds for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

print(f"📊 Dataset: {args.data_percentage}% of {args.input_dir}")
print(f"🧪 Focus: ResNet_Shallow architecture bottleneck analysis")
print("=" * 80)

# ============================================================================
# RESNET_SHALLOW ARCHITECTURE (IDENTICAL TO ORIGINAL)
# ============================================================================

class ResidualBlock(nn.Module):
    """Basic residual block for ResNet implementations"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class ResNetShallowCNN(nn.Module):
    """4-layer ResNet-style CNN with residual blocks - EXACT REPLICA"""
    def __init__(self, base_filters=64):
        super(ResNetShallowCNN, self).__init__()
        self.name = "ResNet_Shallow"
        self.depth = 4
        self.has_skip_connections = True
        self.architecture_type = "ResNet"

        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, base_filters, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.layer1 = ResidualBlock(base_filters, base_filters)
        self.layer2 = ResidualBlock(base_filters, base_filters*2, stride=2)
        self.layer3 = ResidualBlock(base_filters*2, base_filters*4, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(base_filters*4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ============================================================================
# DATASET CLASS
# ============================================================================

class MicrobeadDataset(Dataset):
    """Dataset class for microbead density estimation - Compatible with working comprehensive study"""
    def __init__(self, image_dir, density_csv, dilution_factors=None, transform=None, data_percentage=100, use_all_dilutions=False):
        self.image_dir = image_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # Load CSV data using working comprehensive study pattern
        self.df = pd.read_csv(density_csv)

        # Handle CSV format - ensure proper column names (from working comprehensive study)
        if len(self.df.columns) == 1:
            self.df = self.df.iloc[:, 0].str.split(expand=True)
            self.df.columns = ['filename', 'density']
        elif len(self.df.columns) == 2:
            self.df.columns = ['filename', 'density']

        self.df['density'] = self.df['density'].astype(float)

        # Filter by dilution factors using filename patterns (working comprehensive study approach)
        if not use_all_dilutions and dilution_factors:
            pattern = '|'.join([f'^{factor}_' for factor in dilution_factors])
            mask = self.df['filename'].str.contains(pattern, case=False, na=False)
            self.df = self.df[mask].reset_index(drop=True)

        # Apply data percentage sampling
        if data_percentage < 100:
            n_samples = int(len(self.df) * data_percentage / 100)
            self.df = self.df.sample(n=n_samples, random_state=42).reset_index(drop=True)

        print(f"✅ Loaded {len(self.df)} samples")
        print(f"   Density range: {self.df['density'].min():.1f} - {self.df['density'].max():.1f}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image using filename (working comprehensive study pattern)
        image_path = os.path.join(self.image_dir, row['filename'])
        try:
            image = Image.open(image_path)
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            image = torch.zeros(1, 224, 224)

        density = float(row['density'])
        return image, density

# ============================================================================
# EXPERIMENTAL CONFIGURATIONS
# ============================================================================

def get_experimental_configs():
    """Define systematic hyperparameter experiments"""
    
    base_config = {
        'learning_rate': 3e-4,
        'batch_size': 32,
        'num_workers': 6,
        'weight_decay': 1e-4,
        'gradient_accumulation': 1,
        'lr_scheduler': 'cosine',
        'memory_cleanup_freq': 5,
    }
    
    experiments = []
    
    # 1. BATCH SIZE STUDY - Core hypothesis test
    print("🧪 Experiment Group 1: Batch Size Effects")
    batch_sizes = [16, 32, 64, 96, 128, 192, 256]
    for batch_size in batch_sizes:
        config = base_config.copy()
        config.update({
            'batch_size': batch_size,
            'experiment_group': 'batch_size_study',
            'experiment_id': f'batch_{batch_size}',
            'description': f'Batch size {batch_size} - gradient noise analysis'
        })
        experiments.append(config)
    
    # 2. LEARNING RATE SCALING STUDY - Batch size dependent LR
    print("🧪 Experiment Group 2: Learning Rate Scaling")
    lr_batch_combinations = [
        (32, 3e-4),      # Original successful config
        (64, 6e-4),      # Linear scaling
        (128, 1.2e-3),   # Linear scaling  
        (32, 1e-4),      # Lower LR with small batch
        (32, 1e-3),      # Higher LR with small batch
        (128, 3e-4),     # Original LR with large batch (failed config)
    ]
    for batch_size, lr in lr_batch_combinations:
        config = base_config.copy()
        config.update({
            'batch_size': batch_size,
            'learning_rate': lr,
            'experiment_group': 'lr_scaling_study',
            'experiment_id': f'batch_{batch_size}_lr_{lr:.0e}',
            'description': f'Batch {batch_size} with LR {lr:.0e} - scaling analysis'
        })
        experiments.append(config)
    
    # 3. GRADIENT ACCUMULATION STUDY - True batch size vs effective batch size
    print("🧪 Experiment Group 3: Gradient Accumulation vs True Batch Size")
    accumulation_configs = [
        (32, 1),    # True batch 32
        (16, 2),    # Effective batch 32 via accumulation
        (8, 4),     # Effective batch 32 via accumulation
        (64, 1),    # True batch 64
        (32, 2),    # Effective batch 64 via accumulation
        (16, 4),    # Effective batch 64 via accumulation
        (128, 1),   # True batch 128
        (32, 4),    # Effective batch 128 via accumulation
    ]
    for batch_size, grad_accum in accumulation_configs:
        config = base_config.copy()
        config.update({
            'batch_size': batch_size,
            'gradient_accumulation': grad_accum,
            'experiment_group': 'gradient_accumulation_study',
            'experiment_id': f'true_batch_{batch_size}_accum_{grad_accum}_eff_{batch_size*grad_accum}',
            'description': f'True batch {batch_size}, accumulation {grad_accum}, effective {batch_size*grad_accum}'
        })
        experiments.append(config)
    
    # 4. MEMORY AND DATA LOADING STUDY
    print("🧪 Experiment Group 4: Memory and Data Loading")
    worker_memory_configs = [
        (32, 0, 3),     # No workers, frequent cleanup
        (32, 2, 3),     # Few workers, frequent cleanup  
        (32, 6, 3),     # Moderate workers, frequent cleanup
        (32, 6, 5),     # Moderate workers, normal cleanup
        (32, 6, 10),    # Moderate workers, infrequent cleanup
        (32, 12, 5),    # Many workers, normal cleanup
        (32, 18, 5),    # Original failed config workers
        (128, 18, 5),   # Original failed config exactly
    ]
    for batch_size, workers, cleanup_freq in worker_memory_configs:
        config = base_config.copy()
        config.update({
            'batch_size': batch_size,
            'num_workers': workers,
            'memory_cleanup_freq': cleanup_freq,
            'experiment_group': 'memory_data_study',
            'experiment_id': f'batch_{batch_size}_workers_{workers}_cleanup_{cleanup_freq}',
            'description': f'Batch {batch_size}, workers {workers}, cleanup every {cleanup_freq} batches'
        })
        experiments.append(config)
    
    # 5. LEARNING RATE SCHEDULER STUDY
    print("🧪 Experiment Group 5: Learning Rate Scheduler")
    scheduler_configs = [
        ('cosine', 32),
        ('step', 32), 
        ('exponential', 32),
        ('plateau', 32),
        ('cosine', 128),
        ('step', 128),
    ]
    for scheduler, batch_size in scheduler_configs:
        config = base_config.copy()
        config.update({
            'batch_size': batch_size,
            'lr_scheduler': scheduler,
            'experiment_group': 'scheduler_study', 
            'experiment_id': f'scheduler_{scheduler}_batch_{batch_size}',
            'description': f'{scheduler} scheduler with batch size {batch_size}'
        })
        experiments.append(config)
    
    print(f"📋 Total experiments designed: {len(experiments)}")
    return experiments

# ============================================================================
# ADVANCED TRAINING AND ANALYSIS
# ============================================================================

class GradientAnalyzer:
    """Analyze gradient properties during training"""
    def __init__(self):
        self.gradient_norms = []
        self.gradient_variance = []
        self.gradient_noise_scale = []
        self.batch_gradient_norms = []
        
    def analyze_gradients(self, model, data_loader, criterion, device, num_batches=10):
        """Analyze gradient noise and variance"""
        model.train()
        batch_gradients = []
        
        with torch.no_grad():
            # Collect gradients from multiple batches
            for i, (images, targets) in enumerate(data_loader):
                if i >= num_batches:
                    break
                    
                images, targets = images.to(device), targets.to(device)
                
                # Forward pass
                model.zero_grad()
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs.squeeze(), targets)
                
                # Backward pass
                loss.backward()
                
                # Collect gradient norms
                grad_norm = 0
                grad_vector = []
                for param in model.parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        grad_norm += param_norm.item() ** 2
                        grad_vector.extend(param.grad.data.flatten().cpu().numpy())
                
                grad_norm = grad_norm ** 0.5
                batch_gradients.append(np.array(grad_vector))
                self.batch_gradient_norms.append(grad_norm)
        
        # Calculate gradient statistics
        if len(batch_gradients) > 1:
            batch_gradients = np.array(batch_gradients)
            
            # Gradient variance across batches
            grad_variance = np.var(batch_gradients, axis=0).mean()
            self.gradient_variance.append(grad_variance)
            
            # Gradient noise scale (variance / mean²)  
            grad_mean = np.mean(batch_gradients, axis=0)
            grad_var = np.var(batch_gradients, axis=0)
            noise_scale = np.mean(grad_var / (grad_mean**2 + 1e-8))
            self.gradient_noise_scale.append(noise_scale)
            
            # Overall gradient norm
            overall_norm = np.mean(self.batch_gradient_norms[-num_batches:])
            self.gradient_norms.append(overall_norm)
            
            return {
                'gradient_norm': overall_norm,
                'gradient_variance': grad_variance, 
                'gradient_noise_scale': noise_scale,
                'batch_count': len(batch_gradients)
            }
        
        return None

class MemoryMonitor:
    """Monitor GPU memory usage patterns"""
    def __init__(self):
        self.memory_snapshots = []
        self.peak_memory = []
        
    def snapshot(self, label=""):
        """Take memory snapshot"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            
            snapshot = {
                'label': label,
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'timestamp': time.time()
            }
            self.memory_snapshots.append(snapshot)
            return snapshot
        return None
    
    def get_peak_memory(self):
        """Get peak memory usage"""
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 1024**3
            self.peak_memory.append(peak)
            return peak
        return 0

def enhanced_memory_cleanup(aggressive=False):
    """Enhanced memory cleanup"""
    if aggressive:
        print("🧹 Aggressive memory cleanup...")
    
    for _ in range(3 if aggressive else 1):
        gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        if aggressive:
            torch.cuda.reset_peak_memory_stats()

def train_with_comprehensive_analysis(model, train_loader, val_loader, config, device):
    """Train with comprehensive gradient and memory analysis"""
    
    print(f"🚀 Training with config: {config['experiment_id']}")
    
    # Initialize analyzers
    gradient_analyzer = GradientAnalyzer()
    memory_monitor = MemoryMonitor()
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scaler = GradScaler() if args.mixed_precision else None
    
    # Learning rate scheduler
    if config['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    elif config['lr_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    elif config['lr_scheduler'] == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    else:  # plateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training history
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': [],
        'gradient_analysis': [],
        'memory_snapshots': [],
        'gradient_accumulation_steps': [],
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    epochs = 30 if args.quick_study else args.epochs
    
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        batch_count = 0
        gradient_accumulation = config['gradient_accumulation']
        
        # Memory snapshot at epoch start
        memory_monitor.snapshot(f"epoch_{epoch}_start")
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            
            # Forward pass
            with autocast() if args.mixed_precision else torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs.squeeze(), targets) / gradient_accumulation
            
            # Backward pass
            if args.mixed_precision:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            epoch_train_loss += loss.item() * gradient_accumulation
            batch_count += 1
            
            # Gradient accumulation step
            if (batch_idx + 1) % gradient_accumulation == 0:
                if args.mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                
                training_history['gradient_accumulation_steps'].append(batch_idx)
            
            # Memory cleanup
            if batch_count % config['memory_cleanup_freq'] == 0:
                enhanced_memory_cleanup(aggressive=False)
            
            # Gradient analysis (every 50 batches)
            if batch_idx % 50 == 0 and batch_idx > 0:
                grad_stats = gradient_analyzer.analyze_gradients(
                    model, train_loader, criterion, device, num_batches=5
                )
                if grad_stats:
                    grad_stats['epoch'] = epoch
                    grad_stats['batch'] = batch_idx
                    training_history['gradient_analysis'].append(grad_stats)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_count = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                with autocast() if args.mixed_precision else torch.no_grad():
                    outputs = model(images)
                    loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item()
                val_count += 1
        
        avg_train_loss = epoch_train_loss / batch_count
        avg_val_loss = val_loss / val_count
        
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(avg_val_loss)
        training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Memory snapshot at epoch end
        memory_snapshot = memory_monitor.snapshot(f"epoch_{epoch}_end")
        training_history['memory_snapshots'].append(memory_snapshot)
        
        # Learning rate scheduler
        if config['lr_scheduler'] == 'plateau':
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()
        
        print(f"Epoch {epoch:3d}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    training_time = (time.time() - start_time) / 60
    peak_memory = memory_monitor.get_peak_memory()
    
    return {
        'best_val_loss': best_val_loss,
        'training_time_minutes': training_time,
        'epochs_completed': epoch + 1,
        'peak_memory_gb': peak_memory,
        'final_train_loss': avg_train_loss,
        'history': training_history,
        'gradient_analyzer': gradient_analyzer,
        'memory_monitor': memory_monitor
    }

def evaluate_model(model, test_loader, device):
    """Evaluate model and return comprehensive metrics"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            with autocast() if args.mixed_precision else torch.no_grad():
                outputs = model(images)
            
            all_predictions.extend(outputs.squeeze().cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    r2 = r2_score(all_targets, all_predictions)
    mse = mean_squared_error(all_targets, all_predictions)
    mae = mean_absolute_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    
    # Additional metrics
    mape = np.mean(np.abs((all_targets - all_predictions) / all_targets)) * 100
    max_error = np.max(np.abs(all_targets - all_predictions))
    
    return {
        'r2_score': r2,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'max_error': max_error,
        'predictions': all_predictions,
        'targets': all_targets
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("🔍 Starting ResNet_Shallow Hyperparameter Bottleneck Study")
    
    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, args.output_dir + f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load dataset using working comprehensive study pattern
    images_dir = os.path.join(args.input_dir, 'images')
    csv_file = os.path.join(args.input_dir, 'density.csv')

    full_dataset = MicrobeadDataset(
        image_dir=images_dir,
        density_csv=csv_file,
        dilution_factors=args.dilution_factors,
        data_percentage=args.data_percentage,
        use_all_dilutions=False
    )
    
    # Create data splits
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    
    # Use only specified percentage of data
    if args.data_percentage < 100:
        np.random.seed(args.seed)
        np.random.shuffle(indices)
        split_idx = int(dataset_size * (args.data_percentage / 100))
        indices = indices[:split_idx]
    
    # Train/val/test splits
    train_size = int(0.8 * len(indices))
    val_size = int(0.1 * len(indices))
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size] 
    test_indices = indices[train_size + val_size:]
    
    print(f"📊 Data splits: Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    # Get experimental configurations
    experimental_configs = get_experimental_configs()
    
    # Quick study mode - run fewer experiments
    if args.quick_study:
        experimental_configs = experimental_configs[:10]  # First 10 experiments
        print(f"⚡ Quick study mode: Running {len(experimental_configs)} experiments")
    
    # Run experiments
    all_results = []
    study_start_time = time.time()
    
    for exp_idx, config in enumerate(experimental_configs):
        print(f"\n{'='*80}")
        print(f"🧪 EXPERIMENT {exp_idx + 1}/{len(experimental_configs)}: {config['experiment_id']}")
        print(f"📋 {config['description']}")
        print(f"{'='*80}")
        
        try:
            # Create model (fresh for each experiment)
            model = ResNetShallowCNN().to(device)
            param_count = sum(p.numel() for p in model.parameters())
            print(f"📐 Model parameters: {param_count:,}")
            
            # Create data loaders with experiment-specific config
            train_sampler = SubsetRandomSampler(train_indices)
            val_sampler = SubsetRandomSampler(val_indices)
            test_sampler = SubsetRandomSampler(test_indices)
            
            train_loader = DataLoader(
                full_dataset, 
                batch_size=config['batch_size'],
                sampler=train_sampler,
                num_workers=config['num_workers'],
                pin_memory=True,
                persistent_workers=True if config['num_workers'] > 0 else False
            )
            
            val_loader = DataLoader(
                full_dataset,
                batch_size=config['batch_size'],
                sampler=val_sampler,
                num_workers=config['num_workers'],
                pin_memory=True,
                persistent_workers=True if config['num_workers'] > 0 else False
            )
            
            test_loader = DataLoader(
                full_dataset,
                batch_size=config['batch_size'],
                sampler=test_sampler,
                num_workers=config['num_workers'],
                pin_memory=True
            )
            
            # Train model with comprehensive analysis
            training_results = train_with_comprehensive_analysis(
                model, train_loader, val_loader, config, device
            )
            
            # Evaluate model  
            evaluation_results = evaluate_model(model, test_loader, device)
            
            # Combine results
            experiment_result = {
                'experiment_id': config['experiment_id'],
                'experiment_group': config['experiment_group'],
                'description': config['description'],
                'config': config,
                'training': training_results,
                'evaluation': evaluation_results,
                'model_parameters': param_count
            }
            
            all_results.append(experiment_result)
            
            print(f"✅ Experiment completed successfully")
            print(f"   R² Score: {evaluation_results['r2_score']:.4f}")
            print(f"   MSE: {evaluation_results['mse']:.1f}")
            print(f"   Training Time: {training_results['training_time_minutes']:.1f} min")
            print(f"   Peak Memory: {training_results['peak_memory_gb']:.2f} GB")
            
            # Save individual experiment result
            exp_file = os.path.join(output_dir, f"experiment_{exp_idx + 1:02d}_{config['experiment_id']}_results.json")
            with open(exp_file, 'w') as f:
                json.dump(experiment_result, f, indent=4, default=str)
            
            # Memory cleanup
            del model
            enhanced_memory_cleanup(aggressive=True)
            
        except Exception as e:
            print(f"❌ Experiment {exp_idx + 1} failed: {str(e)}")
            import traceback
            print(traceback.format_exc())
            
            # Save error info
            error_result = {
                'experiment_id': config['experiment_id'],
                'experiment_group': config['experiment_group'],
                'error': str(e),
                'config': config
            }
            all_results.append(error_result)
            
            enhanced_memory_cleanup(aggressive=True)
            continue
    
    # Analysis and visualization
    total_study_time = (time.time() - study_start_time) / 60
    print(f"\n🎯 Study completed in {total_study_time:.1f} minutes")
    
    # Save complete results
    complete_results = {
        'study_info': {
            'total_experiments': len(experimental_configs),
            'successful_experiments': len([r for r in all_results if 'error' not in r]),
            'total_time_minutes': total_study_time,
            'args': vars(args)
        },
        'results': all_results
    }
    
    results_file = os.path.join(output_dir, 'complete_hyperparameter_study.json')
    with open(results_file, 'w') as f:
        json.dump(complete_results, f, indent=4, default=str)
    
    print(f"📁 Results saved to: {output_dir}")
    print(f"📊 Complete study saved to: {results_file}")
    
    # Create summary analysis
    create_analysis_summary(all_results, output_dir)
    
    print("\n🔬 ResNet_Shallow Hyperparameter Bottleneck Study Complete!")

def create_analysis_summary(results, output_dir):
    """Create summary analysis of all experiments"""
    successful_results = [r for r in results if 'error' not in r]
    
    if len(successful_results) == 0:
        print("⚠️ No successful experiments for analysis")
        return
    
    # Create summary DataFrame
    summary_data = []
    for result in successful_results:
        config = result['config']
        training = result['training']
        evaluation = result['evaluation']
        
        summary_data.append({
            'experiment_id': result['experiment_id'],
            'experiment_group': result['experiment_group'],
            'batch_size': config['batch_size'],
            'learning_rate': config['learning_rate'],
            'num_workers': config['num_workers'],
            'gradient_accumulation': config['gradient_accumulation'],
            'lr_scheduler': config['lr_scheduler'],
            'memory_cleanup_freq': config['memory_cleanup_freq'],
            'effective_batch_size': config['batch_size'] * config['gradient_accumulation'],
            'r2_score': evaluation['r2_score'],
            'mse': evaluation['mse'],
            'training_time_min': training['training_time_minutes'],
            'peak_memory_gb': training['peak_memory_gb'],
            'epochs_completed': training['epochs_completed'],
            'best_val_loss': training['best_val_loss']
        })
    
    df = pd.DataFrame(summary_data)
    
    # Save summary CSV
    summary_file = os.path.join(output_dir, 'hyperparameter_study_summary.csv')
    df.to_csv(summary_file, index=False)
    
    # Create analysis report
    report = f"""
# ResNet_Shallow Hyperparameter Study Analysis Report

## Summary Statistics
- Total successful experiments: {len(successful_results)}
- Best R² score: {df['r2_score'].max():.4f}
- Worst R² score: {df['r2_score'].min():.4f}  
- Mean R² score: {df['r2_score'].mean():.4f}
- R² standard deviation: {df['r2_score'].std():.4f}

## Top 5 Performing Configurations:
{df.nlargest(5, 'r2_score')[['experiment_id', 'batch_size', 'learning_rate', 'r2_score', 'mse']].to_string(index=False)}

## Bottom 5 Performing Configurations:
{df.nsmallest(5, 'r2_score')[['experiment_id', 'batch_size', 'learning_rate', 'r2_score', 'mse']].to_string(index=False)}

## Analysis by Experiment Group:
"""
    
    for group in df['experiment_group'].unique():
        group_df = df[df['experiment_group'] == group]
        report += f"""
### {group.replace('_', ' ').title()}:
- Experiments: {len(group_df)}
- Best R²: {group_df['r2_score'].max():.4f}
- Mean R²: {group_df['r2_score'].mean():.4f}
- R² Std: {group_df['r2_score'].std():.4f}
"""
    
    # Save analysis report
    report_file = os.path.join(output_dir, 'analysis_report.md')
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"📊 Summary analysis saved to: {summary_file}")
    print(f"📋 Analysis report saved to: {report_file}")

if __name__ == "__main__":
    main()