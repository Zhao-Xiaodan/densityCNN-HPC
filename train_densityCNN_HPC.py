
#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
import copy
import time
import json
import argparse
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler
import warnings
warnings.filterwarnings('ignore')

# Create argument parser for configuration options
parser = argparse.ArgumentParser(description='Train a CNN for microbead density regression - HPC Optimized')
parser.add_argument('--input_dir', type=str, default='dataset',
                    help='Input directory containing images/ and density.csv (default: dataset)')
parser.add_argument('--output_dir', type=str, default='training_results',
                    help='Directory to save results')
parser.add_argument('--batch_sizes', nargs='+', type=int, default=[256],
                    help='Batch sizes to experiment with')
parser.add_argument('--filter_configs', nargs='+', type=str,
                    default=['64,128,256'],
                    help='Comma-separated filter configurations for each layer')
parser.add_argument('--epochs', type=int, default=50,
                    help='Maximum number of epochs to train')
parser.add_argument('--patience', type=int, default=15,
                    help='Early stopping patience')
parser.add_argument('--learning_rate', type=float, default=3e-4,
                    help='Learning rate for optimizer')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility')
parser.add_argument('--num_workers', type=int, default=18,
                    help='Number of data loading workers')
parser.add_argument('--mixed_precision', action='store_true',
                    help='Use mixed precision training')
parser.add_argument('--cache_dataset', action='store_true',
                    help='Cache dataset in memory for faster loading')
parser.add_argument('--use_enhanced_model', action='store_true',
                    help='Use enhanced CNN architecture')
parser.add_argument('--use_enhanced_loss', action='store_true',
                    help='Use enhanced loss function with regularization')
parser.add_argument('--enhanced_preprocessing', action='store_true',
                    help='Skip resize since images are already 512x512')
parser.add_argument('--data_percentage', type=int, default=100,
                    help='Percentage of data to use (20, 50, or 100)')
parser.add_argument('--dilution_factors', nargs='+', type=str,
                    default=['80x', '160x', '320x', '640x', '1280x', '2560x', '5120x', '10240x'],
                    help='Specific dilution factors to include in training')
parser.add_argument('--use_all_dilutions', action='store_true',
                    help='Use all available dilution factors')

args = parser.parse_args()

# Set random seeds for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

# Enhanced CNN Model
class EnhancedDensityRegressionCNN(nn.Module):
    def __init__(self, filters=[64, 128, 256], input_size=512):
        super(EnhancedDensityRegressionCNN, self).__init__()
        self.filters = filters
        self.input_size = input_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, filters[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(filters[0], filters[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[1], filters[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[2], filters[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(inplace=True)
        )

        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

        self.fc_layers = nn.Sequential(
            nn.Linear(filters[2] * 64, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
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
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# Original CNN Model
class DensityRegressionCNN(nn.Module):
    def __init__(self, filters=[32, 64, 128]):
        super(DensityRegressionCNN, self).__init__()
        self.filters = filters
        self.conv1 = nn.Conv2d(1, filters[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.conv2 = nn.Conv2d(filters[0], filters[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(filters[1])
        self.conv3 = nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(filters[2])
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(filters[2], 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.mean(dim=[2, 3])
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Enhanced Loss Function
class EnhancedLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super(EnhancedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.alpha = alpha

    def forward(self, predictions, targets):
        mse_loss = self.mse(predictions, targets)
        mae_loss = self.mae(predictions, targets)
        return mse_loss + self.alpha * mae_loss

# Optimized Dataset Class
class OptimizedMicrobeadDataset(Dataset):
    def __init__(self, image_dir, density_csv, transform=None, cache_images=False,
                 data_percentage=100, dilution_factors=None, use_all_dilutions=False):
        self.image_dir = image_dir
        self.transform = transform
        self.cache_images = cache_images
        self.image_cache = {}

        self.df = pd.read_csv(density_csv)

        if len(self.df.columns) == 1:
            self.df = self.df.iloc[:, 0].str.split(expand=True)
            self.df.columns = ['filename', 'density']
        elif len(self.df.columns) == 2:
            self.df.columns = ['filename', 'density']
        else:
            raise ValueError("CSV format not recognized. Should have filename and density columns.")

        self.df['density'] = self.df['density'].astype(float)

        if not use_all_dilutions and dilution_factors:
            print(f"Filtering data for dilution factors: {dilution_factors}")
            # Create pattern to match dilution factors at the beginning of filename
            pattern = '|'.join([f'^{factor}_' for factor in dilution_factors])
            mask = self.df['filename'].str.contains(pattern, case=False, na=False)
            self.df = self.df[mask].reset_index(drop=True)
            print(f"After dilution filtering: {len(self.df)} samples")

        if data_percentage < 100:
            sample_size = int(len(self.df) * data_percentage / 100)
            self.df = self.df.sample(n=sample_size, random_state=args.seed).reset_index(drop=True)
            print(f"Using {data_percentage}% of data: {len(self.df)} samples")

        print(f"Final dataset size: {len(self.df)} image-density pairs")

        if self.cache_images:
            print("Caching images in memory...")
            self._cache_all_images()

    def _cache_all_images(self):
        for idx in range(len(self.df)):
            img_name = self.df.iloc[idx]['filename']
            if not img_name.endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                img_name = img_name + '.png'

            img_path = os.path.join(self.image_dir, img_name)
            try:
                image = Image.open(img_path).convert('L')
                self.image_cache[idx] = image
            except Exception as e:
                print(f"Warning: Could not cache image {img_path}: {e}")

        print(f"Cached {len(self.image_cache)} images")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        density = self.df.iloc[idx]['density']

        if self.cache_images and idx in self.image_cache:
            image = self.image_cache[idx]
        else:
            img_name = self.df.iloc[idx]['filename']
            if not img_name.endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                img_name = img_name + '.png'

            img_path = os.path.join(self.image_dir, img_name)
            try:
                image = Image.open(img_path).convert('L')
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                image = Image.new('L', (512, 512), 0)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(density, dtype=torch.float32)

# Optimized preprocessing
def get_transforms(enhanced_preprocessing=False):
    if enhanced_preprocessing:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.RandomAdjustSharpness(sharpness_factor=1.2, p=0.3),
        ])

        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ])

        eval_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    return train_transform, eval_transform

# Training function
def train_model_optimized(model, train_loader, val_loader, config, device='cuda'):
    batch_size = config['batch_size']
    filter_config = config['filter_config']
    num_epochs = config['num_epochs']
    patience = config['patience']
    learning_rate = config['learning_rate']
    use_mixed_precision = config.get('mixed_precision', False)
    use_enhanced_loss = config.get('use_enhanced_loss', False)

    if use_enhanced_loss:
        criterion = EnhancedLoss(alpha=0.1)
    else:
        criterion = nn.MSELoss()

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = GradScaler() if use_mixed_precision else None

    model.to(device)

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    no_improve_epochs = 0
    epochs_completed = 0

    start_time = time.time()

    for epoch in range(num_epochs):
        epochs_completed = epoch + 1

        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0

        for images, densities in train_loader:
            images, densities = images.to(device, non_blocking=True), densities.to(device, non_blocking=True)

            optimizer.zero_grad()

            if use_mixed_precision and scaler:
                with autocast():
                    outputs = model(images).squeeze()
                    loss = criterion(outputs, densities)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images).squeeze()
                loss = criterion(outputs, densities)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss /= num_batches
        history['train_loss'].append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for images, densities in val_loader:
                images, densities = images.to(device, non_blocking=True), densities.to(device, non_blocking=True)

                if use_mixed_precision:
                    with autocast():
                        outputs = model(images).squeeze()
                        loss = criterion(outputs, densities)
                else:
                    outputs = model(images).squeeze()
                    loss = criterion(outputs, densities)

                val_loss += loss.item()
                num_val_batches += 1

        val_loss /= num_val_batches
        history['val_loss'].append(val_loss)

        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0

            model_save_path = os.path.join(
                config['output_dir'],
                f"best_model_batch{batch_size}_filters{'-'.join(map(str, filter_config))}.pth"
            )
            torch.save(model.state_dict(), model_save_path)
            print(f"✅ New best model saved with validation loss: {val_loss:.4f}")
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    time_elapsed = time.time() - start_time
    training_minutes = time_elapsed / 60

    print(f'Training completed in {training_minutes:.2f} minutes')
    print(f'Best validation loss: {best_val_loss:.4f}')

    model.load_state_dict(best_model_wts)

    plt.figure(figsize=(12, 6))
    plt.plot(history['train_loss'], label='Training Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training Progress - Batch: {batch_size}, Filters: {filter_config}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_save_path = os.path.join(
        config['output_dir'],
        f"training_curve_batch{batch_size}_filters{'-'.join(map(str, filter_config))}.png"
    )
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    plt.close()

    training_stats = {
        'config': {
            'batch_size': batch_size,
            'filter_config': filter_config,
            'learning_rate': learning_rate,
            'mixed_precision': use_mixed_precision,
            'enhanced_loss': use_enhanced_loss
        },
        'performance': {
            'epochs_completed': epochs_completed,
            'best_val_loss': float(best_val_loss),
            'training_minutes': training_minutes,
            'peak_memory_gb': torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else None
        },
        'history': {
            'train_loss': [float(x) for x in history['train_loss']],
            'val_loss': [float(x) for x in history['val_loss']]
        }
    }

    return model, training_stats, epochs_completed

# Enhanced evaluation
def evaluate_model_enhanced(model, test_loader, config, device):
    model.eval()
    predictions = []
    actual_values = []

    with torch.no_grad():
        for images, densities in test_loader:
            images, densities = images.to(device, non_blocking=True), densities.to(device, non_blocking=True)

            with autocast() if config.get('mixed_precision', False) else torch.no_grad():
                outputs = model(images).squeeze()

            predictions.extend(outputs.cpu().numpy())
            actual_values.extend(densities.cpu().numpy())

    predictions = np.array(predictions)
    actual_values = np.array(actual_values)

    mse = np.mean((predictions - actual_values) ** 2)
    mae = np.mean(np.abs(predictions - actual_values))
    rmse = np.sqrt(mse)

    ss_res = np.sum((actual_values - predictions) ** 2)
    ss_tot = np.sum((actual_values - np.mean(actual_values)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    mape = np.mean(np.abs((actual_values - predictions) / np.clip(actual_values, 1e-8, None))) * 100
    max_error = np.max(np.abs(actual_values - predictions))

    print(f"📊 Enhanced Evaluation Metrics:")
    print(f"   MSE: {mse:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   R²: {r2:.4f}")
    print(f"   MAPE: {mape:.2f}%")
    print(f"   Max Error: {max_error:.4f}")

    batch_size = config['batch_size']
    filter_config = config['filter_config']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.scatter(actual_values, predictions, alpha=0.6, s=20)
    min_val, max_val = min(min(actual_values), min(predictions)), max(max(actual_values), max(predictions))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax1.set_xlabel('Actual Density')
    ax1.set_ylabel('Predicted Density')
    ax1.set_title(f'Predictions vs Actual\nR² = {r2:.4f}')
    ax1.grid(True, alpha=0.3)

    residuals = actual_values - predictions
    ax2.scatter(actual_values, residuals, alpha=0.6, s=20)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Actual Density')
    ax2.set_ylabel('Residuals')
    ax2.set_title(f'Residual Plot\nMAE = {mae:.4f}')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_save_path = os.path.join(
        config['output_dir'],
        f"enhanced_evaluation_batch{batch_size}_filters{'-'.join(map(str, filter_config))}.png"
    )
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    plt.close()

    eval_metrics = {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'mape': float(mape),
        'max_error': float(max_error)
    }

    return eval_metrics

# Main execution
if __name__ == "__main__":
    print("🚀 Starting HPC-Optimized Density CNN Training")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, args.input_dir)
    image_dir = os.path.join(input_path, 'images')
    density_file = os.path.join(input_path, 'density.csv')

    for path, name in [(input_path, 'Input directory'), (image_dir, 'Images directory'), (density_file, 'Density file')]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} does not exist: {path}")

    print(f"📁 Input directory: {input_path}")
    print(f"🖼️  Images directory: {image_dir}")
    print(f"📊 Density file: {density_file}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    config_summary = {
        'script_args': vars(args),
        'timestamp': timestamp,
        'enhanced_features': {
            'mixed_precision': args.mixed_precision,
            'enhanced_model': args.use_enhanced_model,
            'enhanced_loss': args.use_enhanced_loss,
            'enhanced_preprocessing': args.enhanced_preprocessing,
            'cache_dataset': args.cache_dataset
        }
    }

    with open(os.path.join(output_dir, 'config_summary.json'), 'w') as f:
        json.dump(config_summary, f, indent=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")

    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(0.95)

    train_transform, eval_transform = get_transforms(args.enhanced_preprocessing)

    print("\n📂 Creating datasets...")
    train_val_dataset = OptimizedMicrobeadDataset(
        image_dir, density_file,
        transform=train_transform,
        cache_images=args.cache_dataset,
        data_percentage=args.data_percentage,
        dilution_factors=args.dilution_factors,
        use_all_dilutions=args.use_all_dilutions
    )

    test_dataset = OptimizedMicrobeadDataset(
        image_dir, density_file,
        transform=eval_transform,
        cache_images=args.cache_dataset,
        data_percentage=args.data_percentage,
        dilution_factors=args.dilution_factors,
        use_all_dilutions=args.use_all_dilutions
    )

    total_size = len(train_val_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    indices = list(range(total_size))
    np.random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]

    print(f"📊 Dataset split: {train_size} train, {val_size} val, {test_size} test")

    from torch.utils.data import SubsetRandomSampler

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    all_results = []

    print(f"\n🧪 Starting experiments with {len(args.batch_sizes)} batch sizes and {len(args.filter_configs)} filter configs")

    for batch_size in args.batch_sizes:
        for filter_str in args.filter_configs:
            filter_config = [int(x) for x in filter_str.split(',')]

            print(f"\n{'='*80}")
            print(f"🔬 EXPERIMENT: Batch={batch_size}, Filters={filter_config}")
            print(f"{'='*80}")

            train_loader = DataLoader(
                train_val_dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=True if args.num_workers > 0 else False,
                prefetch_factor=2 if args.num_workers > 0 else 2
            )

            val_loader = DataLoader(
                train_val_dataset,
                batch_size=batch_size,
                sampler=val_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=True if args.num_workers > 0 else False,
                prefetch_factor=2 if args.num_workers > 0 else 2
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                sampler=test_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=True if args.num_workers > 0 else False
            )

            if args.use_enhanced_model:
                input_size = 512 if args.enhanced_preprocessing else 224
                model = EnhancedDensityRegressionCNN(filters=filter_config, input_size=input_size)
                print(f"🔧 Using Enhanced CNN with input size {input_size}")
            else:
                model = DensityRegressionCNN(filters=filter_config)
                print(f"🔧 Using Standard CNN")

            print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

            experiment_config = {
                'batch_size': batch_size,
                'filter_config': filter_config,
                'num_epochs': args.epochs,
                'patience': args.patience,
                'learning_rate': args.learning_rate,
                'output_dir': output_dir,
                'mixed_precision': args.mixed_precision,
                'use_enhanced_loss': args.use_enhanced_loss,
                'data_percentage': args.data_percentage,
                'dilution_factors': args.dilution_factors
            }

            print("\n🚂 Starting training...")
            start_experiment_time = time.time()

            try:
                trained_model, training_stats, epochs_completed = train_model_optimized(
                    model, train_loader, val_loader, experiment_config, device=device
                )

                print("\n📈 Evaluating model...")
                eval_metrics = evaluate_model_enhanced(trained_model, test_loader, experiment_config, device)

                experiment_time = (time.time() - start_experiment_time) / 60

                experiment_results = {
                    'config': {
                        'batch_size': batch_size,
                        'filter_config': filter_config,
                        'data_percentage': args.data_percentage,
                        'dilution_factors': args.dilution_factors,
                        'enhanced_features': {
                            'model': args.use_enhanced_model,
                            'loss': args.use_enhanced_loss,
                            'preprocessing': args.enhanced_preprocessing,
                            'mixed_precision': args.mixed_precision
                        }
                    },
                    'training': training_stats,
                    'evaluation': eval_metrics,
                    'experiment_time_minutes': experiment_time
                }

                all_results.append(experiment_results)

                result_filename = f"results_batch{batch_size}_filters{'-'.join(map(str, filter_config))}_{args.data_percentage}pct.json"
                with open(os.path.join(output_dir, result_filename), 'w') as f:
                    json.dump(experiment_results, f, indent=4)

                print(f"✅ Experiment completed in {experiment_time:.2f} minutes")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"❌ Experiment failed: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

    print(f"\n📋 Saving comprehensive results...")
    with open(os.path.join(output_dir, 'all_experiment_results.json'), 'w') as f:
        json.dump(all_results, f, indent=4)

    if all_results:
        comparison_data = []
        for result in all_results:
            config = result['config']
            training = result['training']['performance']
            evaluation = result['evaluation']

            comparison_data.append({
                'Batch_Size': config['batch_size'],
                'Filter_Config': '-'.join(map(str, config['filter_config'])),
                'Data_Percentage': config['data_percentage'],
                'Enhanced_Model': config['enhanced_features']['model'],
                'Mixed_Precision': config['enhanced_features']['mixed_precision'],
                'Best_Val_Loss': training['best_val_loss'],
                'MSE': evaluation['mse'],
                'MAE': evaluation['mae'],
                'RMSE': evaluation['rmse'],
                'R2_Score': evaluation['r2'],
                'MAPE': evaluation['mape'],
                'Max_Error': evaluation['max_error'],
                'Epochs_Completed': training['epochs_completed'],
                'Training_Time_Min': training['training_minutes'],
                'Total_Experiment_Time_Min': result['experiment_time_minutes'],
                'Peak_Memory_GB': training.get('peak_memory_gb', 'N/A')
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('R2_Score', ascending=False)

        comparison_df.to_csv(os.path.join(output_dir, 'experiment_comparison.csv'), index=False)

        if len(comparison_df) > 0:
            best_config = comparison_df.iloc[0]
            print("\n" + "="*80)
            print("🏆 BEST CONFIGURATION RESULTS:")
            print("="*80)
            print(f"📊 Configuration:")
            print(f"   Batch Size: {best_config['Batch_Size']}")
            print(f"   Filters: {best_config['Filter_Config']}")
            print(f"   Data: {best_config['Data_Percentage']}%")
            print(f"   Enhanced Model: {best_config['Enhanced_Model']}")
            print(f"   Mixed Precision: {best_config['Mixed_Precision']}")
            print(f"\n📈 Performance Metrics:")
            print(f"   R² Score: {best_config['R2_Score']:.4f}")
            print(f"   MSE: {best_config['MSE']:.4f}")
            print(f"   MAE: {best_config['MAE']:.4f}")
            print(f"   RMSE: {best_config['RMSE']:.4f}")
            print(f"   MAPE: {best_config['MAPE']:.2f}%")
            print(f"\n⏱️  Training Info:")
            print(f"   Epochs: {best_config['Epochs_Completed']}")
            print(f"   Training Time: {best_config['Training_Time_Min']:.2f} min")
            print(f"   Total Time: {best_config['Total_Experiment_Time_Min']:.2f} min")
            print(f"   Peak GPU Memory: {best_config['Peak_Memory_GB']}")
            print("="*80)

        if len(comparison_df) > 1:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

            ax1.bar(range(len(comparison_df)), comparison_df['R2_Score'])
            ax1.set_title('R² Score by Configuration')
            ax1.set_ylabel('R² Score')
            ax1.set_xticks(range(len(comparison_df)))
            ax1.set_xticklabels([f"Batch{row['Batch_Size']}\nF{row['Filter_Config']}"
                                for _, row in comparison_df.iterrows()], rotation=45)

            ax2.scatter(comparison_df['Training_Time_Min'], comparison_df['R2_Score'],
                       s=comparison_df['Batch_Size']*2, alpha=0.7)
            ax2.set_xlabel('Training Time (minutes)')
            ax2.set_ylabel('R² Score')
            ax2.set_title('Training Efficiency (bubble size = batch size)')

            memory_data = comparison_df[comparison_df['Peak_Memory_GB'] != 'N/A']
            if len(memory_data) > 0:
                ax3.bar(range(len(memory_data)), memory_data['Peak_Memory_GB'].astype(float))
                ax3.set_title('Peak GPU Memory Usage')
                ax3.set_ylabel('Memory (GB)')
                ax3.set_xticks(range(len(memory_data)))
                ax3.set_xticklabels([f"Batch{row['Batch_Size']}" for _, row in memory_data.iterrows()], rotation=45)

            x_pos = range(len(comparison_df))
            width = 0.35
            ax4.bar([x - width/2 for x in x_pos], comparison_df['MAE'], width, label='MAE', alpha=0.8)
            ax4.bar([x + width/2 for x in x_pos], comparison_df['RMSE'], width, label='RMSE', alpha=0.8)
            ax4.set_title('Error Metrics Comparison')
            ax4.set_ylabel('Error Value')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels([f"B{row['Batch_Size']}" for _, row in comparison_df.iterrows()], rotation=45)
            ax4.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'performance_summary.png'), dpi=300, bbox_inches='tight')
            plt.close()

    total_experiments = len(all_results)
    successful_experiments = len([r for r in all_results if 'evaluation' in r])

    final_report = f"""
🎉 HPC DENSITY CNN TRAINING COMPLETED
{'='*60}

📊 EXPERIMENT SUMMARY:
   Total Experiments: {total_experiments}
   Successful: {successful_experiments}
   Data Used: {args.data_percentage}% of dataset
   Dilution Factors: {', '.join(args.dilution_factors)}

🔧 CONFIGURATION USED:
   Enhanced Model: {args.use_enhanced_model}
   Enhanced Loss: {args.use_enhanced_loss}
   Enhanced Preprocessing: {args.enhanced_preprocessing}
   Mixed Precision: {args.mixed_precision}
   Dataset Caching: {args.cache_dataset}

📁 OUTPUT LOCATION: {output_dir}

📋 KEY FILES GENERATED:
   📊 experiment_comparison.csv - Detailed comparison table
   📈 performance_summary.png - Visual performance analysis
   🏆 best_model_*.pth - Trained model weights
   📄 all_experiment_results.json - Complete results
   ⚙️  config_summary.json - Configuration record

🚀 PERFORMANCE OPTIMIZATIONS APPLIED:
   ✅ GPU memory optimization
   ✅ Mixed precision training (if enabled)
   ✅ Optimized data loading with {args.num_workers} workers
   ✅ Enhanced CNN architecture (if enabled)
   ✅ Efficient preprocessing for 512x512 images
   ✅ Memory pinning and persistent workers
   ✅ Gradient scaling and learning rate scheduling
"""

    print(final_report)

    with open(os.path.join(output_dir, 'final_report.txt'), 'w') as f:
        f.write(final_report)

    print(f"\n✨ Training completed successfully!")
    print(f"📁 All results saved to: {output_dir}")

    if torch.cuda.is_available():
        print(f"\n🎮 Final GPU Memory Usage:")
        print(f"   Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"   Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
