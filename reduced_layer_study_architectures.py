#!/usr/bin/env python3
"""
Reduced-Layer Study Architecture Definitions
Tests architectural robustness by systematically reducing layers while maintaining comparable parameter counts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ReducedBaselineShallowCNN(nn.Module):
    """Baseline Shallow reduced from 4 to 3 layers"""
    def __init__(self, base_filters=64):
        super().__init__()
        self.name = "ReducedBaseline_Shallow"
        self.depth = 3
        self.has_skip_connections = False
        self.original_depth = 4
        self.reduction_factor = 0.75
        
        # 3-layer architecture (reduced from 4)
        self.features = nn.Sequential(
            # Layer 1: Wider to compensate for depth reduction
            nn.Conv2d(3, base_filters * 2, 3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Layer 2: Enhanced feature extraction
            nn.Conv2d(base_filters * 2, base_filters * 4, 3, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Layer 3: Deep feature extraction
            nn.Conv2d(base_filters * 4, base_filters * 8, 3, padding=1),
            nn.BatchNorm2d(base_filters * 8),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(base_filters * 8, base_filters * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(base_filters * 2, 1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.squeeze()

class ReducedBaselineDeepCNN(nn.Module):
    """Baseline Deep reduced from 12 to 8 layers"""
    def __init__(self, base_filters=32):
        super().__init__()
        self.name = "ReducedBaseline_Deep"
        self.depth = 8
        self.has_skip_connections = False
        self.original_depth = 12
        self.reduction_factor = 0.67
        
        self.features = nn.Sequential(
            # Block 1: Input processing
            nn.Conv2d(3, base_filters, 3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2: Feature extraction
            nn.Conv2d(base_filters, base_filters * 2, 3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 2, base_filters * 2, 3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3: Mid-level features  
            nn.Conv2d(base_filters * 2, base_filters * 4, 3, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 4, base_filters * 4, 3, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4: High-level features
            nn.Conv2d(base_filters * 4, base_filters * 8, 3, padding=1),
            nn.BatchNorm2d(base_filters * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 8, base_filters * 16, 3, padding=1),
            nn.BatchNorm2d(base_filters * 16),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(base_filters * 16, base_filters * 8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(base_filters * 8, 1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.squeeze()

class ReducedResNetBlock(nn.Module):
    """Reduced ResNet block with optional skip connection"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out

class ReducedResNetShallowCNN(nn.Module):
    """ResNet Shallow reduced from 4 to 3 layers"""
    def __init__(self, base_filters=64):
        super().__init__()
        self.name = "ReducedResNet_Shallow"
        self.depth = 3
        self.has_skip_connections = True
        self.original_depth = 4
        self.reduction_factor = 0.75
        
        self.conv1 = nn.Conv2d(3, base_filters, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        # Reduced ResNet layers
        self.layer1 = self._make_layer(base_filters, base_filters, 1, stride=1)
        self.layer2 = self._make_layer(base_filters, base_filters * 2, 1, stride=2) 
        self.layer3 = self._make_layer(base_filters * 2, base_filters * 4, 1, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(base_filters * 4, base_filters),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(base_filters, 1)
        )
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(ReducedResNetBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(ReducedResNetBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = self.classifier(x)
        return x.squeeze()

class ReducedResNetDeepCNN(nn.Module):
    """ResNet Deep reduced from 12 to 8 layers"""
    def __init__(self, base_filters=32):
        super().__init__()
        self.name = "ReducedResNet_Deep"
        self.depth = 8
        self.has_skip_connections = True
        self.original_depth = 12
        self.reduction_factor = 0.67
        
        self.conv1 = nn.Conv2d(3, base_filters, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        # Reduced ResNet layers (strategic layer selection)
        self.layer1 = self._make_layer(base_filters, base_filters, 2, stride=1)
        self.layer2 = self._make_layer(base_filters, base_filters * 2, 2, stride=2)
        self.layer3 = self._make_layer(base_filters * 2, base_filters * 4, 2, stride=2)
        self.layer4 = self._make_layer(base_filters * 4, base_filters * 8, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(base_filters * 8, base_filters * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(base_filters * 4, 1)
        )
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(ReducedResNetBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(ReducedResNetBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = self.classifier(x)
        return x.squeeze()

class ReducedUNetChannelReduced(nn.Module):
    """UNet reduced from 4 to 3 encoder-decoder layers"""
    def __init__(self, base_filters=32):
        super().__init__()
        self.name = f"ReducedUNet_channel_reduced_{base_filters}filters"
        self.depth = 3
        self.has_skip_connections = True
        self.original_depth = 4
        self.reduction_factor = 0.75
        
        # Encoder (3 levels instead of 4)
        self.enc1 = self._conv_block(3, base_filters)
        self.enc2 = self._conv_block(base_filters, base_filters * 2)
        self.enc3 = self._conv_block(base_filters * 2, base_filters * 4)
        
        # Bottleneck
        self.bottleneck = self._conv_block(base_filters * 4, base_filters * 8)
        
        # Decoder (3 levels)
        self.dec3 = self._conv_block(base_filters * 8 + base_filters * 4, base_filters * 4)
        self.dec2 = self._conv_block(base_filters * 4 + base_filters * 2, base_filters * 2) 
        self.dec1 = self._conv_block(base_filters * 2 + base_filters, base_filters)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(base_filters, base_filters // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(base_filters // 2, 1)
        )
    
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc1_pool = F.max_pool2d(enc1, 2)
        
        enc2 = self.enc2(enc1_pool)
        enc2_pool = F.max_pool2d(enc2, 2)
        
        enc3 = self.enc3(enc2_pool)
        enc3_pool = F.max_pool2d(enc3, 2)
        
        # Bottleneck
        bottleneck = self.bottleneck(enc3_pool)
        
        # Decoder with skip connections
        dec3 = F.interpolate(bottleneck, scale_factor=2, mode='bilinear', align_corners=True)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = F.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=True)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=True)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Global classification
        output = self.classifier(dec1)
        return output.squeeze()

class ReducedDenseNetStyleCNN(nn.Module):
    """DenseNet style reduced from 4 to 3 layers"""
    def __init__(self, growth_rate=32):
        super().__init__()
        self.name = "ReducedDenseNet_Style"
        self.depth = 3
        self.has_skip_connections = True
        self.original_depth = 4
        self.reduction_factor = 0.75
        self.growth_rate = growth_rate
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, growth_rate * 2, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(growth_rate * 2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        # Reduced dense blocks (3 instead of 4)
        num_channels = growth_rate * 2
        self.block1 = self._make_dense_block(num_channels, 6)
        num_channels += 6 * growth_rate
        self.trans1 = self._make_transition(num_channels, num_channels // 2)
        num_channels = num_channels // 2
        
        self.block2 = self._make_dense_block(num_channels, 12)
        num_channels += 12 * growth_rate
        self.trans2 = self._make_transition(num_channels, num_channels // 2)
        num_channels = num_channels // 2
        
        self.block3 = self._make_dense_block(num_channels, 6)
        num_channels += 6 * growth_rate
        
        self.bn_final = nn.BatchNorm2d(num_channels)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_channels, 1)
        )
    
    def _make_dense_block(self, in_channels, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(self._make_dense_layer(in_channels + i * self.growth_rate))
        return nn.Sequential(*layers)
    
    def _make_dense_layer(self, in_channels):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4 * self.growth_rate, 1, bias=False),
            nn.BatchNorm2d(4 * self.growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * self.growth_rate, self.growth_rate, 3, padding=1, bias=False)
        )
    
    def _make_transition(self, in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.AvgPool2d(2, 2)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Dense blocks with feature concatenation
        x = self._forward_dense_block(x, self.block1)
        x = self.trans1(x)
        
        x = self._forward_dense_block(x, self.block2)
        x = self.trans2(x)
        
        x = self._forward_dense_block(x, self.block3)
        
        x = self.bn_final(x)
        x = F.relu(x, inplace=True)
        x = self.classifier(x)
        return x.squeeze()
    
    def _forward_dense_block(self, x, block):
        features = [x]
        for layer in block:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)

def get_reduced_layer_architectures():
    """Get all reduced-layer architectures for the study"""
    
    architectures = [
        # Baseline architectures (reduced)
        ReducedBaselineShallowCNN(base_filters=64),
        ReducedBaselineDeepCNN(base_filters=32),
        
        # ResNet architectures (reduced)
        ReducedResNetShallowCNN(base_filters=64),
        ReducedResNetDeepCNN(base_filters=32),
        
        # UNet architectures (reduced)
        ReducedUNetChannelReduced(base_filters=32),
        ReducedUNetChannelReduced(base_filters=36),
        
        # DenseNet architecture (reduced)
        ReducedDenseNetStyleCNN(growth_rate=32),
    ]
    
    return architectures

def print_architecture_comparison():
    """Print comparison between original and reduced architectures"""
    
    print("=" * 80)
    print("REDUCED-LAYER STUDY ARCHITECTURE COMPARISON")
    print("=" * 80)
    
    comparisons = [
        ("Baseline_Shallow", "ReducedBaseline_Shallow", "4 → 3 layers", "Wider filters to compensate"),
        ("Baseline_Deep", "ReducedBaseline_Deep", "12 → 8 layers", "Strategic layer selection"),
        ("ResNet_Shallow", "ReducedResNet_Shallow", "4 → 3 layers", "Maintained skip connections"),
        ("ResNet_Deep", "ReducedResNet_Deep", "12 → 8 layers", "Preserved ResNet blocks"),
        ("UNet_Channel_Reduced", "ReducedUNet_Channel_Reduced", "4 → 3 levels", "Encoder-decoder reduction"),
        ("DenseNet_Style", "ReducedDenseNet_Style", "4 → 3 blocks", "Dense connection preservation")
    ]
    
    print(f"{'Original':<20} {'Reduced':<25} {'Reduction':<15} {'Strategy':<30}")
    print("-" * 95)
    
    for orig, reduced, reduction, strategy in comparisons:
        print(f"{orig:<20} {reduced:<25} {reduction:<15} {strategy:<30}")
    
    print(f"\\n🎯 Study Objective: Test architectural robustness under depth constraints")
    print(f"📊 Metrics: Performance degradation, parameter efficiency, training stability")
    print(f"🔬 Hypothesis: Skip connections provide more graceful degradation")

if __name__ == "__main__":
    print_architecture_comparison()
    
    # Test architecture instantiation
    print(f"\\n" + "=" * 50)
    print("ARCHITECTURE INSTANTIATION TEST")
    print("=" * 50)
    
    architectures = get_reduced_layer_architectures()
    
    for arch in architectures:
        try:
            # Count parameters
            total_params = sum(p.numel() for p in arch.parameters())
            trainable_params = sum(p.numel() for p in arch.parameters() if p.requires_grad)
            
            print(f"✅ {arch.name:30} | Params: {total_params:,} | Reduction: {arch.reduction_factor:.0%}")
        except Exception as e:
            print(f"❌ {arch.name:30} | Error: {e}")
    
    print(f"\\n🎉 All architectures ready for reduced-layer study!")