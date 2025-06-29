"""
PyTorch MobileNet architecture for wildlife footprint classification
Matches the TensorFlow version with custom classifier head
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class WildlifeMobileNet(nn.Module):
    """
    PyTorch MobileNet architecture matching the TensorFlow version:
    - MobileNetV3Small backbone (ImageNet pretrained)
    - Custom classifier head with same architecture as TensorFlow version
    """
    
    def __init__(self, num_classes: int = 13, dropout_rate: float = 0.3, pretrained: bool = True):
        super(WildlifeMobileNet, self).__init__()
        
        # Load MobileNetV3Small backbone (equivalent to TensorFlow version)
        if pretrained:
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        else:
            weights = None
            
        self.backbone = models.mobilenet_v3_small(weights=weights)
        
        # Remove the original classifier
        self.features = self.backbone.features
        self.avgpool = self.backbone.avgpool
        
        # Get the number of features from backbone
        # MobileNetV3Small outputs 576 features after avgpool
        backbone_features = 576
        
        # Custom classifier head matching TensorFlow architecture:
        # GlobalAveragePooling2D -> Dense(512) -> BatchNorm -> Dropout -> Dense(256) -> Dropout -> Dense(num_classes)
        self.classifier = nn.Sequential(
            # First dense layer: 576 -> 512 (equivalent to Dense(512) in TF)
            nn.Linear(backbone_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),  # BatchNorm after first dense (like TF version)
            nn.Dropout(dropout_rate),
            
            # Second dense layer: 512 -> 256 (equivalent to Dense(256) in TF)
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            # Output layer: 256 -> num_classes (equivalent to Dense(18) in TF)
            nn.Linear(256, num_classes)
            # Note: No softmax here - will be applied in loss function or during inference
        )
        
        # Initialize weights for custom classifier
        self._initialize_classifier_weights()
        
    def _initialize_classifier_weights(self):
        """Initialize classifier weights (similar to Keras default initialization)"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                # Xavier/Glorot uniform initialization (Keras default)
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        # Extract features using MobileNet backbone
        x = self.features(x)
        
        # Global average pooling (equivalent to GlobalAveragePooling2D in TF)
        x = self.avgpool(x)
        
        # Flatten for classifier
        x = torch.flatten(x, 1)
        
        # Pass through custom classifier
        x = self.classifier(x)
        
        return x
    
    def freeze_backbone(self):
        """Freeze backbone parameters for transfer learning (Stage 1 training)"""
        for param in self.features.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self, unfreeze_last_n_layers: int = 20):
        """
        Unfreeze last N layers of backbone for fine-tuning (Stage 2 training)
        Similar to the TensorFlow version which unfreezes last 20 layers
        """
        # First, freeze all backbone parameters
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Then unfreeze the last N layers
        total_layers = len(list(self.features.modules()))
        layers_to_unfreeze = list(self.features.modules())[-unfreeze_last_n_layers:]
        
        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True
    
    def get_trainable_parameters(self):
        """Get only trainable parameters (useful for optimizer)"""
        return filter(lambda p: p.requires_grad, self.parameters())
    
    def print_trainable_parameters(self):
        """Print information about trainable parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {total_params - trainable_params:,}")


def create_wildlife_mobilenet(num_classes: int = 13, pretrained: bool = True) -> WildlifeMobileNet:
    """
    Factory function to create WildlifeMobileNet model
    
    Args:
        num_classes: Number of wildlife species classes
        pretrained: Whether to use ImageNet pretrained weights
    
    Returns:
        WildlifeMobileNet model
    """
    model = WildlifeMobileNet(num_classes=num_classes, pretrained=pretrained)
    return model