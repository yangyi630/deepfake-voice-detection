import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Simple CNN baseline model for voice anti-spoofing.
    3 convolutional layers with batch normalization and adaptive pooling.
    """
    
    def __init__(self, num_classes=2, n_mels=80):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNetBlock(nn.Module):
    """Residual block for ResNet architecture"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    """
    ResNet-18 model adapted for voice anti-spoofing.
    Provides better gradient flow through residual connections.
    Reference: He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
    """
    
    def __init__(self, num_classes=2):
        super(ResNet18, self).__init__()
        
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(ResNetBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class LightCNNBlock(nn.Module):
    """
    LightCNN block with Max-Feature-Map (MFM) activation.
    Reference: Wu et al., "LightCNN: Deep Facial Recognition in the Wild", 2016
    Applied to voice anti-spoofing for efficient feature extraction.
    """
    
    def __init__(self, in_channels, out_channels):
        super(LightCNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 2, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels * 2)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # Max-Feature-Map activation
        channels = x.size(1) // 2
        x = torch.max(x[:, :channels, :, :], x[:, channels:, :, :])
        return x


class LightCNN(nn.Module):
    """
    LightCNN for voice anti-spoofing with MFM activations.
    Key innovation: Max-Feature-Map activation selects more discriminative features,
    which is particularly effective for detecting subtle artifacts in synthetic speech.
    """
    
    def __init__(self, num_classes=2):
        super(LightCNN, self).__init__()
        
        self.features = nn.Sequential(
            LightCNNBlock(1, 32),
            nn.MaxPool2d(2),
            
            LightCNNBlock(32, 64),
            nn.MaxPool2d(2),
            
            LightCNNBlock(64, 128),
            LightCNNBlock(128, 128),
            nn.MaxPool2d(2),
            
            LightCNNBlock(128, 256),
            LightCNNBlock(256, 256),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
