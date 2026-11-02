from .cnn import SimpleCNN, ResNet18, LightCNN
from .conformer import ConformerModel


def list_models():
    """
    Return list of available model names with descriptions.
    
    Available models:
    - simple_cnn: Baseline CNN with 3 convolutional layers
    - resnet18: ResNet-18 with residual connections
    - lightcnn: LightCNN with Max-Feature-Map activation
    - conformer: Conformer (Transformer + CNN) for long-range dependencies
    """
    return {
        'simple_cnn': {
            'name': 'Simple CNN',
            'description': 'Baseline 3-layer CNN. Fast training, good for simple cases.',
            'params': '~0.5M'
        },
        'resnet18': {
            'name': 'ResNet-18',
            'description': 'Deep residual network with 18 layers. Better feature hierarchy.',
            'params': '~11.2M'
        },
        'lightcnn': {
            'name': 'LightCNN',
            'description': 'LightCNN with Max-Feature-Map activation. Selects discriminative features.',
            'params': '~2.1M'
        },
        'conformer': {
            'name': 'Conformer',
            'description': 'Convolution-augmented Transformer. Captures long-range dependencies.',
            'params': '~1.5M'
        }
    }


def get_model(model_name='resnet18', num_classes=2, **kwargs):
    """
    Factory function to create model by name.
    
    Args:
        model_name: Name of the model to create
        num_classes: Number of output classes (default: 2 for bonafide/spoof)
        **kwargs: Additional model-specific parameters
    
    Returns:
        PyTorch model instance
    """
    model_name = model_name.lower()
    
    if model_name == 'simple_cnn':
        return SimpleCNN(num_classes=num_classes)
    
    elif model_name == 'resnet18':
        return ResNet18(num_classes=num_classes)
    
    elif model_name == 'lightcnn':
        return LightCNN(num_classes=num_classes)
    
    elif model_name == 'conformer':
        return ConformerModel(
            num_classes=num_classes,
            n_mels=kwargs.get('n_mels', 80),
            d_model=kwargs.get('d_model', 144),
            nhead=kwargs.get('nhead', 4),
            num_layers=kwargs.get('num_layers', 6),
            dim_feedforward=kwargs.get('dim_feedforward', 576)
        )
    
    else:
        available = list(list_models().keys())
        raise ValueError(f"Unknown model: {model_name}. Available models: {available}")


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
