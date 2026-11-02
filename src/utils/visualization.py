import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


def plot_training_history(history, save_path=None):
    """
    Plot training history curves (loss and accuracy).
    
    Args:
        history: Dictionary with 'loss', 'val_loss', 'acc', 'val_acc' keys
        save_path: Path to save figure
    
    Returns:
        matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_mel_spectrogram(mel_spec, sr=16000, hop_length=256, save_path=None, 
                          title='Log-Mel Spectrogram'):
    """
    Plot log-mel spectrogram.
    
    Args:
        mel_spec: Log-mel spectrogram (2D array)
        sr: Sample rate
        hop_length: Hop length
        save_path: Path to save
        title: Plot title
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    img = librosa.display.specshow(mel_spec, sr=sr, hop_length=hop_length,
                                   x_axis='time', y_axis='mel', ax=ax)
    
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_waveform(waveform, sr=16000, save_path=None, title='Waveform'):
    """
    Plot audio waveform.
    
    Args:
        waveform: Audio waveform
        sr: Sample rate
        save_path: Path to save
        title: Plot title
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    
    time = np.arange(len(waveform)) / sr
    ax.plot(time, waveform, linewidth=0.5)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_f0_contour(f0, time_axis=None, save_path=None, title='F0 Contour'):
    """
    Plot fundamental frequency (F0) contour.
    
    Args:
        f0: F0 values
        time_axis: Time values (optional)
        save_path: Path to save
        title: Plot title
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    if time_axis is None:
        time_axis = np.arange(len(f0)) * 0.01  # Assume 10ms frame shift
    
    # Only plot non-zero values
    voiced = f0 > 0
    ax.plot(time_axis[voiced], f0[voiced], 'b-', linewidth=1, label='F0')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Frequency (Hz)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_feature_comparison(bonafide_features, spoof_features, feature_name='Feature',
                            save_path=None, title='Feature Comparison'):
    """
    Compare features between bonafide and spoof samples.
    
    Args:
        bonafide_features: Features from bonafide samples
        spoof_features: Features from spoof samples
        feature_name: Name of the feature
        save_path: Path to save
        title: Plot title
    
    Returns:
        matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot example features
    if bonafide_features.ndim == 2:
        img1 = ax1.imshow(bonafide_features, aspect='auto', origin='lower', cmap='viridis')
        ax1.set_title(f'Bonafide {feature_name}', fontsize=12, fontweight='bold')
        fig.colorbar(img1, ax=ax1)
    else:
        ax1.plot(bonafide_features)
        ax1.set_title(f'Bonafide {feature_name}', fontsize=12, fontweight='bold')
    
    if spoof_features.ndim == 2:
        img2 = ax2.imshow(spoof_features, aspect='auto', origin='lower', cmap='viridis')
        ax2.set_title(f'Spoof {feature_name}', fontsize=12, fontweight='bold')
        fig.colorbar(img2, ax=ax2)
    else:
        ax2.plot(spoof_features)
        ax2.set_title(f'Spoof {feature_name}', fontsize=12, fontweight='bold')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_model_comparison(results_dict, metric='eer', save_path=None, 
                          title='Model Performance Comparison'):
    """
    Compare performance across different models.
    
    Args:
        results_dict: Dictionary with model names as keys and metric dicts as values
        metric: Metric to compare ('eer', 'accuracy', 'f1_score', etc.)
        save_path: Path to save
        title: Plot title
    
    Returns:
        matplotlib figure
    """
    models = list(results_dict.keys())
    values = [results_dict[m][metric] for m in models]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(models, values, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'], alpha=0.8)
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    metric_labels = {
        'eer': 'Equal Error Rate (EER)',
        'accuracy': 'Accuracy',
        'f1_score': 'F1 Score',
        'precision': 'Precision',
        'recall': 'Recall'
    }
    
    ax.set_ylabel(metric_labels.get(metric, metric), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
