import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def calculate_eer(labels, predictions):
    """
    Calculate Equal Error Rate (EER).
    
    EER is the point where False Accept Rate (FAR) equals False Rejection Rate (FRR).
    It is a standard metric for voice anti-spoofing and biometric systems.
    
    Lower EER indicates better detection performance.
    
    Args:
        labels: Ground truth labels (0 for bonafide, 1 for spoof)
        predictions: Predicted probabilities for class 1 (spoof)
    
    Returns:
        eer: Equal Error Rate value
        threshold: Threshold at EER point
    """
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    fnr = 1 - tpr
    
    # Find the point where FPR is closest to FNR
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = fpr[eer_idx]
    threshold = thresholds[eer_idx]
    
    return eer, threshold


def compute_metrics(labels, predictions_proba, threshold=None):
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        labels: Ground truth labels
        predictions_proba: Predicted probabilities for class 1
        threshold: Classification threshold (default: 0.5)
    
    Returns:
        Dictionary with all computed metrics
    """
    if threshold is None:
        threshold = 0.5
    
    predictions = (predictions_proba >= threshold).astype(int)
    
    # Calculate EER
    eer, eer_threshold = calculate_eer(labels, predictions_proba)
    
    # Standard metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # FAR and FRR
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    far = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Accept Rate
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Rejection Rate
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'eer': eer,
        'eer_threshold': eer_threshold,
        'far': far,
        'frr': frr,
        'confusion_matrix': cm,
        'threshold': threshold
    }


def plot_confusion_matrix(cm, class_names=None, save_path=None, title='Confusion Matrix'):
    """
    Plot confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix (2x2)
        class_names: List of class names
        save_path: Path to save the figure
        title: Plot title
    
    Returns:
        matplotlib figure
    """
    if class_names is None:
        class_names = ['Bonafide', 'Spoof']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_roc_curve(labels, predictions, save_path=None, title='ROC Curve'):
    """
    Plot ROC curve.
    
    Args:
        labels: Ground truth labels
        predictions: Predicted probabilities
        save_path: Path to save
        title: Plot title
    
    Returns:
        matplotlib figure
    """
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    
    # Calculate AUC
    from sklearn.metrics import auc
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
            label='Random Classifier')
    
    # Mark EER point
    eer, _ = calculate_eer(labels, predictions)
    ax.plot([eer], [1 - eer], 'ro', markersize=10, label=f'EER = {eer:.4f}')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
