# ============================================================================
# evaluator.py - Metrics and Evaluation
# ============================================================================

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import matplotlib.pyplot as plt

def compute_auc(predictions, labels, average='macro'):
    """Compute multi-label AUC"""
    # Handle cases with only one class present
    valid_classes = []
    aucs = []
    
    for i in range(labels.shape[1]):
        if len(np.unique(labels[:, i])) > 1:
            auc = roc_auc_score(labels[:, i], predictions[:, i])
            aucs.append(auc)
            valid_classes.append(i)
    
    if len(aucs) == 0:
        return 0.0
    
    return np.mean(aucs) if average == 'macro' else aucs


def compute_metrics(predictions, labels, threshold=0.5):
    """Compute comprehensive metrics"""
    # AUC-ROC
    auc_roc = compute_auc(predictions, labels, average='macro')
    
    # Average Precision
    ap_scores = []
    for i in range(labels.shape[1]):
        if len(np.unique(labels[:, i])) > 1:
            ap = average_precision_score(labels[:, i], predictions[:, i])
            ap_scores.append(ap)
    mean_ap = np.mean(ap_scores) if ap_scores else 0.0
    
    # F1 Score
    binary_preds = (predictions > threshold).astype(int)
    f1_scores = []
    for i in range(labels.shape[1]):
        if len(np.unique(labels[:, i])) > 1:
            f1 = f1_score(labels[:, i], binary_preds[:, i])
            f1_scores.append(f1)
    mean_f1 = np.mean(f1_scores) if f1_scores else 0.0
    
    return {
        'auc_roc': auc_roc,
        'mean_ap': mean_ap,
        'mean_f1': mean_f1,
        'per_class_auc': aucs if 'aucs' in locals() else []
    }


def plot_training_curves(train_losses, val_losses, val_aucs, save_path=None):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # AUC curve
    axes[1].plot(val_aucs, label='Val AUC', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC')
    axes[1].set_title('Validation AUC')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
