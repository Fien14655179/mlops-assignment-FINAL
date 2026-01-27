import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os


def log_confusion_matrix(y_true, y_pred, class_names, step=None):
    """Compute and log confusion matrix as image artefact."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")

    os.makedirs("artifacts", exist_ok=True)
    fig_path = f"artifacts/confusion_matrix_step{step or 'final'}.png"
    plt.savefig(fig_path)
    plt.close()

    mlflow.log_artifact(fig_path)
