
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ConfusionMatrices:
    
    @staticmethod
    def plot_confusion_matrices(y_true, y_pred, labels):
        for i, label in enumerate(labels):
            cm = confusion_matrix(y_true[:, i], y_pred[:, i])
            plt.figure(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {label}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            plt.show()
