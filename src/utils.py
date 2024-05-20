import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from logistic_regression import sigmoid

def plot_classification_report(X: np.array, theta: np.array, y_true: np.array, target_names:list, train_test:str, show_plot:bool=False):

    if len(target_names) < 3:
        z = np.dot(X, theta)
        h = sigmoid(z)
        y_pred = [0 if i < 0.5 else 1 for i in h]

    report = classification_report(y_true, y_pred=y_pred) #, target_names=target_names
    print(report)

    cm = confusion_matrix(y_true, y_pred=y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f"../plots/confusion_matrix_{train_test}.png")
    if show_plot:
        plt.show()
    plt.close()

def plot_loss_curve(cost_history:list[float], show_plot:bool=False):
    """
    Plot the loss curve.
    """
    plt.plot(cost_history)
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.title('Loss Curve')
    if show_plot:
        plt.show()
    plt.savefig('../plots/loss_curve.png')
    plt.close()

