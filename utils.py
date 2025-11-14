import os
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def standardize(X: np.ndarray):
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    return (X - mean) / std, mean, std

def train_test_split_np(X, y, test_size=0.3, random_state=0):
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    idx = rng.permutation(n)
    n_test = int(round(test_size * n))
    te, tr = idx[:n_test], idx[n_test:]
    return X[tr], X[te], y[tr], y[te]

def plot_accuracy(train_acc, test_acc, outpath: str):
    epochs = np.arange(1, len(train_acc)+1)
    plt.figure()
    plt.plot(epochs, train_acc, marker="o", label="train")
    if test_acc:
        plt.plot(epochs, test_acc, marker="o", label="test")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.ylim(0.0, 1.05)
    plt.title("Accuracy per Epoch"); plt.legend(); plt.tight_layout()
    plt.savefig(outpath); plt.close()
    return outpath

def plot_loss(loss_vals, outpath: str):
    epochs = np.arange(1, len(loss_vals)+1)
    plt.figure()
    plt.plot(epochs, loss_vals, marker="o")
    plt.xlabel("Epoch"); plt.ylabel("Perceptron hinge loss")
    plt.title("Loss per Epoch"); plt.tight_layout()
    plt.savefig(outpath); plt.close()
    return outpath

def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, K: int):
    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        cm[t, p] += 1
    return cm

def plot_confusion_matrix(cm: np.ndarray, outpath: str, class_names=None):
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    if class_names is not None:
        plt.xticks(np.arange(len(class_names)), class_names, rotation=45, ha="right")
        plt.yticks(np.arange(len(class_names)), class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout(); plt.savefig(outpath); plt.close()
    return outpath