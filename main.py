import argparse
import numpy as np
from sklearn.datasets import load_digits  # sklearn only for data
from model import MulticlassPerceptron
from utils import (
    ensure_dir, standardize, train_test_split_np,
    plot_accuracy, plot_loss, confusion_matrix_np, plot_confusion_matrix
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ensure_dir(args.outdir)

    # Load real dataset (10-way classification)
    data = load_digits()
    X, y = data["data"], data["target"]   # X: (n, 64), y: (n,)
    class_names = [str(i) for i in range(10)]

    # Standardize & split
    Xz, _, _ = standardize(X)
    Xtr, Xte, ytr, yte = train_test_split_np(Xz, y, test_size=0.3, random_state=args.seed)

    # Train
    clf = MulticlassPerceptron(lr=args.lr, n_epochs=args.epochs, shuffle=True, random_state=args.seed)
    clf.fit(Xtr, ytr, eval_set=(Xte, yte))

    # Final metrics
    train_acc = clf.history_["train_acc"][-1]
    test_acc  = clf.history_["test_acc"][-1]
    print(f"Final Train acc: {train_acc:.4f} | Final Test acc: {test_acc:.4f}")

    # Plots (only what you asked for)
    plot_accuracy(clf.history_["train_acc"], clf.history_["test_acc"], f"{args.outdir}/accuracy_per_epoch.png")
    plot_loss(clf.history_["loss"], f"{args.outdir}/loss_per_epoch.png")

    # Confusion matrix on test
    y_pred = clf.predict(Xte)
    cm = confusion_matrix_np(yte, y_pred, K=10)
    np.savetxt(f"{args.outdir}/confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    plot_confusion_matrix(cm, f"{args.outdir}/confusion_matrix.png", class_names=class_names)
    print(f"Saved to {args.outdir}")

if __name__ == "__main__":
    main()