import os
import argparse
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)

def save_confusion_matrix(y_true, y_pred, path):
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    disp.ax_.set_title("Confusion Matrix (Test)")
    disp.figure_.tight_layout()
    disp.figure_.savefig(path, dpi=150)

def save_sample_grid(images, y_true, y_pred, indices, path, cols=5):
    # show a grid of sample predictions from test set
    n = min(len(indices), 20)
    idxs = indices[:n]
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.array(axes).reshape(-1)

    for ax_i, idx in enumerate(idxs):
        ax = axes[ax_i]
        ax.imshow(images[idx], cmap="gray")
        title = f"pred {y_pred[ax_i]} / true {y_true[ax_i]}"
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    # hide any leftover axes
    for j in range(len(idxs), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Sample Predictions (Test)", y=0.99)
    fig.tight_layout()
    fig.savefig(path, dpi=150)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts_dir", default="artifacts")
    parser.add_argument("--model_path", default="artifacts/model.joblib")
    parser.add_argument("--cm_path", default="artifacts/confusion_matrix.png")
    parser.add_argument("--samples_path", default="artifacts/samples.png")
    parser.add_argument("--metrics_path", default="artifacts/metrics.txt")
    parser.add_argument("--test_indices_path", default="artifacts/test_indices.npy")
    parser.add_argument("--train_indices_path", default="artifacts/train_indices.npy")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--C", type=float, default=10.0)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.artifacts_dir, exist_ok=True)

    # 1) Load data
    digits = load_digits()
    X = digits.data       # shape (1797, 64)
    y = digits.target     # shape (1797,)
    images = digits.images

    # 2) Train/test split (also keep indices so we can reuse the exact test set later)
    indices = np.arange(len(y))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y
    )

    # 3) Build model pipeline (scale + SVC)
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=args.C, gamma="scale")
    )

    # 4) Train
    model.fit(X_train, y_train)

    # 5) Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    # 6) Save artifacts
    joblib.dump(model, args.model_path)
    np.save(args.test_indices_path, idx_test)
    np.save(args.train_indices_path, idx_train)

    with open(args.metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)

    save_confusion_matrix(y_test, y_pred, args.cm_path)

    # prepare a sample grid (use first 20 samples from test set order)
    # we recompute predictions for those same samples to label the grid
    n = min(20, len(idx_test))
    X_sample = X_test[:n]
    y_sample = y_test[:n]
    y_sample_pred = model.predict(X_sample)
    save_sample_grid(images, y_sample, y_sample_pred, idx_test[:n], args.samples_path)

    print(f"✅ Done! Accuracy: {acc:.4f}")
    print(f"Artifacts saved in: {os.path.abspath(args.artifacts_dir)}")

if __name__ == "__main__":
    main()
