import argparse
import numpy as np
import joblib
import pandas as pd

from sklearn.datasets import load_digits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="artifacts/model.joblib")
    parser.add_argument("--test_indices_path", default="artifacts/test_indices.npy")
    parser.add_argument("--out_csv", default="artifacts/sample_predictions.csv")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    # Load model + test set indices
    model = joblib.load(args.model_path)
    test_indices = np.load(args.test_indices_path)

    digits = load_digits()
    X = digits.data
    y = digits.target

    rng = np.random.RandomState(args.seed)
    pick = rng.choice(test_indices, size=min(args.n, len(test_indices)), replace=False)

    X_sel = X[pick]
    y_true = y[pick]
    y_pred = model.predict(X_sel)

    df = pd.DataFrame({
        "index": pick,
        "true": y_true,
        "pred": y_pred,
        "correct": (y_true == y_pred)
    })

    df.to_csv(args.out_csv, index=False)
    print(df)
    print(f"\nSaved to {args.out_csv}")

if __name__ == "__main__":
    main()
