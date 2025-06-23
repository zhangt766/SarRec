
import pandas as pd
import numpy as np
import argparse
import torch

def calibrate_lambda_from_csv(csv_path, alpha=0.05, epsilon=0.01, lambda_grid=None):
    df = pd.read_csv(csv_path)
    df['probs'] = df['probs'].apply(eval)
    df['cans'] = df['cans'].apply(eval)

    if lambda_grid is None:
        lambda_grid = np.linspace(1.0, 0.0, 200)

    for lam in lambda_grid:
        hits = []
        for _, row in df.iterrows():
            real = row['real'].strip().lower()
            probs = row['probs']
            cans = [c.strip().lower() for c in row['cans']]
            pred_set = [item for item, prob in zip(cans, probs) if prob >= lam]
            hits.append(int(real in pred_set))

        coverage = np.mean(hits)
        risk = 1 - coverage

        if risk <= alpha - epsilon:
            print(f"✅ lambda*: {lam:.3f} | coverage={coverage:.3f}, risk={risk:.3f}")
            return lam

    print("⚠️ No lambda found satisfying coverage. Returning lowest.")
    return lambda_grid[-1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to valid_scored.csv")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--epsilon", type=float, default=0.01)
    args = parser.parse_args()

    lambda_star = calibrate_lambda_from_csv(args.csv, alpha=args.alpha, epsilon=args.epsilon)
    torch.save({"tau": lambda_star}, "checkpoints/trained_lambda.pth")
    print(f"✅ Saved lambda* to checkpoints/trained_lambda.pth")
