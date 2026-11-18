"""Turn detector scores into a percentage: Platt scaling per channel.

The bona fide logit is not a probability: a softmax over the two logits is close to 0 or 1 for almost
every clip. This fits P(spoof | score) = 1 / (1 + exp(a * score + b)) on labelled scores of one channel,
weighting both classes equally (as if a call were equally likely to be genuine or cloned), and adds
the parameters to that channel's policy file. The two thresholds in the file are not changed.

  python -m voiceguard.calibrate --scores results/scores/itw_L14_opus12_4s.txt --labels $ITW/meta.csv \\
      --policy configs/policy/L14_opus12_4s.json
"""
import argparse
import json

import numpy as np

from .compare import read_scores
from .datasets import read_labels


def fit_platt(scores, spoof, iters=100):
    """Weighted logistic regression of spoof (0/1) on score; returns (a, b) of 1 / (1 + exp(a s + b))."""
    s = np.asarray(scores, float)
    y = np.asarray(spoof, float)
    w = np.where(y == 1, 0.5 / max(y.sum(), 1), 0.5 / max((1 - y).sum(), 1))
    mu, sd = s.mean(), s.std() or 1.0
    x = (s - mu) / sd
    beta = np.zeros(2)                                   # logit of spoof = beta0 + beta1 * x
    X = np.stack([np.ones_like(x), x], 1)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ beta)))
        g = X.T @ (w * (y - p))
        H = (X * (w * p * (1 - p))[:, None]).T @ X + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, g)
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    # logit of spoof = beta0 + beta1 (s - mu) / sd = -(a s + b)
    a = -beta[1] / sd
    b = -(beta[0] - beta[1] * mu / sd)
    return float(a), float(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True, help="'name score' file of one model on one channel")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--policy", required=True, help="policy JSON of the same model and channel, updated in place")
    args = ap.parse_args()
    labels, scores = read_labels(args.labels), read_scores(args.scores)
    names = [n for n in scores if n in labels]
    s = np.array([scores[n] for n in names])
    y = np.array([1 - labels[n] for n in names])          # 1 = spoof
    a, b = fit_platt(s, y)
    with open(args.policy, encoding="utf-8") as f:
        pol = json.load(f)
    pct = lambda v: 100 / (1 + np.exp(a * v + b))         # noqa: E731
    pol["platt"] = dict(a=round(a, 6), b=round(b, 6), prior="0.5", n=len(names), scores=args.scores)
    with open(args.policy, "w", encoding="utf-8") as f:
        json.dump(pol, f, ensure_ascii=False, indent=2)
    print(f"{args.policy}: a={a:.4f} b={b:.4f} on {len(names)} clips; "
          f"P(spoof) at t_medium {pct(pol['t_medium']):.1f}%, at t_high {pct(pol['t_high']):.1f}%")


if __name__ == "__main__":
    main()
