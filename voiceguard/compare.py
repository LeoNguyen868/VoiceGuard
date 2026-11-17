"""EER difference between two detectors scored on the same clips, with a paired bootstrap 95% CI.

Each replicate resamples bona fide and spoof clips separately (stratified) and recomputes both EERs on the
same resample, so the CI of the difference is much tighter than the overlap of two independent CIs.
Delta is in percentage points, negative = model better than reference.

  python -m voiceguard.compare --labels $ITW/meta.csv --ref results/scores/itw_base_opus12_4s.txt \\
      --model results/scores/itw_L14_opus12_4s.txt
"""
import argparse
import json

import numpy as np

from .datasets import read_labels


def fast_eer(bona, spoof):
    """EER at the crossing of FRR and FAR over all thresholds; matches voiceguard.metrics.eer."""
    s = np.concatenate([bona, spoof])
    y = np.concatenate([np.ones(len(bona)), np.zeros(len(spoof))])[np.argsort(s, kind="mergesort")]
    frr = np.cumsum(y) / len(bona)
    far = 1 - np.cumsum(1 - y) / len(spoof)
    i = np.argmin(np.abs(frr - far))
    return (frr[i] + far[i]) / 2


def read_scores(path):
    with open(path) as f:
        return {k: float(v) for k, v in (line.split() for line in f)}


def paired_delta(labels, ref, model, n_boot=1000, seed=2026):
    names = sorted(set(ref) & set(model) & set(labels))
    if len(names) < len(set(ref) | set(model)):
        print(f"warning: {len(set(ref) | set(model)) - len(names)} clips are not in both score files; using {len(names)}")
    y = np.array([labels[n] for n in names])
    a = np.array([model[n] for n in names])
    b = np.array([ref[n] for n in names])
    bi, si = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    e_m, e_r = fast_eer(a[bi], a[si]), fast_eer(b[bi], b[si])
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for k in range(n_boot):
        rb, rs = rng.choice(bi, len(bi)), rng.choice(si, len(si))
        d[k] = fast_eer(a[rb], a[rs]) - fast_eer(b[rb], b[rs])
    lo, hi = np.percentile(d, [2.5, 97.5])
    delta = 100 * (e_m - e_r)
    return dict(n_clips=len(names), eer_ref=100 * e_r, eer_model=100 * e_m, delta=delta,
                ci95=[100 * lo, 100 * hi], clear=bool(abs(delta) >= 1 and (lo > 0 or hi < 0)),
                n_boot=n_boot, seed=seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--ref", required=True, help="score file of the reference (e.g. the baseline)")
    ap.add_argument("--model", required=True, help="score file of the model compared with it")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="", help="optional JSON output")
    args = ap.parse_args()
    r = paired_delta(read_labels(args.labels), read_scores(args.ref), read_scores(args.model), args.n_boot, args.seed)
    print(f"{r['n_clips']} clips: EER ref {r['eer_ref']:.2f}%, model {r['eer_model']:.2f}%, "
          f"delta {r['delta']:+.2f} points (95% CI {r['ci95'][0]:+.2f} to {r['ci95'][1]:+.2f})"
          f"{', clear difference' if r['clear'] else ''}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(r, ref=args.ref, model=args.model), f, indent=2)


if __name__ == "__main__":
    main()
