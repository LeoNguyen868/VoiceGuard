"""Evaluation metrics. Score convention: higher = more likely bona fide."""
import numpy as np

# ASVspoof 5 cost model: Cmiss=1, Cfa=10, pi_spoof=0.05 -> normalised DCF = Pmiss + beta * Pfa
ASV5_BETA = (10 * 0.05) / (1 * (1 - 0.05))


def det_curve(bona, spoof):
    """Miss (bona fide rejected) and false-alarm (spoof accepted) rates for every threshold."""
    scores = np.concatenate([bona, spoof])
    labels = np.concatenate([np.ones(len(bona)), np.zeros(len(spoof))])
    order = np.argsort(scores, kind="mergesort")
    labels = labels[order]
    tar_cum = np.cumsum(labels)
    non_cum = len(spoof) - (np.arange(1, len(scores) + 1) - tar_cum)
    frr = np.concatenate([[0], tar_cum / len(bona)])
    far = np.concatenate([[1], non_cum / len(spoof)])
    thr = np.concatenate([[scores[order[0]] - 1e-3], scores[order]])
    return frr, far, thr


def eer(bona, spoof):
    """Equal error rate (fraction) and its threshold, as in the ASVspoof toolkits."""
    frr, far, thr = det_curve(np.asarray(bona), np.asarray(spoof))
    i = np.nanargmin(np.abs(frr - far))
    return float(np.mean([frr[i], far[i]])), float(thr[i])


def min_dcf(bona, spoof, beta=ASV5_BETA):
    frr, far, _ = det_curve(np.asarray(bona), np.asarray(spoof))
    return float(np.min(frr + beta * far))


def bootstrap_eer_ci(bona, spoof, n=1000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    bona, spoof = np.asarray(bona), np.asarray(spoof)
    vals = [eer(rng.choice(bona, len(bona)), rng.choice(spoof, len(spoof)))[0] for _ in range(n)]
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def rates_at(bona, spoof, t):
    """FRR: genuine callers flagged (score < t). FAR: spoofs passed (score >= t)."""
    return float(np.mean(np.asarray(bona) < t)), float(np.mean(np.asarray(spoof) >= t))


def threshold_at_frr(bona, target):
    """Threshold that flags `target` fraction of genuine callers."""
    return float(np.quantile(np.asarray(bona), target))


def summarize(bona, spoof, n_boot=1000):
    e, t_eer = eer(bona, spoof)
    lo, hi = bootstrap_eer_ci(bona, spoof, n=n_boot) if n_boot else (None, None)
    t1 = threshold_at_frr(bona, 0.01)
    frr1, far1 = rates_at(bona, spoof, t1)
    return dict(n_bonafide=len(bona), n_spoof=len(spoof), eer=e, eer_ci95=[lo, hi],
                min_dcf_asv5=min_dcf(bona, spoof), t_eer=t_eer,
                t_frr1=t1, far_at_frr1=far1, frr_at_frr1=frr1)
