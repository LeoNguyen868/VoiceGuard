"""Detector: waveform -> per-window scores -> aggregated score -> risk decision."""
import time

import numpy as np
import torch

from .audio import SR, repeat_pad, rms_dbfs, sliding_windows
from .policy import Policy

MIN_SECONDS = 0.5
MIN_RMS_DBFS = -60.0


class InsufficientAudio(ValueError):
    pass


class Detector:
    def __init__(self, model, policy=None, device="cpu", fp16=False,
                 window_s=4.0, hop_s=2.0, batch_size=8, cuda_graph=False):
        self.model = model.eval()
        self.policy = policy or Policy(t_high=-2.0, t_medium=0.0)
        self.device = device
        self.fp16 = fp16 and str(device).startswith("cuda")
        self.window_s, self.hop_s, self.batch_size = window_s, hop_s, batch_size
        self.win = int(window_s * SR)
        self.graphs = None
        if cuda_graph:
            from .graphs import GraphRunner
            self.graphs = GraphRunner(self.model, device, self.win, fp16=self.fp16)

    @torch.no_grad()
    def logits(self, clips):
        """clips: list of equal-length float32 arrays -> N x 2 logits (float32)."""
        x = torch.from_numpy(np.stack(clips))
        if self.graphs is not None and x.shape[1] == self.win:
            return self.graphs(x).cpu().numpy()
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
            out = self.model(x.to(self.device))
        return out.float().cpu().numpy()

    def windows(self, wav):
        """Window spans and the clips the model sees (short audio tiled to one window)."""
        spans = sliding_windows(len(wav), self.window_s, self.hop_s)
        return spans, [repeat_pad(wav[s:e], self.win) for s, e in spans]

    def check(self, wav):
        dur = len(wav) / SR
        if dur < MIN_SECONDS:
            raise InsufficientAudio(f"âm thanh quá ngắn ({dur:.2f} s < {MIN_SECONDS} s)")
        level = rms_dbfs(wav)
        if level < MIN_RMS_DBFS:
            raise InsufficientAudio(f"tín hiệu quá nhỏ ({level:.1f} dBFS)")

    def analyze(self, wav, mode="window", policy=None):
        """mode='window': sliding windows, mean of window scores (call screening).
        mode='full': one pass over the whole clip (the benchmark protocol)."""
        self.check(wav)
        t0 = time.perf_counter()
        if mode == "full":
            spans = [(0, len(wav))]
            logit_rows = self.logits([wav])
        else:
            # A clip shorter than one window is tiled to the window length, exactly like
            # eval_itw.py --duration 4, so calibrated thresholds apply to the same quantity.
            spans, clips = self.windows(wav)
            logit_rows = np.concatenate([self.logits(clips[i:i + self.batch_size])
                                         for i in range(0, len(clips), self.batch_size)])
        model_ms = (time.perf_counter() - t0) * 1000
        return self.result(wav, mode, spans, logit_rows, model_ms, policy)

    def result(self, wav, mode, spans, logit_rows, model_ms, policy=None):
        """Per-window scores, mean score, risk level. Shared by analyze() and the batched API path."""
        scores = logit_rows[:, 1]
        p_bona = _softmax(logit_rows)[:, 1]
        segments = [dict(start=round(s / SR, 2), end=round(e / SR, 2), score=round(float(sc), 4),
                         p_bonafide=round(float(p), 4))
                    for (s, e), sc, p in zip(spans, scores, p_bona)]
        score = float(np.mean(scores))
        policy = policy or self.policy
        level, action = policy.decide(score)
        worst = int(np.argmin(scores))
        return dict(
            duration_s=round(len(wav) / SR, 2), mode=mode, n_segments=len(segments),
            score=round(score, 4), p_bonafide=round(float(np.mean(p_bona)), 4),
            risk_level=level, recommended_action=action,
            most_suspicious_segment=segments[worst], segments=segments,
            policy=policy.to_dict(),
            timing_ms=dict(model=round(model_ms, 1),
                           rtf=round(model_ms / 1000 / (len(wav) / SR), 4)),
        )


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
