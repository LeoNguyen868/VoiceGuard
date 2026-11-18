"""Risk policy: maps a detector score to a risk level and an action for the operator.

Score = bona fide logit (higher means more likely genuine), as in the authors' code.
Thresholds come from a calibration file produced by python -m voiceguard.evaluate on labelled
data. They are dataset- and channel-specific: a threshold calibrated on clean
In-the-Wild audio is not valid for G.711 calls without re-calibration.
"""
import json
import math
from dataclasses import asdict, dataclass, field

LEVELS = {
    "LOW": "Tiếp tục quy trình bình thường",
    "MEDIUM": "Xác thực bổ sung: gọi lại số đã đăng ký hoặc dùng kênh xác thực thứ hai",
    "HIGH": "Tạm dừng giao dịch, chuyển bộ phận chống gian lận xem xét",
}


@dataclass
class Policy:
    t_high: float  # score < t_high -> HIGH (FRR 1%: few genuine callers are stopped)
    t_medium: float  # t_high <= score < t_medium -> MEDIUM (FRR 7%: extra verification budget)
    calibrated: bool = False
    source: str = "mặc định, chưa hiệu chỉnh"
    # Platt scaling P(spoof | score) = 1 / (1 + exp(a * score + b)), fitted per channel with equal priors
    # (python -m voiceguard.calibrate). Empty: the percentage is estimated from the two thresholds.
    platt: dict = field(default_factory=dict)

    def decide(self, score):
        if score < self.t_high:
            level = "HIGH"
        elif score < self.t_medium:
            level = "MEDIUM"
        else:
            level = "LOW"
        return level, LEVELS[level]

    def spoof_percent(self, score):
        """(Percentage that the audio is spoofed, method). Calibrated when Platt parameters are known;
        otherwise a piecewise estimate anchored at the thresholds: t_medium -> 50%, t_high -> 90%."""
        if self.platt:
            z = self.platt["a"] * score + self.platt["b"]
            return 100.0 / (1.0 + math.exp(max(-60.0, min(60.0, z)))), "platt"
        span = max(self.t_medium - self.t_high, 1e-6)
        if score >= self.t_medium:
            p = 50.0 - 40.0 * (score - self.t_medium) / span      # one span above t_medium -> 10%
        elif score >= self.t_high:
            p = 50.0 + 40.0 * (self.t_medium - score) / span      # t_high -> 90%
        else:
            p = 90.0 + 9.0 * min(1.0, (self.t_high - score) / span)
        return max(1.0, min(99.0, p)), "thresholds"

    def to_dict(self):
        return asdict(self)


def at_least_medium(level):
    """Pre-filter escalation for unknown or hidden numbers: LOW becomes MEDIUM."""
    level = "MEDIUM" if level == "LOW" else level
    return level, LEVELS[level]


def load_policy(path=None):
    """Read thresholds from a calibration JSON; fall back to uncalibrated defaults."""
    if not path:
        return Policy(t_high=-2.0, t_medium=0.0)
    with open(path, encoding="utf-8") as f:
        cal = json.load(f)
    return Policy(t_high=cal["t_high"], t_medium=cal["t_medium"], calibrated=True,
                  source=cal.get("source", path), platt=cal.get("platt") or {})
