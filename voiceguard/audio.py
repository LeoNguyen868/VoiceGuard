"""Audio decoding and windowing. Everything is 16 kHz mono float32, not normalised,
which is the input the Nes2Net-X checkpoint was trained and evaluated with."""
import subprocess

import librosa
import numpy as np

SR = 16000


def load_audio(path):
    """Decode a file to 16 kHz mono float32. Same call as the upstream eval.py."""
    try:
        wav, _ = librosa.load(path, sr=SR, mono=True)
    except Exception:
        wav = decode_with_ffmpeg(path)
    return wav.astype(np.float32)


def decode_with_ffmpeg(path_or_bytes, in_args=()):
    """Decode a path or raw bytes (any format ffmpeg reads) to 16 kHz mono float32."""
    data = None
    src = path_or_bytes
    if isinstance(path_or_bytes, (bytes, bytearray)):
        data, src = bytes(path_or_bytes), "pipe:0"
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", *in_args, "-i", src,
           "-f", "f32le", "-ac", "1", "-ar", str(SR), "pipe:1"]
    out = subprocess.run(cmd, input=data, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def repeat_pad(x, n):
    """Crop to n samples, or tile a short clip up to n samples (upstream pad())."""
    if len(x) >= n:
        return x[:n]
    reps = int(n / len(x)) + 1
    return np.tile(x, reps)[:n]


def crop_duration(x, seconds):
    """'full' keeps the whole clip; a number keeps the first `seconds` (tiled if shorter)."""
    if seconds in (None, "full"):
        return x
    return repeat_pad(x, int(round(float(seconds) * SR)))


def sliding_windows(n_samples, window_s, hop_s):
    """Start/end sample indices. A clip shorter than one window is one segment."""
    win, hop = int(window_s * SR), int(hop_s * SR)
    if n_samples <= win:
        return [(0, n_samples)]
    starts = list(range(0, n_samples - win + 1, hop))
    if starts[-1] + win < n_samples:  # cover the tail
        starts.append(n_samples - win)
    return [(s, s + win) for s in starts]


def rms_dbfs(x):
    return 20 * np.log10(np.sqrt(np.mean(np.square(x), dtype=np.float64)) + 1e-12)
