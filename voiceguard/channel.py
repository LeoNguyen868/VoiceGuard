"""Simulation of enterprise voice channels with ffmpeg.

Each condition encodes a 16 kHz clip with a real codec, decodes it, and returns
16 kHz float32 again, so that the detector sees what it would receive after the
channel. 'nb8k' only band-limits (16 -> 8 -> 16 kHz) without a codec; comparing
it with 'g711u' separates the loss of high frequencies from codec artefacts.
AMR-NB and AMR-WB need an ffmpeg built with libopencore_amrnb and libvo_amrwbenc.
"""
import subprocess

import numpy as np

from .audio import SR

# name: (encoder args, container args for decoding raw streams, description)
CHANNELS = {
    "none": (None, None, "Âm thanh gốc 16 kHz"),
    "nb8k": (["-ar", "8000", "-c:a", "pcm_s16le", "-f", "wav"], [],
             "Giới hạn băng hẹp 8 kHz, không nén codec"),
    "g711u": (["-ar", "8000", "-c:a", "pcm_mulaw", "-f", "wav"], [],
              "G.711 μ-law 64 kbit/s, 8 kHz (Mạng viễn thông cố định PSTN)"),
    "g711a": (["-ar", "8000", "-c:a", "pcm_alaw", "-f", "wav"], [],
              "G.711 A-law 64 kbit/s, 8 kHz (Mạng viễn thông cố định PSTN A-law)"),
    "gsm": (["-ar", "8000", "-c:a", "libgsm", "-f", "gsm"], ["-f", "gsm", "-ar", "8000"],
            "GSM 06.10 13 kbit/s, 8 kHz (Mạng di động 2G)"),
    "amrnb": (["-ar", "8000", "-c:a", "libopencore_amrnb", "-b:a", "12.2k", "-f", "amr"], ["-f", "amr"],
              "AMR-NB 12.2 kbit/s, 8 kHz (Mạng di động 3G, 4G)"),
    "amrwb": (["-ar", "16000", "-c:a", "libvo_amrwbenc", "-b:a", "12.65k", "-f", "amr"], ["-f", "amr"],
              "AMR-WB 12.65 kbit/s, 16 kHz (Thoại di động chất lượng cao, VoLTE)"),
    "g722": (["-ar", "16000", "-c:a", "g722", "-f", "g722"], ["-f", "g722"],
             "G.722 64 kbit/s, 16 kHz (Tổng đài IP / VoIP băng rộng)"),
    "opus12": (["-ar", "16000", "-c:a", "libopus", "-b:a", "12k", "-application", "voip",
                "-f", "ogg"], [], "Opus 12 kbit/s (Ứng dụng gọi OTT: Zalo, Messenger)"),
    "mp3_32": (["-ar", "16000", "-c:a", "libmp3lame", "-b:a", "32k", "-f", "mp3"], [],
               "MP3 32 kbit/s (Tin nhắn thoại OTT, mạng xã hội)"),
}


# Codec augmentation for fine-tuning: each entry draws a bitrate per clip.
# AMR-NB, AMR-WB and G.722 are left out on purpose, to test generalisation to unseen codecs.
TRAIN_POOL = {
    "nb8k": lambda rng: CHANNELS["nb8k"][:2],
    "g711u": lambda rng: CHANNELS["g711u"][:2],
    "g711a": lambda rng: CHANNELS["g711a"][:2],
    "gsm": lambda rng: CHANNELS["gsm"][:2],
    "opus": lambda rng: (["-ar", "16000", "-c:a", "libopus", "-b:a", f"{rng.choice([6, 8, 12, 16, 24])}k",
                          "-application", "voip", "-f", "ogg"], []),
    "mp3": lambda rng: (["-ar", "16000", "-c:a", "libmp3lame", "-b:a", f"{rng.choice([16, 24, 32, 48, 64])}k",
                         "-f", "mp3"], []),
}


def random_channel(wav, rng, pool=tuple(TRAIN_POOL)):
    """Apply one codec drawn uniformly from `pool` (names in TRAIN_POOL). Returns (wav, name)."""
    name = pool[rng.integers(len(pool))]
    enc, dec_in = TRAIN_POOL[name](rng)
    return encode_decode(wav, enc, dec_in), name


def apply_channel(wav, name):
    """Pass a 16 kHz float32 clip through channel `name` and return 16 kHz float32."""
    if name not in CHANNELS:
        raise ValueError(f"unknown channel {name}; choose from {list(CHANNELS)}")
    enc, dec_in, _ = CHANNELS[name]
    if enc is None:
        return wav
    return encode_decode(wav, enc, dec_in)


def encode_decode(wav, enc, dec_in):
    """Encode with ffmpeg arguments `enc`, decode back to 16 kHz float32."""
    raw = np.clip(wav, -1.0, 1.0).astype(np.float32).tobytes()
    base = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error"]
    encoded = subprocess.run(
        base + ["-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0", *enc, "pipe:1"],
        input=raw, capture_output=True, check=True).stdout
    decoded = subprocess.run(
        base + [*dec_in, "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", str(SR), "pipe:1"],
        input=encoded, capture_output=True, check=True).stdout
    return np.frombuffer(decoded, dtype=np.float32).copy()


_AVAILABLE = None


def available_channels():
    """{name: True/False}: which channels this ffmpeg can run (one second of a tone each, cached)."""
    global _AVAILABLE
    if _AVAILABLE is None:
        t = np.arange(SR, dtype=np.float32) / SR
        tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        found = {}
        for name in CHANNELS:
            try:
                apply_channel(tone, name)
                found[name] = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                found[name] = False
        _AVAILABLE = found                                                    # published only when complete
    return _AVAILABLE


def check_channels():
    """Print which channels this ffmpeg can run; True if all of them."""
    ok = available_channels()
    for name, (_, _, desc) in CHANNELS.items():
        print(f"{name:8s} {desc:55s} {'OK' if ok[name] else 'FAILED (encoder not in this ffmpeg)'}")
    return all(ok.values())


if __name__ == "__main__":
    import sys
    sys.exit(0 if check_channels() else 1)
