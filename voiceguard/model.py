"""Nes2Net-X detector = XLS-R 300M frontend (transformers) + Nes2Net-X backend.

Checkpoint loading is strict: any missing or unexpected key raises an error.
The upstream eval.py filters keys silently, which is how a wrong model class
once produced 50.74% EER without any warning.
"""
import hashlib
import os
import re

import torch
import torch.nn as nn

from .backend import Nested_Res2Net_TDNN
from .frontend import FAIRSEQ_PREFIX, XLSRFrontend, map_fairseq_to_hf

# Configuration of the authors' In-the-Wild checkpoint (wav2vec2_Nes2Net_X).
ITW_BACKEND_CFG = dict(Nes_ratio=(8, 8), input_channel=1024, n_output_logits=2,
                       dilation=1, pool_func='mean', SE_ratio=(1,))
BACKEND_PREFIX = "Nested_Res2Net_TDNN."
FORMAT = "voiceguard-nes2netx-v1"


class Nes2NetXDetector(nn.Module):
    """Returns logits B x 2. Column 1 (bona fide) is the score, as in main_itw.py.
    num_layers < 24 keeps only the first Transformer layers of the encoder (truncation)."""

    def __init__(self, backend_cfg=None, tiny=False, num_layers=None):
        super().__init__()
        self.ssl = XLSRFrontend(tiny, num_layers)
        cfg = dict(backend_cfg or ITW_BACKEND_CFG)
        if tiny:
            cfg["input_channel"] = self.ssl.out_dim
        self.backend = Nested_Res2Net_TDNN(**cfg)
        self.backend_cfg, self.tiny = cfg, tiny

    def features(self, wav):
        return self.ssl(wav).transpose(1, 2)  # B x 1024 x T

    def forward(self, wav):
        return self.backend(self.features(wav))


def _strip_module(sd):
    if all(k.startswith("module.") for k in sd):
        return {k[len("module."):]: v for k, v in sd.items()}
    return sd


def convert_original_state_dict(original_sd, model=None):
    """Convert the authors' checkpoint (fairseq naming) to Nes2NetXDetector naming."""
    model = model or Nes2NetXDetector()
    sd = _strip_module(original_sd)
    hf_keys = set(model.ssl.model.state_dict().keys())
    ssl_part = {k[len(FAIRSEQ_PREFIX):]: v for k, v in sd.items() if k.startswith(FAIRSEQ_PREFIX)}
    back_part = {k[len(BACKEND_PREFIX):]: v for k, v in sd.items() if k.startswith(BACKEND_PREFIX)}
    other = [k for k in sd if not k.startswith((FAIRSEQ_PREFIX, BACKEND_PREFIX))]
    if other:
        raise KeyError(f"{len(other)} keys belong to neither frontend nor backend, e.g. {other[:5]}")
    if not ssl_part:
        raise KeyError("checkpoint has no 'ssl_model.model.' keys: not a wav2vec2_Nes2Net_X checkpoint")
    mapped, skipped = map_fairseq_to_hf(ssl_part, hf_keys)
    new_sd = {"ssl.model." + k: v for k, v in mapped.items()}
    new_sd.update({"backend." + k: v for k, v in back_part.items()})
    return new_sd, skipped


_LAYER_KEY = re.compile(r"ssl\.model\.encoder\.layers\.(\d+)\.")


def truncate_state_dict(sd, num_layers):
    """Drop the Transformer layers >= num_layers. Returns (kept, dropped_keys).
    Only whole encoder layers are dropped; every other key must still load strictly."""
    kept, dropped = {}, []
    for k, v in sd.items():
        m = _LAYER_KEY.match(k)
        if m and int(m[1]) >= num_layers:
            dropped.append(k)
        else:
            kept[k] = v
    return kept, dropped


def load_detector(path, device="cpu", num_layers=None):
    """Load a VoiceGuard checkpoint with strict=True.

    num_layers smaller than the checkpoint's depth initialises a truncated encoder from it
    (used before fine-tuning); the dropped layers are listed in info['dropped_layers']."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or ckpt.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} checkpoint; run python -m voiceguard.convert first")
    tiny = bool(ckpt.get("tiny"))
    depth = ckpt.get("num_layers") or (2 if tiny else 24)
    num_layers = num_layers or depth
    if num_layers > depth:
        raise ValueError(f"checkpoint has {depth} encoder layers, cannot build {num_layers}")
    model = Nes2NetXDetector(ckpt.get("backend_cfg"), tiny=tiny, num_layers=num_layers)
    sd = {k: v.float() for k, v in ckpt["state_dict"].items()}
    dropped = []
    if num_layers < depth:
        sd, dropped = truncate_state_dict(sd, num_layers)
    model.load_state_dict(sd, strict=True)
    model.eval().to(device)
    info = {k: ckpt.get(k) for k in ("source", "source_sha256", "converted_at", "trained")}
    info.update(path=os.path.abspath(path), num_layers=num_layers,
                dropped_layers=sorted({int(_LAYER_KEY.match(k)[1]) for k in dropped}))
    return model, info


def save_detector(model, path, **meta):
    """Save in the format load_detector reads (weights, backend config, depth, metadata)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(dict(format=FORMAT, backend_cfg=model.backend_cfg, tiny=model.tiny,
                    num_layers=model.ssl.num_layers, state_dict=model.state_dict(), **meta), path)


def build_random_detector(device="cpu", seed=0, tiny=False, num_layers=None):
    """Random weights. Only for plumbing tests and latency/MAC measurement.
    tiny=True keeps the structure with ~2 M parameters, for tests on weak machines."""
    torch.manual_seed(seed)
    return Nes2NetXDetector(tiny=tiny, num_layers=num_layers).eval().to(device)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
