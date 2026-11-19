"""Build VoiceGuard checkpoints.

Three steps, each a sub-command:

  authors      the authors' Nes2Net-X In-the-Wild checkpoint (fairseq naming) -> VoiceGuard format
               (24 layers, the baseline). Does not need fairseq. Fails on any unmapped or missing tensor.
  export-head  a trained 14-layer checkpoint -> its classifier only (a few MB), with a fingerprint of the
               frozen encoder it was trained on.
  assemble     baseline checkpoint + exported classifier -> the 14-layer detector. The encoder is the first
               14 layers of the baseline (it was frozen during training); the fingerprint check refuses a
               baseline that differs from the one the classifier was trained on.

  python -m voiceguard.convert authors  --src WavLM_Nes2Net_X_ITW.pt --dst weights/nes2netx_itw_vg.pt
  python -m voiceguard.convert assemble --base weights/nes2netx_itw_vg.pt --head weights/L14_head.pt \\
      --dst weights/voiceguard_L14.pt
"""
import argparse
import hashlib
import os
from datetime import datetime, timezone

import torch

from .model import (FORMAT, ITW_BACKEND_CFG, Nes2NetXDetector, convert_original_state_dict, sha256_file,
                    truncate_state_dict)

HEAD_FORMAT = "voiceguard-head-v1"


def encoder_fingerprint(sd):
    """SHA-256 over the encoder tensors (keys 'ssl.*', sorted, as float32 bytes)."""
    h = hashlib.sha256()
    for k in sorted(k for k in sd if k.startswith("ssl.")):
        h.update(k.encode())
        h.update(sd[k].detach().float().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def cmd_authors(args):
    sd = torch.load(args.src, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd and not any(k.startswith("ssl_model") for k in sd):
        sd = sd["state_dict"]
    model = Nes2NetXDetector()
    new_sd, skipped = convert_original_state_dict(sd, model)
    model.load_state_dict(new_sd, strict=True)  # raises on any missing/unexpected key
    print(f"source tensors: {len(sd)}, converted: {len(new_sd)}, "
          f"skipped (unused by features_only): {len(skipped)} {sorted(set(k.split('.')[0] for k in skipped))}")
    if args.fp16:
        new_sd = {k: (v.half() if v.is_floating_point() else v) for k, v in new_sd.items()}
    _save(dict(format=FORMAT, backend_cfg=ITW_BACKEND_CFG, state_dict=new_sd, source=os.path.basename(args.src),
               source_sha256=sha256_file(args.src), converted_at=_now()), args.dst)


def cmd_export_head(args):
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if ckpt.get("format") != FORMAT:
        raise ValueError(f"{args.ckpt} is not a {FORMAT} checkpoint")
    sd = ckpt["state_dict"]
    head = {k[len("backend."):]: v for k, v in sd.items() if k.startswith("backend.")}
    _save(dict(format=HEAD_FORMAT, backend_cfg=ckpt["backend_cfg"], num_layers=ckpt["num_layers"],
               tiny=bool(ckpt.get("tiny")),
               encoder_fingerprint=encoder_fingerprint(sd), state_dict=head,
               trained=ckpt.get("trained"), source=os.path.basename(args.ckpt),
               source_sha256=sha256_file(args.ckpt), exported_at=_now()), args.dst)


def cmd_assemble(args):
    base = torch.load(args.base, map_location="cpu", weights_only=False)
    head = torch.load(args.head, map_location="cpu", weights_only=False)
    if base.get("format") != FORMAT or head.get("format") != HEAD_FORMAT:
        raise ValueError(f"--base must be {FORMAT}, --head {HEAD_FORMAT}")
    L = head["num_layers"]
    enc, _ = truncate_state_dict({k: v for k, v in base["state_dict"].items() if k.startswith("ssl.")}, L)
    # The classifier was trained on an encoder loaded in float32 from the baseline file; a baseline
    # converted in FP16 gives FP16-rounded values, so both readings are tried.
    for name, cand in (("float32", {k: v.float() for k, v in enc.items()}),
                       ("fp16-rounded", {k: v.half().float() for k, v in enc.items()})):
        if encoder_fingerprint(cand) == head["encoder_fingerprint"]:
            break
    else:
        raise ValueError("the baseline's first layers differ from the encoder the classifier was trained on")
    sd = dict(cand)
    sd.update({"backend." + k: v for k, v in head["state_dict"].items()})
    tiny = bool(head.get("tiny"))
    model = Nes2NetXDetector(head["backend_cfg"], tiny=tiny, num_layers=L)
    model.load_state_dict(sd, strict=True)
    print(f"encoder: first {L} layers of {os.path.basename(args.base)} ({name}), fingerprint matches")
    _save(dict(format=FORMAT, backend_cfg=head["backend_cfg"], tiny=tiny, num_layers=L, state_dict=sd,
               trained=head.get("trained"), source=f"{os.path.basename(args.base)} + {os.path.basename(args.head)}",
               base_sha256=sha256_file(args.base), head_sha256=sha256_file(args.head),
               head_source_sha256=head.get("source_sha256"), assembled_at=_now()), args.dst)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _save(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(obj, path)
    print(f"saved {path} ({os.path.getsize(path) / 2**20:.1f} MiB), sha256 {sha256_file(path)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("authors")
    a.add_argument("--src", required=True)
    a.add_argument("--dst", required=True)
    a.add_argument("--fp16", action="store_true", help="store weights in FP16 (half the size)")
    e = sub.add_parser("export-head")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--dst", required=True)
    s = sub.add_parser("assemble")
    s.add_argument("--base", required=True)
    s.add_argument("--head", required=True)
    s.add_argument("--dst", required=True)
    args = ap.parse_args()
    dict(authors=cmd_authors, **{"export-head": cmd_export_head, "assemble": cmd_assemble})[args.cmd](args)


if __name__ == "__main__":
    main()
