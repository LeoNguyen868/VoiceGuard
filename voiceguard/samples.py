"""Build the sample library of the web interface from local copies of the evaluation sets.

Picks, with a fixed seed, N bona fide and N spoofed clips from each dataset given, cuts each to at most
--max_s seconds, writes 16 kHz mono FLAC to --out and a manifest.json the service reads (VG_SAMPLES).
Clips are drawn at random, not chosen by score, so the library shows the model's mistakes too.

  python -m voiceguard.samples --out samples \\
      --itw $ITW --itw_labels $ITW/meta.csv \\
      --asv5 $ASV5 --asv5_protocol $ASV5_PROTOCOL \\
      --sea $SEA/flac --sea_systems $SEA/systems.txt \\
      --own my_clips/ --own_labels my_clips/labels.txt

--own adds clips you recorded or generated yourself ('name bonafide|spoof [system]' per line).
Licences differ: SEA-Spoof is gated and may not be redistributed, so samples/ is git-ignored; keep the
library on the demo machine and do not publish it.
"""
import argparse
import json
import os

import numpy as np
import soundfile as sf

from .audio import SR, load_audio
from .datasets import audio_files, read_labels


def _itw_meta(path):
    """{name: speaker} from ITW meta.csv (file,speaker,label)."""
    out = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            if line.count(",") >= 2:
                file, speaker, _ = line.strip().rsplit(",", 2)
                out[os.path.splitext(file)[0]] = speaker.strip().strip('"')
    return out


def _asv5_meta(path):
    """{name: (label, attack)} from the ASVspoof 5 protocol (column 2 name, 8 attack, 9 key)."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 9:
                out[p[1]] = (p[8], p[7] if p[7] not in ("-", "bonafide") else "")
    return out


def _sea_meta(path):
    """{name: (label, system)} from systems.txt of the SEA-Spoof preparation."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 6:
                label, model, source = p[1], p[4], p[5]
                out[p[0]] = (label, model if label == "spoof" and model != "-" else (source if source != "-" else ""))
    return out


def _own_meta(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 2:
                out[p[0]] = (p[1].lower(), " ".join(p[2:]))
    return out


def pick(names, labels, n, rng):
    chosen = []
    for lab in (1, 0):
        pool = sorted(x for x in names if labels[x] == lab)
        chosen += list(rng.choice(pool, min(n, len(pool)), replace=False))
    return chosen


def add_dataset(entries, key, title, language, data_dir, labels, info, n, rng, out, max_s, licence):
    ext = audio_files(data_dir, labels)
    for i, name in enumerate(pick(sorted(ext), labels, n, rng)):
        wav = load_audio(os.path.join(data_dir, name + ext[name]))
        wav = wav[: int(max_s * SR)]
        label = "bonafide" if labels[name] == 1 else "spoof"
        file = f"{key}_{label}_{i:02d}.flac"
        sf.write(os.path.join(out, file), wav, SR, subtype="PCM_16")
        entries.append(dict(id=f"{key}_{label}_{i:02d}", file=file, dataset=key, dataset_title=title,
                            language=language, label=label, detail=info.get(name, ""),
                            duration_s=round(len(wav) / SR, 2), original=name, licence=licence))
        print(f"{file}: {title}, {label}, {info.get(name, '')}, {len(wav) / SR:.1f} s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="samples")
    ap.add_argument("--n", type=int, default=4, help="clips per label per dataset")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max_s", type=float, default=12.0)
    ap.add_argument("--itw")
    ap.add_argument("--itw_labels")
    ap.add_argument("--asv5")
    ap.add_argument("--asv5_protocol")
    ap.add_argument("--sea")
    ap.add_argument("--sea_systems")
    ap.add_argument("--own")
    ap.add_argument("--own_labels")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    entries = []
    if args.itw:
        speakers = _itw_meta(args.itw_labels)
        add_dataset(entries, "itw", "In-the-Wild", "en", args.itw, read_labels(args.itw_labels), speakers,
                    args.n, rng, args.out, args.max_s, "In-the-Wild (Müller và cộng sự, 2022), chỉ dùng nghiên cứu")
    if args.asv5:
        meta = _asv5_meta(args.asv5_protocol)
        labels = {k: int(v[0] == "bonafide") for k, v in meta.items()}
        add_dataset(entries, "asv5", "ASVspoof 5", "en", args.asv5, labels, {k: v[1] for k, v in meta.items()},
                    args.n, rng, args.out, args.max_s, "ASVspoof 5, xem LICENSE.txt của bản phát hành")
    if args.sea:
        meta = _sea_meta(args.sea_systems)
        labels = {k: int(v[0] == "bonafide") for k, v in meta.items()}
        add_dataset(entries, "sea", "SEA-Spoof tiếng Việt", "vi", args.sea, labels, {k: v[1] for k, v in meta.items()},
                    args.n, rng, args.out, args.max_s, "SEA-Spoof, chỉ dùng học thuật, không phân phối lại")
    if args.own:
        meta = _own_meta(args.own_labels)
        labels = {k: int(v[0] in ("bonafide", "bona-fide", "real", "that")) for k, v in meta.items()}
        add_dataset(entries, "own", "Mẫu tự thu", "vi", args.own, labels, {k: v[1] for k, v in meta.items()},
                    10 ** 6, rng, args.out, args.max_s, "Mẫu của người dùng")
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=1)
    print(f"{len(entries)} samples, manifest at {os.path.join(args.out, 'manifest.json')}")


if __name__ == "__main__":
    main()
