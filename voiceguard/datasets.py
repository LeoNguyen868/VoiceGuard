"""Label files and test subsets for the evaluation sets.

A label file has one line per clip, '<name without extension> bonafide|spoof'. In-the-Wild's meta.csv
(file,speaker,label) is read directly. ASVspoof 5 labels are built from the official protocol:

  python -m voiceguard.datasets asv5 --protocol ASVspoof5.dev.track_1.tsv --out datasets/asv5_dev_labels.txt
  python -m voiceguard.datasets check --data_dir <audio folder> --labels <labels>   # counts, missing files
"""
import argparse
import os
from collections import Counter

import numpy as np

AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg")


def read_labels(path):
    """{name: 1 for bona fide, 0 for spoof} from 'name label' lines or ITW meta.csv (file,speaker,label)."""
    labels = {}
    with open(path, encoding="utf-8") as f:
        if path.endswith(".csv"):
            next(f)
            for line in f:
                file, _, lab = line.strip().rsplit(",", 2)
                labels[os.path.splitext(file)[0]] = int(lab.strip().lower() in ("bona-fide", "bonafide"))
        else:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    labels[parts[0]] = int(parts[-1].lower() in ("bonafide", "bona-fide"))
    return labels


def pick_subset(names, labels, n, seed):
    """Stratified random subset of n clips (same bona fide/spoof ratio), fixed seed; 0 = all clips.
    The ASVspoof 5 dev results use n=20000, seed=2026."""
    if not n or n >= len(names):
        return names
    rng = np.random.default_rng(seed)
    bona = [x for x in names if labels[x] == 1]
    spoof = [x for x in names if labels[x] == 0]
    nb = round(n * len(bona) / len(names))
    keep = set(rng.choice(bona, nb, replace=False)) | set(rng.choice(spoof, n - nb, replace=False))
    return [x for x in names if x in keep]


def audio_files(data_dir, labels):
    """{name: extension} for the labelled audio files present in data_dir."""
    ext = {}
    for f in os.listdir(data_dir):
        stem, e = os.path.splitext(f)
        if e.lower() in AUDIO_EXT and stem in labels:
            ext[stem] = e
    return ext


def asv5_labels(protocol, out):
    """ASVspoof 5 protocol (space separated; column 2 = file name, column 9 = KEY) -> label file."""
    rows = []
    with open(protocol, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 9:
                rows.append((p[1], p[8]))
    keys = Counter(k for _, k in rows)
    if set(keys) - {"bonafide", "spoof"}:
        raise ValueError(f"unexpected KEY values {dict(keys)}: is column 9 the KEY column?")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.writelines(f"{n} {k}\n" for n, k in rows)
    print(f"{out}: {len(rows)} clips, {dict(keys)}")


def check(data_dir, labels_path):
    labels = read_labels(labels_path)
    ext = audio_files(data_dir, labels)
    c = Counter(labels[n] for n in ext)
    print(f"{len(labels)} labelled, {len(ext)} found in {data_dir} "
          f"(bona fide {c[1]}, spoof {c[0]}), {len(labels) - len(ext)} missing")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("asv5")
    a.add_argument("--protocol", required=True)
    a.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("--data_dir", required=True)
    c.add_argument("--labels", required=True)
    args = ap.parse_args()
    if args.cmd == "asv5":
        asv5_labels(args.protocol, args.out)
    else:
        check(args.data_dir, args.labels)


if __name__ == "__main__":
    main()
