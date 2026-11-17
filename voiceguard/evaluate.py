"""Score a labelled audio folder (In-the-Wild, ASVspoof 5, ...) under a channel and one or more durations.

Each clip is decoded, passed through the channel (codec on the whole clip), then cut to every requested
duration: 'full' keeps the whole clip, a number N keeps the first N seconds (tiled if shorter), as the
service does for one 4 s window. Reading and ffmpeg run in a thread pool.

Writes, per model and duration, <out_dir>/scores/<tag>.txt ('name score', score = bona fide logit) and
<out_dir>/<tag>.json (EER with bootstrap 95% CI, minDCF, thresholds at FRR 1%), tag =
<prefix><clean|channel>_<full|Ns>. An interrupted run resumes from the score files.

--ckpt_shared scores a second detector whose encoder is the first L layers of --ckpt (the 14-layer
detector and the baseline) in the same pass: the full encoder runs once and the second classifier reads
hidden state L after the final layer norm. The script refuses checkpoints whose encoders differ.

Full-length clips are batched without padding: the CNN feature encoder of XLS-R has no padding, so clips
with the same number of frames give the same scores alone or in a batch (up to rounding). --group 1 turns
this off.

  python -m voiceguard.evaluate --ckpt weights/voiceguard_L14.pt --data_dir $ITW --labels $ITW/meta.csv \\
      --channel opus12 --durations 4 full --fp16 --prefix itw_L14_
"""
import argparse
import json
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from tqdm import tqdm

from .audio import crop_duration, load_audio
from .channel import apply_channel
from .datasets import audio_files, pick_subset, read_labels
from .metrics import summarize
from .model import load_detector


def check_shared(full, trunc):
    """Number of shared encoder tensors; raises if any differs or the second one is not shallower."""
    if trunc.ssl.num_layers >= full.ssl.num_layers:
        raise ValueError(f"--ckpt_shared has {trunc.ssl.num_layers} layers, --ckpt {full.ssl.num_layers}")
    a, b = full.ssl.model.state_dict(), trunc.ssl.model.state_dict()
    missing = [k for k in b if k not in a]
    differ = [k for k in b if k in a and not torch.equal(a[k].cpu(), b[k].cpu())]
    if missing or differ:
        raise ValueError(f"encoders are not shared: {len(missing)} tensors missing, {len(differ)} differ")
    return len(b)


@torch.no_grad()
def scores_of(models, x):
    """Bona fide scores (logit column 1) of one model, or of a model and a shallower one sharing its encoder."""
    full = models[0]
    if len(models) == 1:
        return [full(x)[:, 1].float().cpu().numpy()]
    trunc = models[1]
    out = full.ssl.model(x, output_hidden_states=True)
    h = trunc.ssl.model.encoder.layer_norm(out.hidden_states[trunc.ssl.num_layers])
    return [full.backend(out.last_hidden_state.transpose(1, 2))[:, 1].float().cpu().numpy(),
            trunc.backend(h.transpose(1, 2))[:, 1].float().cpu().numpy()]


def prepared(names, load, threads, prefetch):
    """Yield (name, load(name)) in order, computed ahead in a thread pool."""
    with ThreadPoolExecutor(threads) as ex:
        it, queue = iter(names), deque()
        for n in names[:prefetch]:
            queue.append((n, ex.submit(load, n)))
            next(it)
        while queue:
            n, fut = queue.popleft()
            nxt = next(it, None)
            if nxt is not None:
                queue.append((nxt, ex.submit(load, nxt)))
            yield n, fut.result()


def frames(n_samples, spec):
    for k, st in spec:
        n_samples = (n_samples - k) // st + 1
    return n_samples


def needed(n_frames, spec):
    """Fewest samples that give n_frames frames: exactly the samples the frames read."""
    for k, st in reversed(spec):
        n_frames = (n_frames - 1) * st + k
    return n_frames


def tag_of(prefix, channel, duration):
    return f"{prefix}{'clean' if channel == 'none' else channel}_{'full' if duration == 'full' else f'{duration}s'}"


def read_scores(path):
    out = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                k, v = line.split()
                out[k] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt_shared", default="", help="optional shallower detector sharing --ckpt's encoder")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--labels", required=True, help="label file or ITW meta.csv")
    ap.add_argument("--channel", default="none")
    ap.add_argument("--durations", nargs="+", default=["4"], help="'full' and/or seconds, e.g. 4 full")
    ap.add_argument("--prefix", default="", help="tag prefix of --ckpt")
    ap.add_argument("--prefix_shared", default="", help="tag prefix of --ckpt_shared")
    ap.add_argument("--subset", type=int, default=0, help="stratified subset size, 0 = all clips")
    ap.add_argument("--seed", type=int, default=2026)
    prec = ap.add_mutually_exclusive_group()
    prec.add_argument("--fp16", action="store_true")
    prec.add_argument("--bf16", action="store_true")
    ap.add_argument("--batch_size", type=int, default=16, help="for fixed durations")
    ap.add_argument("--threads", type=int, default=8, help="threads for reading and ffmpeg")
    ap.add_argument("--prefetch", type=int, default=64)
    ap.add_argument("--group", type=int, default=32, help="max clips per full-length batch, 1 = one at a time")
    ap.add_argument("--chunk", type=int, default=256, help="clips (sorted by length) grouped together")
    ap.add_argument("--max_frames", type=int, default=12000, help="max encoder frames in one full-length batch")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--n_boot", type=int, default=1000)
    args = ap.parse_args()
    if args.ckpt_shared and args.prefix_shared == args.prefix:
        ap.error("--prefix_shared must differ from --prefix")

    labels = read_labels(args.labels)
    ext = audio_files(args.data_dir, labels)
    names = pick_subset(sorted(ext), labels, args.subset, args.seed)

    models, infos = zip(*[load_detector(p, args.device) for p in [args.ckpt] + ([args.ckpt_shared] if args.ckpt_shared else [])])
    n_shared = check_shared(*models) if len(models) == 2 else None
    prefixes = [args.prefix, args.prefix_shared][:len(models)]

    os.makedirs(os.path.join(args.out_dir, "scores"), exist_ok=True)
    tags = {(m, d): tag_of(prefixes[m], args.channel, d) for d in args.durations for m in range(len(models))}
    paths = {k: os.path.join(args.out_dir, "scores", t + ".txt") for k, t in tags.items()}
    scores = {k: read_scores(p) for k, p in paths.items()}
    done = set(names).intersection(*[set(s) for s in scores.values()])
    todo = [n for n in names if n not in done]
    for k, s in scores.items():
        clash = set(todo) & set(s)
        if clash:
            sys.exit(f"{paths[k]} already holds {len(clash)} of the clips to score; remove it or change the prefix")
    if args.group > 1 and "full" in args.durations:  # similar lengths arrive together
        import soundfile as sf
        todo.sort(key=lambda n: sf.info(os.path.join(args.data_dir, n + ext[n])).frames)
    print(f"{len(names)} clips ({len(done)} already scored), channel={args.channel}, durations={args.durations}, "
          f"models={[i['path'] for i in infos]}, device={args.device}")

    def load(n):
        wav = apply_channel(load_audio(os.path.join(args.data_dir, n + ext[n])), args.channel)
        return {d: crop_duration(wav, d) for d in args.durations}

    use_amp = (args.fp16 or args.bf16) and args.device.startswith("cuda")
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    handles = {k: open(p, "a") for k, p in paths.items()}
    batches = {d: [] for d in args.durations if d != "full"}

    def run(d, items):
        x = torch.from_numpy(np.stack([w for _, w in items])).to(args.device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            out = scores_of(models, x)
        for m, col in enumerate(out):
            for (n, _), v in zip(items, col):
                handles[(m, d)].write(f"{n} {v}\n")
                scores[(m, d)][n] = float(v)

    spec = list(zip(models[0].ssl.model.config.conv_kernel, models[0].ssl.model.config.conv_stride))
    pending = []

    def flush_full():
        by_len = {}
        for n, w in pending:
            f = frames(len(w), spec)
            by_len.setdefault(f, []).append((n, w[:needed(f, spec)] if f >= 1 else w))
        for f, items in by_len.items():
            size = max(1, min(args.group, args.max_frames // max(f, 1)))
            for i in range(0, len(items), size):
                run("full", items[i:i + size])
        pending.clear()

    t0 = time.time()
    try:
        for n, wavs in tqdm(prepared(todo, load, args.threads, args.prefetch), total=len(todo), desc=args.channel):
            for d, w in wavs.items():
                if d == "full":
                    pending.append((n, w))
                    if len(pending) >= (args.chunk if args.group > 1 else 1):
                        flush_full()
                else:
                    batches[d].append((n, w))
                    if len(batches[d]) == args.batch_size:
                        run(d, batches[d])
                        batches[d] = []
        if pending:
            flush_full()
        for d, items in batches.items():
            if items:
                run(d, items)
    finally:
        for h in handles.values():
            h.close()
    elapsed = time.time() - t0

    for (m, d), t in tags.items():
        s = scores[(m, d)]
        bona = np.array([s[k] for k in names if labels[k] == 1])
        spoof = np.array([s[k] for k in names if labels[k] == 0])
        res = summarize(bona, spoof, n_boot=args.n_boot)
        res.update(tag=t, channel=args.channel, duration=d, subset=args.subset, seed=args.seed, fp16=args.fp16,
                   bf16=args.bf16, device=args.device, model=infos[m], shared_encoder_tensors=n_shared,
                   gpu=torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
                   clips_scored_this_run=len(todo), seconds_this_run=round(elapsed, 1))
        with open(os.path.join(args.out_dir, f"{t}.json"), "w") as f:
            json.dump(res, f, indent=2, ensure_ascii=False, default=str)
        print(f"{t}: EER {100 * res['eer']:.3f}%  (95% CI {100 * res['eer_ci95'][0]:.2f}-{100 * res['eer_ci95'][1]:.2f})"
              if res["eer_ci95"][0] is not None else f"{t}: EER {100 * res['eer']:.3f}%")


if __name__ == "__main__":
    main()
