"""Simulate a PBX: many concurrent calls streaming audio to a running VoiceGuard service in real time.

Each call picks a caller (registered customer, unknown, hidden, or a known fraudster), a line channel
(Zalo/Messenger = Opus, mobile = AMR-NB, VoLTE = AMR-WB, PBX = G.711), and a clip from the sample
library (samples/manifest.json, spoofed clips go to fraudster calls more often) or a synthetic signal.
The clip is passed through the line's codec with ffmpeg, then sent as 16 kHz 16-bit PCM in chunks at
the speed of speech, exactly as a PBX connector would.

  VG_API_KEY=<key> python -m voiceguard.simulate --url http://localhost:8000 --calls 20 --duration 30

The service must run with the same VG_API_KEY: the simulator authenticates as the PBX connector.
"""
import argparse
import json
import os
import random
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

from .audio import SR, load_audio
from .channel import apply_channel, available_channels

LINES = [("opus12", 0.35), ("amrnb", 0.3), ("amrwb", 0.15), ("g711u", 0.2)]


API_KEY = os.environ.get("VG_API_KEY", "")


def http(url, data=None, body=None, method="POST"):
    if data is not None:
        body = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(url, data=body, method=method)
    if API_KEY:
        req.add_header("Authorization", f"Bearer {API_KEY}")
    if data is None and body is not None:
        req.add_header("Content-Type", "application/octet-stream")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def phone(rng):
    return "09" + "".join(str(rng.integers(10)) for _ in range(8))


def synthetic(rng, seconds, spoof):
    t = np.arange(int(seconds * SR)) / SR
    f0 = 120 + 60 * rng.random()
    vib = (0.5 if spoof else 3.0) * np.sin(2 * np.pi * 4 * t)                 # flatter pitch for "spoof"
    syl = 0.5 + 0.5 * np.sin(2 * np.pi * (3 + rng.random()) * t) ** 2
    x = sum(np.sin(2 * np.pi * k * (f0 + vib) * t) / k for k in range(1, 6)) * syl
    return (0.15 * x / np.abs(x).max() + 0.01 * rng.standard_normal(len(t))).astype(np.float32)


def run_call(i, args, clips, pools, rng, stats, lock):
    kind = rng.choice(["registered", "unknown", "hidden", "fraudster"], p=[0.55, 0.25, 0.05, 0.15])
    number = "" if kind == "hidden" and rng.random() < 0.5 else ("***" if kind == "hidden" else
                                                                 rng.choice(pools[kind]) if pools[kind] else phone(rng))
    lines = args.lines
    ch = rng.choice([c for c, _ in lines], p=np.array([w for _, w in lines]) / sum(w for _, w in lines))
    spoof = kind == "fraudster" or rng.random() < args.spoof_share
    pool = [c for c in clips if (c["label"] == "spoof") == spoof] or clips
    if pool:
        wav = load_audio(os.path.join(args.samples, rng.choice(pool)["file"]))
        wav = np.tile(wav, int(np.ceil(args.duration * SR / max(len(wav), 1))))[: int(args.duration * SR)]
    else:
        wav = synthetic(rng, args.duration, spoof)
    wav = apply_channel(wav, ch)
    pcm = (np.clip(wav, -1, 1) * 32767).astype("<i2").tobytes()
    time.sleep(rng.random() * args.ramp)
    call = http(f"{args.url}/v1/calls", dict(caller_number=number, channel=ch, agent=f"nv{i % 8 + 1:02d}",
                                            recording_ref=f"REC-{int(time.time())}-{i:04d}"))
    cid, step = call["call_id"], int(args.chunk * SR) * 2
    t0 = time.time()
    for k, off in enumerate(range(0, len(pcm), step)):
        r = http(f"{args.url}/v1/calls/{cid}/audio", body=pcm[off:off + step])
        delay = t0 + (k + 1) * args.chunk / args.speed - time.time()
        if delay > 0:
            time.sleep(delay)
        if r.get("risk_level") == "HIGH" and kind == "fraudster" and rng.random() < 0.3:
            break                                                             # fraudster hangs up
    end = http(f"{args.url}/v1/calls/{cid}/end")
    with lock:
        stats[end["risk_level"] or "none"] = stats.get(end["risk_level"] or "none", 0) + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--calls", type=int, default=20, help="calls in total")
    ap.add_argument("--concurrent", type=int, default=0, help="calls at the same time (default: all)")
    ap.add_argument("--duration", type=float, default=30, help="seconds of audio per call")
    ap.add_argument("--chunk", type=float, default=0.5, help="seconds of audio per request")
    ap.add_argument("--speed", type=float, default=1.0, help="2 = twice as fast as speech")
    ap.add_argument("--ramp", type=float, default=10, help="calls start at random within this many seconds")
    ap.add_argument("--spoof_share", type=float, default=0.1, help="share of non-fraudster calls with a spoofed clip")
    ap.add_argument("--samples", default="samples", help="sample library (manifest.json); empty = synthetic audio")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--user", default="", help="console account that registers the simulated customers (password in VG_SIM_PASSWORD)")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    avail = available_channels()                                             # once, before the threads start
    args.lines = [(c, w) for c, w in LINES if avail.get(c)]
    if len(args.lines) < len(LINES):
        print("this ffmpeg lacks", [c for c, _ in LINES if not avail.get(c)], "- those lines are skipped")
    manifest = os.path.join(args.samples, "manifest.json")
    clips = json.load(open(manifest, encoding="utf-8")) if os.path.exists(manifest) else []
    # Number pools: some registered customers and a few repeat fraudsters (not blacklisted yet).
    pools = dict(registered=[phone(rng) for _ in range(30)], fraudster=[phone(rng) for _ in range(3)], unknown=[])
    if args.user:                                                            # number lists need a console account
        jar = urllib.request.build_opener(urllib.request.HTTPCookieProcessor())
        jar.open(f"{args.url}/v1/auth/login", urllib.parse.urlencode(
            dict(username=args.user, password=os.environ.get("VG_SIM_PASSWORD", ""))).encode())
        for n in pools["registered"]:
            jar.open(f"{args.url}/v1/numbers", urllib.parse.urlencode(dict(number=n, list="registered", note="Khách hàng")).encode())
    else:
        pools["registered"] = []
    print(f"{args.calls} calls, {len(clips) or 'synthetic'} clips, {args.duration:.0f} s each, url {args.url}")
    stats, lock, threads = {}, threading.Lock(), []
    sem = threading.Semaphore(args.concurrent or args.calls)

    def worker(i, seed):
        with sem:
            try:
                run_call(i, args, clips, pools, np.random.default_rng(seed), stats, lock)
            except Exception as e:
                print(f"call {i}: {e}")
    for i in range(args.calls):
        t = threading.Thread(target=worker, args=(i, args.seed * 1000 + i))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    print("final risk levels:", stats)


if __name__ == "__main__":
    main()
