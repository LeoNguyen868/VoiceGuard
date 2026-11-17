"""HTTP service.

Environment variables:
  VG_CKPT        converted checkpoint (python -m voiceguard.convert); default weights/voiceguard_L14.pt
  VG_RANDOM=1    use random weights instead (plumbing demo only, scores are meaningless);
                 VG_RANDOM=tiny uses the ~2 M parameter variant (tests on weak machines)
  VG_POLICY      calibration JSON with t_high / t_medium; default configs/policy/L14_clean_4s.json
                 when the default 14-layer checkpoint is used
  VG_DEVICE      cpu | cuda (default: cuda if available)
  VG_FP16=1      FP16 autocast on GPU
  VG_REGISTERED  text file of registered customer numbers, one per line (pre-filter)
  VG_BLACKLIST   text file of blacklisted numbers, one per line (pre-filter)
  VG_AUDIT_LOG   JSONL audit log path (default: logs/audit.jsonl). Audio and phone numbers are never stored.
  VG_CUDA_GRAPH=1    replay CUDA Graphs for 4 s windows (GPU only)
  VG_BATCH=0         run each request alone instead of batching windows across requests and calls
  VG_BATCH_MAX       largest batch of windows (default 16)
  VG_BATCH_WAIT_MS   how long the first queued window waits for others (default 10)

Calls can also be streamed: POST /v1/calls opens a call, POST /v1/calls/{id}/audio sends raw
16 kHz mono 16-bit PCM chunks (a window is scored every 2 s once 4 s are buffered), and
POST /v1/calls/{id}/end closes it and writes one audit row.

Run: uvicorn voiceguard.api:app --host 0.0.0.0 --port 8000
"""
import hashlib
import json
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from .audio import SR, load_audio, repeat_pad, rms_dbfs
from .batcher import WindowBatcher
from .channel import CHANNELS, apply_channel
from .detector import MIN_RMS_DBFS, MIN_SECONDS, Detector, InsufficientAudio
from .model import build_random_detector, load_detector
from .policy import at_least_medium, load_policy
from .prefilter import RULES, classify, load_numbers

STATIC = Path(__file__).parent / "static"
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = ROOT / "weights" / "voiceguard_L14.pt"
DEFAULT_POLICY = ROOT / "configs" / "policy" / "L14_clean_4s.json"
app = FastAPI(title="VoiceGuard - examination service", version="0.2")
_state = {}


def _get():
    if "detector" not in _state:
        device = os.environ.get("VG_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        if os.environ.get("VG_RANDOM") in ("1", "tiny"):
            model = build_random_detector(device, tiny=os.environ["VG_RANDOM"] == "tiny")
            info = {"path": None, "random_weights": True}
        else:
            ckpt = os.environ.get("VG_CKPT") or (str(DEFAULT_CKPT) if DEFAULT_CKPT.exists() else None)
            if not ckpt:
                raise RuntimeError(f"no checkpoint: build {DEFAULT_CKPT} (see weights/README.md), set VG_CKPT, "
                                   "or VG_RANDOM=1 for a demo")
            model, info = load_detector(ckpt, device)
        policy_path = os.environ.get("VG_POLICY")
        if policy_path is None and not os.environ.get("VG_CKPT") and DEFAULT_POLICY.exists():
            policy_path = str(DEFAULT_POLICY)
        policy = load_policy(policy_path)
        det = Detector(model, policy, device, fp16=os.environ.get("VG_FP16") == "1",
                       cuda_graph=os.environ.get("VG_CUDA_GRAPH") == "1")
        _state["detector"] = det
        _state["model_info"] = info
        _state["device"] = device
    if "batcher" not in _state:
        _state["batcher"] = WindowBatcher(_state["detector"].logits,
                                          max_batch=int(os.environ.get("VG_BATCH_MAX", 16)),
                                          max_wait_ms=float(os.environ.get("VG_BATCH_WAIT_MS", 10)),
                                          max_requests=1 if os.environ.get("VG_BATCH") == "0" else None)
        _state["calls"] = {}
    return _state["detector"]


def _lists():
    if "registered" not in _state:
        _state["registered"] = load_numbers(os.environ.get("VG_REGISTERED"))
        _state["blacklist"] = load_numbers(os.environ.get("VG_BLACKLIST"))
    return _state["registered"], _state["blacklist"]


@app.on_event("startup")
def _startup():
    _get()
    _lists()


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    det = _get()
    registered, blacklist = _lists()
    b = _state["batcher"]
    return dict(status="ok", device=_state["device"], fp16=det.fp16, model=_state["model_info"],
                policy=det.policy.to_dict(), channels=list(CHANNELS),
                prefilter=dict(registered=len(registered), blacklist=len(blacklist)),
                inference=dict(cuda_graph=det.graphs is not None, batch_max=b.max_batch,
                               batch_wait_ms=b.max_wait * 1000, across_requests=b.max_requests is None,
                               batches=b.stats["batches"], windows=b.stats["windows"], open_calls=len(_state["calls"])))


@app.post("/v1/analyze")
async def analyze(file: UploadFile = File(...), mode: str = Form("window"),
                  channel: str = Form("none"), caller_number: str = Form("")):
    if mode not in ("window", "full"):
        raise HTTPException(422, "mode phải là 'window' hoặc 'full'")
    if channel not in CHANNELS:
        raise HTTPException(422, f"channel phải thuộc {list(CHANNELS)}")
    det = _get()
    data = await file.read()
    registered, blacklist = _lists()
    category = classify(caller_number, registered, blacklist)

    if category == "blacklist":
        # Stopped by the pre-filter: the model is not run.
        result = dict(duration_s=None, mode=mode, n_segments=0, score=None, p_bonafide=None,
                      risk_level="HIGH", recommended_action=RULES["blacklist"],
                      most_suspicious_segment=None, segments=[], policy=det.policy.to_dict(),
                      timing_ms=dict(decode=0.0, model=0.0, rtf=None), screened_by="prefilter")
    else:
        t0 = time.perf_counter()
        suffix = Path(file.filename or "audio").suffix or ".bin"
        # Closed before decoding: on Windows an open temp file cannot be reopened by ffmpeg.
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
        try:
            wav = load_audio(tmp.name)
        except Exception as e:
            raise HTTPException(415, f"không giải mã được tệp âm thanh: {e}")
        finally:
            os.unlink(tmp.name)
        wav = apply_channel(wav, channel)
        decode_ms = (time.perf_counter() - t0) * 1000
        try:
            det.check(wav)
        except InsufficientAudio as e:
            raise HTTPException(422, str(e))
        if mode == "window":
            t1 = time.perf_counter()
            spans, clips = det.windows(wav)
            logit_rows = await _state["batcher"].submit(clips)
            result = det.result(wav, mode, spans, logit_rows, (time.perf_counter() - t1) * 1000)
        else:
            result = await run_in_threadpool(det.analyze, wav, mode)
        result["timing_ms"]["decode"] = round(decode_ms, 1)
        result["screened_by"] = "model"
        if category in ("unknown", "hidden"):
            level, action = at_least_medium(result["risk_level"])
            result["escalated_by_prefilter"] = level != result["risk_level"]
            result["risk_level"], result["recommended_action"] = level, action

    result["caller_category"] = category
    result["request_id"] = uuid.uuid4().hex
    result["channel_simulated"] = channel
    result["model_random_weights"] = bool(_state["model_info"].get("random_weights"))
    _audit(result, hashlib.sha256(data).hexdigest(), file.filename)
    return result


class _Call:
    """Streaming state of one call. Only the last window of audio is kept in memory."""

    def __init__(self, category):
        self.category, self.buf, self.offset, self.next_start = category, np.zeros(0, np.float32), 0, 0
        self.scores, self.skipped, self.sha = [], 0, hashlib.sha256()
        self.received, self.started = 0, time.time()


def _decision(det, call):
    if call.category == "blacklist":
        return dict(score=None, risk_level="HIGH", recommended_action=RULES["blacklist"], screened_by="prefilter")
    if not call.scores:
        return dict(score=None, risk_level=None, recommended_action=None, screened_by="model")
    score = float(np.mean(call.scores))
    level, action = det.policy.decide(score)
    escalated = False
    if call.category in ("unknown", "hidden"):
        new_level, action = at_least_medium(level)
        escalated, level = new_level != level, new_level
    return dict(score=round(score, 4), risk_level=level, recommended_action=action,
                screened_by="model", escalated_by_prefilter=escalated)


@app.post("/v1/calls")
async def open_call(caller_number: str = Form("")):
    det = _get()
    registered, blacklist = _lists()
    call_id = uuid.uuid4().hex
    call = _Call(classify(caller_number, registered, blacklist))
    _state["calls"][call_id] = call
    return dict(call_id=call_id, caller_category=call.category, **_decision(det, call))


@app.post("/v1/calls/{call_id}/audio")
async def call_audio(call_id: str, request: Request):
    """Body: raw 16 kHz mono signed 16-bit PCM. Scores every new full window (4 s, hop 2 s)."""
    det = _get()
    call = _state["calls"].get(call_id)
    if call is None:
        raise HTTPException(404, "không tìm thấy cuộc gọi")
    data = await request.body()
    if len(data) % 2:
        raise HTTPException(422, "PCM 16-bit phải có số byte chẵn")
    call.sha.update(data)
    call.received += len(data) // 2
    new = []
    if call.category != "blacklist":  # blacklisted calls are never scored, so no audio is kept
        call.buf = np.concatenate([call.buf, np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0])
        win, hop = det.win, int(det.hop_s * SR)
        starts, clips = [], []
        while call.next_start + win <= call.offset + len(call.buf):
            s = call.next_start - call.offset
            clip = call.buf[s:s + win]
            if rms_dbfs(clip) < MIN_RMS_DBFS:
                call.skipped += 1  # silence (hold, mute): no score
            else:
                starts.append(call.next_start)
                clips.append(clip)
            call.next_start += hop
        drop = max(0, call.next_start - call.offset)  # keep only audio a future window still needs
        call.buf, call.offset = call.buf[drop:], call.offset + drop
        if clips:
            t0 = time.perf_counter()
            rows = await _state["batcher"].submit(clips)
            ms = (time.perf_counter() - t0) * 1000
            for st, row in zip(starts, rows):
                call.scores.append(float(row[1]))
                new.append(dict(start=round(st / SR, 2), end=round((st + win) / SR, 2), score=round(float(row[1]), 4)))
            new[-1]["latency_ms"] = round(ms, 1)
    return dict(call_id=call_id, audio_s=round(call.received / SR, 2), new_windows=new,
                n_windows=len(call.scores), skipped_silent=call.skipped, **_decision(det, call))


@app.post("/v1/calls/{call_id}/end")
async def end_call(call_id: str):
    det = _get()
    call = _state["calls"].pop(call_id, None)
    if call is None:
        raise HTTPException(404, "không tìm thấy cuộc gọi")
    if call.category != "blacklist" and not call.scores and call.received >= MIN_SECONDS * SR:
        # Call shorter than one window: score the tiled audio once, as for a short file.
        tail = call.buf if len(call.buf) else np.zeros(1, np.float32)
        if rms_dbfs(tail) >= MIN_RMS_DBFS:
            call.scores.append(float((await _state["batcher"].submit([repeat_pad(tail, det.win)]))[0][1]))
    result = dict(request_id=call_id, duration_s=round(call.received / SR, 2), mode="stream",
                  n_segments=len(call.scores), skipped_silent=call.skipped, caller_category=call.category,
                  channel_simulated="none", timing_ms=dict(wall_s=round(time.time() - call.started, 2)),
                  **_decision(det, call))
    _audit(result, call.sha.hexdigest(), None)
    return result


def _audit(result, sha256, filename):
    path = Path(os.environ.get("VG_AUDIT_LOG", "logs/audit.jsonl"))
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(time=datetime.now(timezone.utc).isoformat(), request_id=result["request_id"],
               file_sha256=sha256, filename=filename, duration_s=result["duration_s"],
               score=result["score"], risk_level=result["risk_level"], mode=result["mode"],
               channel=result["channel_simulated"], caller_category=result["caller_category"],
               screened_by=result["screened_by"], timing_ms=result["timing_ms"])
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
