"""HTTP service.

Environment variables:
  VG_CKPT        converted checkpoint (python -m voiceguard.convert); default weights/voiceguard_L14.pt
  VG_RANDOM=1    use random weights instead (plumbing demo only, scores are meaningless);
                 VG_RANDOM=tiny uses the ~2 M parameter variant (tests on weak machines)
  VG_POLICY      calibration JSON with t_high / t_medium; default configs/policy/L14_clean_4s.json
                 when the default 14-layer checkpoint is used
  VG_POLICY_DIR  folder of per-channel thresholds <VG_POLICY_PREFIX><channel>_4s.json (channel 'none' is
                 'clean'); a request on a channel without a file uses VG_POLICY. Default configs/policy with
                 prefix L14_ when the default 14-layer checkpoint is used
  VG_SAMPLES     folder with manifest.json and sample audio for the web interface (default samples/)
  VG_DEVICE      cpu | cuda (default: cuda if available)
  VG_FP16=1      FP16 autocast on GPU
  VG_REGISTERED  text file of registered customer numbers, one per line, imported at start-up
  VG_BLACKLIST   text file of blacklisted numbers, one per line, imported at start-up
  VG_DB          SQLite file of call history, number lists and feedback (default data/voiceguard.db)
  VG_WEBHOOK_URL URL that receives a JSON POST when a call reaches HIGH or an agent holds or transfers it
  VG_RECORD      flagged (default): keep the audio of calls that reached MEDIUM or HIGH, for review;
                 all: keep every call; none: keep no audio
  VG_RECORD_DAYS recordings older than this are deleted at start-up (default 30)
  VG_API_KEY     key of the PBX connector ("Authorization: Bearer <key>") for /v1/calls and /v1/analyze
  VG_ADMIN_PASSWORD  password of the first admin account, created when the database has no user; without it a
                 random password is written to data/admin-password.txt
  VG_SESSION_HOURS   console session lifetime (default 12)
Console users log in at /login; python -m voiceguard.users adds, lists, locks accounts and changes passwords.
  VG_AUDIT_LOG   JSONL audit log path (default: logs/audit.jsonl); it holds no audio and no phone number.
  VG_CUDA_GRAPH=1    replay CUDA Graphs for 4 s windows (GPU only)
  VG_BATCH=0         run each request alone instead of batching windows across requests and calls
  VG_BATCH_MAX       largest batch of windows (default 16)
  VG_BATCH_WAIT_MS   how long the first queued window waits for others (default 10)

Calls are streamed by the PBX connector: POST /v1/calls opens a call (caller number, line channel), POST
/v1/calls/{id}/audio sends raw 16 kHz mono 16-bit PCM chunks (a window is scored every 2 s once 4 s are
buffered), POST /v1/calls/{id}/end closes it. Every step is stored and published on GET /v1/events
(server-sent events) for the operator console: agents act on live calls, fraud analysts give verdicts,
confirmed fraud blacklists the caller.

Run: uvicorn voiceguard.api:app --host 0.0.0.0 --port 8000
"""
import asyncio
import hashlib
import hmac
import secrets
import json
import os
import tempfile
import threading
import time
import urllib.request
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .audio import SR, load_audio, repeat_pad, rms_dbfs
from .batcher import WindowBatcher
from .channel import CHANNELS, apply_channel, available_channels
from .detector import MIN_RMS_DBFS, MIN_SECONDS, Detector, InsufficientAudio
from .model import build_random_detector, load_detector
from .policy import LEVELS, at_least_medium, load_policy
from .prefilter import RULES, load_numbers, normalize
from .store import LEVEL_RANK, Store

STATIC = Path(__file__).parent / "static"
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = ROOT / "weights" / "voiceguard_L14.pt"
DEFAULT_POLICY = ROOT / "configs" / "policy" / "L14_clean_4s.json"
SAMPLES = Path(os.environ.get("VG_SAMPLES") or ROOT / "samples")
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
        _state["channel_policies"] = _channel_policies(not os.environ.get("VG_CKPT"))
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


def _channel_policies(default_ckpt):
    """{channel: Policy} from VG_POLICY_DIR; empty when no folder is configured."""
    folder = os.environ.get("VG_POLICY_DIR") or (str(DEFAULT_POLICY.parent) if default_ckpt else None)
    prefix = os.environ.get("VG_POLICY_PREFIX", "L14_" if default_ckpt else "")
    out = {}
    if folder:
        for ch in CHANNELS:
            f = Path(folder) / f"{prefix}{'clean' if ch == 'none' else ch}_4s.json"
            if f.exists():
                out[ch] = load_policy(str(f))
    return out


def _policy_for(channel):
    return _state["channel_policies"].get(channel, _state["detector"].policy)


def _store():
    if "store" not in _state:
        st = Store(os.environ.get("VG_DB", str(ROOT / "data" / "voiceguard.db")))
        for lst, env in (("registered", "VG_REGISTERED"), ("blacklist", "VG_BLACKLIST")):
            path = os.environ.get(env)
            if path:
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        if normalize(line):
                            st.add_number(line.strip(), lst, source=f"tệp {os.path.basename(path)}")
        _state["store"] = st
    return _state["store"]


def _rec_dir():
    d = Path(os.environ.get("VG_RECORD_DIR", str(ROOT / "data" / "recordings")))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _purge_recordings():
    days = float(os.environ.get("VG_RECORD_DAYS", 30))
    for f in _rec_dir().glob("*"):
        if f.stat().st_mtime < time.time() - days * 86400:
            f.unlink(missing_ok=True)


def _finish_recording(call):
    """Keep the call's audio as WAV when the policy says so, else delete it."""
    if call.rec is None:
        return False
    call.rec.close()
    raw = _rec_dir() / f"{call.id}.pcm"
    mode = os.environ.get("VG_RECORD", "flagged")
    keep = mode == "all" or (mode == "flagged" and LEVEL_RANK[call.max_level] >= LEVEL_RANK["MEDIUM"])
    if keep and raw.stat().st_size:
        with wave.open(str(_rec_dir() / f"{call.id}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(raw.read_bytes())
    raw.unlink(missing_ok=True)
    return keep


def _classify(number):
    """Caller category from the stored lists (blacklist wins)."""
    if number is None or number.strip() == "":
        return "not_provided"
    if not normalize(number):
        return "hidden"
    lists = _store().lists_of(_store().key(number))
    return "blacklist" if "blacklist" in lists else "registered" if "registered" in lists else "unknown"


class _Bus:
    """Fan-out of events to every open console (server-sent events)."""

    def __init__(self):
        self.subs = set()

    def publish(self, event):
        for q in list(self.subs):
            if q.qsize() < 1000:
                q.put_nowait(event)


BUS = _Bus()


def _webhook(event):
    url = os.environ.get("VG_WEBHOOK_URL")
    if not url:
        return

    def send():
        try:
            req = urllib.request.Request(url, data=json.dumps(event).encode(), method="POST",
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=3).close()
        except Exception:
            pass
    threading.Thread(target=send, daemon=True).start()


# ---------------------------------------------------------------- authentication

COOKIE = "vg_session"
_fails = {}


def _current(request):
    """Logged-in console user, or the PBX connector when the request carries VG_API_KEY, else None."""
    u = _store().session_user(request.cookies.get(COOKIE))
    if u:
        return u
    key, hdr = os.environ.get("VG_API_KEY"), request.headers.get("authorization", "")
    if key and hmac.compare_digest(hdr.encode(), f"Bearer {key}".encode()):
        return dict(username="pbx", full_name="Kết nối tổng đài", connector=True)
    return None


def need(connector=False):
    """Dependency: a logged-in console user, or also the PBX connector when connector is True."""
    def dep(request: Request):
        u = _current(request)
        if u is None:
            raise HTTPException(401, "Chưa đăng nhập")
        if u.get("connector") and not connector:
            raise HTTPException(403, "Khóa kết nối tổng đài không dùng được cho chức năng này")
        return u
    return dep


def _bootstrap_admin():
    st = _store()
    if st.count_users():
        return
    pw = os.environ.get("VG_ADMIN_PASSWORD")
    if not pw:
        pw = secrets.token_urlsafe(12)
        f = ROOT / "data" / "admin-password.txt"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(pw + "\n", encoding="utf-8")
        os.chmod(f, 0o600)
        print(f"VoiceGuard: created user 'admin'; its password is in {f}")
    st.set_user("admin", pw, "Quản trị hệ thống")


@app.get("/login")
def login_page(request: Request):
    if _store().session_user(request.cookies.get(COOKIE)):
        return RedirectResponse("/", 303)
    return FileResponse(STATIC / "login.html", headers={"Cache-Control": "no-cache"})


@app.post("/v1/auth/login")
def login(response: Response, request: Request, username: str = Form(...), password: str = Form(...)):
    name = username.strip().lower()
    n, t0 = _fails.get(name, (0, time.time()))
    if time.time() - t0 > 300:
        n, t0 = 0, time.time()
    if n >= 5:
        raise HTTPException(429, "Đăng nhập sai quá nhiều lần, thử lại sau 5 phút")
    u = _store().check_password(name, password)
    if u is None:
        _fails[name] = (n + 1, t0)
        raise HTTPException(401, "Sai tài khoản hoặc mật khẩu")
    _fails.pop(name, None)
    hours = float(os.environ.get("VG_SESSION_HOURS", 12))
    token = _store().new_session(u["username"], hours)
    response.set_cookie(COOKIE, token, max_age=int(hours * 3600), httponly=True, samesite="strict",
                        secure=request.url.scheme == "https")
    return u


@app.post("/v1/auth/logout")
def logout(response: Response, request: Request):
    _store().end_session(request.cookies.get(COOKIE))
    response.delete_cookie(COOKIE)
    return dict(ok=True)


@app.get("/v1/auth/me")
def me(user=Depends(need())):
    return user


@app.on_event("startup")
def _startup():
    _get()
    _store()
    _bootstrap_admin()
    _purge_recordings()


@app.get("/")
def console(request: Request):
    if not _store().session_user(request.cookies.get(COOKIE)):
        return RedirectResponse("/login", 303)
    return FileResponse(STATIC / "console.html", headers={"Cache-Control": "no-cache"})


@app.get("/check")
def check_page(request: Request):
    if not _store().session_user(request.cookies.get(COOKIE)):
        return RedirectResponse("/login", 303)
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/v1/samples")
def samples(user=Depends(need())):
    """Sample clips for the web interface (manifest.json written by python -m voiceguard.samples)."""
    f = SAMPLES / "manifest.json"
    if not f.exists():
        return []
    return json.loads(f.read_text(encoding="utf-8"))


if SAMPLES.is_dir():
    app.mount("/samples", StaticFiles(directory=SAMPLES), name="samples")


@app.get("/health")
def health():
    det = _get()
    counts = _store().count_numbers()
    b = _state["batcher"]
    return dict(status="ok", device=_state["device"], fp16=det.fp16, model=_state["model_info"],
                policy=det.policy.to_dict(), channels=list(CHANNELS),
                channel_info={c: dict(description=CHANNELS[c][2], available=available_channels()[c],
                                      calibrated=c in _state["channel_policies"]) for c in CHANNELS},
                prefilter=dict(registered=counts.get("registered", 0), blacklist=counts.get("blacklist", 0)),
                inference=dict(cuda_graph=det.graphs is not None, batch_max=b.max_batch,
                               batch_wait_ms=b.max_wait * 1000, across_requests=b.max_requests is None,
                               batches=b.stats["batches"], windows=b.stats["windows"], open_calls=len(_state["calls"])))


@app.post("/v1/analyze")
async def analyze(file: UploadFile = File(...), mode: str = Form("window"),
                  channel: str = Form("none"), caller_number: str = Form(""), user=Depends(need(connector=True))):
    if mode not in ("window", "full"):
        raise HTTPException(422, "mode phải là 'window' hoặc 'full'")
    if channel not in CHANNELS:
        raise HTTPException(422, f"channel phải thuộc {list(CHANNELS)}")
    if not available_channels()[channel]:
        raise HTTPException(422, f"ffmpeg trên máy chủ không có encoder cho kênh {channel}")
    det = _get()
    data = await file.read()
    category = _classify(caller_number)
    policy = _policy_for(channel)

    if category == "blacklist":
        # Stopped by the pre-filter: the model is not run.
        result = dict(duration_s=None, mode=mode, n_segments=0, score=None, p_bonafide=None,
                      risk_level="HIGH", recommended_action=RULES["blacklist"],
                      most_suspicious_segment=None, segments=[], policy=policy.to_dict(),
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
            raise HTTPException(415, f"Không thể giải mã file âm thanh: {e}")
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
            result = det.result(wav, mode, spans, logit_rows, (time.perf_counter() - t1) * 1000, policy)
        else:
            result = await run_in_threadpool(det.analyze, wav, mode, policy)
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

    def __init__(self, call_id, category, channel):
        self.id, self.category, self.channel = call_id, category, channel
        self.buf, self.offset, self.next_start = np.zeros(0, np.float32), 0, 0
        self.scores, self.skipped, self.sha = [], 0, hashlib.sha256()
        self.received, self.started = 0, time.time()
        self.max_level, self.alerted = None, False
        self.rec = None if os.environ.get("VG_RECORD", "flagged") == "none" else open(_rec_dir() / f"{call_id}.pcm", "wb")


def _decision(call):
    if call.category == "blacklist":
        return dict(score=None, spoof_percent=None, risk_level="HIGH", recommended_action=RULES["blacklist"],
                    screened_by="prefilter")
    if not call.scores:
        return dict(score=None, spoof_percent=None, risk_level=None, recommended_action=None, screened_by="model")
    policy = _policy_for(call.channel)
    score = float(np.mean(call.scores))
    level, action = policy.decide(score)
    escalated = False
    if call.category in ("unknown", "hidden"):
        new_level, action = at_least_medium(level)
        escalated, level = new_level != level, new_level
    return dict(score=round(score, 4), spoof_percent=round(policy.spoof_percent(score)[0], 1), risk_level=level,
                recommended_action=action, screened_by="model", escalated_by_prefilter=escalated)


def _summary(call, dec):
    return dict(call_id=call.id, audio_s=round(call.received / SR, 2), n_windows=len(call.scores),
                skipped_silent=call.skipped, max_level=call.max_level, caller_category=call.category,
                channel=call.channel, **dec)


def _after_decision(call, dec):
    """Store and publish the call's state; alert once when it first reaches HIGH."""
    level = dec["risk_level"]
    if LEVEL_RANK[level] > LEVEL_RANK[call.max_level]:
        call.max_level = level
    _store().update_call(call.id, n_windows=len(call.scores), score=dec["score"], spoof_percent=dec["spoof_percent"],
                         risk_level=level, max_level=call.max_level, screened_by=dec["screened_by"],
                         duration_s=round(call.received / SR, 2))
    if level == "HIGH" and not call.alerted:
        call.alerted = True
        ev = dict(type="alert", call=_store().get_call(call.id))
        BUS.publish(ev)
        _webhook(ev)


@app.post("/v1/calls")
async def open_call(caller_number: str = Form(""), channel: str = Form("none"), agent: str = Form(""),
                    recording_ref: str = Form(""), user=Depends(need(connector=True))):
    """Open a call. channel is the line's codec (the audio arrives already through it), so it selects
    the channel's thresholds; it is not simulated."""
    _get()
    if channel not in CHANNELS:
        raise HTTPException(422, f"channel phải thuộc {list(CHANNELS)}")
    call_id = uuid.uuid4().hex
    call = _Call(call_id, _classify(caller_number), channel)
    _state["calls"][call_id] = call
    _store().start_call(call_id, caller_number, call.category, channel, agent, recording_ref)
    dec = _decision(call)
    if call.category == "blacklist":
        _after_decision(call, dec)
    BUS.publish(dict(type="call_started", call=dict(_store().get_call(call_id), policy=_policy_for(channel).to_dict())))
    return _summary(call, dec)


@app.post("/v1/calls/{call_id}/audio")
async def call_audio(call_id: str, request: Request, user=Depends(need(connector=True))):
    """Body: raw 16 kHz mono signed 16-bit PCM. Scores every new full window (4 s, hop 2 s)."""
    det = _get()
    call = _state["calls"].get(call_id)
    if call is None:
        raise HTTPException(404, "Không tìm thấy thông tin cuộc gọi")
    data = await request.body()
    if len(data) % 2:
        raise HTTPException(422, "PCM 16-bit phải có số byte chẵn")
    call.sha.update(data)
    call.received += len(data) // 2
    if call.rec is not None:
        call.rec.write(data)
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
            policy = _policy_for(call.channel)
            for st, row in zip(starts, rows):
                call.scores.append(float(row[1]))
                new.append(dict(start=round(st / SR, 2), end=round((st + win) / SR, 2), score=round(float(row[1]), 4),
                                level=policy.decide(float(row[1]))[0], latency_ms=round(ms, 1)))
            _store().add_windows(call.id, new)
    dec = _decision(call)
    if new:
        _after_decision(call, dec)
        BUS.publish(dict(type="call_update", windows=new, **_summary(call, dec)))
    return dict(new_windows=new, **_summary(call, dec))


@app.post("/v1/calls/{call_id}/end")
async def end_call(call_id: str, user=Depends(need(connector=True))):
    det = _get()
    call = _state["calls"].pop(call_id, None)
    if call is None:
        raise HTTPException(404, "không tìm thấy cuộc gọi")
    if call.category != "blacklist" and not call.scores and call.received >= MIN_SECONDS * SR:
        # Call shorter than one window: score the tiled audio once, as for a short file.
        tail = call.buf if len(call.buf) else np.zeros(1, np.float32)
        if rms_dbfs(tail) >= MIN_RMS_DBFS:
            call.scores.append(float((await _state["batcher"].submit([repeat_pad(tail, det.win)]))[0][1]))
    dec = _decision(call)
    if dec["risk_level"]:
        _after_decision(call, dec)
    _store().update_call(call.id, status="ended", ended_at=time.time(), duration_s=round(call.received / SR, 2),
                         recording=int(_finish_recording(call)))
    result = dict(request_id=call_id, duration_s=round(call.received / SR, 2), mode="stream",
                  n_segments=len(call.scores), skipped_silent=call.skipped, caller_category=call.category,
                  channel_simulated=call.channel, timing_ms=dict(wall_s=round(time.time() - call.started, 2)), **dec)
    _audit(result, call.sha.hexdigest(), None)
    BUS.publish(dict(type="call_ended", call=_store().get_call(call_id)))
    return result


# ---------------------------------------------------------------- console API

@app.get("/v1/events")
async def events(request: Request, user=Depends(need())):
    """Server-sent events: call_started, call_update, alert, call_ended, action, verdict, numbers."""
    q = asyncio.Queue()
    BUS.subs.add(q)

    async def stream():
        try:
            yield "retry: 2000\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            BUS.subs.discard(q)
    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/v1/calls")
def list_calls(status: str = None, min_level: str = None, unreviewed: bool = False, limit: int = 100,
               hours: float = None, user=Depends(need())):
    if min_level and min_level not in LEVELS:
        raise HTTPException(422, "min_level phải là LOW, MEDIUM hoặc HIGH")
    since = time.time() - hours * 3600 if hours else None
    return _store().list_calls(status, min_level, unreviewed, limit, since)


@app.get("/v1/calls/{call_id}/recording")
def call_recording(call_id: str, user=Depends(need())):
    """The call's audio (16 kHz mono WAV), kept for flagged calls only (VG_RECORD)."""
    f = _rec_dir() / f"{call_id}.wav"
    if not f.exists():
        raise HTTPException(404, "Cuộc gọi này không có file ghi âm")
    return FileResponse(f, media_type="audio/wav")


@app.get("/v1/calls/{call_id}")
def get_call(call_id: str, user=Depends(need())):
    c = _store().get_call(call_id)
    if c is None:
        raise HTTPException(404, "Không tìm thấy thông tin cuộc gọi")
    c["policy"] = _policy_for(c["channel"] or "none").to_dict()
    return c


ACTIONS = {"continue": "Đã xác minh · Tiếp tục giao dịch", "verify": "Yêu cầu xác thực bổ sung",
           "hold": "Tạm dừng giao dịch", "transfer": "Chuyển phòng chống gian lận"}


@app.post("/v1/calls/{call_id}/action")
def call_action(call_id: str, action: str = Form(...), user=Depends(need())):
    """An agent's action on a call; hold and transfer are sent to the webhook (CRM, core banking)."""
    if action not in ACTIONS:
        raise HTTPException(422, f"action phải thuộc {list(ACTIONS)}")
    if _store().get_call(call_id) is None:
        raise HTTPException(404, "Không tìm thấy thông tin cuộc gọi")
    _store().update_call(call_id, action=action, action_at=time.time(), agent=user["username"])
    ev = dict(type="action", call=_store().get_call(call_id))
    BUS.publish(ev)
    if action in ("hold", "transfer"):
        _webhook(ev)
    return ev["call"]


@app.post("/v1/calls/{call_id}/verdict")
def call_verdict(call_id: str, verdict: str = Form(...), note: str = Form(""), user=Depends(need())):
    """Fraud analyst's conclusion. Confirmed fraud blacklists the caller (when the number was given)."""
    if verdict not in ("fraud", "genuine"):
        raise HTTPException(422, "verdict phải là 'fraud' hoặc 'genuine'")
    if _store().get_call(call_id) is None:
        raise HTTPException(404, "Không tìm thấy thông tin cuộc gọi")
    _store().update_call(call_id, verdict=verdict, verdict_note=note, verdict_by=user["username"], verdict_at=time.time())
    key, masked = _store().caller_key_of(call_id)
    if verdict == "fraud" and key:
        _store().add_number(key=key, masked=masked, lst="blacklist", source=f"Xác nhận gian lận từ cuộc gọi {call_id[:8]}",
                            note=note)
        BUS.publish(dict(type="numbers"))
    ev = dict(type="verdict", call=_store().get_call(call_id))
    BUS.publish(ev)
    return ev["call"]


@app.get("/v1/numbers")
def list_numbers(list: str = "blacklist", user=Depends(need())):
    if list == "suggested":
        return _store().suggestions()
    if list not in ("blacklist", "registered"):
        raise HTTPException(422, "list phải là blacklist, registered hoặc suggested")
    return _store().numbers(list)


@app.post("/v1/numbers")
def add_number(number: str = Form(""), list: str = Form("blacklist"), note: str = Form(""), key: str = Form(""),
               user=Depends(need())):
    """Add a number by its digits, or a suggested caller by its key."""
    if list not in ("blacklist", "registered"):
        raise HTTPException(422, "list phải là blacklist hoặc registered")
    try:
        if key:
            masked = next((s["masked"] for s in _store().suggestions(min_high=1) if s["key"] == key), "số đã che")
            _store().add_number(key=key, masked=masked, lst=list, source="Duyệt từ danh sách đề xuất", note=note)
        else:
            _store().add_number(number, list, source="Thêm thủ công", note=note)
    except ValueError as e:
        raise HTTPException(422, str(e))
    BUS.publish(dict(type="numbers"))
    return dict(ok=True)


@app.delete("/v1/numbers/{key}")
def remove_number(key: str, list: str = "blacklist", user=Depends(need())):
    if not _store().remove_number(key, list):
        raise HTTPException(404, "Số điện thoại không có trong danh sách")
    BUS.publish(dict(type="numbers"))
    return dict(ok=True)


@app.get("/v1/stats")
def stats(hours: float = 24, user=Depends(need())):
    out = _store().stats(time.time() - hours * 3600)
    out["open_calls"] = len(_state.get("calls", {}))
    out["hours"] = hours
    return out


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
