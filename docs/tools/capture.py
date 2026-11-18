"""Screenshots of the console for the user guide, with the boxes to annotate.

Needs a running service (VG_API_KEY set), the system Python with Playwright and Google Chrome, and a JSON file
of test accounts ({"admin": admin password, "api_key": connector key}).
Every screenshot is the 1600x900 viewport at twice the pixel density; for each one, the position of the parts
to explain is written to boxes.json as fractions of the image (x, y, width, height), which make_figures.py
turns into numbered boxes in LaTeX.

  python3 docs/tools/capture.py --url http://localhost:8766 --accounts accounts.json \\
      --samples samples --out docs/user-guide/img
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
VENV_PY = os.environ.get("VG_PYTHON", sys.executable)
W, H = 1600, 900


def session(url, user, pw):
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))
    op.open(f"{url}/v1/auth/login", urllib.parse.urlencode(dict(username=user, password=pw)).encode())
    return op


def simulate(args, acc, calls, duration, seed, wait):
    env = dict(os.environ, VG_API_KEY=acc["api_key"], VG_SIM_PASSWORD=acc["admin"])
    cmd = [VENV_PY, "-m", "voiceguard.simulate", "--url", args.url, "--calls", str(calls), "--duration", str(duration),
           "--ramp", "3", "--seed", str(seed), "--user", "admin", "--samples", args.samples, "--spoof_share", "0.3"]
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if wait:
        p.wait()
    return p


class Shooter:
    def __init__(self, page, out, only=()):
        self.page, self.out, self.boxes, self.only = page, out, {}, only
        self.W, self.H = page.viewport_size["width"], page.viewport_size["height"]

    def rect(self, target):
        """Bounding box of a selector (first match) or of a JS expression returning an element, in the viewport."""
        if target.startswith("js:"):
            r = self.page.evaluate("(() => { const e = " + target[3:] + "; if (!e) return null; const b = e.getBoundingClientRect(); "
                                   "return [b.x, b.y, b.width, b.height]; })()")
        else:
            r = self.page.evaluate("s => { const e = document.querySelector(s); if (!e) return null; const b = e.getBoundingClientRect(); "
                                   "return [b.x, b.y, b.width, b.height]; }", target)
        return r

    def shot(self, name, parts, pad=4):
        if self.only and name not in self.only:
            return
        rows = []
        for i, t in enumerate(parts, 1):
            r = self.rect(t)
            if not r or r[2] <= 0 or r[3] <= 0:
                print(f"  {name}: part {i} not found ({t})")
                continue
            x, y, w, h = r[0] - pad, r[1] - pad, r[2] + 2 * pad, r[3] + 2 * pad
            W, H = self.W, self.H
            x, y = max(0, x), max(0, y)
            w, h = min(w, W - x), min(h, H - y)
            if w <= 0 or h <= 0:
                print(f"  {name}: part {i} outside the screen ({t})")
                continue
            rows.append([i, round(x / W, 4), round(1 - (y + h) / H, 4), round(w / W, 4), round(h / H, 4)])
        self.page.screenshot(path=str(self.out / f"{name}.png"))
        self.boxes[name] = rows
        print(f"{name}: {len(rows)} boxes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--accounts", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="comma-separated screenshot groups to take (default all)")
    ap.add_argument("--no_seed", action="store_true", help="the database already holds the first wave and the verdicts")
    args = ap.parse_args()
    acc = json.load(open(args.accounts, encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # First wave of calls, then the fraud team's verdicts on some of them.
    an = session(args.url, "admin", acc["admin"])
    if not args.no_seed:
        simulate(args, acc, calls=24, duration=16, seed=21, wait=True)
        flagged = json.loads(an.open(f"{args.url}/v1/calls?min_level=MEDIUM&status=ended&limit=50").read())
        for c, v in zip(flagged[:3], ["fraud", "genuine", "fraud"]):
            an.open(f"{args.url}/v1/calls/{c['id']}/verdict", urllib.parse.urlencode(dict(verdict=v, note="Đã gọi lại số chính chủ" if v == "genuine" else "Khách hàng xác nhận không gọi")).encode())
        an.open(f"{args.url}/v1/numbers", urllib.parse.urlencode(dict(number="0901234567", list="blacklist", note="Báo cáo từ khách hàng")).encode())
    def end_active():
        for c in json.loads(an.open(f"{args.url}/v1/calls?status=active&limit=500").read()):
            an.open(f"{args.url}/v1/calls/{c['id']}/end", b"")
    end_active()                                                             # calls left open by an earlier run
    # Second wave stays on the line while the live screens are captured.
    live = simulate(args, acc, calls=9, duration=150, seed=42, wait=False)
    time.sleep(14)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        ctx = browser.new_context(viewport=dict(width=W, height=H), device_scale_factor=2, locale="vi-VN")
        page = ctx.new_page()
        s = Shooter(page, out, [x for x in args.only.split(",") if x])

        # Login
        page.goto(f"{args.url}/login")
        page.wait_for_timeout(800)
        s.shot("login", ["#u", "#p", "#eye", "#go"])
        page.fill("#u", "admin")
        page.fill("#p", "sai-mat-khau")
        page.click("#go")
        page.wait_for_timeout(600)
        s.shot("login_error", ["#err"])
        page.fill("#p", acc["admin"])
        page.click("#go")
        page.wait_for_url(f"{args.url}/")
        page.wait_for_timeout(2500)

        # Live calls
        page.goto(f"{args.url}/#live")
        page.wait_for_timeout(5000)
        s.shot("layout", ["nav", ".topbar h1", "#clock", ".user", "#logout", "#view"], pad=2)
        s.shot("live", [".toolbar", "js:document.querySelector('#view table thead')", "js:document.querySelector('#view tbody tr.HIGH, #view tbody tr.MEDIUM')",
                        "js:document.querySelector('#view tbody svg.spark')?.closest('td')", "js:document.querySelector('#view tbody .pctcell')",
                        "js:document.querySelector('#view tbody .lvl')", "js:document.querySelector('.section h2')"])
        # Drawer of a live call
        page.evaluate("openCall([...active.values()].sort((a,b)=>(a.risk_level==='HIGH'?-1:1))[0].id)")
        page.wait_for_timeout(1200)
        s.shot("drawer_live", [".drawer .dh", ".drawer .kv", ".drawer .verdict", "js:document.querySelector('.drawer svg.spark')",
                               "js:document.querySelector('.drawer [data-act]')?.parentElement",
                               "js:document.querySelector('.drawer [data-ver]')?.parentElement",
                               "js:[...document.querySelectorAll('.drawer .kv')].pop()"])
        page.evaluate("closeDrawer()")

        # Review queue and a flagged call with its recording
        page.goto(f"{args.url}/#review")
        page.wait_for_timeout(1500)
        s.shot("review", [".tabs", "#refresh", "js:document.querySelector('#view table thead')",
                          "js:document.querySelector('#view tbody tr td:last-child')"])
        page.evaluate("document.querySelector('#view tbody tr[data-id]')?.click()")
        page.wait_for_timeout(1500)
        page.evaluate("document.querySelector('#recAudio')?.scrollIntoView({block:'center'})")
        page.wait_for_timeout(500)
        s.shot("drawer_review", ["#recAudio", "#segs", ".suspects", "#vnote",
                                 "js:document.querySelector('.drawer [data-ver]')?.parentElement"])
        page.evaluate("closeDrawer()")

        # Numbers
        page.goto(f"{args.url}/#numbers")
        page.wait_for_timeout(1500)
        s.shot("numbers", [".tabs", "#addForm", "js:document.querySelector('#view table thead')", "[data-del]"])
        page.evaluate("document.querySelector('[data-t=\"suggested\"]').click()")
        page.wait_for_timeout(1200)
        s.shot("numbers_suggested", [".tabs", "js:document.querySelector('#view table')", "[data-add]"])

        # Overview
        page.goto(f"{args.url}/#overview")
        page.wait_for_timeout(1800)
        s.shot("overview", [".kpis", "js:document.querySelectorAll('.section')[0]", "js:document.querySelectorAll('.section')[1]"])

        # File check, inside the console, on a taller screen so that the input and the result fit
        page.set_viewport_size(dict(width=W, height=1240))
        s.W, s.H = W, 1240
        page.goto(f"{args.url}/#check")
        page.wait_for_timeout(2000)
        fr = page.frame_locator("#checkFrame")
        fr.locator("[data-tab='lib']").click()
        page.wait_for_timeout(400)
        fr.locator(".sample").nth(1).click()
        page.wait_for_timeout(1200)
        fr.locator(".src").nth(1).click()

        def frame_part(sel):
            return ("js:(() => { const f = document.querySelector('#checkFrame'); const e = f.contentDocument.querySelector(" + json.dumps(sel) +
                    "); if (!e) return null; const a = f.getBoundingClientRect(), b = e.getBoundingClientRect(); "
                    "return {getBoundingClientRect: () => ({x:a.x+b.x, y:a.y+b.y, width:b.width, height:b.height})}; })()")
        s.shot("check_input", [frame_part(".seg"), frame_part(".samples"), frame_part("#player"), frame_part("#sources"), frame_part("#go")])
        fr.locator("#go").click()
        page.wait_for_timeout(4500)
        s.shot("check_result", [frame_part(".verdict"), frame_part(".gauge"), frame_part(".todo"), frame_part(".legend"),
                                frame_part(".truth"), frame_part("details.tech")])

        browser.close()

    live.terminate()
    end_active()
    f = out / "boxes.json"
    boxes = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    boxes.update(s.boxes)
    f.write_text(json.dumps(boxes, indent=1), encoding="utf-8")
    print("boxes written to", out / "boxes.json")


if __name__ == "__main__":
    main()
