"""Local dashboard for the live bot.

    ../.venv/bin/python dashboard.py            # then open http://127.0.0.1:8765
    ../.venv/bin/python dashboard.py --port 9000 --state /path/to/state

Reads what the bot writes (bot/state/status.json, bot/state/history-<competition>.jsonl, bot/logs/live-*.log) and
edits bot/overrides.json (halt, scale, per-symbol disable / fair-value adjustment). Listens on 127.0.0.1 only.
No network calls to the exchange: the dashboard keeps working when the bot or the network is down.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CASE = ROOT / "hurricane-season-research" / "case.json"
SYMBOLS = ["PORT", "ISLE", "AQUA", "GRID", "FERRY", "RECON", "MAT", "GEN", "HAVEN", "REIN", "ARCH", "REBLD"]


def tail(path: Path, n: int) -> list[str]:
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 64_000))
            lines = f.read().decode(errors="replace").splitlines()
        return lines[-n:]
    except OSError:
        return []


def validate_overrides(o) -> dict:
    if not isinstance(o, dict):
        raise ValueError("overrides must be an object")
    out = {"halt": bool(o.get("halt", False)), "scale": float(o.get("scale", 1.0)), "symbols": {},
           "arb": {"halt": bool((o.get("arb") or {}).get("halt", False))}}
    if not 0 <= out["scale"] <= 1.0:
        raise ValueError("scale must be between 0 and 1")
    for sym, v in (o.get("symbols") or {}).items():
        if sym not in SYMBOLS or not isinstance(v, dict):
            raise ValueError(f"bad symbol entry {sym}")
        e = {}
        if v.get("disable"):
            e["disable"] = True
        if v.get("mult") not in (None, "", 1, 1.0):
            e["mult"] = float(v["mult"])
        if v.get("add") not in (None, "", 0, 0.0):
            e["add"] = float(v["add"])
        if e:
            out["symbols"][sym] = e
    return out


def make_handler(state: Path, logs: Path, overrides: Path):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body, default=float).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                return self.send(200, (HERE / "dashboard.html").read_bytes(), "text/html; charset=utf-8")
            if path == "/api/status":
                f = state / "status.json"
                try:
                    st = json.loads(f.read_text())
                    st["age"] = time.time() - f.stat().st_mtime
                except (OSError, json.JSONDecodeError):
                    st = {"missing": True}
                return self.send(200, st)
            if path == "/api/arb":
                f = state / "arb_status.json"
                try:
                    st = json.loads(f.read_text())
                    st["age"] = time.time() - f.stat().st_mtime
                except (OSError, json.JSONDecodeError):
                    st = {"missing": True}
                return self.send(200, st)
            if path == "/api/history":
                try:
                    comp = json.loads((state / "status.json").read_text()).get("competition_id")
                except (OSError, json.JSONDecodeError):
                    comp = None
                rows = []
                if comp:
                    for line in tail(state / f"history-{comp}.jsonl", 6000):
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
                return self.send(200, rows[::max(1, len(rows) // 1500)])
            if path == "/api/log":
                files = sorted(logs.glob("live-*.log"), key=lambda p: p.stat().st_mtime)
                return self.send(200, {"file": files[-1].name if files else None,
                                       "lines": tail(files[-1], 80) if files else []})
            if path == "/api/case":
                return self.send(200, CASE.read_bytes())
            if path == "/api/overrides":
                try:
                    return self.send(200, json.loads(overrides.read_text()))
                except (OSError, json.JSONDecodeError):
                    return self.send(200, {"halt": False, "scale": 1.0, "symbols": {}})
            return self.send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/overrides":
                return self.send(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length", 0))
                o = validate_overrides(json.loads(self.rfile.read(n)))
            except (ValueError, TypeError, json.JSONDecodeError) as e:
                return self.send(400, {"error": str(e)})
            tmp = overrides.with_suffix(".tmp")
            tmp.write_text(json.dumps(o, indent=2))
            os.replace(tmp, overrides)
            return self.send(200, o)

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--state", default=str(HERE / "state"))
    ap.add_argument("--logs", default=str(HERE / "logs"))
    ap.add_argument("--overrides", default=str(HERE / "overrides.json"))
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(Path(a.state), Path(a.logs), Path(a.overrides)))
    print(f"dashboard on http://127.0.0.1:{a.port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
