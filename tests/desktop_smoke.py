"""Smoke test of a built / installed stand-alone edition of MAMC ML Studio.

Starts the program exactly as its desktop icon does (but with MAMC_NO_BROWSER=1, so no window opens), then
  1. waits until the program answers,
  2. connects like a program window (a WebSocket, as the browser window does) and checks the program keeps running,
  3. disconnects ("closes the window") and checks that the program then stops by itself.

Usage:  python tests/desktop_smoke.py [--port 8799] <command that starts the program> [arguments ...]
  Windows:  python tests/desktop_smoke.py "C:/Program Files/.../MAMC ML Studio.exe"
  macOS:    python tests/desktop_smoke.py "/Volumes/MAMC ML Studio/MAMC ML Studio.app/Contents/MacOS/MAMCMLStudio"
  source:   python tests/desktop_smoke.py python desktop_launcher.py
"""
import argparse
import base64
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

NAME = "MAMC ML Studio"
if os.name == "nt":
    DATA = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")) / NAME
elif sys.platform == "darwin":
    DATA = Path.home() / "Library" / "Application Support" / NAME
else:
    DATA = Path.home() / ".local" / "share" / "mamc-ml-studio"


def get(url, timeout=2.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return None, ""


def healthy(port):
    status, body = get(f"http://127.0.0.1:{port}/_stcore/health")
    return status == 200 and "ok" in body


def open_websocket(port):
    s = socket.create_connection(("127.0.0.1", port), timeout=10)
    key = base64.b64encode(os.urandom(16)).decode()
    s.sendall((f"GET /_stcore/stream HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
               f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Protocol: streamlit\r\n"
               f"Origin: http://127.0.0.1:{port}\r\n\r\n").encode())
    first = s.recv(4096).decode("latin-1").split("\r\n", 1)[0]
    if " 101 " not in first + " ":
        raise SystemExit(f"FAIL: the program refused the window connection: {first}")
    return s


def show_log():
    log = DATA / "last_run.log"
    print(f"----- {log} -----")
    print(log.read_text(encoding="utf-8", errors="replace")[-6000:] if log.exists() else "(no log file)")
    print("-----")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8799)
    ap.add_argument("--start-timeout", type=int, default=300)
    ap.add_argument("command", nargs="+")
    a = ap.parse_args()
    port = a.port
    if healthy(port):
        raise SystemExit(f"FAIL: something is already running on port {port}")
    env = dict(os.environ, MAMC_NO_BROWSER="1", MAMC_PORT=str(port), MAMC_IDLE_EXIT="5")
    t0 = time.time()
    proc = subprocess.Popen(a.command, env=env)
    ok = False
    try:
        while time.time() - t0 < a.start_timeout:
            if healthy(port):
                break
            time.sleep(1)
        else:
            raise SystemExit(f"FAIL: the program did not start within {a.start_timeout} s")
        print(f"ok  program answers after {time.time() - t0:.1f} s")
        status, page = get(f"http://127.0.0.1:{port}/", 10)
        if status != 200 or "<div id=\"root\"" not in page:
            raise SystemExit(f"FAIL: the start page was not served (HTTP {status})")
        print("ok  start page served")
        ws = open_websocket(port)
        print("ok  window connected")
        time.sleep(12)
        if not healthy(port):
            raise SystemExit("FAIL: the program stopped although a window was open")
        print("ok  program keeps running while the window is open")
        ws.close()
        t1 = time.time()
        while time.time() - t1 < 60:
            if not healthy(port):
                break
            time.sleep(1)
        else:
            raise SystemExit("FAIL: the program did not stop after the window was closed")
        print(f"ok  program stopped by itself {time.time() - t1:.1f} s after the window was closed")
        time.sleep(2)
        if (DATA / "running.json").exists():
            raise SystemExit("FAIL: running.json was not removed")
        ok = True
    finally:
        show_log()
        if proc.poll() is None:
            try:
                proc.wait(15)
            except subprocess.TimeoutExpired:
                proc.kill()
    print("SMOKE TEST PASSED" if ok else "SMOKE TEST FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
