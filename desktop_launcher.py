"""Starts the installed (stand-alone) edition of MAMC ML Studio.

Used by "MAMC ML Studio.exe" on Windows and by "MAMC ML Studio.app" on a Mac. It uses the Python that is bundled
with the installer, so nothing has to be installed on the computer and no internet is needed.

What it does:
  1. If MAMC ML Studio is already running, it simply opens another window.
  2. Otherwise it shows a small "starting" window, starts the program on a free local port (127.0.0.1 only, so the
     program can never be reached from another computer) and opens it in its own window (Edge / Chrome "app" window),
     or in the normal web browser if neither is installed.
  3. When the last window is closed, it stops the program completely.
Everything that happens is written to a log file (see LOG below) to help if something goes wrong.

Environment variables (used by the automatic tests):
  MAMC_NO_BROWSER=1   do not open any window
  MAMC_PORT=8799      use this port
  MAMC_IDLE_EXIT=30   stop this many seconds after the last window is closed (default 30)
"""
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

NAME = "MAMC ML Studio"
APP_DIR = Path(__file__).resolve().parent
IS_WIN = os.name == "nt"
IS_MAC = sys.platform == "darwin"

if IS_WIN:
    DATA = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")) / NAME
elif IS_MAC:
    DATA = Path.home() / "Library" / "Application Support" / NAME
else:
    DATA = Path.home() / ".local" / "share" / "mamc-ml-studio"
DATA.mkdir(parents=True, exist_ok=True)
LOG = DATA / "last_run.log"
STATE = DATA / "running.json"
NO_BROWSER = os.environ.get("MAMC_NO_BROWSER") == "1"
IDLE_EXIT = float(os.environ.get("MAMC_IDLE_EXIT", "30"))
START_TIMEOUT = 240

_log_file = open(LOG, "w", encoding="utf-8", buffering=1)


def log(msg):
    _log_file.write(time.strftime("%H:%M:%S ") + str(msg) + "\n")


# ---------------------------------------------------------------------------- messages
def message(text, error=True):
    log(("ERROR: " if error else "") + text)
    if NO_BROWSER:
        if sys.stderr:
            print(text, file=sys.stderr)
        return
    if IS_WIN:
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, text, NAME, 0x10 if error else 0x40)
            return
        except Exception:  # noqa: BLE001
            pass
    if IS_MAC:
        safe = text.replace('"', "'")
        subprocess.run(["osascript", "-e", f'display dialog "{safe}" buttons {{"OK"}} with title "{NAME}" '
                                            f'with icon {"stop" if error else "note"}'], check=False)
        return
    if sys.stderr:
        print(text, file=sys.stderr)


class Splash:
    """A small 'MAMC ML Studio is starting...' window (Windows / Linux). Never blocks the start if Tk is missing."""

    def __init__(self):
        self.root = None
        if NO_BROWSER or IS_MAC:
            return
        try:
            import tkinter as tk
            from tkinter import ttk
            root = tk.Tk()
            root.title(NAME)
            root.resizable(False, False)
            try:
                ico = APP_DIR / "assets" / "mamc_ml_studio.ico"
                if IS_WIN and ico.exists():
                    root.iconbitmap(str(ico))
            except Exception:  # noqa: BLE001
                pass
            w, h = 420, 150
            root.geometry(f"{w}x{h}+{(root.winfo_screenwidth() - w) // 2}+{(root.winfo_screenheight() - h) // 3}")
            root.configure(bg="#ffffff")
            tk.Label(root, text=NAME, font=("Segoe UI", 16, "bold"), fg="#7b1c2e", bg="#ffffff").pack(pady=(18, 2))
            self.label = tk.Label(root, text="Starting - please wait a few seconds ...", font=("Segoe UI", 10),
                                  fg="#52514e", bg="#ffffff")
            self.label.pack()
            bar = ttk.Progressbar(root, mode="indeterminate", length=320)
            bar.pack(pady=14)
            bar.start(12)
            root.attributes("-topmost", True)
            root.update()
            self.root = root
        except Exception as e:  # noqa: BLE001
            log(f"no splash window ({e})")

    def pump(self, text=None):
        if self.root is not None:
            try:
                if text:
                    self.label.configure(text=text)
                self.root.update()
            except Exception:  # noqa: BLE001
                self.root = None

    def close(self):
        if self.root is not None:
            try:
                self.root.destroy()
            except Exception:  # noqa: BLE001
                pass
            self.root = None


# ---------------------------------------------------------------------------- server
def healthy(url, timeout=1.0) -> bool:
    try:
        with urllib.request.urlopen(url + "/_stcore/health", timeout=timeout) as r:
            return r.status == 200 and b"ok" in r.read(100)
    except Exception:  # noqa: BLE001
        return False


def free_port() -> int:
    if os.environ.get("MAMC_PORT"):
        return int(os.environ["MAMC_PORT"])
    for p in range(8501, 8600):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def server_python() -> str:
    exe = Path(sys.executable)
    if IS_WIN and exe.name.lower() == "pythonw.exe" and (exe.parent / "python.exe").exists():
        return str(exe.parent / "python.exe")  # console Python, but started without a window (see CREATE_NO_WINDOW)
    return str(exe)


def start_server(port):
    cmd = [server_python(), "-m", "streamlit", "run", str(APP_DIR / "app.py"),
           "--server.port", str(port), "--server.address", "127.0.0.1", "--server.headless", "true",
           "--browser.gatherUsageStats", "false", "--global.developmentMode", "false"]
    env = dict(os.environ, MAMC_DESKTOP="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    kw = {}
    if IS_WIN:
        kw["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    log("starting server: " + " ".join(cmd))
    return subprocess.Popen(cmd, cwd=str(APP_DIR), env=env, stdout=_log_file, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, **kw)


def window_connections(server_pid, port):
    """Number of open browser connections to the program (each open window keeps one), or None if unknown."""
    try:
        import psutil
    except ImportError:
        return None
    try:
        proc = psutil.Process(server_pid)
        conns = proc.net_connections(kind="tcp") if hasattr(proc, "net_connections") else proc.connections(kind="tcp")
    except Exception as e:  # noqa: BLE001
        try:  # some systems only allow the computer-wide list (it needs no special rights on Windows)
            conns = psutil.net_connections(kind="tcp")
        except Exception:  # noqa: BLE001
            log(f"cannot read connections ({e})")
            return None
    return sum(1 for c in conns if c.status == psutil.CONN_ESTABLISHED and c.laddr and c.laddr.port == port)


# ---------------------------------------------------------------------------- window
def find_app_browser():
    if IS_WIN:
        roots = [os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"), os.environ.get("LOCALAPPDATA")]
        cands = []
        for r in filter(None, roots):
            cands += [Path(r) / "Microsoft" / "Edge" / "Application" / "msedge.exe",
                      Path(r) / "Google" / "Chrome" / "Application" / "chrome.exe"]
    elif IS_MAC:
        cands = [Path(p) for p in ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                                   "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
                                   str(Path.home() / "Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
                                   "/Applications/Chromium.app/Contents/MacOS/Chromium")]
    else:
        import shutil
        cands = [Path(p) for p in (shutil.which("google-chrome"), shutil.which("chromium"), shutil.which("chromium-browser"),
                                   shutil.which("microsoft-edge")) if p]
    for c in cands:
        if c.exists():
            return str(c)
    return None


def open_window(url):
    if NO_BROWSER:
        return None
    b = find_app_browser()
    if b:
        prof = DATA / "window-profile"
        prof.mkdir(exist_ok=True)
        args = [b, f"--app={url}", f"--user-data-dir={prof}", "--window-size=1440,900", "--no-first-run",
                "--no-default-browser-check", "--disable-features=Translate"]
        try:
            log("opening window: " + b)
            return subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:  # noqa: BLE001
            log(f"could not start {b}: {e}")
    log("opening the default web browser")
    webbrowser.open(url)
    return None


# ---------------------------------------------------------------------------- main
def read_state():
    try:
        return json.loads(STATE.read_text())
    except Exception:  # noqa: BLE001
        return None


def main():
    log(f"{NAME} launcher, Python {sys.version.split()[0]}, app folder {APP_DIR}")
    st = read_state()
    if st and healthy(st.get("url", ""), 2):
        log("already running - opening another window")
        open_window(st["url"])
        return 0

    splash = Splash()
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    proc = start_server(port)
    t0 = time.time()
    while not healthy(url, 0.8):
        splash.pump("Starting - please wait ..." if time.time() - t0 < 15 else
                    "Still starting (the first start after installation can take up to a minute) ...")
        if proc.poll() is not None:
            splash.close()
            message(f"{NAME} could not start.\n\nDetails are saved in:\n{LOG}")
            return 1
        if time.time() - t0 > START_TIMEOUT:
            splash.close()
            proc.kill()
            message(f"{NAME} did not start within {START_TIMEOUT // 60} minutes.\n\nDetails are saved in:\n{LOG}")
            return 1
        time.sleep(0.25)
    log(f"server ready at {url} after {time.time() - t0:.1f} s")
    STATE.write_text(json.dumps({"url": url, "pid": proc.pid, "port": port}))
    win = open_window(url)
    for _ in range(12):  # keep the splash visible until the window has had time to appear
        splash.pump("Opening the window ...")
        time.sleep(0.25)
    splash.close()

    # ---- watch: stop when no window has been connected for IDLE_EXIT seconds
    seen = False
    idle_since = None
    opened = time.time()
    try:
        while proc.poll() is None:
            time.sleep(2)
            n = window_connections(proc.pid, port)
            if n is None:
                # psutil missing: fall back to the app window process (if we have one)
                if win is not None and win.poll() is not None:
                    log("window closed")
                    break
                continue
            if n > 0:
                seen, idle_since = True, None
                continue
            if not seen and time.time() - opened < 600 and not NO_BROWSER:
                continue  # waiting for the first window
            if NO_BROWSER and not seen:
                continue  # automatic tests: keep running until stopped
            idle_since = idle_since or time.time()
            if time.time() - idle_since >= IDLE_EXIT:
                log("no window open any more - stopping")
                break
    except KeyboardInterrupt:
        pass
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(10)
            except Exception:  # noqa: BLE001
                proc.kill()
        try:
            if read_state() and read_state().get("pid") == proc.pid:
                STATE.unlink()
        except Exception:  # noqa: BLE001
            pass
        log("stopped")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001
        import traceback
        log(traceback.format_exc())
        message(f"{NAME} could not start: {e}\n\nDetails are saved in:\n{LOG}")
        sys.exit(1)
