"""Starts MAMC ML Studio.

What it does (all automatic):
  1. Creates a private Python environment in the folder '.venv' next to this file (first start only).
  2. Installs the required components listed in requirements.txt (first start only - needs internet).
  3. Starts the app and opens it in your web browser. Everything runs on THIS computer.
Close the black window to stop the app.
"""
import hashlib
import os
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
VENV_PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
REQ = ROOT / "requirements.txt"
MARK = VENV / "mamc_ml_studio_installed.txt"
NEEDED = ["streamlit", "pandas", "numpy", "sklearn", "scipy", "plotly", "matplotlib", "openpyxl", "docx", "joblib"]


def fix_conda_dll_path():
    """Anaconda/Miniconda on Windows: make sure the DLLs of the base installation (e.g. for SSL) are found."""
    if os.name != "nt":
        return
    base = Path(sys.base_prefix)
    if not ((base / "conda-meta").exists() or (base / "Library" / "bin").exists()):
        return
    extra = [base, base / "Library" / "mingw-w64" / "bin", base / "Library" / "usr" / "bin", base / "Library" / "bin",
             base / "Scripts"]
    os.environ["PATH"] = os.pathsep.join([str(p) for p in extra if p.exists()] + [os.environ.get("PATH", "")])
    os.environ["CONDA_DLL_SEARCH_MODIFICATION_ENABLE"] = "1"


def say(msg=""):
    print(msg, flush=True)


def pause_and_exit(code=1):
    try:
        input("\nPress Enter to close this window...")
    except EOFError:
        pass
    sys.exit(code)


def inside_our_venv() -> bool:
    try:
        return Path(sys.prefix).resolve() == VENV.resolve()
    except OSError:
        return False


def missing_packages() -> list:
    out = []
    for m in NEEDED:
        try:
            __import__(m)
        except Exception:  # noqa: BLE001
            out.append(m)
    return out


def install_requirements(python: str) -> bool:
    say("Installing the components MAMC ML Studio needs (first start only; needs internet; 5-10 minutes) ...")
    subprocess.call([python, "-m", "pip", "install", "--upgrade", "pip"])
    rc = subprocess.call([python, "-m", "pip", "install", "-r", str(REQ)])
    return rc == 0


def free_port(start=8501) -> int:
    for p in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return start


def run_app():
    port = free_port()
    url = f"http://localhost:{port}"
    cmd = [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"), "--server.port", str(port),
           "--server.headless", "true", "--server.address", "localhost", "--browser.gatherUsageStats", "false"]
    say("Starting MAMC ML Studio ...")
    proc = subprocess.Popen(cmd, cwd=str(ROOT))
    for _ in range(120):
        try:
            urllib.request.urlopen(url, timeout=1)
            break
        except Exception:  # noqa: BLE001
            if proc.poll() is not None:
                say("The app stopped unexpectedly. See the messages above.")
                pause_and_exit(1)
            time.sleep(0.5)
    webbrowser.open(url)
    say("")
    say("=" * 66)
    say(f"  MAMC ML Studio is running at  {url}")
    say("  If the browser did not open, copy that address into Chrome or Edge.")
    say("  Keep this window open while you work. Close it to stop the app.")
    say("=" * 66)
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()


def main():
    os.chdir(ROOT)
    fix_conda_dll_path()
    say("MAMC ML Studio - Maulana Azad Medical College, New Delhi")
    say("-" * 66)
    if sys.version_info < (3, 10):
        say(f"Python {sys.version.split()[0]} is too old. Please install Python 3.10 or newer from https://www.python.org")
        pause_and_exit(1)

    if not inside_our_venv():
        if not VENV_PY.exists():
            say("Creating a private Python environment for the app (first start only) ...")
            rc = subprocess.call([sys.executable, "-m", "venv", str(VENV)])
            if rc != 0 or not VENV_PY.exists():
                say("Could not create the private environment; using this Python directly instead.")
                if missing_packages() and not install_requirements(sys.executable):
                    say("Installation failed. Check the internet connection and try again.")
                    pause_and_exit(1)
                run_app()
                return
        # restart this launcher inside the private environment
        sys.exit(subprocess.call([str(VENV_PY), str(Path(__file__).resolve())] + sys.argv[1:]))

    req_hash = hashlib.sha256(REQ.read_bytes()).hexdigest()
    if not MARK.exists() or MARK.read_text().strip() != req_hash or missing_packages():
        if not install_requirements(sys.executable):
            say("Installation failed. Check the internet connection (or proxy) and start again.")
            pause_and_exit(1)
        MARK.write_text(req_hash)
    run_app()


if __name__ == "__main__":
    main()
