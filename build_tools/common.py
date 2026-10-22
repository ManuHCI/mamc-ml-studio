"""Shared steps for building the stand-alone (no-installation) editions of MAMC ML Studio.

Each edition carries its own private copy of Python plus every package the program needs, so the user installs
nothing else and needs no internet. The Python used is an official CPython build from the
python-build-standalone project (https://github.com/astral-sh/python-build-standalone) - the same relocatable
Python builds used by the `uv` tool. Package versions are pinned in build_tools/constraints-desktop.txt so every
edition is built and tested with exactly the same libraries.
"""
import json
import os
import re
import shutil
import subprocess
import tarfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP_VERSION = re.search(r'APP_VERSION\s*=\s*"([^"]+)"', (ROOT / "mlstudio" / "__init__.py").read_text(encoding="utf-8")).group(1)
APP_NAME = "MAMC ML Studio"
PY_MINOR = os.environ.get("MAMC_PY_MINOR", "3.12")
PBS_REPO = "astral-sh/python-build-standalone"
CONSTRAINTS = ROOT / "build_tools" / "constraints-desktop.txt"
# what goes into the "app" folder of every edition
APP_ITEMS = ["app.py", "desktop_launcher.py", "mlstudio", "app_pages", "assets", "sample_data", ".streamlit", "LICENSE"]
GUIDE = ROOT / "docs" / "MAMC_ML_Studio_User_Guide.docx"


def log(msg):
    print(f"[build] {msg}", flush=True)


def run(cmd, **kw):
    log(" ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], **kw)


def _open(url, token=False):
    headers = {"User-Agent": "MAMC-ML-Studio-builder"}
    if token and os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["GITHUB_TOKEN"]
        headers["Accept"] = "application/vnd.github+json"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=600)


def fetch(url, token=False, text=False, tries=4):
    for i in range(tries):
        try:
            with _open(url, token) as r:
                data = r.read()
            return data.decode("utf-8", "replace") if text else data
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            log(f"  retrying {url} ({e})")
            time.sleep(5 * (i + 1))


# ---------------------------------------------------------------------------------------------------- Python
def _latest_tag():
    if os.environ.get("PBS_TAG"):
        return os.environ["PBS_TAG"]
    try:  # github.com/<repo>/releases/latest redirects to .../releases/tag/<tag>
        with _open(f"https://github.com/{PBS_REPO}/releases/latest") as r:
            tag = r.geturl().rstrip("/").rsplit("/", 1)[-1]
        if re.fullmatch(r"\d{8}", tag):
            return tag
    except Exception as e:  # noqa: BLE001
        log(f"  could not read the latest release page ({e})")
    return json.loads(fetch(f"https://api.github.com/repos/{PBS_REPO}/releases/latest", token=True, text=True))["tag_name"]


def _asset_names(tag):
    names = set()
    try:
        html = fetch(f"https://github.com/{PBS_REPO}/releases/expanded_assets/{tag}", text=True)
        names = {urllib.parse.unquote(n) for n in re.findall(r'/releases/download/[^/"]+/(cpython-[^"<>\s]+?\.tar\.gz)', html)}
    except Exception as e:  # noqa: BLE001
        log(f"  could not read the asset page ({e})")
    if not names:  # fall back to the GitHub API
        rel = json.loads(fetch(f"https://api.github.com/repos/{PBS_REPO}/releases/tags/{tag}", token=True, text=True))
        page = 1
        while True:
            items = json.loads(fetch(f"https://api.github.com/repos/{PBS_REPO}/releases/{rel['id']}/assets?per_page=100&page={page}",
                                     token=True, text=True))
            names |= {a["name"] for a in items}
            if len(items) < 100:
                break
            page += 1
    return names


def download_python(triple, dest: Path) -> str:
    """Download and unpack CPython <PY_MINOR>.x for `triple` into `dest` (dest/python/...). Returns the version."""
    tag = _latest_tag()
    names = _asset_names(tag)
    pat = re.compile(rf"cpython-({re.escape(PY_MINOR)}\.(\d+))\+{tag}-{re.escape(triple)}-install_only(_stripped)?\.tar\.gz")
    found = [(int(m.group(2)), bool(m.group(3)), n, m.group(1)) for n in names if (m := pat.fullmatch(n))]
    if not found:
        raise SystemExit(f"No Python {PY_MINOR} build for {triple} in release {tag}")
    found.sort()
    _, _, name, version = found[-1]  # newest patch release, stripped build preferred
    url = f"https://github.com/{PBS_REPO}/releases/download/{tag}/{urllib.parse.quote(name)}"
    log(f"downloading Python {version}: {url}")
    archive = dest.parent / name
    archive.write_bytes(fetch(url))
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    with tarfile.open(archive) as tf:
        if hasattr(tarfile, "data_filter"):
            tf.extractall(dest, filter="data")
        else:
            tf.extractall(dest)
    archive.unlink()
    if not (dest / "python").is_dir():
        raise SystemExit("unexpected Python archive layout")
    return version


def install_packages(python_exe: Path, site_packages: Path, platform_tag: str):
    """Install the program's packages (pre-built wheels only, pinned versions) into the bundled Python."""
    log(f"installing packages for {platform_tag} into {site_packages}")
    r = run([python_exe, "-m", "pip", "install", "--disable-pip-version-check", "--no-input", "--no-cache-dir",
             "--only-binary=:all:", "--platform", platform_tag, "--target", site_packages, "--upgrade",
             "-r", ROOT / "requirements.txt", "psutil>=5.9", "-c", CONSTRAINTS])
    if r.returncode:
        raise SystemExit("package installation failed")
    for d in ("bin", "Scripts"):  # command-line wrappers with build-machine paths - not used by the program
        shutil.rmtree(site_packages / d, ignore_errors=True)


def slim(python_dir: Path, site_packages: Path, stdlib: Path):
    """Remove test suites and other files the program never uses (about 150 MB)."""
    before = folder_size(python_dir)
    for top in (site_packages, stdlib):
        for d in sorted(top.rglob("__pycache__"), reverse=True):
            shutil.rmtree(d, ignore_errors=True)
    for pkg in site_packages.iterdir():
        if not pkg.is_dir():
            continue
        for d in sorted(pkg.rglob("tests"), key=lambda p: len(p.parts), reverse=True):
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
    for rel in ("pyarrow/include", "pyarrow/src", "pip"):
        shutil.rmtree(site_packages / rel, ignore_errors=True)
    for f in list(site_packages.glob("pip-*.dist-info")):
        shutil.rmtree(f, ignore_errors=True)
    for rel in ("test", "idlelib", "turtledemo", "lib2to3/tests"):
        shutil.rmtree(stdlib / rel, ignore_errors=True)
    for f in python_dir.rglob("*.pdb"):
        f.unlink()
    log(f"slimmed Python folder: {before / 1e6:.0f} MB -> {folder_size(python_dir) / 1e6:.0f} MB")


def stage_app(app_dir: Path):
    log(f"copying the program into {app_dir}")
    shutil.rmtree(app_dir, ignore_errors=True)
    app_dir.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")
    for item in APP_ITEMS:
        src = ROOT / item
        if src.is_dir():
            shutil.copytree(src, app_dir / item, ignore=ignore)
        else:
            shutil.copy2(src, app_dir / item)


def precompile(python_exe: Path, *folders: Path):
    """Pre-compile all .py files so the first start is quick (hash-based .pyc: valid whatever the file dates)."""
    log("pre-compiling Python files")
    run([python_exe, "-m", "compileall", "-q", "-j", "0", "--invalidation-mode", "unchecked-hash", *folders],
        stdout=subprocess.DEVNULL)


def check_imports(python_exe: Path, app_dir: Path):
    """The bundled Python must import every library the program uses, without any other Python on the PATH."""
    code = ("import streamlit, sklearn, pandas, numpy, scipy, plotly, matplotlib, openpyxl, xlrd, docx, joblib, psutil, pyarrow;"
            "import sys; sys.path.insert(0, r'%s'); import mlstudio;"
            "print('imports ok - Python', sys.version.split()[0], '- streamlit', streamlit.__version__, '- scikit-learn',"
            " sklearn.__version__, '- pandas', pandas.__version__)") % app_dir
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    r = run([python_exe, "-I", "-c", code], env=env)
    if r.returncode:
        raise SystemExit("the bundled Python cannot import the program's libraries")


def folder_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink())
