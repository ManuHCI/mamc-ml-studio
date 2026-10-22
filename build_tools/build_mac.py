"""
Build the stand-alone macOS edition of MAMC ML Studio: a .dmg containing "MAMC ML Studio.app".

The .app contains its own Python and every package the program needs, so the user installs nothing else and the
program works fully offline. Works on macOS 12 (Monterey) and newer.

Run on a Mac (an Apple-silicon Mac makes the Apple-silicon edition, an Intel Mac the Intel edition):
    python3 build_tools/build_mac.py
Needs: Python 3.10+ (only to run this script), internet, the Xcode command-line tools (sips, iconutil, codesign, hdiutil)
Output: dist/MAMC_ML_Studio_<version>_macOS-AppleSilicon.dmg   or   ..._macOS-Intel.dmg
"""
import os
import platform
import plistlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (APP_NAME, APP_VERSION, GUIDE, PY_MINOR, ROOT, check_imports, download_python,  # noqa: E402
                    folder_size, install_packages, log, precompile, run, slim, stage_app)

ARCH = "arm64" if platform.machine() == "arm64" else "x86_64"
TRIPLE = {"arm64": "aarch64-apple-darwin", "x86_64": "x86_64-apple-darwin"}[ARCH]
MIN_MACOS = "12.0"
PLATFORM_TAG = f"macosx_{MIN_MACOS.replace('.', '_')}_{ARCH}"
LABEL = "AppleSilicon" if ARCH == "arm64" else "Intel"
EXE_NAME = "MAMCMLStudio"

LAUNCHER = r'''#!/bin/bash
# MAMC ML Studio for macOS - starts the program with the Python that is inside this app.
RES="$(cd "$(dirname "$0")/../Resources" && pwd)"
APP="$(cd "$RES/../.." && pwd)"
if [ "__ARCH__" = "arm64" ] && [ "$(sysctl -n hw.optional.arm64 2>/dev/null)" != "1" ]; then
  osascript -e 'display dialog "This copy of MAMC ML Studio is for Macs with an Apple chip (M1, M2, M3, M4 ...).\n\nThis Mac has an Intel processor - please download the Intel edition (the file whose name ends in macOS-Intel.dmg)." buttons {"OK"} default button "OK" with title "MAMC ML Studio" with icon stop' >/dev/null 2>&1
  exit 1
fi
# The user has already allowed this app to open; remove the "downloaded from the internet" flag from the files
# inside it as well, so macOS does not ask again for each of them (no administrator rights needed).
if xattr -p com.apple.quarantine "$RES/python/bin/python3" >/dev/null 2>&1; then
  xattr -dr com.apple.quarantine "$APP" 2>/dev/null
fi
exec "$RES/python/bin/python3" "$RES/app/desktop_launcher.py"
'''

README = """MAMC ML Studio {version} for macOS ({label})
Maulana Azad Medical College & Govind Ballabh Pant Hospital, New Delhi

1. Drag "MAMC ML Studio" onto the "Applications" folder.
2. Open Applications, RIGHT-click (or Control-click) MAMC ML Studio and choose Open, then Open again.
   (Only the first time - macOS asks because the app is not from the App Store.)
   On macOS 15 or newer: if there is no Open button, click Done, then open System Settings > Privacy & Security,
   scroll down and click "Open Anyway".
3. The program opens in its own window after a few seconds. Close the window to quit.

Nothing else needs to be installed and no internet is needed - Python and all libraries are inside the app.
Needs macOS {min_macos} (Monterey) or newer.
If something goes wrong, the details are in  ~/Library/Application Support/MAMC ML Studio/last_run.log

Free and open source (GNU GPL-3.0). Your data never leave your computer.
"""


def make_icns(dest: Path):
    src = ROOT / "installer" / "mamc_icon_512.png"
    iconset = dest.parent / "AppIcon.iconset"
    shutil.rmtree(iconset, ignore_errors=True)
    iconset.mkdir()
    for s in (16, 32, 64, 128, 256, 512):
        run(["sips", "-z", s, s, src, "--out", iconset / f"icon_{s}x{s}.png"], stdout=subprocess.DEVNULL, check=True)
        if s <= 256:
            run(["sips", "-z", 2 * s, 2 * s, src, "--out", iconset / f"icon_{s}x{s}@2x.png"], stdout=subprocess.DEVNULL,
                check=True)
    run(["iconutil", "-c", "icns", iconset, "-o", dest], check=True)
    shutil.rmtree(iconset)


def is_macho(p: Path) -> bool:
    try:
        with open(p, "rb") as f:
            head = f.read(4)
    except OSError:
        return False
    return head in (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xfe\xed\xfa\xcf", b"\xfe\xed\xfa\xce", b"\xca\xfe\xba\xbe")


def sign(app: Path):
    """Ad-hoc signature for every binary and then the whole app (Apple-silicon Macs refuse unsigned native code)."""
    binaries = [p for p in app.rglob("*") if p.is_file() and not p.is_symlink() and is_macho(p)]
    log(f"signing {len(binaries)} binaries (ad-hoc)")
    for i in range(0, len(binaries), 200):
        run(["codesign", "--force", "--sign", "-", *binaries[i:i + 200]], check=True, stdout=subprocess.DEVNULL)
    run(["codesign", "--force", "--sign", "-", app], check=True)
    run(["codesign", "--verify", "--strict", app])


def make_dmg(src: Path, out: Path):
    out.unlink(missing_ok=True)
    for attempt in range(4):  # hdiutil is sometimes "busy" right after a large copy
        r = run(["hdiutil", "create", "-volname", APP_NAME, "-srcfolder", src, "-ov", "-format", "UDZO",
                 "-imagekey", "zlib-level=9", out])
        if r.returncode == 0:
            return
        time.sleep(10 * (attempt + 1))
    raise SystemExit("hdiutil could not make the disk image")


def main():
    if platform.system() != "Darwin":
        raise SystemExit("run this on a Mac")
    log(f"{APP_NAME} {APP_VERSION} for macOS {MIN_MACOS}+ ({ARCH})")
    work = ROOT / "build" / f"mac-{ARCH}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    dmgsrc = work / "dmg"
    app = dmgsrc / f"{APP_NAME}.app"
    contents = app / "Contents"
    res = contents / "Resources"
    (contents / "MacOS").mkdir(parents=True)
    res.mkdir(parents=True)

    pv = download_python(TRIPLE, work / "pbs")
    shutil.move(str(work / "pbs" / "python"), str(res / "python"))
    pydir = res / "python"
    py = pydir / "bin" / "python3"
    stdlib = pydir / "lib" / f"python{PY_MINOR}"
    site = stdlib / "site-packages"
    install_packages(py, site, PLATFORM_TAG)
    slim(pydir, site, stdlib)
    for f in pydir.glob("bin/*"):  # keep only the interpreter itself
        if not f.name.startswith("python"):
            f.unlink()
    stage_app(res / "app")
    check_imports(py, res / "app")
    precompile(py, stdlib, res / "app")
    (pydir / "BUNDLED_PYTHON.txt").write_text(f"CPython {pv} (python-build-standalone, {TRIPLE})\n")
    shutil.copy2(ROOT / "LICENSE", res / "LICENSE.txt")

    exe = contents / "MacOS" / EXE_NAME
    exe.write_text(LAUNCHER.replace("__ARCH__", ARCH))
    exe.chmod(0o755)
    make_icns(res / "AppIcon.icns")
    plist = {
        "CFBundleName": APP_NAME, "CFBundleDisplayName": APP_NAME, "CFBundleIdentifier": "in.ac.mamc.mlstudio",
        "CFBundleVersion": APP_VERSION, "CFBundleShortVersionString": APP_VERSION, "CFBundlePackageType": "APPL",
        "CFBundleExecutable": EXE_NAME, "CFBundleIconFile": "AppIcon", "LSMinimumSystemVersion": MIN_MACOS,
        "LSApplicationCategoryType": "public.app-category.education", "NSHighResolutionCapable": True,
        "NSHumanReadableCopyright": "Maulana Azad Medical College & GB Pant Hospital, New Delhi - GNU GPL-3.0",
    }
    with open(contents / "Info.plist", "wb") as f:
        plistlib.dump(plist, f)
    sign(app)
    log(f"app size: {folder_size(app) / 1e6:.0f} MB")

    os.symlink("/Applications", dmgsrc / "Applications")
    shutil.copy2(GUIDE, dmgsrc / "User Guide.docx")
    (dmgsrc / "READ ME FIRST.txt").write_text(README.format(version=APP_VERSION, label=LABEL, min_macos=MIN_MACOS))
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    out = dist / f"MAMC_ML_Studio_{APP_VERSION}_macOS-{LABEL}.dmg"
    make_dmg(dmgsrc, out)
    log(f"done -> {out} ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
