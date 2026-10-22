"""
Build the stand-alone Windows 10/11 edition of MAMC ML Studio.

The result contains its own Python and every package the program needs, so the user installs nothing else and the
program works fully offline.

Usage (on Windows, from the repository folder):
    python build_tools/build_windows.py
Needs: Python 3.10+ (only to run this script), internet, NSIS (makensis) on the PATH  (choco install nsis)

Output (in dist/):
    MAMC_ML_Studio_Setup_<version>_Windows10-11.exe   installer (no administrator rights needed)
    MAMC_ML_Studio_<version>_Windows_portable.zip     unzip and double-click "MAMC ML Studio.exe"

    python build_tools/build_windows.py --nsis-only    only re-make the two .exe files from an existing build folder
"""
import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (APP_NAME, APP_VERSION, GUIDE, ROOT, check_imports, download_python, folder_size,  # noqa: E402
                    install_packages, log, precompile, run, slim, stage_app)

TRIPLE = "x86_64-pc-windows-msvc"
PLATFORM_TAG = "win_amd64"
ICON = ROOT / "installer" / "mamc_ml_studio.ico"
VC_DLLS = ["vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "concrt140.dll"]
PUBLISHER = "Maulana Azad Medical College, New Delhi"

# --------------------------------------------------------------------------- "MAMC ML Studio.exe" (tiny starter)
STUB_NSI = r'''Unicode true
Name "MAMC ML Studio"
OutFile "{outfile}"
Icon "{icon}"
RequestExecutionLevel user
SilentInstall silent
SetCompressor /SOLID lzma
VIProductVersion "{version4}"
VIAddVersionKey "ProductName" "MAMC ML Studio"
VIAddVersionKey "CompanyName" "{publisher}"
VIAddVersionKey "FileDescription" "MAMC ML Studio"
VIAddVersionKey "FileVersion" "{version}"
VIAddVersionKey "ProductVersion" "{version}"
VIAddVersionKey "LegalCopyright" "GNU GPL-3.0"

Section
  IfFileExists "$EXEDIR\python\pythonw.exe" found
    MessageBox MB_ICONSTOP "MAMC ML Studio cannot find its program files.$\r$\n$\r$\nIf you opened it from inside a zip file: right-click the zip file, choose 'Extract All', then open MAMC ML Studio.exe from the extracted folder.$\r$\n$\r$\nOtherwise please install MAMC ML Studio again."
    Abort
  found:
  SetOutPath "$EXEDIR\app"
  Exec '"$EXEDIR\python\pythonw.exe" "$EXEDIR\app\desktop_launcher.py"'
SectionEnd
'''

# --------------------------------------------------------------------------- installer
SETUP_NSI = r'''Unicode true
!include "MUI2.nsh"
!include "FileFunc.nsh"
Name "MAMC ML Studio"
OutFile "{outfile}"
InstallDir "$LOCALAPPDATA\Programs\MAMC ML Studio"
InstallDirRegKey HKCU "Software\MAMC ML Studio" "InstallDir"
RequestExecutionLevel user
SetCompressor /SOLID lzma
SetCompressorDictSize 64
ManifestDPIAware true
BrandingText "MAMC ML Studio {version} - MAMC & GB Pant Hospital, New Delhi"
VIProductVersion "{version4}"
VIAddVersionKey "ProductName" "MAMC ML Studio"
VIAddVersionKey "CompanyName" "{publisher}"
VIAddVersionKey "FileDescription" "MAMC ML Studio installer (Windows 10/11)"
VIAddVersionKey "FileVersion" "{version}"
VIAddVersionKey "ProductVersion" "{version}"
VIAddVersionKey "LegalCopyright" "GNU GPL-3.0"
!define UNINST "Software\Microsoft\Windows\CurrentVersion\Uninstall\MAMCMLStudio"
!define MUI_ICON "{icon}"
!define MUI_UNICON "{icon}"
!define MUI_ABORTWARNING
!define MUI_WELCOMEPAGE_TITLE "MAMC ML Studio {version}"
!define MUI_WELCOMEPAGE_TEXT "Free, open-source machine-learning software for medical students, residents and researchers - no coding needed.$\r$\n$\r$\nMaulana Azad Medical College & Govind Ballabh Pant Hospital, New Delhi.$\r$\n$\r$\nEverything needed (including Python) is included - nothing else has to be installed and no internet is required.$\r$\n$\r$\nClick Next to continue."
!define MUI_FINISHPAGE_RUN
!define MUI_FINISHPAGE_RUN_FUNCTION LaunchApp
!define MUI_FINISHPAGE_RUN_TEXT "Start MAMC ML Studio now"
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "{license}"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

Function LaunchApp
  Exec '"$INSTDIR\MAMC ML Studio.exe"'
FunctionEnd

Section "Install"
  SetOutPath "$INSTDIR"
  ; an older version: remove its program files first so no outdated library is left behind
  RMDir /r "$INSTDIR\python"
  RMDir /r "$INSTDIR\app"
  File /r "{stage}\*.*"
  WriteUninstaller "$INSTDIR\Uninstall.exe"
  CreateDirectory "$SMPROGRAMS\MAMC ML Studio"
  CreateShortCut "$SMPROGRAMS\MAMC ML Studio\MAMC ML Studio.lnk" "$INSTDIR\MAMC ML Studio.exe" "" "$INSTDIR\mamc_ml_studio.ico"
  CreateShortCut "$SMPROGRAMS\MAMC ML Studio\User Guide.lnk" "$INSTDIR\MAMC ML Studio - User Guide.docx"
  CreateShortCut "$SMPROGRAMS\MAMC ML Studio\Sample data.lnk" "$INSTDIR\app\sample_data"
  CreateShortCut "$SMPROGRAMS\MAMC ML Studio\Uninstall MAMC ML Studio.lnk" "$INSTDIR\Uninstall.exe"
  CreateShortCut "$DESKTOP\MAMC ML Studio.lnk" "$INSTDIR\MAMC ML Studio.exe" "" "$INSTDIR\mamc_ml_studio.ico"
  WriteRegStr HKCU "Software\MAMC ML Studio" "InstallDir" "$INSTDIR"
  WriteRegStr HKCU "${{UNINST}}" "DisplayName" "MAMC ML Studio"
  WriteRegStr HKCU "${{UNINST}}" "DisplayVersion" "{version}"
  WriteRegStr HKCU "${{UNINST}}" "Publisher" "{publisher}"
  WriteRegStr HKCU "${{UNINST}}" "DisplayIcon" "$INSTDIR\mamc_ml_studio.ico"
  WriteRegStr HKCU "${{UNINST}}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "${{UNINST}}" "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegStr HKCU "${{UNINST}}" "QuietUninstallString" '"$INSTDIR\Uninstall.exe" /S'
  WriteRegDWORD HKCU "${{UNINST}}" "NoModify" 1
  WriteRegDWORD HKCU "${{UNINST}}" "NoRepair" 1
  ${{GetSize}} "$INSTDIR" "/S=0K" $0 $1 $2
  IntFmt $0 "0x%08X" $0
  WriteRegDWORD HKCU "${{UNINST}}" "EstimatedSize" "$0"
SectionEnd

Section "Uninstall"
  Delete "$DESKTOP\MAMC ML Studio.lnk"
  RMDir /r "$SMPROGRAMS\MAMC ML Studio"
  ; only the files this installer put there (safe even if a shared folder was chosen)
  RMDir /r "$INSTDIR\python"
  RMDir /r "$INSTDIR\app"
{uninstall_files}
  Delete "$INSTDIR\Uninstall.exe"
  RMDir "$INSTDIR"
  ; log file and the program window's settings
  RMDir /r "$LOCALAPPDATA\MAMC ML Studio"
  DeleteRegKey HKCU "${{UNINST}}"
  DeleteRegKey HKCU "Software\MAMC ML Studio"
SectionEnd
'''

README = """MAMC ML Studio {version} - portable edition for Windows 10 / 11
Maulana Azad Medical College & Govind Ballabh Pant Hospital, New Delhi

How to start
  Double-click "MAMC ML Studio.exe" in this folder.
  The program opens in its own window after a few seconds. Close the window to quit.

Nothing needs to be installed and no internet is needed - Python and all libraries are inside the "python" folder.
Keep all files of this folder together (you may copy the whole folder to a pen drive).

If Windows shows "Windows protected your PC": click "More info", then "Run anyway".
If something goes wrong, the details are in  %LOCALAPPDATA%\\MAMC ML Studio\\last_run.log

Free and open source (GNU GPL-3.0). Your data never leave your computer.
"""


def makensis():
    m = shutil.which("makensis") or r"C:\Program Files (x86)\NSIS\makensis.exe"
    if not Path(m).exists() and not shutil.which(m):
        raise SystemExit("NSIS (makensis) is not installed - on Windows: choco install nsis")
    return m


def version4():
    return ".".join((APP_VERSION.split(".") + ["0", "0", "0"])[:4])


def build_stub(stage: Path, work: Path):
    nsi = work / "starter.nsi"
    nsi.write_text(STUB_NSI.format(outfile=stage / f"{APP_NAME}.exe", icon=ICON, version=APP_VERSION,
                                   version4=version4(), publisher=PUBLISHER), encoding="utf-8")
    r = run([makensis(), "-V2", nsi])
    if r.returncode:
        raise SystemExit("could not make the starter .exe")


def build_setup(stage: Path, work: Path, dist: Path) -> Path:
    out = dist / f"MAMC_ML_Studio_Setup_{APP_VERSION}_Windows10-11.exe"
    top_files = sorted(p.name for p in stage.iterdir() if p.is_file())
    uninstall_files = "\n".join(f'  Delete "$INSTDIR\\{n}"' for n in top_files)
    nsi = work / "installer.nsi"
    nsi.write_text(SETUP_NSI.format(outfile=out, icon=ICON, license=ROOT / "LICENSE", stage=stage, version=APP_VERSION,
                                    version4=version4(), publisher=PUBLISHER, uninstall_files=uninstall_files),
                   encoding="utf-8")
    log("making the installer with NSIS (this takes a few minutes)")
    r = run([makensis(), "-V2", nsi])
    if r.returncode:
        raise SystemExit("NSIS failed")
    return out


def copy_vc_runtime(python_dir: Path):
    """Put the Microsoft C++ runtime next to python.exe so the program also runs on a PC that never had it installed."""
    sysdir = Path(r"C:\Windows\System32")
    for dll in VC_DLLS:
        if not (python_dir / dll).exists() and (sysdir / dll).exists():
            shutil.copy2(sysdir / dll, python_dir / dll)
            log(f"  added {dll}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsis-only", action="store_true", help="only re-make the .exe files from build/windows")
    a = ap.parse_args()
    work = ROOT / "build" / "windows"
    stage = work / "MAMC_ML_Studio"
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)

    if not a.nsis_only:
        if platform.system() != "Windows":
            raise SystemExit("run this on Windows (the bundled Python must run to pre-compile the program)")
        log(f"{APP_NAME} {APP_VERSION} for Windows 10/11")
        shutil.rmtree(work, ignore_errors=True)
        stage.mkdir(parents=True)
        pv = download_python(TRIPLE, work / "pbs")
        shutil.move(str(work / "pbs" / "python"), str(stage / "python"))
        py = stage / "python" / "python.exe"
        site = stage / "python" / "Lib" / "site-packages"
        install_packages(py, site, PLATFORM_TAG)
        copy_vc_runtime(stage / "python")
        slim(stage / "python", site, stage / "python" / "Lib")
        shutil.rmtree(stage / "python" / "Scripts", ignore_errors=True)
        stage_app(stage / "app")
        check_imports(py, stage / "app")
        precompile(py, stage / "python" / "Lib", stage / "app")
        shutil.copy2(ICON, stage / "mamc_ml_studio.ico")
        shutil.copy2(ROOT / "LICENSE", stage / "LICENSE.txt")
        shutil.copy2(GUIDE, stage / "MAMC ML Studio - User Guide.docx")
        (stage / "README.txt").write_text(README.format(version=APP_VERSION).replace("\n", "\r\n"), encoding="utf-8",
                                          newline="")
        (stage / "python" / "BUNDLED_PYTHON.txt").write_text(f"CPython {pv} (python-build-standalone, {TRIPLE})\r\n")
        build_stub(stage, work)
        log(f"program folder: {folder_size(stage) / 1e6:.0f} MB")

        log("making the portable zip")
        zip_base = dist / f"MAMC_ML_Studio_{APP_VERSION}_Windows_portable"
        shutil.make_archive(str(zip_base), "zip", root_dir=work, base_dir=stage.name)
    elif not (stage / f"{APP_NAME}.exe").exists():
        build_stub(stage, work)

    out = build_setup(stage, work, dist)
    for f in sorted(dist.iterdir()):
        log(f"  {f.name}: {f.stat().st_size / 1e6:.0f} MB")
    log(f"done -> {out}")


if __name__ == "__main__":
    main()
