@echo off
setlocal EnableExtensions
title MAMC ML Studio
cd /d "%~dp0"

rem ----------------------------------------------------------------------
rem  MAMC ML Studio - double-click this file to start the software.
rem  The first start sets everything up (needs internet, about 5-10 min).
rem ----------------------------------------------------------------------

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" launcher.py
    goto :end
)

set "PYEXE="
where py >nul 2>nul
if not errorlevel 1 (
    py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
    if not errorlevel 1 set "PYEXE=py -3"
)
if not defined PYEXE (
    python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
    if not errorlevel 1 set "PYEXE=python"
)
if not defined PYEXE (
    for %%D in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%ProgramData%\anaconda3" "%ProgramData%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%LOCALAPPDATA%\miniconda3") do (
        if not defined PYEXE if exist "%%~D\python.exe" set "PYEXE="%%~D\python.exe""
    )
)
if not defined PYEXE (
    for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do (
        if exist "%%~D\python.exe" set "PYEXE="%%~D\python.exe""
    )
)
if not defined PYEXE (
    echo.
    echo  Python was not found on this computer.
    echo  Please install Python 3.10 or newer from https://www.python.org/downloads/
    echo  ^(tick "Add python.exe to PATH" during installation^), then double-click this file again.
    echo.
    pause
    goto :end
)

%PYEXE% launcher.py

:end
endlocal
