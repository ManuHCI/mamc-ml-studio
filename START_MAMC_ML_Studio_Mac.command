#!/bin/bash
# MAMC ML Studio - double-click to start on a Mac (first start needs internet).
cd "$(dirname "$0")"
if [ -x ".venv/bin/python" ]; then
  exec ".venv/bin/python" launcher.py
fi
if command -v python3 >/dev/null 2>&1; then
  exec python3 launcher.py
fi
echo "Python 3 was not found. Install it from https://www.python.org/downloads/ and double-click this file again."
read -p "Press Enter to close..."
