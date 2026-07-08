#!/usr/bin/env bash
# Face Flashing MVP — quick start (Linux / macOS)
# Requires Python 3.11 or 3.12

set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
pip install -r requirements.txt
python server/app.py
