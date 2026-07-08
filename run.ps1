# Face Flashing MVP — quick start (Windows PowerShell)
# Requires Python 3.11 or 3.12

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

& .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python server\app.py
