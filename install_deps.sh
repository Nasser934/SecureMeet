#!/usr/bin/env bash
# Simple helper script to install SecureMeet dependencies
set -euo pipefail

# Install system packages (Ubuntu/Debian)
if command -v apt-get >/dev/null; then
    sudo apt-get update
    sudo apt-get install -y portaudio19-dev python3-tk
fi

# Install Python packages
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
