#!/usr/bin/env bash
# Simple helper script to install SecureMeet dependencies
set -e

# Install system packages (Ubuntu/Debian)
if command -v apt-get >/dev/null; then
    sudo apt-get update
    sudo apt-get install -y portaudio19-dev python3-tk
fi

# Install Python packages
pip install -r requirements.txt
