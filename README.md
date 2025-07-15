# SecureMeet

SecureMeet is an offline meeting transcription system with real-time speaker identification and emotion detection. To run the application you need to install several Python packages.

## Installation

1. Ensure you have Python 3.11 and development headers for PortAudio installed (for `pyaudio`). On Ubuntu you can run:
   ```bash
   sudo apt-get update
   sudo apt-get install -y portaudio19-dev python3-tk
   ```
2. Install Python dependencies (or run `./install_deps.sh`):
   ```bash
   ./install_deps.sh
   ```
   The installation may take a while as several large packages (TensorFlow,
   PyTorch, Whisper, PyAnnote, etc.) are downloaded. Ensure you have a stable
   internet connection and sufficient disk space.

Some optional features rely on the `rnnoise` library which is not available on PyPI. If you require noise suppression, build and install an appropriate `rnnoise` wrapper manually.

## Running

After installing the dependencies you can run the application with:

```bash
python3 securemeet.py
```

Note that running the GUI requires a desktop environment.
