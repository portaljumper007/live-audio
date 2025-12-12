# Real-time Audio Spectrogram Visualiser 🎵

Visualises system audio frequencies using a GPU-accelerated Mel-scale spectrogram. Built with Python, Flask, and Socket.IO. Backend uses CuPy for fast signal processing and sends incremental updates to a web-based frontend for low-latency rendering.

To use the visualiser, ensure you have a CUDA-capable GPU and the necessary drivers installed. Install the dependencies via `pip install flask flask-socketio sounddevice cupy-cuda12x librosa` (adjusting the CuPy version to match your CUDA toolkit). Run the script and the dashboard will launch at `http://127.0.0.1:8050`, automatically detecting your system's loopback or microphone input.
