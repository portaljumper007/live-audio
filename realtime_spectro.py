#!/usr/bin/env python3
"""
Jamie Slater

Real-time Audio Spectrogram Visualiser
"""

import numpy as np
import cupy as cp
import sounddevice as sd
from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit
import threading
import logging
from dataclasses import dataclass
from typing import Tuple, Optional, List
import sys
import time
import librosa

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class SpectrogramConfig:
    sample_rate: int = 44100
    fft_size: int = 8192
    hop_size: int = 1024
    window_function: str = 'hann'
    time_window: float = 20.0
    waveform_window: float = 20.0
    freq_range: Tuple[float, float] = (20, 20000)
    colourmap: str = 'Viridis'
    db_range: Tuple[float, float] = (-80, 0)
    channel: str = 'stereo'
    n_mels: int = 128  #Mel bins


def hz_to_mel(hz):
    """Convert Hz to Mel scale."""
    return 2595 * np.log10(1 + hz / 700)


def mel_to_hz(mel):
    """Convert Mel to Hz."""
    return 700 * (10**(mel / 2595) - 1)


class AudioDeviceManager:
    @staticmethod
    def find_loopback_device() -> Optional[int]:
        devices = sd.query_devices()
        loopback_keywords = ['Stereo Mix', 'stereo mix', 'Wave Out', 'Loopback', 'loopback', 'What U Hear', 'Monitor']
        
        for idx, device in enumerate(devices):
            if device['max_input_channels'] >= 2:
                for keyword in loopback_keywords:
                    if keyword in device['name']:
                        logger.info(f"Found loopback device: {device['name']} (index: {idx})")
                        return idx
        
        logger.warning("No loopback device found")
        return None
    
    @staticmethod
    def list_devices() -> List[dict]:
        devices = sd.query_devices()
        device_list = []
        for idx, device in enumerate(devices):
            device_list.append({
                'index': idx,
                'name': device['name'],
                'channels': device['max_input_channels'],
                'is_loopback': 'Stereo Mix' in device['name'] or 'Loopback' in device['name']
            })
        return device_list


class WaveformBuffer:
    def __init__(self, sample_rate: int, duration: float):
        self.sample_rate = sample_rate
        self.duration = duration
        self.buffer_size = int(sample_rate * duration)
        self.left_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.right_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.write_index = 0
        self.lock = threading.Lock()
        
        self.target_points = 1024
        self.downsample_factor = max(1, self.buffer_size // self.target_points)
        self.output_points = self.buffer_size // self.downsample_factor
        
        logger.info(f"Waveform buffer: {self.buffer_size} samples ({duration}s) → {self.output_points} points")
    
    def add_samples(self, left_data: np.ndarray, right_data: np.ndarray) -> None:
        with self.lock:
            num_samples = min(len(left_data), len(right_data))
            for i in range(num_samples):
                self.left_buffer[self.write_index] = left_data[i]
                self.right_buffer[self.write_index] = right_data[i]
                self.write_index = (self.write_index + 1) % self.buffer_size
    
    def get_waveform(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with self.lock:
            end_idx = self.write_index
            left = np.concatenate([self.left_buffer[end_idx:], self.left_buffer[:end_idx]])
            right = np.concatenate([self.right_buffer[end_idx:], self.right_buffer[:end_idx]])
            
            left_ds = left[::self.downsample_factor]
            right_ds = right[::self.downsample_factor]
            time_axis = np.linspace(0, self.duration, len(left_ds))
            
            return time_axis, left_ds.copy(), right_ds.copy()


class SpectrogramBuffer:
    def __init__(self, num_freq_bins: int, num_time_frames: int):
        self.num_freq_bins = num_freq_bins
        self.num_time_frames = num_time_frames
        
        #Initialise with NAN for empty regions
        self.buffer = np.full((num_freq_bins, num_time_frames), np.nan, dtype=np.float32)
        
        self.write_index = 0
        self.lock = threading.Lock()
        self.frame_counter = 0
        self.last_sent_index = 0  #Track last sent column
        
        logger.info(f"Spectrogram buffer: {num_freq_bins} freq × {num_time_frames} time (NaN initialised)")
    
    def add_frame(self, magnitude_data: np.ndarray) -> None:
        with self.lock:
            if len(magnitude_data) != self.num_freq_bins:
                logger.error(f"Frame size mismatch: expected {self.num_freq_bins}, got {len(magnitude_data)}")
                return
            self.buffer[:, self.write_index] = magnitude_data
            self.write_index = (self.write_index + 1) % self.num_time_frames
            self.frame_counter += 1
    
    def get_incremental_update(self) -> Tuple[np.ndarray, int, int]:
        """
        Get only new columns since last sent, with percentile computation on GPU.
        Returns: (percentile_uint8_data, start_col, end_col)
        """
        with self.lock:
            current_index = self.write_index
            
            #Compute new columns
            if current_index >= self.last_sent_index:
                #No wraparound
                new_columns = slice(self.last_sent_index, current_index)
                start_col = self.last_sent_index
                end_col = current_index
            else:
                #Wraparound case - send from last_sent to end, then 0 to current
                #send entire buffer (happens once per 20s)
                new_columns = slice(0, self.num_time_frames)
                start_col = 0
                end_col = self.num_time_frames
            
            if start_col == end_col:
                return None, start_col, end_col
            
            new_data = self.buffer[:, new_columns].copy()
            
            data_gpu = cp.asarray(new_data)
            
            valid_mask = ~cp.isnan(data_gpu) #NAN mask
            
            if cp.sum(valid_mask) == 0:
                percentile_uint8 = cp.zeros_like(data_gpu, dtype=cp.uint8)
            else:
                flat_valid = data_gpu[valid_mask]
                
                sorted_valid = cp.sort(flat_valid)
                n_valid = len(sorted_valid)
                
                percentile_data = cp.zeros_like(data_gpu, dtype=cp.float32)
                
                #For each value, find its percentile rank
                for i in range(data_gpu.shape[0]):
                    for j in range(data_gpu.shape[1]):
                        if valid_mask[i, j]:
                            val = data_gpu[i, j]
                            #Binary search for rank
                            rank = cp.searchsorted(sorted_valid, val)
                            percentile = (rank / n_valid) * 100
                            percentile_data[i, j] = percentile
                        else:
                            percentile_data[i, j] = 0  # NaN becomes 0
                
                #Normalize/convert to uint8 (0-255)
                percentile_uint8 = (percentile_data * 2.55).astype(cp.uint8)
            
            #Transfer back to CPU only at the end
            result = cp.asnumpy(percentile_uint8)
            
            #Update last sent index
            self.last_sent_index = current_index
            
            return result, start_col, end_col
    
    def get_statistics(self) -> dict:
        with self.lock:
            data = self.buffer.copy()
            valid_data = data[~np.isnan(data)]
            if len(valid_data) == 0:
                return {'min': 0, 'max': 0, 'mean': 0, 'median': 0, 'frames': self.frame_counter, 'shape': f"{data.shape[0]}×{data.shape[1]}"}
            return {
                'min': float(np.min(valid_data)), 
                'max': float(np.max(valid_data)),
                'mean': float(np.mean(valid_data)),
                'median': float(np.median(valid_data)),
                'frames': self.frame_counter,
                'shape': f"{data.shape[0]}×{data.shape[1]}"
            }


class AudioBuffer:
    def __init__(self, buffer_size: int, sample_rate: int):
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate
        self.left_buffer = np.zeros(buffer_size, dtype=np.float32)
        self.right_buffer = np.zeros(buffer_size, dtype=np.float32)
        self.write_index = 0
        self.lock = threading.Lock()
        self.samples_received = 0
    
    def add_samples(self, left_data: np.ndarray, right_data: np.ndarray) -> None:
        """Fast audio callback"""
        with self.lock:
            num_samples = min(len(left_data), len(right_data))
            for i in range(num_samples):
                self.left_buffer[self.write_index] = left_data[i]
                self.right_buffer[self.write_index] = right_data[i]
                self.write_index = (self.write_index + 1) % self.buffer_size
            self.samples_received += num_samples
    
    def get_latest_samples(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        with self.lock:
            end_idx = self.write_index
            start_idx = (end_idx - num_samples) % self.buffer_size
            if start_idx < end_idx:
                left = self.left_buffer[start_idx:end_idx].copy()
                right = self.right_buffer[start_idx:end_idx].copy()
            else:
                left = np.concatenate([self.left_buffer[start_idx:], self.left_buffer[:end_idx]])
                right = np.concatenate([self.right_buffer[start_idx:], self.right_buffer[:end_idx]])
            return left, right
    
    def get_statistics(self) -> dict:
        with self.lock:
            left_rms = float(np.sqrt(np.mean(self.left_buffer**2)))
            right_rms = float(np.sqrt(np.mean(self.right_buffer**2)))
            return {
                'samples_received': self.samples_received,
                'left': {'rms': left_rms}, 
                'right': {'rms': right_rms}
            }


class STFTProcessor:
    def __init__(self, config: SpectrogramConfig):
        self.config = config
        self.fft_size = config.fft_size
        self.window = np.hanning(self.fft_size).astype(np.float32)
        self.window_gpu = cp.asarray(self.window)
        
        #Linear frequency bins
        self.freq_bins = np.fft.rfftfreq(self.fft_size, 1.0 / config.sample_rate)
        
        self.mel_filterbank = self._create_mel_filterbank(config)
        
        self.mel_filterbank_gpu = cp.asarray(self.mel_filterbank)
        
        #Output frequency labels (Mel scale)
        self.mel_frequencies = librosa.mel_frequencies(n_mels=config.n_mels, 
                                                        fmin=config.freq_range[0],
                                                        fmax=config.freq_range[1])
        
        logger.info(f"STFT Processor:")
        logger.info(f"  FFT size: {self.fft_size}")
        logger.info(f"  Hop size: {config.hop_size}")
        logger.info(f"  Linear freq bins: {len(self.freq_bins)}")
        logger.info(f"  Mel bins: {config.n_mels} ({config.freq_range[0]}-{config.freq_range[1]} Hz)")
        logger.info(f"  Hop interval: {config.hop_size / config.sample_rate * 1000:.2f}ms")
    
    def _create_mel_filterbank(self, config):
        """Create Mel filterbank matrix."""
        mel_fb = librosa.filters.mel(
            sr=config.sample_rate,
            n_fft=self.fft_size,
            n_mels=config.n_mels,
            fmin=config.freq_range[0],
            fmax=config.freq_range[1]
        )
        return mel_fb.astype(np.float32)
    
    def get_frequency_labels(self) -> np.ndarray:
        """Get Mel-scaled frequency labels."""
        return self.mel_frequencies
    
    def process(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Process audio with STFT on GPU, then apply Mel filterbank.
        
        Validates hop/FFT relationship:
        - Takes last FFT_SIZE samples from audio buffer
        - This represents the most recent window
        - Called every HOP_SIZE samples by the processing loop
        """
        try:
            if len(audio_data) < self.fft_size:
                return np.full(self.config.n_mels, self.config.db_range[0], dtype=np.float32)
            
            #Get most recent FFT_SIZE samples for this hop
            frame = audio_data[-self.fft_size:]
            
            frame_gpu = cp.asarray(frame)
            windowed_frame = frame_gpu * self.window_gpu
            fft_result = cp.fft.rfft(windowed_frame, n=self.fft_size)
            
            #Power spectrum
            power_spectrum = cp.abs(fft_result) ** 2
            
            #Apply Mel filterbank (linear to Mel scale)
            mel_spectrum = cp.dot(self.mel_filterbank_gpu, power_spectrum)
            
            #Convert to dB
            mel_db = 10 * cp.log10(cp.maximum(mel_spectrum, 1e-10))
            
            #Clip to range
            min_db, max_db = self.config.db_range
            mel_db_clipped = cp.clip(mel_db, min_db, max_db)
            
            #Transfer to CPU
            result = cp.asnumpy(mel_db_clipped).astype(np.float32)
            
            return result
            
        except Exception as e:
            logger.error(f"STFT error: {e}")
            return np.full(self.config.n_mels, self.config.db_range[0], dtype=np.float32)


class RealtimeSpectrogramSystem:
    def __init__(self, config: SpectrogramConfig, device_id: Optional[int] = None):
        self.config = config
        self.running = False
        self.device_id = device_id
        
        self.audio_buffer_size = int(config.sample_rate * 5.0)
        self.num_time_frames = int(config.time_window * config.sample_rate / config.hop_size)
        
        logger.info(f"System Configuration:")
        logger.info(f"  Time window: {config.time_window}s → {self.num_time_frames} frames")
        logger.info(f"  Processing rate: {config.sample_rate / config.hop_size:.2f} fps")
        logger.info(f"  Audio buffer: {self.audio_buffer_size} samples")
        
        self.stft_processor = STFTProcessor(config)
        num_freq_bins = config.n_mels
        
        self.audio_buffer = AudioBuffer(self.audio_buffer_size, config.sample_rate)
        self.spectrogram_buffer = SpectrogramBuffer(num_freq_bins, self.num_time_frames)
        self.waveform_buffer = WaveformBuffer(config.sample_rate, config.waveform_window)
        
        self.stream = None
        self.processing_thread = None
        
        logger.info("System initialised")
    
    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """
        CRITICAL: Keep this FAST - runs in real-time audio thread!
        Only copy data, no processing.
        """
        if status:
            logger.warning(f"Audio status: {status}")
        
        try:
            #Copy data
            if indata.shape[1] >= 2:
                left_data, right_data = indata[:, 0], indata[:, 1]
            else:
                left_data = right_data = indata[:, 0]
            
            #Fast buffer write
            self.audio_buffer.add_samples(left_data, right_data)
            self.waveform_buffer.add_samples(left_data, right_data)
        except Exception as e:
            logger.error(f"Audio callback error: {e}")
    
    def _processing_loop(self) -> None:
        """
        Separate thread for STFT processing.
        Runs independently from audio callback.
        """
        logger.info("STFT processing loop started")
        hop_interval = self.config.hop_size / self.config.sample_rate
        
        last_log_time = time.time()
        frame_count = 0
        
        while self.running:
            try:
                loop_start = time.time()
                
                #Get latest audio samples for FFT window
                left, right = self.audio_buffer.get_latest_samples(self.config.fft_size)
                audio_data = (left + right) / 2.0
                
                #Process with GPU (everything stays on GPU until final result)
                magnitude_data = self.stft_processor.process(audio_data)
                
                #Add frame to buffer
                self.spectrogram_buffer.add_frame(magnitude_data)
                frame_count += 1
                
                #Regular logging
                current_time = time.time()
                if current_time - last_log_time > 10.0:
                    elapsed = current_time - last_log_time
                    actual_fps = frame_count / elapsed
                    stats = self.audio_buffer.get_statistics()
                    
                    logger.info(f"Processing: {actual_fps:.2f} fps (target: {1/hop_interval:.2f}), "
                              f"RMS L/R: {stats['left']['rms']:.4f}/{stats['right']['rms']:.4f}")
                    
                    last_log_time = current_time
                    frame_count = 0
                
                #Sleep for hop interval
                elapsed = time.time() - loop_start
                sleep_time = max(0, hop_interval - elapsed)
                if sleep_time > 0:
                    threading.Event().wait(sleep_time)
                
            except Exception as e:
                logger.error(f"Processing loop error: {e}", exc_info=True)
                time.sleep(0.1)
        
        logger.info("STFT processing loop stopped")
    
    def start(self) -> None:
        try:
            self.running = True
            
            device_config = {
                'samplerate': self.config.sample_rate,
                'channels': 2,
                'callback': self._audio_callback,
                'blocksize': self.config.hop_size,
                'dtype': np.float32
            }
            
            if self.device_id is not None:
                device_config['device'] = self.device_id
                device_info = sd.query_devices(self.device_id)
                logger.info(f"Using device: {device_info['name']}")
            
            self.stream = sd.InputStream(**device_config)
            self.stream.start()
            logger.info("Audio stream started")
            
            #Processing thread (seperate to not block audio thread)
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True, name="STFT")
            self.processing_thread.start()
            logger.info("Processing thread started")
            
        except Exception as e:
            logger.error(f"Failed to start: {e}")
            self.stop()
            raise
    
    def stop(self) -> None:
        logger.info("Stopping system...")
        self.running = False
        
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except:
                pass
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        logger.info("System stopped")
    
    def get_incremental_data(self):
        """Get ONLY new data since last update (incremental)."""
        try:
            start_time = time.time()
            
            #Get only new spectrogram columns (GPU-computed percentiles)
            new_spec_data, start_col, end_col = self.spectrogram_buffer.get_incremental_update()
            
            if new_spec_data is None:
                return None
            
            #Get frequency labels (Mel scale)
            freq_labels = self.stft_processor.get_frequency_labels()
            
            #Calculate time labels for new columns only
            time_labels = np.linspace(
                start_col * self.config.hop_size / self.config.sample_rate,
                end_col * self.config.hop_size / self.config.sample_rate,
                new_spec_data.shape[1]
            )
            
            #Waveform
            time_axis, left_wave, right_wave = self.waveform_buffer.get_waveform()
            
            #Statistics
            stats = self.audio_buffer.get_statistics()
            spec_stats = self.spectrogram_buffer.get_statistics()
            
            data = {
                'spectrogram': {
                    'z': new_spec_data.tolist(),  #uint8 only new data
                    'x': time_labels.tolist(),
                    'y': freq_labels.tolist(),
                    'start_col': int(start_col),
                    'end_col': int(end_col),
                    'is_incremental': True
                },
                'waveform': {
                    'time': time_axis.tolist(),
                    'left': left_wave.tolist(),
                    'right': right_wave.tolist()
                },
                'stats': {
                    'frames': spec_stats['frames'],
                    'samples': stats['samples_received'],
                    'rms_left': stats['left']['rms'],
                    'rms_right': stats['right']['rms'],
                    'spec_min_db': spec_stats['min'],
                    'spec_max_db': spec_stats['max'],
                    'spec_median_db': spec_stats['median'],
                    'spec_shape': spec_stats['shape']
                }
            }
            
            serialisation_time = (time.time() - start_time) * 1000
            if serialisation_time > 20:
                logger.warning(f"Slow serialisation: {serialisation_time:.1f}ms")
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting data: {e}", exc_info=True)
            return None


loopback_device = AudioDeviceManager.find_loopback_device()
config = SpectrogramConfig()
system = RealtimeSpectrogramSystem(config, device_id=loopback_device)

UPDATE_INTERVAL_MS = 50 
logger.info(f"WebSocket update interval: {UPDATE_INTERVAL_MS}ms (20 FPS)")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'spectrogram-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', 
                    ping_timeout=60, ping_interval=25)

HTML_TEMPLATE = f'''
<!DOCTYPE html>
<html>
<head>
    <title>Real-time Audio Spectrogram - High Performance</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #0f1419;
            color: #e2e8f0;
        }}
        .header {{
            background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
            padding: 40px 20px 30px 20px;
            border-bottom: 3px solid #667eea;
            text-align: center;
        }}
        h1 {{
            font-size: 36px;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
        }}
        .subtitle {{
            margin-top: 10px;
            font-size: 14px;
            color: #a0aec0;
        }}
        .live-badge {{
            display: inline-block;
            background: rgba(72, 187, 120, 0.2);
            color: #48bb78;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 700;
            margin-top: 8px;
        }}
        .container {{
            display: flex;
            padding: 30px;
            gap: 20px;
        }}
        .main-content {{
            flex: 3;
        }}
        .sidebar {{
            flex: 1;
        }}
        .card {{
            background: #1a202c;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            margin-bottom: 20px;
        }}
        .control-card {{
            background: #2d3748;
            border: 1px solid #4a5568;
        }}
        #spectrogram {{ height: 500px; }}
        #waveform {{ height: 300px; }}
        .status-item {{
            padding: 8px 0;
            border-bottom: 1px solid #4a5568;
            font-size: 13px;
        }}
        .status-item:last-child {{ border-bottom: none; }}
        .status-label {{ color: #a0aec0; }}
        .status-value {{ color: #48bb78; font-weight: 600; float: right; }}
        .control-group {{
            margin-bottom: 20px;
        }}
        label {{
            display: block;
            font-weight: 600;
            font-size: 13px;
            margin-bottom: 8px;
            color: #e2e8f0;
        }}
        select {{
            width: 100%;
            padding: 8px;
            background: #1a202c;
            color: #e2e8f0;
            border: 1px solid #4a5568;
            border-radius: 6px;
        }}
        .footer {{
            background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
            padding: 25px;
            text-align: center;
            color: #718096;
            font-size: 13px;
            border-top: 1px solid #4a5568;
        }}
        .highlight {{ color: #667eea; font-weight: 600; }}
        .section-title {{
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
        }}
        .status-dot {{
            color: #48bb78;
            font-size: 20px;
            margin-right: 10px;
        }}
        .connection-status {{
            position: fixed;
            top: 10px;
            right: 10px;
            padding: 8px 16px;
            border-radius: 20px;
            background: rgba(72, 187, 120, 0.2);
            color: #48bb78;
            font-size: 12px;
            font-weight: 600;
            z-index: 1000;
        }}
        .connection-status.disconnected {{
            background: rgba(252, 129, 129, 0.2);
            color: #fc8181;
        }}
        .info-badge {{
            background: #1a202c;
            padding: 10px 15px;
            border-radius: 8px;
            border: 1px solid #667eea;
            margin-bottom: 15px;
            font-size: 12px;
            line-height: 1.6;
        }}
        .info-badge strong {{
            color: #667eea;
        }}
    </style>
</head>
<body>
    <div class="connection-status" id="connection-status">● CONNECTED</div>
    
    <div class="header">
        <h1>🎵 Real-time Audio Spectrogram</h1>
        <div class="subtitle">CUDA STFT • 20-Second Window • Percentile-Based Colouring</div>
        <div class="live-badge">● LIVE @ {UPDATE_INTERVAL_MS}ms (20 FPS)</div>
    </div>
    
    <div class="container">
        <div class="main-content">
            <div class="card">
                <div id="spectrogram"></div>
            </div>
            <div class="card">
                <div id="waveform"></div>
            </div>
        </div>
        
        <div class="sidebar">
            <div class="card control-card">
                <div class="section-title">
                    <span class="status-dot">●</span>System Status
                </div>
                <div id="status"></div>
            </div>
            
            <div class="card control-card">
                <div class="section-title"> Display Settings</div>
                
                <div class="control-group">
                    <label>Colourmap</label>
                    <select id="colourmap" onchange="updateColourmap()">
                        <option value="Viridis"> Viridis</option>
                        <option value="Hot"> Hot</option>
                        <option value="Plasma"> Plasma</option>
                        <option value="Inferno"> Inferno</option>
                        <option value="Turbo"> Turbo</option>
                        <option value="Jet"> Jet</option>
                    </select>
                </div>
            </div>
        </div>
    </div>
    
    <div class="footer">
        Built with <span class="highlight">Flask + Socket.IO</span> • 
        GPU-Accelerated with <span class="highlight">CuPy</span>
    </div>

    <script>
        const socket = io({{
            reconnection: true,
            reconnectionDelay: 1000,
            reconnectionDelayMax: 5000,
            reconnectionAttempts: Infinity
        }});
        
        let currentColourmap = 'Viridis';
        const UPDATE_INTERVAL = {UPDATE_INTERVAL_MS};
        let isUpdating = false;
        let updateCount = 0;
        let lastFpsUpdate = Date.now();
        let currentFps = 0;
        
        //Client side buffer for the incremental updates
        let spectrogramBuffer = null;
        let timeLabels = null;
        let freqLabels = null;
        const BUFFER_SIZE = 861; //20s * 43fps
        
        const specLayout = {{
            title: '<b>Audio Spectrogram (Mel Scale)</b> | FFT: {config.fft_size} | Hop: {config.hop_size} | 20s',
            xaxis: {{ title: '<b>Time (seconds)</b>', color: '#cbd5e0', gridcolor: '#2d3748' }},
            yaxis: {{ 
                title: '<b>Frequency (Hz - Mel Scale)</b>', 
                color: '#cbd5e0', 
                gridcolor: '#2d3748',
                type: 'log'
            }},
            plot_bgcolor: '#1a202c',
            paper_bgcolor: '#1a202c',
            font: {{ color: '#e2e8f0', size: 12 }},
            margin: {{ l: 60, r: 100, t: 50, b: 50 }}
        }};
        
        const waveLayout = {{
            title: '<b>Audio Waveform</b> | Last 20 seconds',
            xaxis: {{ title: '<b>Time (seconds)</b>', color: '#cbd5e0', gridcolor: '#2d3748' }},
            yaxis: {{ title: '<b>Amplitude</b>', color: '#cbd5e0', gridcolor: '#2d3748', range: [-1, 1] }},
            plot_bgcolor: '#1a202c',
            paper_bgcolor: '#1a202c',
            font: {{ color: '#e2e8f0', size: 12 }},
            margin: {{ l: 60, r: 60, t: 50, b: 50 }},
            showlegend: true,
            legend: {{ orientation: 'h', y: 1.02, x: 1, bgcolor: 'rgba(45,55,72,0.8)', font: {{ color: '#e2e8f0' }} }}
        }};
        
        Plotly.newPlot('spectrogram', [], specLayout, {{ displayModeBar: true, displaylogo: false }});
        Plotly.newPlot('waveform', [], waveLayout, {{ displayModeBar: true, displaylogo: false }});
        
        socket.on('connect', function() {{
            console.log('Connected - incremental mode');
            document.getElementById('connection-status').textContent = '● CONNECTED';
            document.getElementById('connection-status').className = 'connection-status';
            
            //Reset buffer on reconnect
            spectrogramBuffer = null;
        }});
        
        socket.on('disconnect', function() {{
            console.log('Disconnected');
            document.getElementById('connection-status').textContent = '● DISCONNECTED';
            document.getElementById('connection-status').className = 'connection-status disconnected';
        }});
        
        function updateData() {{
            if (!isUpdating) {{
                isUpdating = true;
                socket.emit('request_data');
            }}
        }}
        
        socket.on('data_update', function(data) {{
            try {{
                if (!data) {{
                    isUpdating = false;
                    return;
                }}
                
                updateCount++;
                const now = Date.now();
                if (now - lastFpsUpdate > 1000) {{
                    currentFps = updateCount / ((now - lastFpsUpdate) / 1000);
                    updateCount = 0;
                    lastFpsUpdate = now;
                }}
                
                //INCREMENTAL UPDATE Merge new columns into buffer
                if (data.spectrogram.is_incremental) {{
                    const newData = data.spectrogram.z;
                    const startCol = data.spectrogram.start_col;
                    const endCol = data.spectrogram.end_col;
                    
                    //Initialise buffer
                    if (!spectrogramBuffer || !freqLabels) {{
                        const numFreq = newData.length;
                        spectrogramBuffer = Array(numFreq).fill(0).map(() => Array(BUFFER_SIZE).fill(0));
                        freqLabels = data.spectrogram.y;
                        timeLabels = Array(BUFFER_SIZE).fill(0).map((_, i) => i * {config.hop_size} / {config.sample_rate});
                    }}
                    
                    //Insert new columns
                    for (let i = 0; i < newData.length; i++) {{
                        for (let j = 0; j < newData[i].length; j++) {{
                            const col = startCol + j;
                            if (col < BUFFER_SIZE) {{
                                spectrogramBuffer[i][col] = newData[i][j];
                            }}
                        }}
                    }}
                }}
                
                //Plot full buffer
                const specData = [{{
                    type: 'heatmap',
                    z: spectrogramBuffer,
                    x: timeLabels,
                    y: freqLabels,
                    colorscale: currentColourmap,
                    zmin: 0,
                    zmax: 255,
                    colorbar: {{
                        title: '<b>Percentile</b>',
                        titleside: 'right',
                        tickfont: {{ color: '#e2e8f0', size: 10 }},
                        titlefont: {{ color: '#e2e8f0', size: 11 }},
                        tickvals: [0, 64, 128, 192, 255],
                        ticktext: ['0%', '25%', '50%', '75%', '100%']
                    }},
                    hovertemplate: '<b>Time:</b> %{{x:.2f}}s<br><b>Freq:</b> %{{y:.0f}} Hz<br><b>Percentile:</b> %{{z:.0f}}<extra></extra>'
                }}];
                
                const waveData = [
                    {{
                        x: data.waveform.time,
                        y: data.waveform.left,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Left',
                        line: {{ color: '#667eea', width: 1.5 }}
                    }},
                    {{
                        x: data.waveform.time,
                        y: data.waveform.right,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Right',
                        line: {{ color: '#f093fb', width: 1.5 }}
                    }}
                ];
                
                Plotly.react('spectrogram', specData, specLayout);
                Plotly.react('waveform', waveData, waveLayout);
                
                const stats = data.stats;
                const leftColor = stats.rms_left > 0.01 ? '#667eea' : '#ed8936';
                const rightColor = stats.rms_right > 0.01 ? '#f093fb' : '#ed8936';
                
                document.getElementById('status').innerHTML = `
                    <div class="status-item">
                        <span class="status-label">  Frames:</span>
                        <span class="status-value">${{stats.frames.toLocaleString()}}</span>
                    </div>
                    <div class="status-item">
                        <span class="status-label">  Samples:</span>
                        <span class="status-value">${{stats.samples.toLocaleString()}}</span>
                    </div>
                    <div class="status-item">
                        <span class="status-label">  Shape:</span>
                        <span class="status-value">${{stats.spec_shape}}</span>
                    </div>
                    <div class="status-item">
                        <span class="status-label">  RMS (L):</span>
                        <span class="status-value" style="color: ${{leftColor}}">${{stats.rms_left.toFixed(4)}}</span>
                    </div>
                    <div class="status-item">
                        <span class="status-label">  RMS (R):</span>
                        <span class="status-value" style="color: ${{rightColor}}">${{stats.rms_right.toFixed(4)}}</span>
                    </div>
                    <div class="status-item">
                        <span class="status-label">  dB Range:</span>
                        <span class="status-value">${{stats.spec_min_db.toFixed(1)}} to ${{stats.spec_max_db.toFixed(1)}}</span>
                    </div>
                    <div class="status-item">
                        <span class="status-label">  FPS:</span>
                        <span class="status-value">${{currentFps.toFixed(1)}}</span>
                    </div>
                    <div class="status-item" style="margin-top: 15px; border-top: 1px solid #4a5568; padding-top: 15px;">
                        <span class="status-label">  Status:</span>
                        <span class="status-value" style="color: #48bb78; font-weight: 700;">LIVE</span>
                    </div>
                `;
            }} catch (e) {{
                console.error('Error updating:', e);
            }} finally {{
                isUpdating = false;
            }}
        }});
        
        function updateColourmap() {{
            currentColourmap = document.getElementById('colourmap').value;
        }}
        
        //Fast updates with incremental data
        setInterval(updateData, UPDATE_INTERVAL);
        setTimeout(updateData, 500);
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@socketio.on('request_data')
def handle_data_request():
    try:
        data = system.get_incremental_data()
        if data:
            emit('data_update', data)
    except Exception as e:
        logger.error(f"Error sending data: {e}", exc_info=True)


def main():
    logger.info("="*60)
    logger.info("Real-time Audio Spectrogram")
    logger.info("="*60)
    
    devices = AudioDeviceManager.list_devices()
    logger.info("\nAvailable audio devices:")
    for dev in devices:
        indicator = " [LOOPBACK]" if dev['is_loopback'] else ""
        logger.info(f"  [{dev['index']}] {dev['name']}{indicator}")
    
    try:
        system.start()
        
        logger.info("\n" + "="*60)
        logger.info("🚀 Starting server...")
        logger.info("="*60)
        logger.info("\n   Open browser: http://127.0.0.1:8050")
        logger.info("     20 Hz - 20 kHz Freq Range")
        logger.info(f"     Update rate: {UPDATE_INTERVAL_MS}ms (20 FPS)")
        logger.info("\n  Hop/FFT Relationship:")
        logger.info(f"   FFT Size: {config.fft_size} samples")
        logger.info(f"   Hop Size: {config.hop_size} samples")
        logger.info(f"     Every {config.hop_size} samples, compute FFT on last {config.fft_size} samples")
        logger.info(f"     Processing rate: {config.sample_rate/config.hop_size:.2f} fps")
        logger.info("\n" + "="*60 + "\n")
        
        socketio.run(app, host='127.0.0.1', port=8050, debug=False, allow_unsafe_werkzeug=True)
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        system.stop()
    
    logger.info("Terminated successfully")


if __name__ == "__main__":
    main()