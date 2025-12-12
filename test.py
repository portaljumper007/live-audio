import numpy as np
import cupy as cp
import sounddevice as sd
import threading
import time
from collections import deque

class RealtimeAudioFFT:
    def __init__(self, sample_rate=44100, fft_interval=0.1):
        self.sample_rate = sample_rate
        self.fft_interval = fft_interval  #0.1 seconds = 10 times per second
        self.buffer_duration = 1.0  #1 second of audio data
        self.buffer_size = int(self.sample_rate * self.buffer_duration)
        
        #Circular buffers for left and right channels
        self.left_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.right_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.buffer_index = 0
        self.buffer_lock = threading.Lock()
        
        #Audio stream
        self.stream = None
        self.running = False
        
    def audio_callback(self, indata, frames, time, status):
        """Audio callback function - called by sounddevice for each audio block"""
        if status:
            print(f"Audio status: {status}")
        
        #Left and right channels
        left_data = indata[:, 0]
        right_data = indata[:, 1]
        
        with self.buffer_lock:
            #Add new data to circular buffers
            for i in range(frames):
                self.left_buffer[self.buffer_index] = left_data[i]
                self.right_buffer[self.buffer_index] = right_data[i]
                self.buffer_index = (self.buffer_index + 1) % self.buffer_size
    
    def get_current_buffers(self):
        """Get current 1-second buffers in correct chronological order"""
        with self.buffer_lock:
            #Create properly ordered arrays (oldest to newest samples)
            left_ordered = np.concatenate([
                self.left_buffer[self.buffer_index:],
                self.left_buffer[:self.buffer_index]
            ])
            right_ordered = np.concatenate([
                self.right_buffer[self.buffer_index:],
                self.right_buffer[:self.buffer_index]
            ])
        return left_ordered.copy(), right_ordered.copy()
    
    def process_fft(self):
        """Process FFT on current audio buffers using CuPy"""
        #Get current 1-second audio data
        left_data, right_data = self.get_current_buffers()
        
        #Transfer to GPU
        left_gpu = cp.asarray(left_data)
        right_gpu = cp.asarray(right_data)
        
        #Perform real FFT on GPU
        left_fft = cp.fft.rfft(left_gpu)
        right_fft = cp.fft.rfft(right_gpu)
        
        #Calculate amplitudes
        left_amplitudes = cp.abs(left_fft)
        right_amplitudes = cp.abs(right_fft)
        
        #Optional: Transfer back to CPU if needed for further processing
        #left_amplitudes_cpu = cp.asnumpy(left_amplitudes)
        #right_amplitudes_cpu = cp.asnumpy(right_amplitudes)
        
        print(f"FFT processed - Left: {left_amplitudes.shape}, Right: {right_amplitudes.shape}")
        return left_amplitudes, right_amplitudes
    
    def fft_loop(self):
        """Main FFT processing loop - runs 10 times per second"""
        while self.running:
            start_time = time.time()
            
            try:
                self.process_fft()
            except Exception as e:
                print(f"FFT processing error: {e}")
            
            #Calculate sleep time to maintain 10 Hz processing rate
            elapsed = time.time() - start_time
            sleep_time = max(0, self.fft_interval - elapsed)
            time.sleep(sleep_time)
    
    def start(self):
        """Start audio recording and FFT processing"""
        self.running = True
        
        #Start audio stream
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=2,  #Stereo
            callback=self.audio_callback,
            blocksize=1024,  #Process in 1024-sample blocks
            dtype=np.float32
        )
        
        self.stream.start()
        print(f"Audio stream started - Sample rate: {self.sample_rate} Hz")
        
        #Start FFT processing thread
        self.fft_thread = threading.Thread(target=self.fft_loop, daemon=True)
        self.fft_thread.start()
        print("FFT processing started - 10 Hz rate")
    
    def stop(self):
        """Stop audio recording and FFT processing"""
        self.running = False
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
        
        print("Audio processing stopped")

def main():
    processor = RealtimeAudioFFT(sample_rate=44100)
    
    try:
        processor.start()
        
        #Keep the main thread alive
        print("Processing audio... Press Ctrl+C to stop")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
        processor.stop()

if __name__ == "__main__":
    main()