import sounddevice as sd
sd.default.device = 3 
print(sd.query_devices())

import numpy as np
import librosa

SAMPLE_RATE = 22050
BLOCK_SIZE = 2048

# Thresholds you may need to tune:
ENERGY_THRESHOLD = 0.01   # how loud must sound be
POLYPHONY_THRESHOLD = 3   # how many distinct notes → consider it a chord

def is_chord(y, sr):
    # Compute short-time Fourier transform
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    
    # Find peaks in spectrum
    spectral_peaks = np.sum(S > np.max(S) * 0.3, axis=0)  # peaks per frame
    avg_peaks = np.mean(spectral_peaks)
    
    # Compute energy
    energy = np.mean(y**2)
    
    # Heuristic: if loud enough & has multiple harmonics → chord
    if energy > ENERGY_THRESHOLD and avg_peaks > POLYPHONY_THRESHOLD:
        print("Detected a chord!")
    else:
        print("No chord detected - energy:", energy, "peaks:", avg_peaks)


def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_data = indata[:, 0]  # mono
    
    if is_chord(audio_data, SAMPLE_RATE):
        print("Chord detected!")

# Start live stream
with sd.InputStream(callback=audio_callback,
                    channels=1,
                    samplerate=SAMPLE_RATE,
                    blocksize=BLOCK_SIZE):
    print("Listening for chords... Press Ctrl+C to stop.")
    import time
    while True:
        time.sleep(0.1)
