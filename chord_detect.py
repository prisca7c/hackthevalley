# chord_detector.py
import sounddevice as sd
import numpy as np
import librosa
import time

# -----------------------------
# Config
# -----------------------------
SAMPLE_RATE = 22050
BLOCK_SIZE = 2048
ENERGY_THRESHOLD = 0.00001     # adjust to your mic/environment
POLYPHONY_THRESHOLD = 3        # number of distinct pitches to consider a chord
DETECTION_FRAMES = 2           # chord must persist for these many consecutive frames

# -----------------------------
# Buffer to reduce false positives
# -----------------------------
chord_buffer = []

# Callback function to trigger when chord is detected
_chord_event_handler = None


# -----------------------------
# Chord detection function
# -----------------------------
def is_chord(y, sr=SAMPLE_RATE):
    if np.mean(y**2) < ENERGY_THRESHOLD:
        return False
    
    y_harmonic, _ = librosa.effects.hpss(y)
    pitches, magnitudes = librosa.piptrack(y=y_harmonic, sr=sr)
    num_notes = np.sum(np.max(magnitudes, axis=1) > 0.1)
    
    return num_notes >= POLYPHONY_THRESHOLD


# -----------------------------
# Audio callback
# -----------------------------
def audio_callback(indata, frames, time_info, status):
    global chord_buffer, _chord_event_handler
    if status:
        print(status)
    
    audio_data = indata[:, 0]  # mono
    
    if is_chord(audio_data, SAMPLE_RATE):
        chord_buffer.append(1)
    else:
        chord_buffer.append(0)
    
    if len(chord_buffer) > DETECTION_FRAMES:
        chord_buffer.pop(0)
    
    if sum(chord_buffer) == DETECTION_FRAMES:
        if _chord_event_handler:
            _chord_event_handler()  # Call user-defined event
        else:
            print("Chord detected!")  # Fallback


# -----------------------------
# Public API
# -----------------------------
def start_listening(device_index=None, on_chord_detected=None):
    """
    Start listening for chords using a live microphone stream.
    :param device_index: Optional microphone index.
    :param on_chord_detected: Function to call when chord is detected.
    """
    global _chord_event_handler
    _chord_event_handler = on_chord_detected

    if device_index is not None:
        sd.default.device = device_index
    
    print("Using device:", sd.query_devices(sd.default.device))
    with sd.InputStream(callback=audio_callback,
                        channels=1,
                        samplerate=SAMPLE_RATE,
                        blocksize=BLOCK_SIZE):
        print("Listening for chords... Press Ctrl+C to stop.")
        while True:
            time.sleep(0.1)


# -----------------------------
# Run standalone
# -----------------------------
if __name__ == "__main__":
    def example_handler():
        print(">>> Custom chord event triggered!")
    start_listening(device_index=1, on_chord_detected=example_handler)
