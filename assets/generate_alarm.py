# assets/generate_alarm.py
# Generates a valid alarm.wav using numpy
# Run once: python assets/generate_alarm.py

import numpy as np
import wave
import struct
import os

def generate_alarm(output_path: str = "assets/alarm.wav"):
    """
    Generate a beeping alarm sound in PCM WAV format.
    This format is guaranteed to work with pygame.

    Creates a 2-second alarm with alternating high/low beeps.
    """
    # Audio settings
    sample_rate = 44100   # Hz — standard CD quality
    duration    = 5.0     # seconds
    n_samples   = int(sample_rate * duration)

    # Generate alternating beep pattern
    # 880 Hz for 0.25s then 440 Hz for 0.25s, repeat
    samples = np.zeros(n_samples)

    for i in range(n_samples):
        t = i / sample_rate
        # Alternate between 880Hz and 440Hz every 0.25 seconds
        cycle = int(t / 0.25) % 2
        freq  = 880 if cycle == 0 else 440

        # Sine wave
        samples[i] = np.sin(2 * np.pi * freq * t)

    # Apply fade in/out to avoid clicking
    fade_samples = int(0.01 * sample_rate)   # 10ms fade
    for i in range(fade_samples):
        factor = i / fade_samples
        samples[i]                  *= factor
        samples[n_samples - 1 - i] *= factor

    # Scale to 16-bit PCM range
    samples = (samples * 32767 * 0.7).astype(np.int16)

    # Write WAV file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with wave.open(output_path, 'w') as wav_file:
        wav_file.setnchannels(1)           # mono
        wav_file.setsampwidth(2)           # 16-bit
        wav_file.setframerate(sample_rate) # 44100 Hz
        wav_file.writeframes(samples.tobytes())

    size_kb = os.path.getsize(output_path) / 1024
    print(f"✅ Alarm sound generated: {output_path}")
    print(f"   Format: PCM 16-bit mono @ {sample_rate}Hz")
    print(f"   Duration: {duration}s")
    print(f"   Size: {size_kb:.1f} KB")
    print(f"   This format is guaranteed to work with pygame.")


if __name__ == "__main__":
    generate_alarm()