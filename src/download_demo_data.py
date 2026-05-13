"""
download_demo_data.py
======================
Downloads public‑domain speech and generates synthetic background noise
for testing the visual‑guided voice anonymization pipeline.

What it does:
  1. Downloads a random utterance from LibriSpeech dev‑clean (CC‑BY‑4.0)
     and saves it as 'interferer.wav' in the project root.
  2. Generates pink noise (1/f) and saves it as 'noise.wav' in the project root.

After running this script, you can use create_mixture.py with your own
target video (my_video.mp4) to produce the full mixture – no personal
audio files are needed.

Requirements:
  - torch, torchaudio (already in requirements.txt)
  - numpy, scipy, soundfile (already in requirements.txt)

Usage:
  python src/download_demo_data.py
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from scipy.signal import lfilter
import torch
import torchaudio
import torchaudio.functional as F
from torchaudio.datasets import LIBRISPEECH

# =============================================================================
# Project‑relative paths (works on any machine)
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INTERFERER_PATH = PROJECT_ROOT / "interferer.wav"
NOISE_PATH      = PROJECT_ROOT / "noise.wav"

SAMPLE_RATE = 16000          # everything runs at 16 kHz
NOISE_DURATION_SEC = 10.0    # enough to cover most test videos

# =============================================================================
# Interferer download
# =============================================================================

def download_interferer():
    """
    Download a random LibriSpeech dev‑clean utterance and save as mono 16 kHz.
    """
    print("Downloading a random LibriSpeech dev‑clean utterance ...")
    dataset = LIBRISPEECH(root=str(PROJECT_ROOT / "data"), url="dev-clean", download=True)
    idx = np.random.randint(0, len(dataset))
    waveform, sr, _, _, _, _ = dataset[idx]

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if necessary
    if sr != SAMPLE_RATE:
        waveform = F.resample(waveform, sr, SAMPLE_RATE)

    audio = waveform.squeeze(0).numpy().astype(np.float32)
    sf.write(str(INTERFERER_PATH), audio, SAMPLE_RATE)
    print(f"✅ Saved interfering speech to '{INTERFERER_PATH}' "
          f"({len(audio)} samples, {len(audio)/SAMPLE_RATE:.1f} s)")

# =============================================================================
# Noise generation
# =============================================================================

def generate_pink_noise(duration_sec: float) -> np.ndarray:
    """
    Generate a 1/f (pink) noise signal of given duration.

    Args:
        duration_sec: length in seconds.

    Returns:
        1D float32 numpy array with unit variance.
    """
    length = int(duration_sec * SAMPLE_RATE)
    white = np.random.randn(length)

    # Pink‑noise IIR filter coefficients
    b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
    a = np.array([1.0, -2.494956002, 2.017265875, -0.522189400])
    pink = lfilter(b, a, white)

    # Normalise (zero mean, unit variance)
    pink -= np.mean(pink)
    pink /= np.std(pink)
    return pink.astype(np.float32)


def create_noise():
    """
    Generate pink noise and save as 'noise.wav'.
    """
    print(f"Generating {NOISE_DURATION_SEC:.0f} seconds of pink noise ...")
    noise = generate_pink_noise(NOISE_DURATION_SEC)
    sf.write(str(NOISE_PATH), noise, SAMPLE_RATE)
    print(f"✅ Saved background noise to '{NOISE_PATH}' "
          f"({len(noise)} samples, {len(noise)/SAMPLE_RATE:.1f} s)")

# =============================================================================
if __name__ == "__main__":
    download_interferer()
    create_noise()
    print("\nAll demo data ready. You can now run create_mixture.py with your own video (my_video.mp4).")