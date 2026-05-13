"""
evaluate.py
===========
Objective evaluation of the voice extraction + anonymization pipeline.

Computes:
  - STOI (Short‑Time Objective Intelligibility) – higher is better (0‑1).
  - A 2×2 spectrogram comparison saved as 'spectrograms.png'.

Requires:
  - clean_target.wav      (ground truth, from create_mixture.py)
  - extracted_target.wav  (from separate_with_lips.py)
  - enhanced_target.wav   (optional, from denoise.py)
  - anonymized.wav        (optional, from anonymize.py)

If enhanced_target.wav or anonymized.wav are missing, the corresponding
metrics and subplots are gracefully skipped.
"""

import numpy as np
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
from pystoi import stoi

# =============================================================================
# Project‑relative paths (works on any machine)
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLEAN_TARGET     = PROJECT_ROOT / "smoke_test_audio" / "clean_target.wav"
EXTRACTED_TARGET = PROJECT_ROOT / "smoke_test_audio" / "extracted_target.wav"
ENHANCED_TARGET  = PROJECT_ROOT / "smoke_test_audio" / "enhanced_target.wav"
ANONYMIZED       = PROJECT_ROOT / "smoke_test_audio" / "anonymized.wav"

SAMPLE_RATE = 16000

# =============================================================================
# Helper functions
# =============================================================================

def load_wav(path: Path) -> np.ndarray:
    """
    Load a WAV file as float32 mono at 16 kHz.

    Args:
        path: path to the WAV file.

    Returns:
        1D float32 numpy array.
    """
    sr, data = wavfile.read(str(path))
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    else:
        data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)          # force mono
    if sr != SAMPLE_RATE:
        from scipy.signal import resample
        data = resample(data, int(len(data) * SAMPLE_RATE / sr))
    return data


def compute_stoi(clean: np.ndarray, degraded: np.ndarray) -> float:
    """
    Compute STOI between clean and degraded speech signals.

    Args:
        clean: reference speech.
        degraded: processed speech to evaluate.

    Returns:
        STOI score (0 to 1, higher = better intelligibility).
    """
    return stoi(clean, degraded, SAMPLE_RATE, extended=False)


def make_spectrogram_plot(clean: np.ndarray,
                          extracted: np.ndarray,
                          enhanced: np.ndarray | None,
                          anonymized: np.ndarray | None) -> None:
    """
    Save a 2×2 spectrogram comparison figure to PROJECT_ROOT/spectrograms.png.

    If enhanced or anonymized are None, the corresponding subplot shows the
    extracted signal (so the plot is always complete).
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    titles = ["Clean Target", "Extracted Target",
              "Enhanced (Denoised)" if enhanced is not None else "Extracted (no denoising)",
              "Anonymized" if anonymized is not None else "Extracted (no anonymization)"]
    signals = [
        clean,
        extracted,
        enhanced if enhanced is not None else extracted,
        anonymized if anonymized is not None else extracted
    ]

    for ax, sig, title in zip(axes.flatten(), signals, titles):
        f, t, Sxx = spectrogram(sig, SAMPLE_RATE, nperseg=512, noverlap=256)
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        ax.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='inferno')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [sec]')
        ax.set_title(title)

    plt.tight_layout()
    output_path = PROJECT_ROOT / "spectrograms.png"
    plt.savefig(str(output_path), dpi=150)
    plt.close()
    print(f"Spectrogram saved to {output_path}")

# =============================================================================
# Main evaluation
# =============================================================================

def main():
    print("Loading files ...")
    clean = load_wav(CLEAN_TARGET)
    extracted = load_wav(EXTRACTED_TARGET)

    # Optional pipeline outputs – load if they exist
    enhanced = load_wav(ENHANCED_TARGET) if ENHANCED_TARGET.exists() else None
    anonymized = load_wav(ANONYMIZED) if ANONYMIZED.exists() else None

    # Trim all signals to the same length
    min_len = len(clean)
    for sig in [extracted, enhanced, anonymized]:
        if sig is not None:
            min_len = min(min_len, len(sig))
    clean = clean[:min_len]
    extracted = extracted[:min_len]
    if enhanced is not None:
        enhanced = enhanced[:min_len]
    if anonymized is not None:
        anonymized = anonymized[:min_len]

    # --- STOI ---
    print("\n--- STOI (higher = better) ---")
    stoi_ext = compute_stoi(clean, extracted)
    print(f"Extracted vs Clean:  {stoi_ext:.3f}")

    if enhanced is not None:
        stoi_enh = compute_stoi(clean, enhanced)
        print(f"Enhanced vs Clean:   {stoi_enh:.3f}")
    if anonymized is not None:
        stoi_anon = compute_stoi(clean, anonymized)
        print(f"Anonymized vs Clean: {stoi_anon:.3f}")

    # --- Spectrograms ---
    make_spectrogram_plot(clean, extracted, enhanced, anonymized)

if __name__ == "__main__":
    if not CLEAN_TARGET.exists():
        raise FileNotFoundError(f"Clean target not found: {CLEAN_TARGET}")
    if not EXTRACTED_TARGET.exists():
        raise FileNotFoundError(f"Extracted speech not found: {EXTRACTED_TARGET}")
    main()