"""
anonymize.py
============
Voice anonymization using the WORLD vocoder.

Given a clean speech file (16 kHz mono), it:
  1. Analyses the signal: fundamental frequency (F0), spectral envelope,
     and aperiodicity.
  2. Applies anonymization transforms:
        - Pitch shifting (random factor between 0.8 and 1.2)
        - Formant warping (shifts the spectral envelope along frequency axis)
  3. Synthesizes the anonymised speech.

The input is automatically chosen:
  - If 'enhanced_target.wav' exists (from optional denoising), it is used.
  - Otherwise, the raw 'extracted_target.wav' is used.

The output is saved as 'anonymized.wav'.

Runs entirely on CPU – no GPU, no internet. Ideal for edge deployment.
"""

import numpy as np
import pyworld as pw
import soundfile as sf
from pathlib import Path
from scipy.io import wavfile

# =============================================================================
# Project‑relative paths (works on any machine)
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Try the denoised version first; fall back to raw extracted if not available
if (PROJECT_ROOT / "smoke_test_audio" / "enhanced_target.wav").exists():
    INPUT_PATH = PROJECT_ROOT / "smoke_test_audio" / "enhanced_target.wav"
else:
    INPUT_PATH = PROJECT_ROOT / "smoke_test_audio" / "extracted_target.wav"

OUTPUT_PATH = PROJECT_ROOT / "smoke_test_audio" / "anonymized.wav"

SAMPLE_RATE = 16000

# Anonymisation parameters (randomised for each run)
PITCH_SHIFT_RANGE = (0.8, 1.2)          # multiply F0 by a random factor
FORMANT_WARP_RANGE = (0.9, 1.1)         # stretch/compress spectral envelope

# =============================================================================
def anonymize(input_wav: Path, output_wav: Path,
              pitch_factor: float = None,
              formant_shift: float = None) -> None:
    """
    Anonymize a speech file.

    Args:
        input_wav: path to input WAV (16 kHz mono).
        output_wav: path to save the anonymized WAV.
        pitch_factor: factor to multiply F0 (random if None).
        formant_shift: factor to stretch/compress spectral envelope (>1 = higher).
    """
    # --- 1. Load audio (scipy for robust Windows reading) ---
    sr, x = wavfile.read(str(input_wav))
    if x.dtype == np.int16:
        x = x.astype(np.float64) / 32768.0
    else:
        x = x.astype(np.float64)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {sr}")
    if x.ndim > 1:
        x = x.mean(axis=1)          # force mono

    # --- 2. WORLD analysis ---
    print("Analysing speech ...")
    f0, t = pw.dio(x, SAMPLE_RATE)                       # raw pitch
    f0 = pw.stonemask(x, f0, t, SAMPLE_RATE)             # refine pitch
    sp = pw.cheaptrick(x, f0, t, SAMPLE_RATE)            # spectral envelope
    ap = pw.d4c(x, f0, t, SAMPLE_RATE)                   # aperiodicity

    # --- 3. Anonymisation transforms ---
    if pitch_factor is None:
        pitch_factor = np.random.uniform(*PITCH_SHIFT_RANGE)
    if formant_shift is None:
        formant_shift = np.random.uniform(*FORMANT_WARP_RANGE)

    print(f"Pitch factor: {pitch_factor:.2f}  |  Formant shift: {formant_shift:.2f}")

    # Pitch shifting (only on voiced frames)
    f0_shifted = f0.copy()
    voiced = f0 > 0
    f0_shifted[voiced] *= pitch_factor
    f0_shifted[voiced] = np.clip(f0_shifted[voiced], 50, 400)   # realistic human range

    # Formant warping: shift spectral envelope along frequency axis
    num_frames, fft_size = sp.shape
    sp_warped = np.zeros_like(sp)
    original_freq_axis = np.arange(fft_size)
    for i in range(num_frames):
        warped_axis = original_freq_axis * formant_shift
        sp_warped[i, :] = np.interp(original_freq_axis, warped_axis, sp[i, :],
                                    left=sp[i, 0], right=sp[i, -1])

    # --- 4. WORLD synthesis ---
    print("Synthesising anonymised speech ...")
    y_anon = pw.synthesize(f0_shifted, sp_warped, ap, SAMPLE_RATE)
    y_anon = y_anon.astype(np.float32)

    # --- 5. Save ---
    sf.write(str(output_wav), y_anon, SAMPLE_RATE)
    print(f"✅ Anonymised speech saved to '{output_wav}'")

# =============================================================================
if __name__ == "__main__":
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}\n"
            "Please run separate_with_lips.py first to generate extracted_target.wav"
        )
    print(f"Anonymizing file: {INPUT_PATH}")
    anonymize(INPUT_PATH, OUTPUT_PATH)