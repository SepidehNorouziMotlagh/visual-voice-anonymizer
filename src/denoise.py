"""
denoise.py
==========
(Optional) Apply stationary noise reduction to the extracted target speech.

Uses the 'noisereduce' library to remove residual background noise from
the separated speech, making it cleaner before anonymization.

Input:
    smoke_test_audio/extracted_target.wav

Output:
    smoke_test_audio/enhanced_target.wav

If you skip this step, anonymize.py will automatically use the raw extracted file.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from scipy.io import wavfile
import noisereduce as nr

# =============================================================================
# Project‑relative paths (works on any machine)
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH   = PROJECT_ROOT / "smoke_test_audio" / "extracted_target.wav"
OUTPUT_PATH  = PROJECT_ROOT / "smoke_test_audio" / "enhanced_target.wav"

# Denoising strength – lower values preserve more speech but leave more noise.
# Recommended range: 0.5 (mild) to 1.0 (aggressive).
PROP_DECREASE = 0.7

# =============================================================================
def main():
    # --- 1. Load audio (scipy for robust Windows reading) ---
    sr, data = wavfile.read(str(INPUT_PATH))
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    else:
        data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)          # force mono

    # --- 2. Denoise ---
    print("Applying stationary noise reduction ...")
    enhanced = nr.reduce_noise(y=data, sr=sr, stationary=True,
                               prop_decrease=PROP_DECREASE)

    # --- 3. Save ---
    sf.write(str(OUTPUT_PATH), enhanced, sr)
    print(f"✅ Denoised speech saved to '{OUTPUT_PATH}'")

# =============================================================================
if __name__ == "__main__":
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}\n"
            "Please run separate_with_lips.py first to generate extracted_target.wav"
        )
    main()