"""
create_mixture.py
=================
Generates a synthetic 2‑speaker + noise mixture for speaker extraction research.

Prerequisites:
    - my_video.mp4         : a short video of the target speaker (you)
    - interferer.wav       : a different speaker's speech (obtained by running
                              download_demo_data.py)
    - (optional) noise.wav : a background noise file; if missing, pink noise
                              is generated automatically.

Output (in smoke_test_audio/):
    clean_target.wav       : isolated target speech (ground truth)
    interferer.wav         : interfering speaker speech (copy for reference)
    mixture.wav            : the combined signal to be separated

The script:
  - extracts mono audio at 16 kHz from the video using moviepy + soundfile
  - loads the interferer audio (16 kHz mono)
  - loads/generates background noise
  - mixes all three at realistic signal‑to‑noise ratios
  - saves the three WAV files

"""

import os
import tempfile
import numpy as np
import soundfile as sf
from moviepy.editor import VideoFileClip, AudioFileClip
from scipy.signal import lfilter
from pathlib import Path

# =============================================================================
# Project‑relative paths (works on any machine)
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

TARGET_VIDEO          = PROJECT_ROOT / "my_video.mp4"        # your recording
INTERFERER_AUDIO      = PROJECT_ROOT / "interferer.wav"      # downloaded interfering speech
NOISE_FILE            = PROJECT_ROOT / "noise.wav"           # optional, else pink noise
OUTPUT_DIR            = PROJECT_ROOT / "smoke_test_audio"    # output folder

TARGET_SAMPLE_RATE    = 16000
TARGET_TO_INTERFERER_SNR_DB = 0.0      # equal loudness
TARGET_TO_NOISE_SNR_DB      = 20.0     # noise is 20 dB quieter

# =============================================================================
# Audio extraction utilities
# =============================================================================

def extract_audio_from_video(video_path: str, sample_rate: int = 16000) -> np.ndarray:
    """
    Extract mono audio from a video file, resampled to `sample_rate`.

    Uses moviepy to open the video and writes a temporary WAV file
    (16‑bit PCM) at the desired sample rate, then loads it with soundfile.
    This approach avoids compatibility issues with torchaudio backends
    and the `to_soundarray()` method of moviepy.

    Args:
        video_path: path to the input video (e.g. .mp4, .webm)
        sample_rate: target sample rate in Hz (default 16000)

    Returns:
        1D numpy array of float32 audio samples (range approx. [-1, 1])
    """
    clip = VideoFileClip(video_path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        temp_wav = tmp.name
    try:
        clip.audio.write_audiofile(
            temp_wav,
            fps=sample_rate,
            nbytes=2,
            codec='pcm_s16le',
            verbose=False,
            logger=None
        )
        audio, _ = sf.read(temp_wav)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)         # convert to mono
    finally:
        os.unlink(temp_wav)                    # clean up temporary file
    return audio.astype(np.float32)


def load_audio_mono(file_path: str, sample_rate: int = 16000) -> np.ndarray:
    """
    Load any audio file (m4a, mp3, wav, ...) as mono float32 at `sample_rate`.

    Uses the same robust moviepy → temporary WAV → soundfile pipeline as
    `extract_audio_from_video`, so it works regardless of the moviepy/numpy version.

    Args:
        file_path: path to the audio file
        sample_rate: target sample rate in Hz

    Returns:
        1D float32 numpy array of audio samples
    """
    clip = AudioFileClip(file_path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        temp_wav = tmp.name
    try:
        clip.write_audiofile(
            temp_wav,
            fps=sample_rate,
            nbytes=2,
            codec='pcm_s16le',
            verbose=False,
            logger=None
        )
        audio, _ = sf.read(temp_wav)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
    finally:
        os.unlink(temp_wav)
    return audio.astype(np.float32)

# =============================================================================
# Noise generation / loading
# =============================================================================

def generate_pink_noise(length: int) -> np.ndarray:
    """
    Generate a 1/f (pink) noise signal of given length.

    Pink noise has a natural, softer sound than white noise and resembles
    ambient background hum. It is created by filtering white noise with a
    carefully designed IIR filter.

    Args:
        length: number of samples to generate

    Returns:
        1D float32 numpy array with unit variance and zero mean.
    """
    white = np.random.randn(length)
    b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
    a = np.array([1.0, -2.494956002, 2.017265875, -0.522189400])
    pink = lfilter(b, a, white)
    pink -= np.mean(pink)
    pink /= np.std(pink)
    return pink.astype(np.float32)


def load_or_generate_noise(target_length: int) -> np.ndarray:
    """
    Load noise from NOISE_FILE if it exists; otherwise generate pink noise.

    Args:
        target_length: desired length in samples

    Returns:
        1D float32 numpy array of noise.
    """
    if NOISE_FILE.exists():
        print("   Loading provided noise file ...")
        noise = load_audio_mono(str(NOISE_FILE), TARGET_SAMPLE_RATE)
        # Match length
        if len(noise) >= target_length:
            noise = noise[:target_length]
        else:
            noise = np.tile(noise, int(np.ceil(target_length / len(noise))))[:target_length]
    else:
        print("   No noise file found, generating pink noise ...")
        noise = generate_pink_noise(target_length)
    return noise

# =============================================================================
# Signal mixing
# =============================================================================

def mix_to_snr(target: np.ndarray,
               interferer: np.ndarray,
               noise: np.ndarray,
               t2i_snr: float,
               t2n_snr: float) -> np.ndarray:
    """
    Scale interferer and noise relative to target according to given SNRs.

    Args:
        target: clean target speech
        interferer: competing speech (will be scaled)
        noise: background noise (will be scaled)
        t2i_snr: target‑to‑interferer SNR in dB (positive = target louder)
        t2n_snr: target‑to‑noise SNR in dB

    Returns:
        Mixed signal (target + scaled interferer + scaled noise), normalised
        to prevent clipping.
    """
    def rms(x):
        return np.sqrt(np.mean(x ** 2))

    rms_t = rms(target)

    # Scale interferer
    rms_i = rms(interferer)
    desired_rms_i = rms_t / (10.0 ** (t2i_snr / 20.0))
    scale_i = desired_rms_i / (rms_i + 1e-8)

    # Scale noise
    rms_n = rms(noise)
    desired_rms_n = rms_t / (10.0 ** (t2n_snr / 20.0))
    scale_n = desired_rms_n / (rms_n + 1e-8)

    mixture = target + interferer * scale_i + noise * scale_n

    # Prevent clipping
    max_val = np.max(np.abs(mixture))
    if max_val > 0.95:
        mixture = mixture * (0.95 / max_val)

    return mixture.astype(np.float32)


def match_length(source: np.ndarray, target_length: int) -> np.ndarray:
    """
    Trim or loop `source` so its length equals `target_length`.

    Args:
        source: input 1D array
        target_length: desired length in samples

    Returns:
        Source trimmed (if longer) or tiled (if shorter).
    """
    if len(source) >= target_length:
        return source[:target_length]
    else:
        repeats = int(np.ceil(target_length / len(source)))
        return np.tile(source, repeats)[:target_length]

# =============================================================================
# Main pipeline
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Extract target speech from video
    print("1. Extracting target speech from video ...")
    target = extract_audio_from_video(TARGET_VIDEO, TARGET_SAMPLE_RATE)
    print(f"   Target: {len(target)} samples ({len(target)/TARGET_SAMPLE_RATE:.2f} s)")

    # 2. Load interfering speaker
    print("2. Loading interfering speaker ...")
    interferer = load_audio_mono(INTERFERER_AUDIO, TARGET_SAMPLE_RATE)
    print(f"   Interferer raw: {len(interferer)} samples ({len(interferer)/TARGET_SAMPLE_RATE:.2f} s)")
    interferer = match_length(interferer, len(target))
    print(f"   After length matching: {len(interferer)} samples")

    # 3. Load or generate background noise
    print("3. Preparing background noise ...")
    noise = load_or_generate_noise(len(target))

    # 4. Mix all sources
    print("4. Mixing signals ...")
    mixture = mix_to_snr(target, interferer, noise,
                         TARGET_TO_INTERFERER_SNR_DB,
                         TARGET_TO_NOISE_SNR_DB)

    # 5. Save output files
    sf.write(str(OUTPUT_DIR / "clean_target.wav"), target, TARGET_SAMPLE_RATE)
    sf.write(str(OUTPUT_DIR / "interferer.wav"), interferer, TARGET_SAMPLE_RATE)
    sf.write(str(OUTPUT_DIR / "mixture.wav"), mixture, TARGET_SAMPLE_RATE)

    print("\n✅ Done! Files saved in", OUTPUT_DIR)
    print("   clean_target.wav   – target speaker isolated voice (ground truth)")
    print("   interferer.wav     – the interfering speaker")
    print("   mixture.wav        – the mixed audio to separate")

if __name__ == "__main__":
    main()