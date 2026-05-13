# Lightweight Visual-Guided Voice Anonymization

A privacy‑preserving pipeline that extracts a target speaker's voice from a noisy video **by watching their lips**, then anonymizes it using the WORLD vocoder — all running **offline on CPU**.

## 🚀 Overview

1. **Lip‑activity scoring** with MediaPipe Face Landmarker.
2. **Speech separation** using a pre‑trained Conv‑TasNet (Sepformer).
3. **Audio‑visual speaker selection** via energy‑lip correlation.
4. **Optional denoising** with `noisereduce`.
5. **Voice anonymization** using the WORLD vocoder (pitch shifting + formant warping).

Designed for resource‑constrained devices and fully offline.

---

## 📁 Project Structure
.
├── my_video.mp4 # Your recorded target video (not committed)
├── create_mixture.py # Mix target + interferer + noise
├── download_demo_data.py # Fetches public interferer & generates noise
├── src/
│ ├── lip_scorer.py # Visual lip‑activity estimator
│ ├── separate_with_lips.py # Lip‑guided speaker extraction
│ ├── denoise.py # Stationary noise reduction (optional)
│ ├── anonymize.py # WORLD‑based voice anonymization
│ └── evaluate.py # STOI intelligibility & spectrograms
├── requirements.txt
├── README.md
├── spectrograms.png # Generated after evaluation
└── smoke_test_audio/ # Output folder (generated, not committed)

## 🧪 Quick Start

### 1. Prerequisites
- Python 3.9+
- Working webcam (to record a short test video)
- Git (optional, for cloning)

### 2. Clone & install
```bash
git clone https://github.com/yourusername/visual-voice-anonymizer.git
cd visual-voice-anonymizer
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
3. Download required models & data
Face landmarker
Download face_landmarker.task and place it in the project root.

Separation model
Create pretrained_models/sepformer-whamr16k/ and copy all files from Hugging Face into that folder.

Demo interferer & noise

bash
python src/download_demo_data.py
This downloads a random LibriSpeech utterance (interferer.wav) and generates pink noise (noise.wav).

4. Prepare your target video
Record a short (~5 second) video of yourself speaking clearly, looking at the camera.
Save it as my_video.mp4 in the project root.

5. Run the pipeline
bash
# Generate mixture
python create_mixture.py

# Extract target speaker guided by lip movements
python src/separate_with_lips.py

# (Optional) Denoise the extracted speech
python src/denoise.py

# Anonymize the voice
python src/anonymize.py

# Evaluate intelligibility & generate spectrograms
python src/evaluate.py
The anonymised speech will be saved in smoke_test_audio/anonymized.wav.

📊 Evaluation Results
Metric	STOI
Extracted vs Clean Target	0.527
Anonymized vs Clean Target	0.449
STOI > 0.4 indicates good intelligibility. The pipeline correctly selects the target speaker via lip‑activity correlation and alters vocal identity while preserving words.

https://spectrograms.png

🔮 Future Work (Planned for TEEP Internship)
Replace WORLD vocoder with a neural voice conversion model (e.g., StarGAN‑VC) conditioned on speaker embeddings for more natural anonymization.

Deploy the system on a Raspberry Pi for real‑time edge processing.

Extend to mixed‑lingual scenarios, integrating mixed‑lingual ASR techniques.

📚 References
SpeechBrain Sepformer: speechbrain/sepformer-whamr16k

MediaPipe Face Landmarker: Google MediaPipe

WORLD vocoder: mmorise/World