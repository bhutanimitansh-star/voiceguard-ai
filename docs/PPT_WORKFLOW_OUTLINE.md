# VoiceGuard AI — PPT-Ready Workflow Outline

Use this outline directly as slide content (one heading = one slide).

---

**Slide 1 — Title**
VoiceGuard AI
Deep Learning System for Human vs AI-Generated Voice Detection

**Slide 2 — Problem Statement**
- Voice cloning & TTS are now indistinguishable from real speech to the human ear
- Growing risk: fraud, misinformation, impersonation
- Need: fast, explainable, deployable detection system

**Slide 3 — Solution Overview**
- End-to-end deep learning pipeline: Human vs AI Voice classification
- Explainable predictions (not a black box)
- Sub-3-second inference, REST API, web dashboard

**Slide 4 — System Architecture (high-level)**
Client (React) → FastAPI → Preprocessing → Feature Extraction →
CNN+BiLSTM+Attention Model → Explainability (Grad-CAM/Attention) →
SQLite History → JSON Response

**Slide 5 — Audio Preprocessing Pipeline**
1. Decode (.wav/.mp3/.flac/.m4a)
2. Resample to 16kHz mono
3. VAD-based silence removal
4. Noise reduction (spectral gating)
5. Loudness normalization
6. Fixed-window segmentation

**Slide 6 — Feature Engineering**
- Mel Spectrogram (primary model input)
- MFCC (40 coefficients)
- Chroma
- Spectral Centroid & Rolloff
- Zero Crossing Rate
- Pitch (F0)
- Harmonic-to-Noise Ratio (HNR)

**Slide 7 — Model Architecture**
Mel Spectrogram → Residual CNN Blocks → BiLSTM (x2) → Attention →
Dense Fusion (+ acoustic stats) → Softmax [Human / AI Voice]

**Slide 8 — Why CNN + BiLSTM + Attention?**
- CNN: local time-frequency artifact detection
- BiLSTM: long-range prosody/pitch dependency modeling
- Attention: performance boost + free explainability signal

**Slide 9 — Explainable AI**
- Attention weights → suspicious region timeline on waveform
- Grad-CAM → time-frequency heatmap over Mel spectrogram
- Builds user trust, supports human-in-the-loop review

**Slide 10 — Training Strategy**
- Datasets: Fake-or-Real, ASVspoof, LibriSpeech, ElevenLabs/TTS
- Stratified K-Fold Cross-Validation
- Data Augmentation: noise, time-shift, gain, SpecAugment
- Early Stopping + ReduceLROnPlateau

**Slide 11 — Evaluation Metrics**
- Precision / Recall / F1 Score
- ROC-AUC
- Confusion Matrix
- Cross-validated averages across folds

**Slide 12 — Backend (FastAPI)**
- POST /api/predict
- GET /api/history
- DELETE /api/history
- GET /api/health
- SQLite persistence via SQLAlchemy

**Slide 13 — Frontend (React + Tailwind)**
- Dashboard — activity overview & trends
- Voice Analyzer — upload & real-time verdict
- History — full prediction log
- About Model — architecture & pipeline transparency
- Dark glassmorphism UI, purple + cyan accents

**Slide 14 — Performance Targets**
- < 3 seconds inference for a 10-second clip
- Modular, swappable model checkpoint
- Deployment-ready (Docker-friendly structure)

**Slide 15 — Tech Stack Summary**
PyTorch · FastAPI · React · Vite · Tailwind CSS · Recharts ·
SQLite/SQLAlchemy · Librosa · WebRTC VAD

**Slide 16 — Future Enhancements**
- Real-time streaming detection
- Multi-lingual robustness testing
- Adversarial robustness against voice-conversion attacks
- Model distillation for on-device/edge deployment

**Slide 17 — Thank You / Q&A**
