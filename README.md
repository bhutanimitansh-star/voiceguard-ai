# 🛡️ VoiceGuard AI

**Deep learning system that detects whether an audio clip is Human or AI-generated speech**, with explainable predictions, a FastAPI backend, and a React dashboard.

> Built as a portfolio-level, production-style project. See `docs/ARCHITECTURE.md` for diagrams, `docs/API_DOCUMENTATION.md` for the API reference, and `docs/RESUME_DESCRIPTION.md` for a ready-to-use project write-up.

---

## ✨ Features

- Upload `.wav`, `.mp3`, `.flac`, `.m4a` audio
- Automatic 16kHz mono conversion, VAD silence trimming, denoising, loudness normalization
- **CNN + BiLSTM + Attention** deep learning classifier (PyTorch)
- Confidence score + per-class probabilities (Human / AI Voice)
- Waveform + Mel spectrogram visualization
- **Explainable AI**: attention-based suspicious-region highlighting + Grad-CAM heatmaps
- Prediction history stored in SQLite
- Dark glassmorphism dashboard (React + Tailwind + Recharts)
- Target inference latency: **< 3 seconds** for a 10-second clip

---

## 📁 Project Structure

```
voiceguard-ai/
├── backend/            # FastAPI + PyTorch backend
│   ├── api/            # Routes & Pydantic schemas
│   ├── models/          # Model architecture, inference engine, explainability
│   ├── preprocessing/   # Audio cleaning & feature extraction
│   ├── training/         # Dataset prep, PyTorch Dataset, training loop
│   ├── config.py
│   ├── database.py
│   ├── app.py
│   └── requirements.txt
├── frontend/           # React + Vite + Tailwind dashboard
├── dataset/            # raw/ (your corpora) + processed/ (cached features)
├── notebooks/          # EDA / experimentation
├── docs/               # Architecture, API docs, PPT outline, resume text
└── README.md
```

---

## 🚀 Quick Start

### 1. Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# ffmpeg is required by pydub for audio decoding:
#   macOS:   brew install ffmpeg
#   Ubuntu:  sudo apt-get install ffmpeg
#   Windows: choco install ffmpeg

uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API is now live at `http://localhost:8000` (Swagger docs at `/docs`).

> **Note:** On first run, no trained checkpoint exists yet, so the model
> runs with randomly initialized weights (the server logs a warning).
> Train a real model first — see below — for meaningful predictions.

### 2. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Visit `http://localhost:5173`. Vite is pre-configured to proxy `/api` calls
to `http://localhost:8000`.

### 3. Train the Model (optional but required for real predictions)

```bash
cd backend/training

# 1. Organize raw datasets under dataset/raw/ (see prepare_dataset.py header
#    for expected sub-folder layout matching LibriSpeech, Fake-or-Real,
#    ASVspoof, and TTS-generated samples)

python prepare_dataset.py --raw_dir ../../dataset/raw --out_dir ../../dataset/processed

python train.py --manifest ../../dataset/processed/cached_manifest.csv --epochs 60 --k_folds 5
```

This trains `VoiceGuardNet` with stratified K-fold cross-validation, early
stopping, and LR scheduling, then saves the best checkpoint to
`backend/models/checkpoints/voiceguard_cnn_bilstm_attn.pt` — which
`app.py` automatically loads on the next backend restart.

---

## 🧠 Model Architecture

```
Mel Spectrogram (1, 128, T)
        │
   Residual CNN Blocks (32 → 64 → 128 channels, BatchNorm + Dropout)
        │
   Reshape (collapse frequency → sequence over time)
        │
   BiLSTM ×2 (hidden=128, bidirectional)
        │
   Additive Attention (context vector + explainability weights)
        │
   Concat with acoustic auxiliary features (MFCC/Chroma/Centroid/
   Rolloff/ZCR/Pitch/HNR summary stats)
        │
   Dense(64) → Dropout → Softmax
        │
   [Human, AI Voice]
```

Full diagrams: `docs/ARCHITECTURE.md`

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/predict` | Upload audio, get Human/AI verdict + explainability |
| GET | `/api/history` | List past predictions |
| DELETE | `/api/history` | Clear prediction history |
| GET | `/api/health` | Service/model health check |

Full request/response examples: `docs/API_DOCUMENTATION.md`

---

## 🎨 Frontend Pages

| Page | Purpose |
|------|---------|
| **Dashboard** | Scan counts, verdict distribution, confidence trend |
| **Voice Analyzer** | Upload audio, view verdict, spectrogram, Grad-CAM, suspicious regions |
| **History** | Full SQLite-backed prediction log |
| **About Model** | Architecture, pipeline stages, evaluation metrics |

---

## 🧪 Datasets Used for Training

- [Fake-or-Real Voice Dataset](https://bil.eecs.yorku.ca/datasets/)
- [ASVspoof](https://www.asvspoof.org/)
- [LibriSpeech](https://www.openslr.org/12) (human speech)
- ElevenLabs / other TTS-generated samples (synthetic class)

See `backend/training/prepare_dataset.py` for the expected folder layout
and manifest generation logic.

---

## 📊 Evaluation

Training reports the following per fold and averaged across folds:
- Precision, Recall, F1 Score
- ROC-AUC
- Confusion Matrix

All metrics are logged to TensorBoard (`backend/training/runs/`).

---

## 🛣️ Roadmap

- Real-time streaming detection
- Multi-lingual robustness benchmarking
- Adversarial robustness against voice-conversion attacks
- Model distillation for edge/on-device deployment

---

## 📄 License

MIT — use freely for portfolio, learning, or as a foundation for production systems.
