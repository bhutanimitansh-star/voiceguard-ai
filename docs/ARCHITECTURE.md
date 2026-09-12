# VoiceGuard AI — System Architecture

## 1. High-Level System Diagram

```mermaid
flowchart TB
    subgraph Client["React + Vite + Tailwind Frontend"]
        A1[Voice Analyzer Page]
        A2[Dashboard Page]
        A3[History Page]
        A4[About Model Page]
    end

    subgraph API["FastAPI Backend"]
        B1["POST /api/predict"]
        B2["GET /api/history"]
        B3["DELETE /api/history"]
        B4["GET /api/health"]
    end

    subgraph Pipeline["Preprocessing Pipeline"]
        C1[Decode: wav/mp3/flac/m4a] --> C2[Resample to 16kHz mono]
        C2 --> C3[VAD Silence Removal]
        C3 --> C4[Noise Reduction]
        C4 --> C5[Loudness Normalization]
        C5 --> C6[Fixed-Window Segmentation]
    end

    subgraph Features["Feature Engineering"]
        D1[Mel Spectrogram]
        D2[MFCC - 40 coeff]
        D3[Chroma]
        D4[Spectral Centroid/Rolloff]
        D5[Zero Crossing Rate]
        D6[Pitch F0]
        D7[Harmonic-to-Noise Ratio]
    end

    subgraph Model["VoiceGuardNet (PyTorch)"]
        E1[Residual CNN Blocks] --> E2[BiLSTM x2]
        E2 --> E3[Additive Attention]
        E3 --> E4[Dense Fusion Head]
        E4 --> E5["Softmax: Human / AI Voice"]
    end

    subgraph XAI["Explainability"]
        F1[Attention -> Timeline Highlighting]
        F2[Grad-CAM -> Spectrogram Heatmap]
    end

    subgraph Storage["SQLite"]
        G1[(prediction_history table)]
    end

    A1 -->|multipart upload| B1
    B1 --> C1
    C6 --> Features
    Features --> D1
    D1 --> E1
    Features -.aux stats.-> E4
    E5 --> F1
    E5 --> F2
    F1 --> B1
    F2 --> B1
    B1 --> G1
    B1 -->|JSON response| A1
    A3 --> B2
    A3 --> B3
    B2 --> G1
    B3 --> G1
    A2 --> B2
    A4 --> B4
```

## 2. Model Architecture Detail

```mermaid
flowchart LR
    In["Log-Mel Spectrogram\n(1, 128, T)"] --> Stem["Conv2D Stem\n+BN+ReLU"]
    Stem --> R1["ResidualBlock 1\n(32ch, stride 1)"]
    R1 --> R2["ResidualBlock 2\n(64ch, stride 2)"]
    R2 --> R3["ResidualBlock 3\n(128ch, stride 2)"]
    R3 --> Reshape["Reshape:\ncollapse Freq into Channel\n-> sequence over Time"]
    Reshape --> LSTM["BiLSTM x2\n(hidden=128, bidirectional)"]
    LSTM --> Attn["Additive Attention\n(context + weights)"]
    Attn --> Fuse["Concat with\nAcoustic Aux Features (9-dim)"]
    Fuse --> Dense["Dense(64) + ReLU + Dropout"]
    Dense --> Out["Softmax\n[Human, AI Voice]"]

    Attn -. attention weights .-> XAI1[Timeline Highlighting]
    R3 -. Grad-CAM hook .-> XAI2[Spectrogram Heatmap]
```

## 3. Request Lifecycle (POST /api/predict)

1. User uploads audio via drag-and-drop in the **Voice Analyzer** page.
2. FastAPI validates extension (`.wav/.mp3/.flac/.m4a`) and size (≤ 25MB).
3. `preprocessing.audio_preprocessing.preprocess_audio()` decodes, resamples,
   trims silence (VAD), denoises, normalizes loudness, and segments into
   4-second windows.
4. For each window, `preprocessing.feature_extraction.extract_features()`
   computes the Mel spectrogram (model input) plus MFCC/Chroma/Centroid/
   Rolloff/ZCR/Pitch/HNR (auxiliary features + UI display).
5. `models.inference.VoiceGuardInference.predict()` runs each window through
   `VoiceGuardNet`, averages window-level softmax probabilities into a
   clip-level verdict, and computes:
   - Attention-derived suspicious region timeline
   - Grad-CAM heatmap over the last conv block
   - Waveform + Mel spectrogram + Grad-CAM overlay PNGs (base64-encoded)
6. Result is persisted to SQLite (`prediction_history` table) and returned
   as JSON to the frontend, which renders the verdict banner, confidence
   gauge, probability bars, waveform/spectrogram images, and suspicious
   timeline.

End-to-end target latency for a 10-second clip: **< 3 seconds** on CPU for
inference alone (excluding network transfer), achieved via a lightweight
3-block residual CNN, single-pass windowed inference, and caching the
model in memory at process startup (no cold-start reload per request).

## 4. Folder Structure

```
voiceguard-ai/
│
├── backend/
│   ├── api/                  # FastAPI routes + Pydantic schemas
│   ├── models/                # Architecture, inference engine, explainability
│   ├── preprocessing/         # Audio cleaning + feature extraction
│   ├── training/               # Dataset prep, PyTorch Dataset, training loop
│   ├── config.py               # Central configuration
│   ├── database.py             # SQLite models + CRUD helpers
│   ├── app.py                  # FastAPI entrypoint
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── components/         # Sidebar, Topbar, Dropzone, charts, timeline
│   │   ├── pages/               # Dashboard, VoiceAnalyzer, History, AboutModel
│   │   └── services/api.js      # Axios API layer
│   ├── package.json
│   └── tailwind.config.js
│
├── dataset/
│   ├── raw/                     # User-provided raw corpora (gitignored)
│   └── processed/                # Cached waveforms + manifest CSVs
│
├── notebooks/                    # EDA / experimentation notebooks
├── docs/                          # This file, API docs, resume description
└── README.md
```
