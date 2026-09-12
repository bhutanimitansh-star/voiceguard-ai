# Resume / Portfolio Description

## Short version (for a resume bullet list)

**VoiceGuard AI — Deep Learning System for Human vs AI-Generated Voice Detection**
*PyTorch, FastAPI, React, SQLite | Personal Project*

- Designed and built an end-to-end deep learning system that classifies audio
  as human or AI-generated speech using a custom CNN + BiLSTM + Attention
  architecture trained on Mel spectrograms, achieving explainable, sub-3-second
  inference on 10-second clips.
- Engineered a full audio preprocessing pipeline (resampling, VAD-based
  silence removal, spectral-gating noise reduction, loudness normalization,
  windowed segmentation) and a multi-feature extraction suite (MFCC, Chroma,
  spectral centroid/rolloff, ZCR, pitch/F0, harmonic-to-noise ratio).
- Implemented Grad-CAM and attention-based explainability to visualize which
  time-frequency regions of the spectrogram drove each prediction, surfaced
  as highlighted "suspicious regions" directly on the audio timeline.
- Built a production-style FastAPI backend (REST endpoints, SQLite
  persistence, Pydantic schemas, async lifespan model loading) and a
  responsive React + Tailwind dashboard with Recharts visualizations,
  drag-and-drop upload, and a full prediction history view.
- Set up a reproducible training pipeline with stratified K-fold
  cross-validation, early stopping, LR scheduling, data augmentation
  (Gaussian noise, time-shift, gain perturbation, SpecAugment), and
  standard evaluation reporting (Precision/Recall/F1, ROC-AUC, confusion
  matrix).

## Longer version (for a portfolio site / case study page)

**VoiceGuard AI** is a full-stack, production-style application that detects
whether an audio clip contains human or AI-generated (synthetic/TTS/vocoder)
speech. The project spans the entire ML product lifecycle: dataset
preparation, deep learning model design, explainability, backend API
design, and frontend UX.

**Problem.** As TTS and voice-cloning technology has become widely
accessible, distinguishing authentic human speech from synthetic audio has
become a practical need across trust & safety, fraud prevention, and media
verification use cases.

**Approach.** Rather than a shallow classifier over hand-crafted features,
VoiceGuard AI uses a hybrid deep architecture: a residual convolutional
front-end extracts local time-frequency patterns from log-Mel spectrograms
(useful for catching vocoder artifacts and unnatural spectral textures); a
bidirectional LSTM models long-range temporal dependencies in prosody and
pitch contour; and an additive attention layer both improves classification
performance and doubles as a built-in explainability mechanism, since its
per-timestep weights indicate exactly which parts of a clip most influenced
the verdict. A Grad-CAM pass over the final convolutional layer adds a
complementary frequency-localized heatmap.

**Engineering.** The system is split cleanly into a PyTorch/FastAPI backend
(REST API, SQLite persistence, modular preprocessing and inference
pipelines) and a React/Vite/Tailwind frontend (dashboard, analyzer,
history, and model-info pages) styled with a dark glassmorphism aesthetic
and purple/cyan accent palette. Inference is optimized to stay under the
3-second target for a 10-second clip by keeping the model resident in
memory, using a compact 3-block residual CNN, and batching windowed
inference efficiently.

**Skills demonstrated:** deep learning architecture design (CNN/RNN/
Attention), audio signal processing, PyTorch training pipelines (K-fold CV,
augmentation, LR scheduling, early stopping), explainable AI (Grad-CAM,
attention visualization), REST API design with FastAPI, relational data
modeling with SQLAlchemy/SQLite, and modern React frontend engineering with
data visualization (Recharts).
