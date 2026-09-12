"""
VoiceGuard AI - Global Configuration
=====================================
Centralized configuration for audio processing, model, and API settings.
Using a single source of truth avoids magic numbers scattered across the
codebase and makes the system easy to re-tune for new datasets/hardware.
"""
import torch
import os
from pathlib import Path

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
MODEL_DIR = BASE_DIR / "models" / "checkpoints"
DB_PATH = BASE_DIR / "voiceguard.db"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Audio preprocessing
# ------------------------------------------------------------------
SAMPLE_RATE = 16_000          # target sample rate (Hz)
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128
N_MFCC = 40
WINDOW_SECONDS = 4.0          # fixed-length window fed to the model
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)
MAX_AUDIO_SECONDS = 60        # reject/trim clips longer than this
VAD_AGGRESSIVENESS = 2        # 0 (least aggressive) - 3 (most aggressive)
TARGET_LOUDNESS_DB = -23.0    # EBU R128-ish target for loudness normalization

# ------------------------------------------------------------------
# Model / Inference
# ------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() and os.environ.get("FORCE_CPU") != "1" else "cpu"
NUM_CLASSES = 2               # 0 = Human, 1 = AI-generated
CLASS_NAMES = ["Human", "AI Voice"]
CHECKPOINT_PATH = MODEL_DIR / "voiceguard_cnn_bilstm_attn.pt"
INFERENCE_BATCH_SIZE = 8
CONFIDENCE_DECIMALS = 2

# ------------------------------------------------------------------
# API
# ------------------------------------------------------------------
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a"}
MAX_UPLOAD_MB = 25
CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]

# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------
TRAIN_CONFIG = {
    "batch_size": 32,
    "num_epochs": 60,
    "learning_rate": 3e-4,
    "weight_decay": 1e-5,
    "early_stopping_patience": 8,
    "lr_scheduler_patience": 4,
    "lr_scheduler_factor": 0.5,
    "k_folds": 5,
    "val_split": 0.15,
    "test_split": 0.15,
    "seed": 42,
}
