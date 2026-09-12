"""
VoiceGuard AI - Inference Service
====================================
Loads the trained VoiceGuardNet checkpoint once at startup and exposes a
single `predict()` entry point that the FastAPI layer calls. Handles:

  - Preprocessing -> feature extraction -> tensor construction
  - Windowed inference with score aggregation for clips longer than one window
  - Attention-based timeline highlighting
  - Grad-CAM heatmap generation
  - Waveform + Mel spectrogram PNG rendering (base64) for the frontend
"""

from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa.display

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import (
    DEVICE, CHECKPOINT_PATH, CLASS_NAMES, SAMPLE_RATE,
    N_MELS, HOP_LENGTH, CONFIDENCE_DECIMALS,
)
from preprocessing import preprocess_audio, extract_features
from models.architecture import build_model
from models.explainability import (
    GradCAM, attention_to_timeline, find_suspicious_regions,
    resize_heatmap_to_spectrogram,
)


@dataclass
class PredictionResult:
    prediction: str
    confidence: float
    human_probability: float
    ai_probability: float
    processing_time: str
    suspicious_regions: List[Dict[str, Any]] = field(default_factory=list)
    waveform_png_base64: str = ""
    melspectrogram_png_base64: str = ""
    gradcam_png_base64: str = ""
    acoustic_features: Dict[str, float] = field(default_factory=dict)
    duration_sec: float = 0.0


class VoiceGuardInference:
    """Singleton-style inference wrapper. Instantiate once at app startup."""

    def __init__(self, checkpoint_path: Path = CHECKPOINT_PATH, device: str = DEVICE):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.model = build_model()
        self._load_checkpoint(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        self.gradcam = GradCAM(self.model, target_layer=self.model.block3)

    def _load_checkpoint(self, checkpoint_path: Path):
        if checkpoint_path.exists():
            state = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(state.get("model_state_dict", state))
            print(f"[VoiceGuardInference] Loaded checkpoint from {checkpoint_path}")
        else:
            print(f"[VoiceGuardInference] WARNING: no checkpoint found at {checkpoint_path}. "
                  f"Using randomly initialized weights - run training/train.py first "
                  f"to produce a real model for production use.")

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _fig_to_base64(fig) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                    facecolor="#0f0f1a", transparent=False)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def _render_waveform(self, waveform: np.ndarray, suspicious_regions: List[Dict]) -> str:
        fig, ax = plt.subplots(figsize=(9, 2.4))
        fig.patch.set_facecolor("#0f0f1a")
        ax.set_facecolor("#0f0f1a")
        t = np.linspace(0, len(waveform) / SAMPLE_RATE, num=len(waveform))
        ax.plot(t, waveform, linewidth=0.6, color="#22d3ee")  # cyan
        for region in suspicious_regions:
            ax.axvspan(region["start_sec"], region["end_sec"],
                       color="#a855f7", alpha=0.35 * region["intensity"] + 0.15)
        ax.set_xlabel("Time (s)", color="white")
        ax.set_ylabel("Amplitude", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#4b5563")
        return self._fig_to_base64(fig)

    def _render_melspectrogram(self, mel_db: np.ndarray, gradcam_overlay: np.ndarray = None) -> str:
        fig, ax = plt.subplots(figsize=(9, 3))
        fig.patch.set_facecolor("#0f0f1a")
        img = librosa.display.specshow(
            mel_db, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
            x_axis="time", y_axis="mel", ax=ax, cmap="magma",
        )
        if gradcam_overlay is not None:
            ax.imshow(gradcam_overlay, aspect="auto", origin="lower",
                      cmap="cool", alpha=0.45,
                      extent=[0, mel_db.shape[1] * HOP_LENGTH / SAMPLE_RATE, 0, mel_db.shape[0]])
        ax.set_facecolor("#0f0f1a")
        ax.tick_params(colors="white")
        cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
        return self._fig_to_base64(fig)

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------
    def predict(self, file_bytes: bytes, filename: str) -> PredictionResult:
        start_time = time.time()

        pre = preprocess_audio(file_bytes, filename)

        window_probs = []
        attn_all = []
        last_features = None
        last_mel_tensor = None

        for window in pre.windows:
            features = extract_features(window)
            last_features = features
            mel_tensor = torch.tensor(features.mel_spectrogram_db).unsqueeze(0).unsqueeze(0).to(self.device)
            aux_tensor = torch.tensor(
                list(features.summary_stats().values()), dtype=torch.float32
            ).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits, attn_weights = self.model(mel_tensor, aux_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            window_probs.append(probs)
            attn_all.append(attn_weights[0])
            last_mel_tensor = (mel_tensor, aux_tensor)

        # Aggregate window-level probabilities (mean) into a clip-level score
        avg_probs = np.mean(window_probs, axis=0)
        pred_idx = int(np.argmax(avg_probs))
        prediction_label = CLASS_NAMES[pred_idx]
        confidence = float(avg_probs[pred_idx] * 100)
        human_prob = float(avg_probs[0] * 100)
        ai_prob = float(avg_probs[1] * 100)

        # Explainability: use the last window's attention for timeline highlighting
        timeline = attention_to_timeline(attn_all[-1], num_samples=len(pre.waveform))
        suspicious_regions = find_suspicious_regions(timeline, sr=SAMPLE_RATE)

        # Grad-CAM heatmap on the last analyzed window
        mel_tensor, aux_tensor = last_mel_tensor
        cam = self.gradcam.generate(mel_tensor, aux_tensor, target_class=pred_idx)
        cam_resized = resize_heatmap_to_spectrogram(
            cam, target_shape=(last_features.mel_spectrogram_db.shape[0],
                                last_features.mel_spectrogram_db.shape[1])
        )

        waveform_png = self._render_waveform(pre.waveform, suspicious_regions)
        mel_png = self._render_melspectrogram(last_features.mel_spectrogram_db)
        gradcam_png = self._render_melspectrogram(last_features.mel_spectrogram_db, cam_resized)

        elapsed = time.time() - start_time

        return PredictionResult(
            prediction=prediction_label,
            confidence=round(confidence, CONFIDENCE_DECIMALS),
            human_probability=round(human_prob, CONFIDENCE_DECIMALS),
            ai_probability=round(ai_prob, CONFIDENCE_DECIMALS),
            processing_time=f"{elapsed:.2f} s",
            suspicious_regions=suspicious_regions,
            waveform_png_base64=waveform_png,
            melspectrogram_png_base64=mel_png,
            gradcam_png_base64=gradcam_png,
            acoustic_features=last_features.summary_stats(),
            duration_sec=round(pre.duration_sec, 2),
        )


# Module-level singleton, imported by the FastAPI app at startup.
_inference_engine: "VoiceGuardInference | None" = None


def get_inference_engine() -> VoiceGuardInference:
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = VoiceGuardInference()
    return _inference_engine
