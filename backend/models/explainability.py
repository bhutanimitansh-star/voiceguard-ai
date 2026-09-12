"""
VoiceGuard AI - Explainable AI (XAI) Module
==============================================
Two complementary explainability signals are produced per prediction:

1. Attention-based timeline highlighting
   The BiLSTM attention weights directly tell us which time steps the
   model relied on most - upsampled back to the original waveform's
   time axis, these become "suspicious region" highlights in the UI.

2. Grad-CAM over the final residual conv block
   Produces a coarse time-frequency heatmap over the Mel spectrogram,
   showing *which frequency bands and time windows* pushed the
   prediction toward "AI Voice" - useful for spotting vocoder artifacts
   concentrated in specific frequency bands (e.g. high-frequency
   spectral smearing common in some neural vocoders).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def attention_to_timeline(attn_weights: torch.Tensor, num_samples: int) -> np.ndarray:
    """Upsample attention weights (per CNN time-step) to per-audio-sample
    resolution so the frontend can highlight suspicious regions directly
    on the waveform timeline.

    Args:
        attn_weights: (T',) attention weights for a single clip
        num_samples: number of raw audio samples in the analyzed window

    Returns:
        np.ndarray of shape (num_samples,) with values in [0, 1]
    """
    weights = attn_weights.detach().cpu().numpy()
    weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-9)
    # Linear interpolation from T' attention steps to num_samples
    x_old = np.linspace(0, 1, num=len(weights))
    x_new = np.linspace(0, 1, num=num_samples)
    timeline = np.interp(x_new, x_old, weights)
    return timeline.astype(np.float32)


def find_suspicious_regions(timeline: np.ndarray, sr: int, threshold: float = 0.65,
                             min_region_sec: float = 0.15) -> list:
    """Convert a per-sample attention timeline into discrete
    (start_sec, end_sec, intensity) suspicious regions above `threshold`.
    """
    above = timeline >= threshold
    regions = []
    start = None
    for i, flag in enumerate(above):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            end = i
            if (end - start) / sr >= min_region_sec:
                intensity = float(np.mean(timeline[start:end]))
                regions.append({
                    "start_sec": round(start / sr, 3),
                    "end_sec": round(end / sr, 3),
                    "intensity": round(intensity, 3),
                })
            start = None
    if start is not None:
        end = len(above)
        if (end - start) / sr >= min_region_sec:
            intensity = float(np.mean(timeline[start:end]))
            regions.append({
                "start_sec": round(start / sr, 3),
                "end_sec": round(end / sr, 3),
                "intensity": round(intensity, 3),
            })
    return regions


class GradCAM:
    """Grad-CAM implementation targeting the last residual conv block of
    VoiceGuardNet (`model.block3`). Produces a (freq, time) heatmap
    aligned with the Mel spectrogram axes.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, mel_spectrogram: torch.Tensor, aux_features: torch.Tensor,
                 target_class: int) -> np.ndarray:
        """
        Args:
            mel_spectrogram: (1, 1, n_mels, T) - single example, requires_grad not needed
            aux_features: (1, 9)
            target_class: index of the class to explain (0=Human, 1=AI Voice)

        Returns:
            heatmap: (freq', time') normalized to [0, 1]
        """
        self.model.eval()
        mel_spectrogram = mel_spectrogram.clone().requires_grad_(True)

        logits, _ = self.model(mel_spectrogram, aux_features)
        score = logits[0, target_class]

        self.model.zero_grad()
        score.backward()

        # Global-average-pool gradients over spatial dims -> channel weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = F.relu((weights * self.activations).sum(dim=1)).squeeze(0)  # (F', T')

        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)
        return cam


def resize_heatmap_to_spectrogram(heatmap: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Nearest/bilinear-free numpy resize of the Grad-CAM heatmap back to
    the original Mel spectrogram's (n_mels, T) resolution for overlay."""
    heatmap_t = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0)  # (1,1,F',T')
    resized = F.interpolate(heatmap_t, size=target_shape, mode="bilinear", align_corners=False)
    return resized.squeeze().numpy()
