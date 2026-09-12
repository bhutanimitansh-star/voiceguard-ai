"""
VoiceGuard AI - Feature Engineering
====================================
Extracts the acoustic feature set used both for model input (Mel Spectrogram)
and for auxiliary/explainability + statistical signals (MFCC, Chroma,
Spectral Centroid/Rolloff, ZCR, Pitch/F0, Harmonic-to-Noise Ratio).

The Mel Spectrogram is the primary tensor fed to the CNN+BiLSTM+Attention
network. The remaining hand-crafted features are exposed in the API
response payload as supplementary signals and are also concatenated as
extra channels/summary stats for the model's auxiliary head.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np
import librosa

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS, N_MFCC


@dataclass
class AcousticFeatures:
    mel_spectrogram_db: np.ndarray   # (n_mels, T) - primary model input
    mfcc: np.ndarray                 # (n_mfcc, T)
    chroma: np.ndarray               # (12, T)
    spectral_centroid: np.ndarray    # (T,)
    spectral_rolloff: np.ndarray     # (T,)
    zero_crossing_rate: np.ndarray   # (T,)
    pitch_f0: np.ndarray             # (T,) with NaN for unvoiced frames
    hnr: float                       # scalar harmonic-to-noise ratio (dB)

    def summary_stats(self) -> Dict[str, float]:
        """Collapse frame-level features into scalar stats for logging /
        the auxiliary dense head in the model."""
        def _safe(arr, fn):
            arr = arr[~np.isnan(arr)] if np.issubdtype(arr.dtype, np.floating) else arr
            return float(fn(arr)) if len(arr) else 0.0

        return {
            "mfcc_mean": _safe(self.mfcc, np.mean),
            "mfcc_std": _safe(self.mfcc, np.std),
            "chroma_mean": _safe(self.chroma, np.mean),
            "spectral_centroid_mean": _safe(self.spectral_centroid, np.mean),
            "spectral_rolloff_mean": _safe(self.spectral_rolloff, np.mean),
            "zcr_mean": _safe(self.zero_crossing_rate, np.mean),
            "pitch_mean_hz": _safe(self.pitch_f0, np.nanmean),
            "pitch_std_hz": _safe(self.pitch_f0, np.nanstd),
            "hnr_db": self.hnr,
        }


def compute_mel_spectrogram(waveform: np.ndarray) -> np.ndarray:
    """Log-scaled Mel spectrogram - the core input tensor for the CNN."""
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=SAMPLE_RATE, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)


def compute_hnr(waveform: np.ndarray) -> float:
    """Harmonic-to-Noise Ratio estimate via harmonic/percussive separation.

    HNR = 10 * log10( harmonic_energy / (noise_energy + eps) )
    A cleaner, more "tonal" human voice tends to have a distinct HNR
    profile compared to some TTS/vocoder artifacts, making it a useful
    auxiliary discriminative feature.
    """
    harmonic, percussive = librosa.effects.hpss(waveform)
    harmonic_energy = np.sum(harmonic ** 2)
    noise_energy = np.sum(percussive ** 2)
    hnr_db = 10 * np.log10((harmonic_energy + 1e-9) / (noise_energy + 1e-9))
    return float(hnr_db)


def extract_features(waveform: np.ndarray) -> AcousticFeatures:
    """Run the full acoustic feature extraction suite on a waveform window."""
    mel_db = compute_mel_spectrogram(waveform)

    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    ).astype(np.float32)

    chroma = librosa.feature.chroma_stft(
        y=waveform,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    ).astype(np.float32)

    spectral_centroid = librosa.feature.spectral_centroid(
        y=waveform,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )[0].astype(np.float32)

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=waveform,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )[0].astype(np.float32)

    zcr = librosa.feature.zero_crossing_rate(
        y=waveform,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH,
    )[0].astype(np.float32)

    # Pitch/F0 tracking.
    # pYIN can fail on some NumPy/Numba/librosa combinations,
    # so use a safe fallback instead of crashing training.
    try:
        f0, voiced_flag, _ = librosa.pyin(
            waveform,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=SAMPLE_RATE,
        )

        if f0 is None:
            f0 = np.zeros(1, dtype=np.float32)
        else:
            f0 = np.asarray(f0, dtype=np.float32)

    except Exception:
        f0 = np.zeros(1, dtype=np.float32)

    # Replace NaN/Inf values with safe numerical values.
    f0 = np.nan_to_num(
        f0,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    hnr = compute_hnr(waveform)

    return AcousticFeatures(
        mel_spectrogram_db=mel_db,
        mfcc=mfcc,
        chroma=chroma,
        spectral_centroid=spectral_centroid,
        spectral_rolloff=spectral_rolloff,
        zero_crossing_rate=zcr,
        pitch_f0=f0,
        hnr=hnr,
    )