"""
VoiceGuard AI - Audio Preprocessing Pipeline
=============================================
Responsible for turning an arbitrary uploaded audio file into a clean,
normalized, fixed-format waveform ready for feature extraction.

Pipeline stages:
    1. Decode (any container -> raw PCM) via pydub (ffmpeg backend)
    2. Resample to 16 kHz mono
    3. Voice Activity Detection (VAD) based silence trimming
    4. Noise reduction (spectral gating)
    5. Loudness normalization (RMS/EBU-ish target)
    6. Fixed-length windowing (for batched CNN input)
"""

from __future__ import annotations

import io
import struct
import wave
from dataclasses import dataclass
from typing import List

import numpy as np
import librosa
import noisereduce as nr
import webrtcvad
from pydub import AudioSegment

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    MAX_AUDIO_SECONDS,
    VAD_AGGRESSIVENESS,
    TARGET_LOUDNESS_DB,
)


@dataclass
class PreprocessResult:
    waveform: np.ndarray          # float32, mono, 16kHz, cleaned
    windows: List[np.ndarray]     # fixed-length chunks for the model
    duration_sec: float
    trimmed_silence_sec: float


def load_any_audio(file_bytes: bytes, filename: str) -> AudioSegment:
    """Decode wav/mp3/flac/m4a bytes into a pydub AudioSegment.

    pydub shells out to ffmpeg, which transparently handles all four
    required container formats without us needing per-format branches.
    """
    fmt = filename.split(".")[-1].lower()
    audio = AudioSegment.from_file(io.BytesIO(file_bytes), format=fmt)
    return audio


def to_mono_16k(audio: AudioSegment) -> np.ndarray:
    """Convert a pydub AudioSegment to mono float32 PCM at SAMPLE_RATE."""
    audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples /= 32768.0  # int16 -> [-1, 1]

    max_len = MAX_AUDIO_SECONDS * SAMPLE_RATE
    if len(samples) > max_len:
        samples = samples[:max_len]
    return samples


def apply_vad_trim(waveform: np.ndarray, frame_ms: int = 30) -> np.ndarray:
    """Remove leading/trailing/interior silence using WebRTC VAD.

    WebRTC VAD requires 16-bit PCM mono audio at 8/16/32/48kHz with
    frames of 10/20/30ms - all satisfied by our pipeline.
    """
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    frame_len = int(SAMPLE_RATE * frame_ms / 1000)
    pcm16 = (waveform * 32768).astype(np.int16)

    voiced_frames = []
    for start in range(0, len(pcm16) - frame_len, frame_len):
        frame = pcm16[start:start + frame_len]
        frame_bytes = struct.pack("<%dh" % len(frame), *frame)
        try:
            is_speech = vad.is_speech(frame_bytes, SAMPLE_RATE)
        except Exception:
            is_speech = True  # fail-open: keep frame if VAD errors
        if is_speech:
            voiced_frames.append(frame)

    if not voiced_frames:
        # Nothing detected as speech (e.g. very quiet clip) -> keep original
        return waveform

    return np.concatenate(voiced_frames).astype(np.float32) / 32768.0


def reduce_noise(waveform: np.ndarray) -> np.ndarray:
    """Spectral-gating noise reduction using the first 0.5s as noise profile
    when the clip is long enough, otherwise a stationary estimate."""
    if len(waveform) < SAMPLE_RATE * 0.5:
        return waveform
    try:
        reduced = nr.reduce_noise(y=waveform, sr=SAMPLE_RATE, stationary=True)
        return reduced.astype(np.float32)
    except Exception:
        return waveform


def normalize_loudness(waveform: np.ndarray, target_db: float = TARGET_LOUDNESS_DB) -> np.ndarray:
    """RMS-based loudness normalization to a target dBFS level."""
    rms = np.sqrt(np.mean(waveform ** 2)) + 1e-9
    current_db = 20 * np.log10(rms)
    gain_db = target_db - current_db
    gain_linear = 10 ** (gain_db / 20)
    normalized = waveform * gain_linear
    # Prevent clipping
    peak = np.max(np.abs(normalized)) + 1e-9
    if peak > 1.0:
        normalized = normalized / peak
    return normalized.astype(np.float32)


def segment_fixed_windows(waveform: np.ndarray, window_samples: int = WINDOW_SAMPLES,
                           hop_ratio: float = 0.5) -> List[np.ndarray]:
    """Split a long waveform into fixed-length, overlapping windows.

    Short clips are center-padded (zero-pad) up to one full window so the
    model always receives a fixed-size tensor.
    """
    if len(waveform) <= window_samples:
        pad = window_samples - len(waveform)
        left = pad // 2
        right = pad - left
        return [np.pad(waveform, (left, right), mode="constant")]

    hop = int(window_samples * hop_ratio)
    windows = []
    for start in range(0, len(waveform) - window_samples + 1, hop):
        windows.append(waveform[start:start + window_samples])

    # Make sure the tail of the clip is captured
    if (len(waveform) - window_samples) % hop != 0:
        windows.append(waveform[-window_samples:])

    return windows


def preprocess_audio(file_bytes: bytes, filename: str) -> PreprocessResult:
    """Full preprocessing pipeline: decode -> resample -> VAD -> denoise ->
    normalize -> window. Returns everything downstream stages need."""
    audio_segment = load_any_audio(file_bytes, filename)
    raw_waveform = to_mono_16k(audio_segment)
    raw_duration = len(raw_waveform) / SAMPLE_RATE

    trimmed = apply_vad_trim(raw_waveform)
    trimmed_duration = len(trimmed) / SAMPLE_RATE
    silence_removed = max(0.0, raw_duration - trimmed_duration)

    denoised = reduce_noise(trimmed)
    normalized = normalize_loudness(denoised)
    windows = segment_fixed_windows(normalized)

    return PreprocessResult(
        waveform=normalized,
        windows=windows,
        duration_sec=raw_duration,
        trimmed_silence_sec=silence_removed,
    )
