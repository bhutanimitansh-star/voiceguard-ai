"""
VoiceGuard AI - PyTorch Dataset & Augmentation
================================================

Loads precomputed features from packed memory-mapped .npy files.

Expected structure:

data/processed/packed_cache/
    training_mel.npy
    training_aux.npy
    training_labels.npy
    validation_mel.npy
    validation_aux.npy
    validation_labels.npy
    testing_mel.npy
    testing_aux.npy
    testing_labels.npy

Training augmentation:
    - SpecAugment: frequency masking + time masking

No librosa feature extraction happens during training.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def spec_augment(
    mel_db: np.ndarray,
    freq_mask_pct=0.1,
    time_mask_pct=0.1,
    num_masks=2
) -> np.ndarray:
    """
    Apply SpecAugment directly to a cached Mel spectrogram.
    """

    mel_db = mel_db.copy()

    n_mels, n_frames = mel_db.shape

    for _ in range(num_masks):

        # Frequency masking
        f = int(
            n_mels * freq_mask_pct * random.random()
        )

        if f > 0:
            f0 = random.randint(
                0,
                max(0, n_mels - f)
            )

            mel_db[
                f0:f0 + f,
                :
            ] = mel_db.min()

        # Time masking
        t = int(
            n_frames * time_mask_pct * random.random()
        )

        if t > 0:
            t0 = random.randint(
                0,
                max(0, n_frames - t)
            )

            mel_db[
                :,
                t0:t0 + t
            ] = mel_db.min()

    return mel_db


class VoiceGuardDataset(Dataset):

    def __init__(
        self,
        manifest_csv: str,
        train: bool = True,
        augment_prob: float = 0.5,
        split: str = None,
    ):

        self.df = pd.read_csv(manifest_csv)

        # -------------------------------------------------
        # Select requested split
        # -------------------------------------------------

        if split is not None:
            self.df = self.df[
                self.df["split"] == split
            ].reset_index(drop=True)

        self.train = train
        self.augment_prob = augment_prob

        # -------------------------------------------------
        # Locate packed cache
        # -------------------------------------------------

        processed_dir = (
            Path(manifest_csv).resolve().parent
        )

        self.packed_cache_dir = (
            processed_dir / "packed_cache"
        )

        if not self.packed_cache_dir.exists():
            raise FileNotFoundError(
                f"Packed feature cache not found:\n"
                f"{self.packed_cache_dir}\n\n"
                f"Run pack_feature_cache.py first."
            )

        if split is None:
            raise ValueError(
                "split must be provided. "
                "Use training, validation, or testing."
            )

        # -------------------------------------------------
        # Load memory-mapped arrays
        # -------------------------------------------------

        mel_path = (
            self.packed_cache_dir
            / f"{split}_mel.npy"
        )

        aux_path = (
            self.packed_cache_dir
            / f"{split}_aux.npy"
        )

        labels_path = (
            self.packed_cache_dir
            / f"{split}_labels.npy"
        )

        for path in [
            mel_path,
            aux_path,
            labels_path
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Packed cache file missing:\n{path}"
                )

        print(
            f"Loading packed {split} features..."
        )

        # mmap_mode='r' means:
        # - data stays on disk
        # - only requested portions are loaded
        # - RAM usage stays low

        self.mel = np.load(
            mel_path,
            mmap_mode="r"
        )

        self.aux = np.load(
            aux_path,
            mmap_mode="r"
        )

        self.labels = np.load(
            labels_path,
            mmap_mode="r"
        )

        # -------------------------------------------------
        # Safety check
        # -------------------------------------------------

        if len(self.df) != len(self.mel):
            raise ValueError(
                f"Manifest/cache size mismatch for {split}: "
                f"manifest={len(self.df)}, "
                f"cache={len(self.mel)}"
            )

        if len(self.mel) != len(self.aux):
            raise ValueError(
                "Mel and auxiliary feature counts do not match."
            )

        if len(self.mel) != len(self.labels):
            raise ValueError(
                "Mel and label counts do not match."
            )

        print(
            f"{split}: {len(self.mel)} samples loaded"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        # -------------------------------------------------
        # Direct memory-mapped access
        # -------------------------------------------------

        mel_db = np.array(
            self.mel[idx],
            dtype=np.float32,
            copy=True
        )

        aux = np.array(
            self.aux[idx],
            dtype=np.float32,
            copy=True
        )

        label = int(
            self.labels[idx]
        )

        # -------------------------------------------------
        # Safety against NaN / infinity
        # -------------------------------------------------

        aux = np.nan_to_num(
            aux,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # -------------------------------------------------
        # Fast SpecAugment
        # -------------------------------------------------

        if (
            self.train
            and random.random() < self.augment_prob
        ):
            mel_db = spec_augment(
                mel_db
            )

        # -------------------------------------------------
        # Convert to tensors
        # -------------------------------------------------

        mel_tensor = torch.from_numpy(
            mel_db
        ).unsqueeze(0)

        aux_tensor = torch.from_numpy(
            aux
        )

        label_tensor = torch.tensor(
            label,
            dtype=torch.long
        )

        return (
            mel_tensor,
            aux_tensor,
            label_tensor
        )