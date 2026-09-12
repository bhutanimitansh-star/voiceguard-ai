"""
Precompute VoiceGuard features.

Creates cached Mel spectrograms and 9 auxiliary features
for every waveform in the processed manifest.

Run from backend:
python training/precompute_features.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from training.dataset import fit_to_window
from preprocessing.feature_extraction import (
    compute_mel_spectrogram,
    extract_features,
)


def main():

    # ---------------------------------------------------------
    # Paths
    # ---------------------------------------------------------

    manifest_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "processed"
        / "cached_manifest.csv"
    )

    cache_dir = (
        manifest_path.parent
        / "feature_cache"
    )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Load manifest
    # ---------------------------------------------------------

    import pandas as pd

    df = pd.read_csv(manifest_path)

    print(f"Total samples: {len(df)}")
    print(f"Cache directory: {cache_dir}")

    # ---------------------------------------------------------
    # Process every sample
    # ---------------------------------------------------------

    processed = 0
    skipped = 0
    failed = 0

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Precomputing features"
    ):

        waveform_path = row["waveform_path"]

        # Same cache key used by dataset.py
        cache_key = hashlib.md5(
            str(waveform_path).encode()
        ).hexdigest()

        cache_path = (
            cache_dir
            / f"{cache_key}.npz"
        )

        # Already cached
        if cache_path.exists():

            skipped += 1
            continue

        try:

            # -------------------------------------------------
            # Load waveform
            # -------------------------------------------------

            waveform = np.load(
                waveform_path
            ).astype(np.float32)

            # IMPORTANT:
            # deterministic crop for caching
            waveform = fit_to_window(waveform)

            # -------------------------------------------------
            # Mel spectrogram
            # -------------------------------------------------

            mel_db = compute_mel_spectrogram(
                waveform
            ).astype(np.float32)

            # -------------------------------------------------
            # Auxiliary features
            # -------------------------------------------------

            features = extract_features(
                waveform
            )

            aux = np.array(
                list(
                    features.summary_stats().values()
                ),
                dtype=np.float32
            )

            aux = np.nan_to_num(
                aux,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

            # -------------------------------------------------
            # Save
            # -------------------------------------------------

            np.savez(
                cache_path,
                mel=mel_db,
                aux=aux
            )

            processed += 1

        except Exception as e:

            failed += 1

            print(
                f"\nFAILED: {waveform_path}"
            )
            print(
                f"ERROR: {e}"
            )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print("\n===================================")
    print("PRECOMPUTATION COMPLETE")
    print("===================================")

    print(f"Total samples : {len(df)}")
    print(f"Newly cached  : {processed}")
    print(f"Already cached: {skipped}")
    print(f"Failed        : {failed}")

    print(
        f"\nCache location:\n{cache_dir}"
    )


if __name__ == "__main__":
    main()