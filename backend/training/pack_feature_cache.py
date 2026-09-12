"""
Pack individual feature-cache .npz files into memory-mapped .npy arrays.

This greatly reduces training I/O overhead by replacing thousands of
small file reads with a few large memory-mapped files.
"""

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd


def get_cache_path(feature_cache_dir, waveform_path):
    cache_key = hashlib.md5(
        str(waveform_path).encode()
    ).hexdigest()

    return feature_cache_dir / f"{cache_key}.npz"


def pack_split(manifest_path, split, output_dir):
    print(f"\n===== Packing {split} split =====")

    df = pd.read_csv(manifest_path)

    df = df[
        df["split"] == split
    ].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(
            f"No samples found for split: {split}"
        )

    feature_cache_dir = (
        Path(manifest_path).resolve().parent
        / "feature_cache"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # -------------------------------------------------
    # Inspect first cache file to determine shapes
    # -------------------------------------------------

    first_path = get_cache_path(
        feature_cache_dir,
        df.iloc[0]["waveform_path"]
    )

    if not first_path.exists():
        raise FileNotFoundError(
            f"Missing cache file:\n{first_path}"
        )

    with np.load(first_path) as first:
        mel_shape = first["mel"].shape
        aux_shape = first["aux"].shape

    print(f"Samples : {len(df)}")
    print(f"Mel shape: {mel_shape}")
    print(f"Aux shape: {aux_shape}")

    # -------------------------------------------------
    # Create memory-mapped output arrays
    # -------------------------------------------------

    mel_path = output_dir / f"{split}_mel.npy"
    aux_path = output_dir / f"{split}_aux.npy"
    labels_path = output_dir / f"{split}_labels.npy"

    mel_array = np.lib.format.open_memmap(
        mel_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(df), *mel_shape)
    )

    aux_array = np.lib.format.open_memmap(
        aux_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(df), *aux_shape)
    )

    labels_array = np.lib.format.open_memmap(
        labels_path,
        mode="w+",
        dtype=np.int64,
        shape=(len(df),)
    )

    # -------------------------------------------------
    # Copy individual cache files
    # -------------------------------------------------

    for i, row in df.iterrows():

        cache_path = get_cache_path(
            feature_cache_dir,
            row["waveform_path"]
        )

        if not cache_path.exists():
            raise FileNotFoundError(
                f"Missing cache file:\n{cache_path}"
            )

        with np.load(cache_path) as cached:

            mel = cached["mel"].astype(
                np.float32
            )

            aux = cached["aux"].astype(
                np.float32
            )

        aux = np.nan_to_num(
            aux,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        mel_array[i] = mel
        aux_array[i] = aux
        labels_array[i] = int(row["label"])

        if (i + 1) % 500 == 0:
            print(
                f"Processed {i + 1}/{len(df)}"
            )

    # Flush memory-mapped files
    mel_array.flush()
    aux_array.flush()
    labels_array.flush()

    del mel_array
    del aux_array
    del labels_array

    print(f"✓ Finished {split}")


def main():

    manifest_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "processed"
        / "cached_manifest.csv"
    )

    output_dir = (
        Path(manifest_path).parent
        / "packed_cache"
    )

    print("Manifest:")
    print(manifest_path)

    print("\nOutput:")
    print(output_dir)

    for split in [
        "training",
        "validation",
        "testing"
    ]:
        pack_split(
            str(manifest_path),
            split,
            output_dir
        )

    print("\n================================")
    print("PACKING COMPLETE")
    print("================================")


if __name__ == "__main__":
    main()