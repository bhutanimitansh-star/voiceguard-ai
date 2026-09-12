"""
VoiceGuard AI - Dataset Preparation Script
=============================================
Builds a unified manifest CSV (filepath,label,source_dataset) from several
raw human/AI voice corpora, then applies the same preprocessing pipeline
used at inference time so train/serve skew is minimized.

Expected raw layout (adjust RAW_SOURCES below to your local paths):

    dataset/raw/
        librispeech/**/*.flac              -> label=human
        fake_or_real/real/*.wav            -> label=human
        fake_or_real/fake/*.wav            -> label=ai
        asvspoof/bonafide/*.flac           -> label=human
        asvspoof/spoof/*.flac              -> label=ai
        elevenlabs_tts/*.wav               -> label=ai

Usage:
    python prepare_dataset.py --raw_dir ../../dataset/raw --out_dir ../../dataset/processed
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from preprocessing import preprocess_audio

# Map of (relative_glob_pattern -> label). Extend this to match whichever
# subset of Fake-or-Real / ASVspoof / LibriSpeech / TTS corpora you have
# downloaded locally. Labels: 0 = Human, 1 = AI-generated.
RAW_SOURCES = {
    "training/real/**/*": ("human", 0, "training"),
    "training/fake/**/*": ("ai", 1, "training"),

    "validation/real/**/*": ("human", 0, "validation"),
    "validation/fake/**/*": ("ai", 1, "validation"),

    "testing/real/**/*": ("human", 0, "testing"),
    "testing/fake/**/*": ("ai", 1, "testing"),
}


def build_manifest(raw_dir: Path, out_dir: Path) -> Path:
    """Scan raw_dir according to RAW_SOURCES and write manifest.csv."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"

    rows = []
    for pattern, (source_name, label, split) in RAW_SOURCES.items():
        matches = list(raw_dir.glob(pattern))
        print(f"  {pattern:45s} -> {len(matches):6d} files (label={label}, {source_name})")
        for path in matches:
         rows.append({
    "filepath": str(path),
    "label": label,
    "source": source_name,
    "split": split
})

    if not rows:
        print("\n[WARN] No files matched any RAW_SOURCES pattern.")
        print("       Point --raw_dir at a folder containing the expected "
              "sub-directories, or edit RAW_SOURCES to match your local layout.")

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
    f,
    fieldnames=["filepath", "label", "source", "split"]
)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nManifest written to {manifest_path} ({len(rows)} total files)")
    return manifest_path


def precompute_features(manifest_path: Path, out_dir: Path):
    """Run the shared preprocessing pipeline on every manifest entry and
    cache the cleaned waveform as .npy so training doesn't redo ffmpeg
    decode + VAD + denoise on every epoch."""
    cache_dir = out_dir / "waveform_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    df = pd.read_csv(manifest_path)

    cached_rows = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        src_path = Path(row["filepath"])
        try:
            with open(src_path, "rb") as f:
                file_bytes = f.read()
            result = preprocess_audio(file_bytes, src_path.name)
            cache_path = cache_dir / f"{i}_{src_path.stem}.npy"
            np.save(cache_path, result.waveform)
            cached_rows.append({
    "waveform_path": str(cache_path),
    "label": row["label"],
    "source": row["source"],
    "split": row["split"],
    "original_path": str(src_path),
})
        except Exception as e:
            print(f"[SKIP] {src_path}: {e}")

    import pandas as pd
    cached_df = pd.DataFrame(cached_rows)
    cached_manifest_path = out_dir / "cached_manifest.csv"
    cached_df.to_csv(cached_manifest_path, index=False)
    print(f"Cached {len(cached_df)} preprocessed waveforms -> {cached_manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare VoiceGuard AI training dataset")
    parser.add_argument("--raw_dir", type=str, default="../../dataset/raw")
    parser.add_argument("--out_dir", type=str, default="../../dataset/processed")
    parser.add_argument("--skip_precompute", action="store_true",
                         help="Only build the manifest CSV, skip waveform caching")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    print(f"Scanning raw dataset directory: {raw_dir}")
    manifest_path = build_manifest(raw_dir, out_dir)

    if not args.skip_precompute:
        precompute_features(manifest_path, out_dir)
