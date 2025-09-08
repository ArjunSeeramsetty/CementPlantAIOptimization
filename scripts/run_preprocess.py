#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cement_ai_platform.data import create_unified_platform


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline")
    parser.add_argument("--input", required=True, help="Input CSV/Parquet file")
    parser.add_argument("--outdir", default="artifacts", help="Output directory")
    parser.add_argument("--no-handle-missing", action="store_true", help="Disable missing handling")
    parser.add_argument("--split", action="store_true", help="Perform train/test split")
    parser.add_argument("--synth", action="store_true", help="Generate synthetic artifacts if available")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load input
    in_path = Path(args.input)
    if in_path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(in_path)
    else:
        df = pd.read_csv(in_path)

    platform = create_unified_platform()
    res = platform.preprocess(
        df,
        handle_missing=not args.no_handle_missing,
        do_split=args.split,
        synth=args.synth,
    )

    # Save outputs
    res["data"].to_csv(outdir / "preprocessed.csv", index=False)
    if res.get("split"):
        split = res["split"]
        if split and isinstance(split, dict):
            if "train" in split:
                split["train"].to_csv(outdir / "train.csv", index=False)
            if "test" in split:
                split["test"].to_csv(outdir / "test.csv", index=False)
    # Save artifacts summary
    artifacts = res.get("artifacts", {})
    try:
        (outdir / "preprocess_artifacts.json").write_text(json.dumps(artifacts, indent=2))
    except Exception:
        pass

    print("Preprocessing complete. Outputs in:", outdir)


if __name__ == "__main__":
    main()


