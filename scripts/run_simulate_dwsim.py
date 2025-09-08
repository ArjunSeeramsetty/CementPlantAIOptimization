#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cement_ai_platform.data import create_unified_platform


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DWSIM simulation stage (if available)")
    parser.add_argument("--input", required=True, help="Input CSV/Parquet file")
    parser.add_argument("--outdir", default="artifacts", help="Output directory")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    in_path = Path(args.input)
    if in_path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(in_path)
    else:
        df = pd.read_csv(in_path)

    platform = create_unified_platform()
    sim = platform.simulate_dwsim(df)
    (outdir / "dwsim_results.json").write_text(json.dumps(sim, indent=2, default=str))
    print("DWSIM simulation complete (best-effort). Outputs in:", outdir)


if __name__ == "__main__":
    main()


