#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from cement_ai_platform.data import create_unified_platform
from cement_ai_platform.data.data_pipeline.chemistry_data_generator import (
    CementChemistry,
    OptimizationDataPrep as ChemOptimizationDataPrep,
)
from cement_ai_platform.models.optimization.multi_objective_prep import (
    OptimizationDataPrep as MOOOptimizationDataPrep,
)


def summarize_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "missing_total": int(df.isna().sum().sum()),
        "missing_by_column": {c: int(df[c].isna().sum()) for c in df.columns},
        "basic_stats": {
            c: {
                "mean": float(df[c].mean()),
                "std": float(df[c].std()),
                "min": float(df[c].min()),
                "max": float(df[c].max()),
            }
            for c in numeric_cols[:10]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate data pipeline validation report")
    parser.add_argument("--samples", type=int, default=200, help="Number of samples to generate")
    parser.add_argument("--outdir", default="artifacts", help="Output directory")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    platform = create_unified_platform()
    df = platform.enhanced_generator.generate_complete_dataset(args.samples)

    # Schema and basic stats
    schema_report = summarize_dataframe(df)

    # Chemistry validation on a few samples
    chem = CementChemistry()
    sample_checks = []
    for i in [0, len(df) // 2, len(df) - 1]:
        row = df.iloc[i]
        report = chem.validate_chemistry(
            {
                "CaO": row.get("CaO", 0.0),
                "SiO2": row.get("SiO2", 0.0),
                "Al2O3": row.get("Al2O3", 0.0),
                "Fe2O3": row.get("Fe2O3", 0.0),
            }
        )
        sample_checks.append({"index": i, **{k: bool(v) for k, v in report.items()}})

    # Optimization targets from chemistry prep
    chem_prep = ChemOptimizationDataPrep(df)
    targets_df = chem_prep.create_targets()
    targets_summary = summarize_dataframe(targets_df[[
        "energy_efficiency_score",
        "quality_score",
        "sustainability_score",
    ]].copy())

    # Multi-objective optimization dataset
    moo_prep = MOOOptimizationDataPrep(df)
    ds = moo_prep.create_optimization_dataset()
    moo_summary = ds.get("summary", {})

    report: Dict[str, Any] = {
        "schema": schema_report,
        "chemistry_validation": sample_checks,
        "targets_summary": targets_summary,
        "optimization_summary": moo_summary,
    }

    (outdir / "data_pipeline_report.json").write_text(json.dumps(report, indent=2))
    print("Data pipeline report written to", outdir / "data_pipeline_report.json")


if __name__ == "__main__":
    main()
