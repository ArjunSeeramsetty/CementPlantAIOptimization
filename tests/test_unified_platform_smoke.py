from __future__ import annotations

import pandas as pd


def test_unified_platform_smoke():
    from cement_ai_platform.data import create_unified_platform
    from cement_ai_platform.models.optimization.multi_objective_prep import (
        OptimizationDataPrep,
    )

    platform = create_unified_platform()

    # Generate a tiny dataset via the enhanced generator
    base = platform.enhanced_generator.generate_complete_dataset(30)
    assert isinstance(base, pd.DataFrame)
    assert len(base) == 30

    # Preprocess (no split for speed)
    pre = platform.preprocess(base, handle_missing=True, do_split=False)
    assert "data" in pre and isinstance(pre["data"], pd.DataFrame)

    # Validate (best-effort)
    reports = platform.validate(pre["data"])  # returns dict of reports (may contain None)
    assert isinstance(reports, dict)

    # Train simple baselines (fallback models)
    # Create a dummy target if not present
    df = pre["data"].copy()
    target = "_dummy_target_"
    df[target] = (df.get("free_lime", pd.Series([1.0] * len(df))) * 10).fillna(1.0)
    res = platform.train_baselines(df, target=target)
    assert isinstance(res, dict) and "decision_tree" in res

    # Optimization report (build a minimal optimization dataset for summary)
    opt_prep = OptimizationDataPrep(base)
    ds = opt_prep.create_optimization_dataset()
    summary = platform.optimization_report(ds)
    assert isinstance(summary, dict)


