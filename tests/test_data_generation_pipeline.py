from __future__ import annotations

import pandas as pd

from cement_ai_platform.data import create_unified_platform
from cement_ai_platform.data.data_pipeline.chemistry_data_generator import (
    CementChemistry,
    OptimizationDataPrep as ChemOptimizationDataPrep,
)
from cement_ai_platform.models.optimization.multi_objective_prep import (
    OptimizationDataPrep as MOOOptimizationDataPrep,
)


def test_generator_outputs_schema():
    platform = create_unified_platform()
    df = platform.enhanced_generator.generate_complete_dataset(100)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100

    required_cols = [
        "CaO",
        "SiO2",
        "Al2O3",
        "Fe2O3",
        "LSF",
        "SM",
        "AM",
        "C3S",
        "C2S",
        "kiln_temperature",
        "kiln_speed",
        "coal_feed_rate",
        "burnability_index",
        "heat_consumption",
        "free_lime",
    ]
    for c in required_cols:
        assert c in df.columns
        assert df[c].isna().sum() == 0

    # Sanity ranges (allow broader tolerance)
    assert (df["LSF"].between(0.8, 1.1)).mean() > 0.8
    assert (df["SM"].between(1.5, 4.0)).mean() > 0.9
    assert (df["kiln_temperature"].between(1350, 1500)).mean() > 0.95


def test_chemistry_validation_flags_present():
    chem = CementChemistry()
    platform = create_unified_platform()
    df = platform.enhanced_generator.generate_complete_dataset(20)
    row = df.iloc[0]

    report = chem.validate_chemistry(
        {"CaO": row["CaO"], "SiO2": row["SiO2"], "Al2O3": row["Al2O3"], "Fe2O3": row["Fe2O3"]}
    )
    for key in [
        "cao_range",
        "sio2_range",
        "al2o3_range",
        "fe2o3_range",
        "lsf_range",
        "sm_range",
        "am_range",
        "major_total",
    ]:
        assert key in report
        # Accept numpy boolean-like values
        assert report[key] in (True, False)


def test_create_targets_and_optimization_dataset():
    platform = create_unified_platform()
    df = platform.enhanced_generator.generate_complete_dataset(50)

    # Chemistry targets
    chem_prep = ChemOptimizationDataPrep(df)
    targets_df = chem_prep.create_targets()
    for col in [
        "energy_efficiency_score",
        "quality_score",
        "sustainability_score",
        "dv_fuel_flow_rate",
        "dv_kiln_speed",
        "dv_raw_material_feed",
        "dv_oxygen_content",
        "dv_alternative_fuel_rate",
    ]:
        assert col in targets_df.columns

    # Multi-objective optimization dataset
    moo_prep = MOOOptimizationDataPrep(df)
    ds = moo_prep.create_optimization_dataset()
    assert isinstance(ds, dict)
    for key in ["objectives", "constraints", "decision_variables", "original_data", "summary"]:
        assert key in ds
    assert len(ds["objectives"]) == len(df)
    assert len(ds["constraints"]) == len(df)
    assert len(ds["decision_variables"]) == len(df)

    summary = ds["summary"]
    assert "objectives_stats" in summary
    assert "constraint_violations" in summary
    assert "decision_variable_ranges" in summary
