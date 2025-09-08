from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from cement_ai_platform.data.data_pipeline.chemistry_data_generator import (
    EnhancedCementDataGenerator,
)
from cement_ai_platform.models.optimization.multi_objective_prep import (
    DecisionVariables,
    OptimizationDataPrep,
)


def run_analysis(n_samples: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    gen = EnhancedCementDataGenerator(seed=42)
    cement_dataset = gen.generate_complete_dataset(n_samples)

    subset = cement_dataset.head(min(1000, n_samples))
    prep = OptimizationDataPrep(subset)
    data = prep.create_optimization_dataset()

    objectives_df: pd.DataFrame = data["objectives"]
    constraints_df: pd.DataFrame = data["constraints"]
    decision_df: pd.DataFrame = data["decision_variables"]
    summary = data["summary"]

    # Print basic stats to console
    print("Objective stats:")
    for name, stats in summary["objectives_stats"].items():
        print(name, {k: round(v, 3) for k, v in stats.items()})
    print("Constraint violation rates:")
    for name, cnt in summary["constraint_violations"].items():
        rate = (cnt / max(1, len(constraints_df))) * 100.0
        print(f" {name}: {cnt}/{len(constraints_df)} ({rate:.1f}%)")

    # Plots
    sns.set_palette("husl")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Multi-Objective Optimization: Trade-off Analysis", fontsize=16, fontweight="bold")

    ax1 = axes[0, 0]
    s1 = ax1.scatter(
        objectives_df["energy_efficiency"],
        objectives_df["quality_score"],
        c=objectives_df["sustainability_score"],
        cmap="viridis",
        alpha=0.7,
        s=30,
    )
    ax1.set_xlabel("Energy Efficiency (lower = better)")
    ax1.set_ylabel("Quality Score (higher = better)")
    ax1.set_title("Energy vs Quality (color = Sustainability)")
    plt.colorbar(s1, ax=ax1, label="Sustainability Score")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    s2 = ax2.scatter(
        objectives_df["energy_efficiency"],
        objectives_df["sustainability_score"],
        c=objectives_df["quality_score"],
        cmap="plasma",
        alpha=0.7,
        s=30,
    )
    ax2.set_xlabel("Energy Efficiency (lower = better)")
    ax2.set_ylabel("Sustainability Score (lower = better)")
    ax2.set_title("Energy vs Sustainability (color = Quality)")
    plt.colorbar(s2, ax=ax2, label="Quality Score")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    s3 = ax3.scatter(
        objectives_df["quality_score"],
        objectives_df["sustainability_score"],
        c=objectives_df["energy_efficiency"],
        cmap="coolwarm",
        alpha=0.7,
        s=30,
    )
    ax3.set_xlabel("Quality Score (higher = better)")
    ax3.set_ylabel("Sustainability Score (lower = better)")
    ax3.set_title("Quality vs Sustainability (color = Energy)")
    plt.colorbar(s3, ax=ax3, label="Energy Efficiency")
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    corr = decision_df.iloc[:, :-1].corr() if "sample_idx" in decision_df.columns else decision_df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax4, cbar_kws={"label": "Correlation"}, fmt=".2f")
    ax4.set_title("Decision Variables Correlation Matrix")
    ax4.tick_params(axis="x", rotation=45)
    ax4.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    fig_path = output_dir / "multi_objective_tradeoffs.png"
    plt.savefig(fig_path, dpi=160)
    print(f"Saved trade-off plots to {fig_path}")

    # Example constraint evaluation
    sample = subset.iloc[0]
    dv = DecisionVariables(3200, 3.2, 100, 4.0, 15.0)
    temp_c = prep.temperature_constraint(dv, sample)
    qual_c = prep.quality_constraint(dv, sample)
    chem_c = prep.chemistry_constraints(dv, sample)
    op_c = prep.operational_constraints(dv)
    print("Sample constraint eval:", {
        "temperature": temp_c,
        "quality": qual_c,
        "chemistry_satisfied": int(sum(c > 0 for c in chem_c)),
        "operational_satisfied": int(sum(c > 0 for c in op_c)),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=800)
    parser.add_argument("--out", default="artifacts")
    args = parser.parse_args()
    run_analysis(args.samples, Path(args.out))


if __name__ == "__main__":
    main()


