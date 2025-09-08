from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


def check_paths(paths):
    ok = True
    for p in paths:
        exists = os.path.exists(p)
        print(("✅" if exists else "❌"), p)
        ok = ok and exists
    return ok


def main() -> int:
    print("🔍 Verifying integration...")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))

    dirs = [
        "src/cement_ai_platform/models/timegan",
        "src/cement_ai_platform/models/predictive",
        "src/cement_ai_platform/data/scenarios",
        "src/cement_ai_platform/poc",
    ]
    files = [
        "src/cement_ai_platform/data/data_pipeline/chemistry_data_generator.py",
        "src/cement_ai_platform/models/timegan/complete_timegan.py",
        "src/cement_ai_platform/models/predictive/quality_predictor.py",
        "src/cement_ai_platform/models/predictive/energy_predictor.py",
        "src/cement_ai_platform/poc/complete_pipeline.py",
    ]

    if not check_paths(dirs + files):
        return 1

    print("\n🧪 Import tests...")
    try:
        importlib.import_module("cement_ai_platform.poc.complete_pipeline")
        importlib.import_module("cement_ai_platform.models.timegan.complete_timegan")
        importlib.import_module("cement_ai_platform.models.predictive.quality_predictor")
        importlib.import_module("cement_ai_platform.models.predictive.energy_predictor")
        print("✅ Imports OK")
    except Exception as exc:
        print("❌ Import failed:", exc)
        return 1

    print("\n🎉 Integration verification PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


