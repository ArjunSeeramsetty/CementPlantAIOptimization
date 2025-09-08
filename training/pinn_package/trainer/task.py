import argparse
import json
from pathlib import Path

import numpy as np


def train(output_dir: str, steps: int = 100) -> None:
    # Placeholder: pretend to "train" and write an artifact
    rng = np.random.default_rng(42)
    weights = rng.normal(size=(3,)).tolist()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.json").write_text(json.dumps({"weights": weights}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    train(output_dir=args.output_dir, steps=args.steps)


if __name__ == "__main__":
    main()


