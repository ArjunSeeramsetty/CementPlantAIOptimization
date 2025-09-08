"""Package the training code and submit a Vertex AI custom training job.

Usage (requires gcloud auth and configured project):
  python scripts/submit_vertex_training.py \
      --bucket gs://YOUR_BUCKET \
      --region us-central1 \
      --container gcr.io/cloud-aiplatform/training/tf-cpu.2-13:latest
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from cement_ai_platform.vertex_ai.model_trainer import submit_custom_training_job


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True, help="GCS bucket (gs://...")
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--container", default="gcr.io/cloud-aiplatform/training/tf-cpu.2-13:latest")
    parser.add_argument("--display_name", default="cement-pinn-training-demo")
    parser.add_argument("--output", default="gs://tmp-cement-pinn-output")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1] / "training" / "pinn_package"
    pkg_dir = root
    dist_dir = root / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)

    # Build source distribution
    run([sys.executable, "-m", "pip", "install", "--upgrade", "build"])
    run([sys.executable, "-m", "build"],)

    # Find built artifact (tar.gz)
    artifacts = sorted(dist_dir.glob("*.tar.gz"))
    if not artifacts:
        raise SystemExit("No package artifact produced")
    artifact = artifacts[-1]

    # Upload to GCS
    gcs_uri = f"{args.bucket}/packages/{artifact.name}"
    run(["gsutil", "cp", str(artifact), gcs_uri])

    if args.dry_run:
        print("DRY RUN: would submit training job with", gcs_uri)
        return

    job = submit_custom_training_job(
        display_name=args.display_name,
        package_gcs_uri=gcs_uri,
        module_name="trainer.task",
        container_uri=args.container,
        location=args.region,
        args=["--output_dir", args.output],
    )
    print("Submitted job:", getattr(job, "resource_name", job))


if __name__ == "__main__":
    main()


