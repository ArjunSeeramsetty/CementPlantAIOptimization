import argparse
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path


UUID_REGEX = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\.py$")


def is_uuid_named_python_file(path: Path) -> bool:
    return path.is_file() and path.suffix == ".py" and UUID_REGEX.match(path.name) is not None


def slugify(name: str) -> str:
    s = name.strip().replace("/", ".").replace(" ", "_")
    s = s.replace(".", "_")
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s.lower()


def guess_destination(path: Path) -> Path:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lowered = text.lower()

    # Simple heuristics
    if "validate_temperature" in lowered or "temperature" in lowered:
        return Path("src/cement_ai_platform/data/processors/temperature_validator.py")
    if "bigquery" in lowered:
        return Path("src/cement_ai_platform/data/data_pipeline/bigquery_connector.py")
    if "vertex ai" in lowered or "aiplatform" in lowered:
        return Path("src/cement_ai_platform/vertex_ai/model_trainer.py")
    if "timegan" in lowered or "gan" in lowered:
        return Path("src/cement_ai_platform/models/generative/timegan.py")

    # Default bucket for unknown utilities
    return Path("src/cement_ai_platform/data/processors/quality_analyzer.py")


def load_ledger(ledger_path: Path) -> dict:
    if ledger_path.exists():
        try:
            return json.loads(ledger_path.read_text(encoding="utf-8"))
        except Exception:
            return {"migrated": {}}
    return {"migrated": {}}


def save_ledger(ledger_path: Path, ledger: dict) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze and relocate UUID-named Python files")
    parser.add_argument("--root", type=str, required=True, help="Root directory to scan")
    parser.add_argument("--dry-run", action="store_true", help="Only print proposed moves")
    parser.add_argument("--apply", action="store_true", help="Execute file moves")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files to propose/apply")
    parser.add_argument("--ledger", type=str, default=".migration_ledger.json", help="Path to migration ledger JSON")
    parser.add_argument("--yaml", type=str, default="22685ee7-c317-4938-8cd6-16009a57eb19/canvas.yaml", help="Path to canvas YAML for id→name mapping")
    parser.add_argument("--rename-existing", action="store_true", help="Rename previously migrated files using YAML names")
    parser.add_argument("--no-yaml-names", action="store_true", help="Disable YAML-based naming for new moves")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    ledger_path = Path(args.ledger)
    ledger = load_ledger(ledger_path)
    migrated_map = ledger.get("migrated", {})

    # Optional YAML id→name mapping
    id_to_name: dict[str, str] = {}
    yaml_path = Path(args.yaml)
    if yaml_path.exists():
        try:
            import yaml  # type: ignore

            def _collect_pairs(node):
                pairs = {}
                if isinstance(node, dict):
                    if "id" in node and "name" in node and isinstance(node["id"], str) and isinstance(node["name"], str):
                        pairs[node["id"]] = node["name"]
                    for v in node.values():
                        pairs.update(_collect_pairs(v))
                elif isinstance(node, list):
                    for item in node:
                        pairs.update(_collect_pairs(item))
                return pairs

            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if data is not None:
                id_to_name = _collect_pairs(data)
        except Exception:
            id_to_name = {}

    moves = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            path = Path(dirpath) / filename
            if is_uuid_named_python_file(path):
                # Skip if already migrated (by absolute normalized path)
                key = str(path.resolve())
                if key in migrated_map:
                    continue
                dst = guess_destination(path)
                moves.append((path, dst))

    # Deterministic ordering
    moves.sort(key=lambda t: str(t[0]))

    if args.limit is not None:
        moves = moves[: max(0, int(args.limit))]

    if not moves:
        print("No UUID-named Python files found.")
        # Allow rename-existing flow to proceed even if no new moves
        if not 'args' in locals() or not getattr(args, 'rename_existing', False):
            return

    print("Proposed moves:")
    for src, dst in moves:
        print(f"  {src} -> {dst}")

    if args.dry_run and not args.apply:
        return

    if args.apply:
        for src, dst in moves:
            dst.parent.mkdir(parents=True, exist_ok=True)
            # Generate a YAML-based or unique migrated filename when canonical target already exists
            canonical_dst = dst

            use_yaml = (not args.no_yaml_names) and bool(id_to_name)
            yaml_name = id_to_name.get(src.stem) if use_yaml else None

            if canonical_dst.exists():
                if yaml_name:
                    target_name = f"{slugify(yaml_name)}.py"
                else:
                    target_name = f"{canonical_dst.stem}_{src.stem}.py"
                final_dst = canonical_dst.with_name(target_name)
                if final_dst.exists():
                    final_dst = canonical_dst.with_name(f"{final_dst.stem}_{src.stem}.py")
            else:
                final_dst = canonical_dst

            shutil.copy2(src, final_dst)
            # Record in ledger
            ledger.setdefault("migrated", {})[str(src.resolve())] = {
                "destination": str(final_dst.resolve()),
                "canonical_target": str(canonical_dst.resolve()),
                "source_uuid": src.stem,
                "yaml_name": yaml_name,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
        save_ledger(ledger_path, ledger)
        print("Moves applied (copied). Review diffs and integrate incrementally.")

    # Optional: rename existing migrated files using YAML names
    if args.rename_existing and id_to_name:
        updated = 0
        for src_key, meta in list(ledger.get("migrated", {}).items()):
            try:
                current_dst = Path(meta.get("destination", ""))
                if not current_dst.exists():
                    continue
                src_uuid = Path(src_key).stem
                yaml_name = id_to_name.get(src_uuid)
                if not yaml_name:
                    continue
                canonical_target = Path(meta.get("canonical_target", current_dst))
                desired = canonical_target.with_name(f"{slugify(yaml_name)}.py")
                if desired.resolve() == current_dst.resolve():
                    continue
                if desired.exists():
                    desired = canonical_target.with_name(f"{slugify(yaml_name)}_{src_uuid}.py")
                current_dst.rename(desired)
                meta["destination"] = str(desired.resolve())
                meta["yaml_name"] = yaml_name
                updated += 1
            except Exception:
                continue
        save_ledger(ledger_path, ledger)
        print(f"Renamed {updated} existing migrated files using YAML names.")


if __name__ == "__main__":
    main()


