from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-time bootstrap training for submission package")
    parser.add_argument("--model-path", type=Path, default=Path("artifacts_all_retrain/stepnet_tcn_best.pt"))
    parser.add_argument("--training-root", type=Path, default=Path("training"))
    parser.add_argument("--source-root", type=Path, default=Path("../testdata/OxWalk_Dec2022"))
    parser.add_argument("--seed", type=int, default=20260408)
    parser.add_argument("--n-participants", type=int, default=30)
    parser.add_argument("--sample-rate", type=float, default=50.0)
    parser.add_argument("--force", action="store_true", help="Force re-prepare and retrain")
    return parser.parse_args()


def run_cmd(cmd: list[str], cwd: Path) -> None:
    LOGGER.info("Running command: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = parse_args()

    submission_dir = Path(__file__).resolve().parent
    repo_root = submission_dir.parent

    model_path = (submission_dir / args.model_path).resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if model_path.exists() and not args.force:
        LOGGER.info("Using existing checkpoint: %s", model_path)
        return

    train_out_dir = model_path.parent
    dataset_root = (submission_dir / args.source_root).resolve()

    train_script = repo_root / "train_nn.py"
    if not train_script.exists():
        raise FileNotFoundError(f"Missing train script: {train_script}")

    train_cmd = [
        sys.executable,
        str(train_script),
        "--dataset-root",
        str(dataset_root),
        "--n-participants",
        str(args.n_participants),
        "--seed",
        str(args.seed),
        "--sample-rate",
        str(args.sample_rate),
        "--out-dir",
        str(train_out_dir),
        "--model",
        "tcn",
    ]
    if args.force:
        train_cmd.append("--rebuild-packed-cache")

    run_cmd(train_cmd, cwd=repo_root)

    if not model_path.exists():
        raise FileNotFoundError(f"Expected checkpoint was not created: {model_path}")

    summary = {
        "model_path": str(model_path),
        "training_root": str((submission_dir / args.training_root).resolve()),
        "source_root": str((submission_dir / args.source_root).resolve()),
        "n_participants": int(args.n_participants),
        "seed": int(args.seed),
        "sample_rate_hz": float(args.sample_rate),
    }
    summary_path = train_out_dir / "bootstrap_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    LOGGER.info("Bootstrap training complete. Model: %s", model_path)
    LOGGER.info("Bootstrap summary: %s", summary_path)


if __name__ == "__main__":
    main()
