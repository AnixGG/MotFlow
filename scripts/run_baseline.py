from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline import run_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=(ROOT / "configs" / "run_baseline.yaml"),
        help="Path to baseline YAML config (default: configs/run_baseline.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_baseline(args.config)


if __name__ == "__main__":
    main()
