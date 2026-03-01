from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experiment config through baseline runner")
    parser.add_argument("--config", required=True, type=Path, help="Path to experiment YAML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(root / "scripts" / "run_baseline.py"), "--config", str(args.config.resolve())]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
