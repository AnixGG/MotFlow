from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation.metrics import evaluate_sequence, summarize_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MOT tracks folder against GT")
    parser.add_argument("--data", required=True, type=Path, help="MOT split root with sequence/gt/gt.txt")
    parser.add_argument("--tracks", required=True, type=Path, help="Folder with <sequence>.txt in MOT format")
    parser.add_argument("--sequences", required=True, nargs="+", help="Sequence names to evaluate")
    parser.add_argument("--out", required=True, type=Path, help="Output CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data.expanduser().resolve()
    tracks_root = args.tracks.expanduser().resolve()

    accumulators = {}
    for seq in args.sequences:
        gt_path = data_root / seq / "gt" / "gt.txt"
        pred_path = tracks_root / f"{seq}.txt"
        accumulators[seq] = evaluate_sequence(gt_path, pred_path)

    summary = summarize_metrics(accumulators)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence", "MOTA", "IDF1", "IDS"])
        writer.writeheader()
        for index_name, row in summary.iterrows():
            writer.writerow(
                {
                    "sequence": index_name,
                    "MOTA": round(float(row["mota"]), 6),
                    "IDF1": round(float(row["idf1"]), 6),
                    "IDS": int(row["num_switches"]),
                }
            )


if __name__ == "__main__":
    main()
