"""Compare two saved TrackEval evaluations with a paired bootstrap over source videos."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def read_rows(path: Path, key: str) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {row[key]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate {key} in {path}")
    return indexed


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Directory created by evaluate_tracks.py")
    parser.add_argument("--candidate", type=Path, required=True, help="Directory created by evaluate_tracks.py")
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    parser.add_argument("--groups", type=Path, help="Optional CSV with sequence and group columns. Default is the SportsMOT video prefix.")
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def combined_metrics(counts: np.ndarray, stats) -> dict[str, np.ndarray]:
    tp = counts @ stats["HOTA_TP"]
    fp = counts @ stats["HOTA_FP"]
    fn = counts @ stats["HOTA_FN"]
    assa = (counts @ (stats["AssA"] * stats["HOTA_TP"])) / np.maximum(1, tp)
    deta = tp / np.maximum(1, tp + fp + fn)
    return {
        "HOTA": np.sqrt(assa * deta).mean(axis=-1),
        "AssA": assa.mean(axis=-1),
        "DetA": deta.mean(axis=-1),
    }


def compare(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise FileExistsError(f"Output already exists: {args.out}")
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    baseline = read_rows(args.baseline / "metrics.csv", "sequence")
    candidate = read_rows(args.candidate / "metrics.csv", "sequence")

    with np.load(args.baseline / "hota_stats.npz", allow_pickle=False) as base_stats, np.load(
        args.candidate / "hota_stats.npz", allow_pickle=False
    ) as candidate_stats:
        for key in ("sequences", "frames", "thresholds", "gt_sha256"):
            if not np.array_equal(base_stats[key], candidate_stats[key]):
                raise ValueError(f"The evaluations have different {key}")
        sequences = base_stats["sequences"].tolist()
        expected_rows = set(sequences) | {"OVERALL"}
        if set(baseline) != expected_rows or set(candidate) != expected_rows:
            raise ValueError("Each metrics.csv must cover its HOTA statistics and contain OVERALL")

        if args.groups is None:
            labels = [seq.rsplit("_c", 1)[0] for seq in sequences]
        else:
            groups = read_rows(args.groups, "sequence")
            if set(sequences) - groups.keys():
                raise ValueError("The group CSV does not cover every sequence")
            labels = [groups[seq]["group"].strip() for seq in sequences]
        _, group_index = np.unique(labels, return_inverse=True)
        n_groups = len(set(labels))
        if not all(labels) or n_groups < 2:
            raise ValueError("The bootstrap requires at least two distinct, nonempty groups")

        rng = np.random.default_rng(args.seed)
        # Both methods get identical multiplicities for every source video.
        group_counts = rng.multinomial(n_groups, np.full(n_groups, 1 / n_groups), size=args.samples)
        counts = group_counts[:, group_index].astype(float)
        samples_base = combined_metrics(counts, base_stats)
        samples_candidate = combined_metrics(counts, candidate_stats)
        overall_base = combined_metrics(np.ones(len(sequences)), base_stats)
        overall_candidate = combined_metrics(np.ones(len(sequences)), candidate_stats)

    summary = []
    for metric in ("HOTA", "AssA", "DetA", "IDF1", "MOTA", "IDS"):
        baseline_value = float(baseline["OVERALL"][metric])
        candidate_value = float(candidate["OVERALL"][metric])
        low = high = ""
        if metric in overall_base:
            if abs(baseline_value - overall_base[metric]) > 1e-6 or abs(candidate_value - overall_candidate[metric]) > 1e-6:
                raise ValueError(f"Recombined {metric} does not agree with metrics.csv")
            interval = np.quantile(samples_candidate[metric] - samples_base[metric], (0.025, 0.975))
            low, high = (f"{value:.6f}" for value in interval)
        summary.append({
            "metric": metric,
            "baseline": f"{baseline_value:.6f}",
            "candidate": f"{candidate_value:.6f}",
            "delta": f"{candidate_value - baseline_value:.6f}",
            "ci95_low": low,
            "ci95_high": high,
            "clips": len(sequences),
            "groups": n_groups,
            "grouping": str(args.groups) if args.groups else "sportsmot_video_prefix",
            "bootstrap_samples": args.samples,
            "seed": args.seed,
        })

    per_clip = [
        {
            "sequence": seq,
            "group": label,
            **{
                f"delta_{key}": f"{float(candidate[seq][key]) - float(baseline[seq][key]):.6f}"
                for key in ("HOTA", "AssA", "DetA", "IDF1", "IDS")
            },
        }
        for seq, label in zip(sequences, labels)
    ]
    args.out.mkdir(parents=True)
    write_rows(args.out / "comparison.csv", summary)
    write_rows(args.out / "per_clip.csv", per_clip)
    print(f"Paired evaluation of {len(sequences)} clips from {n_groups} groups")
    print(args.out)


if __name__ == "__main__":
    try:
        compare(parse_args())
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(str(error)) from error
