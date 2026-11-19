"""Compare boxes and confidences in MOT tracks to identify an archived detector run."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment


def read_tracks(path: Path) -> dict[int, np.ndarray]:
    frames = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            frame = int(row[0])
            frames[frame].append([float(field) for field in row[2:7]])
    if not frames:
        raise ValueError(f"No predictions in {path}")
    return {frame: np.array(values) for frame, values in frames.items()}


def compare(reference: dict, candidate: dict) -> dict:
    n_reference = sum(map(len, reference.values()))
    n_candidate = sum(map(len, candidate.values()))
    matched = near_exact = 0
    confidence_errors = []
    for frame in reference.keys() & candidate.keys():
        old = reference[frame]
        new = candidate[frame]
        old_right = old[:, :2] + old[:, 2:4]
        new_right = new[:, :2] + new[:, 2:4]
        top_left = np.maximum(old[:, None, :2], new[None, :, :2])
        bottom_right = np.minimum(old_right[:, None], new_right[None, :])
        intersection = np.maximum(0, bottom_right - top_left).prod(axis=-1)
        union = old[:, 2:4].prod(axis=-1)[:, None] + new[:, 2:4].prod(axis=-1)[None, :] - intersection
        iou = intersection / np.maximum(union, 1e-12)
        errors = np.abs(old[:, None, 4] - new[None, :, 4])
        # Confidence breaks ties between nearly coincident boxes.
        i, j = linear_sum_assignment(iou - 0.001 * errors, maximize=True)
        assigned_iou, assigned_error = iou[i, j], errors[i, j]
        aligned = assigned_iou >= 0.5
        matched += int(aligned.sum())
        near_exact += int(((assigned_iou >= 0.99) & (assigned_error <= 0.001)).sum())
        confidence_errors.extend(assigned_error[aligned])
    return {
        "reference_boxes": n_reference,
        "candidate_boxes": n_candidate,
        "matched_iou50": matched,
        "near_exact_boxes": near_exact,
        "near_exact_f1": f"{2 * near_exact / (n_candidate + n_reference):.6f}",
        "matched_confidence_mae": f"{np.mean(confidence_errors):.6f}" if matched else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True, help="Repeat for each run of the same clip")
    args = parser.parse_args()
    reference = read_tracks(args.reference)
    rows = [
        {"candidate": str(path), **compare(reference, read_tracks(path))}
        for path in args.candidate
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, IndexError) as error:
        raise SystemExit(str(error)) from error
