"""Evaluate saved MOT predictions against SportsMOT annotations on CPU."""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True, help="Split directory containing gt/gt.txt and seqinfo.ini per clip")
    parser.add_argument("--tracks", type=Path, required=True, help="Directory containing one MOT text file per clip")
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    parser.add_argument("--sequence", action="append", help="Evaluate a named subset. Repeat for each clip.")
    parser.add_argument("--trackeval-kit", type=Path, help="TrackEval directory from the official SportsMOT repository")
    return parser.parse_args()


def load_trackeval(kit: Path | None):
    if kit is not None:
        if not (kit / "trackeval" / "__init__.py").is_file():
            raise ValueError(f"Could not find the trackeval package under {kit}")
        sys.path.insert(0, str(kit.resolve()))
    import trackeval

    # The official SportsMOT fork predates NumPy 2.
    for name, scalar_type in {"float": float, "int": int, "bool": bool}.items():
        if name not in vars(np):
            setattr(np, name, scalar_type)
    return trackeval


def format_metrics(sequence, scores):
    hota, clear, identity = scores
    return {
        "sequence": sequence,
        "MOTA": f"{clear['MOTA']:.6f}",
        "IDF1": f"{identity['IDF1']:.6f}",
        "HOTA": f"{np.mean(hota['HOTA']):.6f}",
        "IDS": int(clear["IDSW"]),
        "AssA": f"{np.mean(hota['AssA']):.6f}",
        "DetA": f"{np.mean(hota['DetA']):.6f}",
        "FP": int(clear["CLR_FP"]),
        "FN": int(clear["CLR_FN"]),
    }


def evaluate(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise FileExistsError(f"Output already exists: {args.out}")
    tracks = args.tracks.resolve()
    gt = args.gt.resolve()
    if not tracks.is_dir() or not gt.is_dir():
        raise ValueError("Both --gt and --tracks must be existing directories")

    gt_sequences = {directory.name for directory in gt.iterdir() if directory.is_dir()}
    sequences = sorted(set(args.sequence) if args.sequence else gt_sequences)
    if not sequences or not set(sequences) <= gt_sequences:
        raise ValueError("--sequence must select one or more existing ground-truth clips")

    prediction_sequences = {path.stem for path in tracks.glob("*.txt")}
    missing = set(sequences) - prediction_sequences
    extra = prediction_sequences - gt_sequences
    if missing or extra:
        raise ValueError(f"Missing predictions: {sorted(missing)}. Unknown sequences: {sorted(extra)}")

    gt_digest = hashlib.sha256()
    for sequence in sequences:
        for relative_path in ("gt/gt.txt", "seqinfo.ini"):
            path = gt / sequence / relative_path
            gt_digest.update(f"{sequence}/{relative_path}\0".encode())
            gt_digest.update(hashlib.sha256(path.read_bytes()).digest())

    trackeval = load_trackeval(args.trackeval_kit)
    run = tracks.parent.name
    dataset = trackeval.datasets.MotChallenge2DBox({
        "BENCHMARK": "sportsmot",
        "SPLIT_TO_EVAL": "val",
        "GT_FOLDER": str(gt),
        "TRACKERS_FOLDER": str(tracks.parent.parent),
        "TRACKERS_TO_EVAL": [run],
        "TRACKER_SUB_FOLDER": tracks.name,
        "SKIP_SPLIT_FOL": True,
        "SEQ_INFO": dict.fromkeys(sequences),
        "PRINT_CONFIG": False,
    })
    metrics = (
        trackeval.metrics.HOTA(),
        trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
        trackeval.metrics.Identity({"PRINT_CONFIG": False}),
    )
    sequence_scores = []
    total_frames = sum(dataset.seq_lengths.values())
    processed_frames = 0
    started = time.monotonic()
    print(f"[TrackEval] CPU evaluation of {len(sequences)} clips and {total_frames} frames", flush=True)
    for index, sequence in enumerate(sequences, 1):
        raw = dataset.get_raw_seq_data(run, sequence)
        data = dataset.get_preprocessed_seq_data(raw, "pedestrian")
        sequence_scores.append([metric.eval_sequence(data) for metric in metrics])
        processed_frames += dataset.seq_lengths[sequence]
        if index == 1 or index % 5 == 0 or index == len(sequences):
            elapsed = time.monotonic() - started
            eta = timedelta(seconds=round(elapsed * (total_frames - processed_frames) / max(1, processed_frames)))
            print(f"[TrackEval] {index}/{len(sequences)} clips | {processed_frames}/{total_frames} frames | ETA {eta}", flush=True)

    overall = [
        metric.combine_sequences({seq: values[index] for seq, values in zip(sequences, sequence_scores)})
        for index, metric in enumerate(metrics)
    ]
    rows = [format_metrics(seq, values) for seq, values in zip(sequences, sequence_scores)]
    rows.append(format_metrics("OVERALL", overall))

    args.out.mkdir(parents=True)
    with (args.out / "metrics.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(
        args.out / "hota_stats.npz",
        sequences=sequences,
        gt_sha256=gt_digest.hexdigest(),
        frames=[dataset.seq_lengths[seq] for seq in sequences],
        thresholds=metrics[0].array_labels,
        **{
            key: np.stack([values[0][key] for values in sequence_scores])
            for key in ("HOTA_TP", "HOTA_FP", "HOTA_FN", "AssA")
        },
    )
    print(f"{len(sequences)} clips, {total_frames} frames, HOTA {rows[-1]['HOTA']}")
    print(f"TrackEval: {trackeval.__file__}")
    print(args.out)


if __name__ == "__main__":
    try:
        evaluate(parse_args())
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
