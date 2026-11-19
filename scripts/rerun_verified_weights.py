"""Run BoT-SORT and official SportsMOT evaluation with SHA-256-pinned weights."""

from __future__ import annotations

import argparse
import configparser
import csv
import hashlib
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "none": {"gmc": "none"},
    "sparse4": {"gmc": "sparseOptFlow", "gmc_downscale": 4},
    "sparse8": {"gmc": "sparseOptFlow", "gmc_downscale": 8},
    "raft_large128": {"gmc": "raft"},
    "raft_large128_onnx": {"gmc": "raft"},
    "sift2": {"gmc": "sift", "gmc_downscale": 2},
}


def parse_args(checkpoints: dict) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, choices=checkpoints)
    parser.add_argument("--weights", type=Path, help="Override the checkpoint location. The SHA-256 must still match.")
    parser.add_argument("--data", type=Path, required=True, help="SportsMOT val directory with img1, gt and seqinfo.ini")
    parser.add_argument("--trackeval-kit", type=Path, required=True, help="Official SportsMOT TrackEval directory")
    parser.add_argument("--out", type=Path, required=True, help="New directory for these runs")
    parser.add_argument("--device", default="cuda:0", help="cuda or cuda:N, or cpu for a smoke test")
    parser.add_argument("--method", action="append", choices=METHODS, help="Repeat per method. Default: sparse8 and raft_large128.")
    parser.add_argument("--raft-onnx", type=Path, help="Validated RAFT-large ONNX graph for the raft_large128_onnx method")
    parser.add_argument("--sequence", action="append", help="Repeat for a subset. Default: all 45 val clips.")
    parser.add_argument("--dry-run", action="store_true", help="Check the weight hash and print the configs without requiring images")
    return parser.parse_args()


def validate_sequence(directory: Path) -> None:
    if not (directory / "gt/gt.txt").is_file():
        raise ValueError(f"Missing annotation for {directory.name}")
    info = configparser.ConfigParser()
    info.read(directory / "seqinfo.ini")
    if "Sequence" not in info:
        raise ValueError(f"Missing or invalid seqinfo.ini in {directory}")
    expected = info["Sequence"].getint("seqLength")
    extension = info["Sequence"].get("imExt", ".jpg")
    image_directory = directory / "img1"
    # The Ultralytics directory loader uses glob.glob, which skips hidden files.
    images = sorted(
        path.name for path in image_directory.glob(f"*{extension}")
        if path.is_file() and not path.name.startswith(".")
    )
    expected_images = [f"{frame:06d}{extension}" for frame in range(1, expected + 1)]
    if expected < 1 or images != expected_images:
        raise ValueError(f"{directory.name}: need all {expected} SportsMOT frames in {image_directory}, found {len(images)}")


def make_config(base: dict, method: str, args: argparse.Namespace, sequences: list[str], weight: Path, digest: str) -> dict:
    config = {
        **base,
        **METHODS[method],
        "data": str(args.data.resolve()),
        "outdir": str(args.out.resolve() / method),
        "sequences": sequences,
        "model": str(weight),
        "model_sha256": digest,
        "device": args.device,
        "tracker_config": str(ROOT / "configs/botsort_baseline.yaml"),
        "visualization": {"enabled": False},
    }
    if config["gmc"] == "raft":
        config["raft_gmc"] = {**config["raft_gmc"], "model": "large", "device": args.device}
        if method == "raft_large128_onnx":
            config["raft_gmc"]["onnx_model"] = str(args.raft_onnx.resolve())
    else:
        config.pop("raft_gmc", None)
    return config


def run() -> None:
    checkpoints = yaml.safe_load((ROOT / "configs/verified_checkpoints.yaml").read_text())
    args = parse_args(checkpoints)
    checkpoint = checkpoints[args.checkpoint]
    weight = (args.weights or ROOT / checkpoint["path"]).expanduser().resolve(strict=True)
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    if digest != checkpoint["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {weight}: expected {checkpoint['sha256']}, got {digest}")

    base = yaml.safe_load((ROOT / "configs/run_experiment.yaml").read_text())
    sequences = list(dict.fromkeys(args.sequence or base["sequences"]))
    unknown = set(sequences) - set(base["sequences"])
    if unknown:
        raise ValueError(f"Unknown SportsMOT val sequences: {sorted(unknown)}")
    methods = dict.fromkeys(args.method or ("sparse8", "raft_large128"))
    if "raft_large128_onnx" in methods and args.raft_onnx is None:
        raise ValueError("raft_large128_onnx requires --raft-onnx")
    configs = {method: make_config(base, method, args, sequences, weight, digest) for method in methods}
    if args.dry_run:
        print(yaml.safe_dump(configs, sort_keys=False), end="")
        return

    if args.out.exists():
        raise FileExistsError(f"Output already exists: {args.out}")
    if "raft_large128_onnx" in methods and not args.raft_onnx.is_file():
        raise FileNotFoundError(args.raft_onnx)
    kit = args.trackeval_kit.resolve()
    if not (kit / "trackeval/__init__.py").is_file():
        raise ValueError(f"Missing official TrackEval package in {kit}")
    for sequence in sequences:
        validate_sequence(args.data / sequence)
    if args.device != "cpu":
        import torch

        device = torch.device(args.device)
        if device.type != "cuda" or (device.index or 0) >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {args.device} is unavailable")

    args.out.mkdir(parents=True)
    summary = []
    for index, (method, config) in enumerate(configs.items(), 1):
        print(f"[Benchmark {index}/{len(configs)}] {method}: tracking {len(sequences)} clips", flush=True)
        run_dir = Path(config["outdir"])
        run_dir.mkdir()
        config_path = run_dir / "run_config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        subprocess.run([sys.executable, str(ROOT / "scripts/run_experiment.py"), "--config", str(config_path)], check=True)
        evaluate = [
            sys.executable, str(ROOT / "scripts/evaluate_tracks.py"),
            "--gt", config["data"], "--tracks", str(run_dir / "tracks"),
            "--out", str(run_dir / "trackeval"), "--trackeval-kit", str(kit),
        ]
        for sequence in sequences:
            evaluate.extend(("--sequence", sequence))
        print(f"[Benchmark {index}/{len(configs)}] {method}: running TrackEval", flush=True)
        subprocess.run(evaluate, check=True)
        with (run_dir / "trackeval/metrics.csv").open(newline="") as handle:
            quality = next(row for row in csv.DictReader(handle) if row["sequence"] == "OVERALL")
        with (run_dir / "timing.csv").open(newline="") as handle:
            timing = next(row for row in csv.DictReader(handle) if row["sequence"] == "OVERALL")
        summary.append({
            "method": method,
            "frames": timing["frames"],
            **{key: quality[key] for key in ("HOTA", "IDF1", "AssA", "DetA", "IDS")},
            **{key: timing[key] for key in (
                "wall_latency_ms_mean", "wall_latency_ms_p95", "wall_over_40ms_fraction",
                "raft_gmc_latency_ms_mean",
            )},
        })
    with (args.out / "summary.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0])
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    try:
        run()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
