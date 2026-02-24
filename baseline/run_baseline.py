from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from core.cli import build_baseline_config, parse_args, parse_run_settings
from core.config import LOCAL_ULTRALYTICS_ROOT, REPO_ROOT, ensure_local_ultralytics_path
from core.env_info import write_env_info
from core.logging_io import dump_run_config, read_yaml, write_metrics_csv, write_timing_csv
from core.metrics import evaluate_sequence, summarize_metrics
from core.timing import attach_raw_timing, build_overall_timing_row
from core.tracking import resolve_sequence_dir, run_sequence
from core.visualization import render_sequence_video


def setup_logger(outdir: Path) -> logging.Logger:
    logger = logging.getLogger("baseline_runner")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(outdir / "run.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def main() -> None:
    ensure_local_ultralytics_path()
    args = parse_args()
    run_cfg = read_yaml(args.config.expanduser().resolve())
    data_root, outdir, sequences = parse_run_settings(run_cfg)
    config = build_baseline_config(run_cfg)

    outdir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(outdir)

    from ultralytics import YOLO

    tracker_params = read_yaml(Path(config.tracker))
    dump_run_config(outdir / "config.yaml", config, sequences, tracker_params, data_root)
    write_env_info(outdir / "env_info.txt", REPO_ROOT, LOCAL_ULTRALYTICS_ROOT)
    vis_cfg = run_cfg.get("visualization", {}) if isinstance(run_cfg.get("visualization", {}), dict) else {}
    vis_enabled = bool(vis_cfg.get("enabled", False))
    vis_fps = float(vis_cfg.get("fps", 30.0))
    vis_max_frames = vis_cfg.get("max_frames")
    vis_max_frames = int(vis_max_frames) if vis_max_frames is not None else None

    logger.info("[baseline] data=%s", data_root)
    logger.info("[baseline] sequences=%s", ", ".join(sequences))
    logger.info("[baseline] outdir=%s", outdir)
    logger.info("[baseline] device=%s", config.device)
    logger.info("[baseline] visualization=%s", "enabled" if vis_enabled else "disabled")

    model = YOLO(config.model)
    track_dir = outdir / "tracks"
    track_dir.mkdir(parents=True, exist_ok=True)

    accumulators: dict[str, Any] = {}
    timing_rows: list[dict[str, Any]] = []

    for sequence in sequences:
        seq_dir = resolve_sequence_dir(data_root, sequence)
        pred_path = track_dir / f"{sequence}.txt"
        logger.info("[sequence] %s: tracking", sequence)
        run_info = run_sequence(model, seq_dir, pred_path, config)
        logger.info("[sequence] %s: wrote %s (%s rows)", sequence, pred_path, run_info["rows"])
        logger.info("[sequence] %s: evaluating", sequence)
        accumulators[sequence] = evaluate_sequence(seq_dir / "gt" / "gt.txt", pred_path)
        timing_rows.append(attach_raw_timing(run_info["timing"], run_info["wall_latencies_ms"], run_info["model_latencies_ms"]))
        if vis_enabled:
            vis_path = outdir / "vis" / f"{sequence}.mp4"
            logger.info("[sequence] %s: rendering video -> %s", sequence, vis_path)
            render_sequence_video(seq_dir, pred_path, vis_path, fps=vis_fps, max_frames=vis_max_frames)

    metrics_summary = summarize_metrics(accumulators)
    write_metrics_csv(outdir / "metrics.csv", metrics_summary)

    all_wall_latencies = [lat for item in timing_rows for lat in item.pop("_wall_latencies_ms", [])]
    all_model_latencies = [lat for item in timing_rows for lat in item.pop("_model_latencies_ms", [])]
    all_frames = sum(int(row["frames"]) for row in timing_rows)
    if all_frames:
        timing_rows.append(build_overall_timing_row(all_wall_latencies, all_model_latencies, all_frames))
    write_timing_csv(outdir / "timing.csv", timing_rows)

    logger.info("[done] metrics=%s", outdir / "metrics.csv")
    logger.info("[done] timing=%s", outdir / "timing.csv")


if __name__ == "__main__":
    main()
