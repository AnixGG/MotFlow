from __future__ import annotations

from pathlib import Path

from evaluation.metrics import compute_hota_scores, evaluate_sequence, summarize_metrics
from evaluation.visualization import render_sequence_video
from tracking.botsort_runner import run_botsort_sequence_baseline
from tracking.ultralytics_runtime import patch_botsort_gmc, load_yolo_model
from utils.logging import setup_logger
from utils.run_config import build_baseline_config, build_raft_gmc_config, parse_run_settings, dump_run_config, resolve_sequence_dir
from utils.config import LOCAL_ULTRALYTICS_ROOT, REPO_ROOT
from utils.env_info import write_env_info
from utils.io import read_yaml, write_metrics_csv, write_timing_csv, write_yaml
from utils.timing import attach_raw_timing, build_overall_timing_row


def run_pipeline(config_path: Path, experimental_mode=False) -> None:
    run_config = read_yaml(config_path.expanduser().resolve())
    data_root, outdir, sequences = parse_run_settings(run_config)
    config = build_baseline_config(run_config)

    outdir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(outdir)

    current_mode = "baseline"
    if experimental_mode:
        current_mode = "experiment"

    tracker_params = read_yaml(Path(config.tracker))
    tracker_params["gmc_method"] = config.gmc
    updated_tracker_path = outdir / "tracker.yaml"
    write_yaml(updated_tracker_path, tracker_params)
    config.tracker = str(updated_tracker_path)

    dump_run_config(outdir / "config.yaml", config, sequences, tracker_params, data_root)
    write_env_info(outdir / "env_info.txt", REPO_ROOT, LOCAL_ULTRALYTICS_ROOT)

    vis_config = run_config.get("visualization", {}) if isinstance(run_config.get("visualization", {}), dict) else {}
    vis_enabled = bool(vis_config.get("enabled", False))
    vis_fps = float(vis_config.get("fps", 30.0))
    vis_max_frames = vis_config.get("max_frames")
    vis_max_frames = int(vis_max_frames) if vis_max_frames is not None else None

    logger.info(f"[{current_mode}] data=%s", data_root)
    logger.info(f"[{current_mode}] sequences=%s", ", ".join(sequences))
    logger.info(f"[{current_mode}] outdir=%s", outdir)
    logger.info(f"[{current_mode}] device=%s", config.device)
    logger.info(f"[{current_mode}] warmup_frames=%s", config.warmup_frames)
    logger.info(f"[{current_mode}] visualization=%s", "enabled" if vis_enabled else "disabled")

    model = load_yolo_model(config.model)
    track_dir = outdir / "tracks"
    track_dir.mkdir(parents=True, exist_ok=True)

    raft_config = None
    if config.gmc == "raft":
        raft_section = run_config.get("raft_gmc", {})
        raft_config = build_raft_gmc_config(raft_section if isinstance(raft_section, dict) else {})
        logger.info(
            "[%s] gmc=raft model=%s scale_gmc=%s sample_step=%s",
            current_mode,
            raft_config.model_name,
            raft_config.scale_gmc,
            raft_config.sample_step,
        )
    else:
        logger.info("[%s] gmc=%s gmc_downscale=%s", current_mode, config.gmc, config.gmc_downscale)
    gmc_context = patch_botsort_gmc(raft_config, default_downscale=config.gmc_downscale)

    accumulators = {}
    hota_pairs = {}
    timing_rows = []

    with gmc_context:
        for sequence in sequences:
            seq_dir = resolve_sequence_dir(data_root, sequence)
            pred_path = track_dir / f"{sequence}.txt"
            gt_path = seq_dir / "gt" / "gt.txt"

            logger.info("[sequence] %s: tracking", sequence)

            run_info = run_botsort_sequence_baseline(model, seq_dir, pred_path, config)
            
            logger.info("[sequence] %s: wrote %s (%s rows)", sequence, pred_path, run_info["rows"])
            logger.info("[sequence] %s: evaluating", sequence)
            
            accumulators[sequence] = evaluate_sequence(gt_path, pred_path)
            hota_pairs[sequence] = (gt_path, pred_path)
            timing_rows.append(
                attach_raw_timing(
                    run_info["timing"],
                    wall_latencies=run_info["wall_latencies_ms"],
                    preprocess_latencies=run_info["preprocess_latencies_ms"],
                    inference_latencies=run_info["inference_latencies_ms"],
                    postprocess_latencies=run_info["postprocess_latencies_ms"],
                    model_latencies=run_info["model_latencies_ms"],
                    residual_latencies=run_info["residual_latencies_ms"],
                    raft_gmc_latencies=run_info["raft_gmc_latencies_ms"],
                )
            )
            
            if vis_enabled:
                vis_path = outdir / "vis" / f"{sequence}.mp4"
                logger.info("[sequence] %s: rendering video -> %s", sequence, vis_path)
                render_sequence_video(seq_dir, pred_path, vis_path, fps=vis_fps, max_frames=vis_max_frames)

    hota_scores = compute_hota_scores(hota_pairs)
    metrics_summary = summarize_metrics(accumulators, hota_scores=hota_scores)
    
    if any(value is not None for value in hota_scores.values()):
        logger.info("[metrics] HOTA computed")
    else:
        logger.info("[metrics] HOTA unavailable (install trackeval to enable)")
    
    write_metrics_csv(outdir / "metrics.csv", metrics_summary)

    all_wall_latencies = [lat for item in timing_rows for lat in item.pop("_wall_latencies_ms", [])]
    
    all_preprocess_latencies = [lat for item in timing_rows for lat in item.pop("_preprocess_latencies_ms", [])]
    all_inference_latencies = [lat for item in timing_rows for lat in item.pop("_inference_latencies_ms", [])]
    all_postprocess_latencies = [lat for item in timing_rows for lat in item.pop("_postprocess_latencies_ms", [])]
    
    all_model_latencies = [lat for item in timing_rows for lat in item.pop("_model_latencies_ms", [])]
    all_residual_latencies = [lat for item in timing_rows for lat in item.pop("_residual_latencies_ms", [])]
    all_raft_gmc_latencies = [lat for item in timing_rows for lat in item.pop("_raft_gmc_latencies_ms", [])]
    
    all_frames = sum(int(row["frames"]) for row in timing_rows)
    if all_frames:
        timing_rows.append(
            build_overall_timing_row(
                all_wall_latencies,
                all_preprocess_latencies,
                all_inference_latencies,
                all_postprocess_latencies,
                all_model_latencies,
                all_residual_latencies,
                all_raft_gmc_latencies,
                all_frames,
            )
        )
    write_timing_csv(outdir / "timing.csv", timing_rows)

    logger.info("[done] metrics=%s", outdir / "metrics.csv")
    logger.info("[done] timing=%s", outdir / "timing.csv")
