from __future__ import annotations

import statistics
from typing import Any


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0
    
    if len(values) == 1:
        return float(values[0])
    
    sorted_values = sorted(values)

    position = (len(sorted_values) - 1) * q

    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower

    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0


def summarize_timing(
    sequence: str,
    wall_latencies_ms: list[float],
    preprocess_latencies_ms: list[float],
    inference_latencies_ms: list[float],
    postprocess_latencies_ms: list[float],
    model_latencies_ms: list[float],
    residual_latencies_ms: list[float],
    raft_gmc_latencies_ms: list[float],
) -> dict[str, float | int | str]:
    
    mean_wall = _mean(wall_latencies_ms)

    mean_preprocess = _mean(preprocess_latencies_ms)
    mean_inference = _mean(inference_latencies_ms)
    mean_postprocess = _mean(postprocess_latencies_ms)

    mean_model = _mean(model_latencies_ms)
    mean_residual = _mean(residual_latencies_ms)
    mean_raft_gmc = _mean(raft_gmc_latencies_ms)

    return {
        "sequence": sequence,
        "frames": len(wall_latencies_ms),
        "wall_latency_ms_mean": round(mean_wall, 4),
        "wall_latency_ms_p50": round(percentile(wall_latencies_ms, 0.50), 4),
        "wall_latency_ms_p95": round(percentile(wall_latencies_ms, 0.95), 4),
        "wall_fps_mean": round(1000.0 / mean_wall, 4) if mean_wall > 0 else 0.0,
        "preprocess_latency_ms_mean": round(mean_preprocess, 4),
        "preprocess_latency_ms_p50": round(percentile(preprocess_latencies_ms, 0.50), 4),
        "preprocess_latency_ms_p95": round(percentile(preprocess_latencies_ms, 0.95), 4),
        "inference_latency_ms_mean": round(mean_inference, 4),
        "inference_latency_ms_p50": round(percentile(inference_latencies_ms, 0.50), 4),
        "inference_latency_ms_p95": round(percentile(inference_latencies_ms, 0.95), 4),
        "postprocess_latency_ms_mean": round(mean_postprocess, 4),
        "postprocess_latency_ms_p50": round(percentile(postprocess_latencies_ms, 0.50), 4),
        "postprocess_latency_ms_p95": round(percentile(postprocess_latencies_ms, 0.95), 4),
        "model_latency_ms_mean": round(mean_model, 4),
        "model_latency_ms_p50": round(percentile(model_latencies_ms, 0.50), 4),
        "model_latency_ms_p95": round(percentile(model_latencies_ms, 0.95), 4),
        "residual_latency_ms_mean": round(mean_residual, 4),
        "residual_latency_ms_p50": round(percentile(residual_latencies_ms, 0.50), 4),
        "residual_latency_ms_p95": round(percentile(residual_latencies_ms, 0.95), 4),
        "raft_gmc_latency_ms_mean": round(mean_raft_gmc, 4),
        "raft_gmc_latency_ms_p50": round(percentile(raft_gmc_latencies_ms, 0.50), 4),
        "raft_gmc_latency_ms_p95": round(percentile(raft_gmc_latencies_ms, 0.95), 4),
    }


def build_overall_timing_row(
    all_wall_latencies: list[float],
    all_preprocess_latencies: list[float],
    all_inference_latencies: list[float],
    all_postprocess_latencies: list[float],
    all_model_latencies: list[float],
    all_residual_latencies: list[float],
    all_raft_gmc_latencies: list[float],
    all_frames: int,
) -> dict[str, float | int | str]:
    overall_wall_mean = _mean(all_wall_latencies)

    overall_preprocess_mean = _mean(all_preprocess_latencies)
    overall_inference_mean = _mean(all_inference_latencies)
    overall_postprocess_mean = _mean(all_postprocess_latencies)

    overall_model_mean = _mean(all_model_latencies)
    overall_residual_mean = _mean(all_residual_latencies)
    overall_raft_gmc_mean = _mean(all_raft_gmc_latencies)

    return {
        "sequence": "OVERALL",
        "frames": all_frames,
        "wall_latency_ms_mean": round(overall_wall_mean, 4),
        "wall_latency_ms_p50": round(percentile(all_wall_latencies, 0.50), 4),
        "wall_latency_ms_p95": round(percentile(all_wall_latencies, 0.95), 4),
        "wall_fps_mean": round(1000.0 / overall_wall_mean, 4) if overall_wall_mean > 0 else 0.0,
        "preprocess_latency_ms_mean": round(overall_preprocess_mean, 4),
        "preprocess_latency_ms_p50": round(percentile(all_preprocess_latencies, 0.50), 4),
        "preprocess_latency_ms_p95": round(percentile(all_preprocess_latencies, 0.95), 4),
        "inference_latency_ms_mean": round(overall_inference_mean, 4),
        "inference_latency_ms_p50": round(percentile(all_inference_latencies, 0.50), 4),
        "inference_latency_ms_p95": round(percentile(all_inference_latencies, 0.95), 4),
        "postprocess_latency_ms_mean": round(overall_postprocess_mean, 4),
        "postprocess_latency_ms_p50": round(percentile(all_postprocess_latencies, 0.50), 4),
        "postprocess_latency_ms_p95": round(percentile(all_postprocess_latencies, 0.95), 4),
        "model_latency_ms_mean": round(overall_model_mean, 4),
        "model_latency_ms_p50": round(percentile(all_model_latencies, 0.50), 4),
        "model_latency_ms_p95": round(percentile(all_model_latencies, 0.95), 4),
        "residual_latency_ms_mean": round(overall_residual_mean, 4),
        "residual_latency_ms_p50": round(percentile(all_residual_latencies, 0.50), 4),
        "residual_latency_ms_p95": round(percentile(all_residual_latencies, 0.95), 4),
        "raft_gmc_latency_ms_mean": round(overall_raft_gmc_mean, 4),
        "raft_gmc_latency_ms_p50": round(percentile(all_raft_gmc_latencies, 0.50), 4),
        "raft_gmc_latency_ms_p95": round(percentile(all_raft_gmc_latencies, 0.95), 4),
    }


def attach_raw_timing(
    row: dict[str, Any],
    *,
    wall_latencies: list[float],
    preprocess_latencies: list[float],
    inference_latencies: list[float],
    postprocess_latencies: list[float],
    model_latencies: list[float],
    residual_latencies: list[float],
    raft_gmc_latencies: list[float],
) -> dict[str, Any]:
    timing_entry = dict(row)
    timing_entry["_wall_latencies_ms"] = wall_latencies

    timing_entry["_preprocess_latencies_ms"] = preprocess_latencies
    timing_entry["_inference_latencies_ms"] = inference_latencies
    timing_entry["_postprocess_latencies_ms"] = postprocess_latencies
    
    timing_entry["_model_latencies_ms"] = model_latencies
    timing_entry["_residual_latencies_ms"] = residual_latencies
    timing_entry["_raft_gmc_latencies_ms"] = raft_gmc_latencies
    return timing_entry
