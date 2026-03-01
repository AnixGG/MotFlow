from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

from .config import BaselineConfig, default_model_path, default_tracker_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small BoTSORT baseline on MOT-style sequences")
    parser.add_argument("--config", required=True, type=Path, help="Path to run YAML config")
    return parser.parse_args()


def normalize_sequences(raw_values: Iterable[str]) -> list[str]:
    sequences: list[str] = []
    for value in raw_values:
        parts = [item.strip() for item in value.split(",")]
        sequences.extend([item for item in parts if item])
    if not sequences:
        raise ValueError("No sequences were provided")
    return sequences


def parse_run_settings(run_cfg: dict[str, Any]) -> tuple[Path, Path, list[str]]:
    if "data" not in run_cfg:
        raise ValueError("Missing required config key: data")
    if "outdir" not in run_cfg:
        raise ValueError("Missing required config key: outdir")
    if "sequences" not in run_cfg:
        raise ValueError("Missing required config key: sequences")
    data_root = Path(str(run_cfg["data"])).expanduser().resolve()
    outdir = Path(str(run_cfg["outdir"])).expanduser().resolve()
    raw_sequences = run_cfg["sequences"]
    if isinstance(raw_sequences, str):
        sequences = normalize_sequences([raw_sequences])
    elif isinstance(raw_sequences, list):
        sequences = normalize_sequences([str(item) for item in raw_sequences])
    else:
        raise ValueError("Config key sequences must be string or list")
    return data_root, outdir, sequences


def build_baseline_config(run_cfg: dict[str, Any]) -> BaselineConfig:
    if "device" not in run_cfg:
        raise ValueError("Missing required config key: device")
    imgsz = int(run_cfg.get("imgsz", 1280))
    return BaselineConfig(
        model=str(run_cfg.get("model", default_model_path())),
        tracker=str(Path(str(run_cfg.get("tracker_config", default_tracker_path()))).expanduser().resolve()),
        conf=float(run_cfg.get("conf", 0.25)),
        iou=float(run_cfg.get("iou", 0.7)),
        imgsz=imgsz,
        classes=[int(cls_id) for cls_id in run_cfg.get("classes", [0])],
        device=str(run_cfg["device"]),
        resize=f"letterbox_to_{imgsz}",
    )
