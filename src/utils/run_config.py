from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml

from .config import BaselineConfig, RaftGMCConfig, default_model_path, default_tracker_path


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

def dump_run_config(path: Path, config: BaselineConfig, sequences: list[str], tracker_params: dict[str, Any], data_root: Path) -> None:
    payload = {
        "data_root": str(data_root),
        "sequences": sequences,
        "detector": {
            "model": config.model,
            "conf": config.conf,
            "iou": config.iou,
            "classes": config.classes,
            "device": config.device,
        },
        "resize": {
            "imgsz": config.imgsz,
            "mode": config.resize,
        },
        "tracker": tracker_params,
    }
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


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

def build_raft_gmc_config(section: dict[str, Any]) -> RaftGMCConfig:
    device_raw = section.get("device")
    device = str(device_raw) if device_raw is not None else None
    return RaftGMCConfig(
        model_name=str(section.get("model", "small")),
        device=device,
        mixed_precision=bool(section.get("mixed_precision", False)),
        scale_gmc=float(section.get("scale_gmc", 1)),
        scale=float(section.get("scale", 1)),
        sample_step=int(section.get("sample_step", 8)),
        ransac_reproj_threshold=float(section.get("ransac_reproj_threshold", 3)),
    )

def resolve_sequence_dir(data_root: Path, sequence: str) -> Path:
    seq_dir = data_root / sequence
    img_dir = seq_dir / "img1"
    gt_path = seq_dir / "gt" / "gt.txt"
    if not img_dir.exists():
        raise FileNotFoundError(f"Missing frames directory: {img_dir}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing ground truth file: {gt_path}")
    return seq_dir
