from __future__ import annotations

import csv
from pathlib import Path

import cv2


def _color_for_track(track_id: int) -> tuple[int, int, int]:
    return ((37 * track_id) % 255, (17 * track_id) % 255, (29 * track_id) % 255)


def _load_tracks(path: Path) -> dict[int, list[tuple[int, float, float, float, float]]]:
    by_frame: dict[int, list[tuple[int, float, float, float, float]]] = {}
    if not path.exists() or path.stat().st_size == 0:
        return by_frame

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 6:
                continue
            frame = int(float(row[0]))
            track_id = int(float(row[1]))
            x = float(row[2])
            y = float(row[3])
            w = float(row[4])
            h = float(row[5])
            by_frame.setdefault(frame, []).append((track_id, x, y, w, h))
    return by_frame


def render_sequence_video(
    seq_dir: Path,
    tracks_path: Path,
    out_path: Path,
    fps: float = 30.0,
    max_frames: int | None = None,
) -> None:
    img_dir = seq_dir / "img1"
    frame_paths = sorted(img_dir.glob("*.jpg"))
    if not frame_paths:
        frame_paths = sorted(img_dir.glob("*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in: {img_dir}")

    tracks = _load_tracks(tracks_path)
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise RuntimeError(f"Failed to read frame: {frame_paths[0]}")
    height, width = first.shape[:2]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open output video: {out_path}")

    try:
        for idx, frame_path in enumerate(frame_paths, start=1):
            if max_frames is not None and idx > max_frames:
                break
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            for track_id, x, y, w, h in tracks.get(idx, []):
                x1 = int(round(x))
                y1 = int(round(y))
                x2 = int(round(x + w))
                y2 = int(round(y + h))
                color = _color_for_track(track_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"ID {track_id}",
                    (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            writer.write(frame)
    finally:
        writer.release()
