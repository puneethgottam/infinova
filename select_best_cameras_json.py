#!/usr/bin/env python3
"""Pick the most consistent cameras.json across an episode's frames.

The 5 physical cameras do not move during an episode, so each frame's
cameras.json should in principle describe the same 5 extrinsics. In practice
every frame's extrinsics come from an independent per-frame pose estimate, so
they vary slightly. This script treats the per-camera median position and
mean rotation (across all frames) as the consensus pose, then ranks each
frame's cameras.json by how far it deviates from that consensus: primarily by
mean position error (meters), with mean rotation error (degrees) as a
tie-breaker. The winning frame's cameras.json is copied to
GAUSSIANS_DIR/cameras.json (or --out, if given).

Usage:
  select_best_cameras_json.py GAUSSIANS_DIR [--pattern frame_*_cameras.json] [--out PATH]

GAUSSIANS_DIR is the instantsplat_gaussians folder produced by
run_instantsplat_episode.sh, containing one cameras.json per frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_cameras_json(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8") as f:
        entries = json.load(f)
    return {entry["img_name"]: entry for entry in entries}


def rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    m = rotation
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def mean_quaternion(quaternions: list[np.ndarray]) -> np.ndarray:
    ref = quaternions[0]
    aligned = [q if np.dot(q, ref) >= 0 else -q for q in quaternions]
    mean = np.mean(aligned, axis=0)
    return mean / np.linalg.norm(mean)


def rotation_angle_deg(r_a: np.ndarray, r_b: np.ndarray) -> float:
    relative = r_a.T @ r_b
    cos_angle = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the cameras.json whose extrinsics are most consistent across an episode's frames."
    )
    parser.add_argument("gaussians_dir", type=Path, help="Folder containing per-frame cameras.json files.")
    parser.add_argument(
        "--pattern",
        default="frame_*_cameras.json",
        help="Glob pattern for per-frame camera JSON files. Default: frame_*_cameras.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to save a copy of the best cameras.json. Default: GAUSSIANS_DIR/cameras.json",
    )
    args = parser.parse_args()

    paths = sorted(args.gaussians_dir.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching {args.pattern} found under {args.gaussians_dir}")

    per_frame = {path: load_cameras_json(path) for path in paths}
    camera_names = sorted(next(iter(per_frame.values())).keys())
    for path, cameras in per_frame.items():
        missing = set(camera_names) - set(cameras.keys())
        if missing:
            raise ValueError(f"{path} is missing camera(s): {sorted(missing)}")

    consensus_position = {}
    consensus_rotation = {}
    for camera_name in camera_names:
        positions = np.stack(
            [np.asarray(per_frame[path][camera_name]["position"], dtype=np.float64) for path in paths]
        )
        consensus_position[camera_name] = np.median(positions, axis=0)

        quaternions = [
            rotation_to_quaternion(np.asarray(per_frame[path][camera_name]["rotation"], dtype=np.float64))
            for path in paths
        ]
        consensus_rotation[camera_name] = quaternion_to_rotation(mean_quaternion(quaternions))

    results = []
    for path in paths:
        position_errors = []
        rotation_errors = []
        for camera_name in camera_names:
            entry = per_frame[path][camera_name]
            position = np.asarray(entry["position"], dtype=np.float64)
            rotation = np.asarray(entry["rotation"], dtype=np.float64)

            position_errors.append(np.linalg.norm(position - consensus_position[camera_name]))
            rotation_errors.append(rotation_angle_deg(rotation, consensus_rotation[camera_name]))

        results.append(
            {
                "path": path,
                "mean_position_error_m": float(np.mean(position_errors)),
                "max_position_error_m": float(np.max(position_errors)),
                "mean_rotation_error_deg": float(np.mean(rotation_errors)),
                "max_rotation_error_deg": float(np.max(rotation_errors)),
            }
        )

    results.sort(key=lambda r: (r["mean_position_error_m"], r["mean_rotation_error_deg"]))

    print(f"{'rank':>4}  {'frame file':<40}  {'mean_pos_err_m':>15}  {'max_pos_err_m':>14}  {'mean_rot_err_deg':>17}  {'max_rot_err_deg':>16}")
    for rank, r in enumerate(results, start=1):
        print(
            f"{rank:>4}  {r['path'].name:<40}  {r['mean_position_error_m']:>15.5f}  "
            f"{r['max_position_error_m']:>14.5f}  {r['mean_rotation_error_deg']:>17.3f}  {r['max_rotation_error_deg']:>16.3f}"
        )

    best = results[0]
    out_path = args.out if args.out is not None else args.gaussians_dir / "cameras.json"
    out_path.write_text(best["path"].read_text(encoding="utf-8"), encoding="utf-8")
    print(f"\nBest: {best['path']}")
    print(f"Copied to: {out_path}")


if __name__ == "__main__":
    main()
