#!/usr/bin/env python3
"""
Randomly sample camera positions between selected camera pairs.

Example with Y and Z jitter ranges:
    python sample_camera_positions.py \
        --json cameras_2.json \
        --pairs 6,2 2,4 4,1 1,5 \
        --samples-per-pair 20 \
        --y-jitter-range -0.05 0.40 \
        --z-jitter-range -0.10 0.20 \
        --output sampled_positions.txt

Example with old symmetric Z jitter:
    python sample_camera_positions.py \
        --json cameras_2.json \
        --pairs 6,2 2,4 4,1 1,5 \
        --samples-per-pair 20 \
        --z-jitter 0.15 \
        --output sampled_positions.txt

Output format:
    x,y,z
    x,y,z
    ...
"""

import argparse
import json
import random
from pathlib import Path


def load_camera_positions(json_path):
    """Load camera positions from a JSON file and return dict like {'camera1': [x,y,z]}."""
    with open(json_path, "r") as f:
        cameras = json.load(f)

    positions = {}
    for cam in cameras:
        name = cam["img_name"]
        positions[name] = cam["position"]

    return positions


def parse_pair(pair_text):
    """
    Convert pair string like '6,2' into ('cam6', 'cam2').
    Also accepts 'cam6,cam2'.
    """
    parts = [p.strip() for p in pair_text.split(",") if p.strip()]

    if len(parts) != 2:
        raise ValueError(f"Expected pair in format 'a,b' but got '{pair_text}'")

    a, b = parts

    if not a.startswith("cam"):
        a = f"cam{a}"
    if not b.startswith("cam"):
        b = f"cam{b}"

    return a, b


def interpolate_position(p1, p2, t):
    """Linear interpolation between two 3D positions."""
    return [
        p1[i] * (1.0 - t) + p2[i] * t
        for i in range(3)
    ]


def validate_range(range_values, name):
    """Validate a range argument like [min, max]."""
    if range_values is None:
        return None

    low, high = range_values

    if low > high:
        raise ValueError(f"{name} minimum must be <= maximum. Got {low} > {high}")

    return low, high


def sample_positions_between_pair(
    p1,
    p2,
    samples_per_pair,
    y_jitter_range=None,
    z_jitter_range=None,
    t_min=0.02,
    t_max=0.98,
):
    """
    Sample random positions between p1 and p2.

    y_jitter_range:
        Example: (-0.05, 0.40) means y += random(-0.05, 0.40)

    z_jitter_range:
        Example: (-0.10, 0.20) means z += random(-0.10, 0.20)

    t_min/t_max:
        Avoid exact endpoints. For example, 0.02 to 0.98 means samples are
        between the cameras but not exactly on either camera.
    """
    sampled = []

    for _ in range(samples_per_pair):
        # Step 1: sample the camera position between the two cameras
        t = random.uniform(t_min, t_max)
        p = interpolate_position(p1, p2, t)

        # Step 2: apply random Y offset, if requested
        if y_jitter_range is not None:
            p[1] += random.uniform(y_jitter_range[0], y_jitter_range[1])

        # Step 3: apply random Z offset, if requested
        if z_jitter_range is not None:
            p[2] += random.uniform(z_jitter_range[0], z_jitter_range[1])

        return_position = p
        sampled.append(return_position)

    return sampled


def main():
    parser = argparse.ArgumentParser(
        description="Sample random camera positions between selected camera pairs."
    )

    parser.add_argument(
        "--json",
        required=True,
        help="Path to camera JSON file, e.g. cameras_2.json",
    )

    parser.add_argument(
        "--pairs",
        nargs="+",
        required=True,
        help="Camera pairs like: 6,2 2,4 4,1 1,5",
    )

    parser.add_argument(
        "--samples-per-pair",
        type=int,
        default=20,
        help="Number of random positions to sample between each pair.",
    )

    parser.add_argument(
        "--y-jitter-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("Y_MIN", "Y_MAX"),
        help="Random Y offset range. Example: --y-jitter-range -0.05 0.40",
    )

    parser.add_argument(
        "--z-jitter-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("Z_MIN", "Z_MAX"),
        help="Random Z offset range. Example: --z-jitter-range -0.10 0.20",
    )

    parser.add_argument(
        "--z-jitter",
        type=float,
        default=None,
        help=(
            "Old shortcut for symmetric Z jitter. "
            "Example: --z-jitter 0.15 means --z-jitter-range -0.15 0.15. "
            "Ignored if --z-jitter-range is provided."
        ),
    )

    parser.add_argument(
        "--output",
        default="sampled_camera_positions.txt",
        help="Output TXT file path.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible samples. Omit for different samples every run.",
    )

    parser.add_argument(
        "--t-min",
        type=float,
        default=0.02,
        help="Minimum interpolation factor. Keeps samples away from first endpoint.",
    )

    parser.add_argument(
        "--t-max",
        type=float,
        default=0.98,
        help="Maximum interpolation factor. Keeps samples away from second endpoint.",
    )

    parser.add_argument(
        "--sort-by-pair",
        action="store_true",
        help="Sort samples along each pair from first camera to second camera.",
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.samples_per_pair <= 0:
        raise ValueError("--samples-per-pair must be greater than 0")

    if args.t_min < 0 or args.t_max > 1 or args.t_min >= args.t_max:
        raise ValueError("Require 0 <= t_min < t_max <= 1")

    y_jitter_range = validate_range(args.y_jitter_range, "--y-jitter-range")

    if args.z_jitter_range is not None:
        z_jitter_range = validate_range(args.z_jitter_range, "--z-jitter-range")
    elif args.z_jitter is not None:
        if args.z_jitter < 0:
            raise ValueError("--z-jitter must be >= 0")
        z_jitter_range = (-args.z_jitter, args.z_jitter)
    else:
        z_jitter_range = None

    positions = load_camera_positions(args.json)

    all_samples = []

    for pair_text in args.pairs:
        cam_a, cam_b = parse_pair(pair_text)

        if cam_a not in positions:
            raise KeyError(f"{cam_a} not found in {args.json}")
        if cam_b not in positions:
            raise KeyError(f"{cam_b} not found in {args.json}")

        p1 = positions[cam_a]
        p2 = positions[cam_b]

        pair_samples = sample_positions_between_pair(
            p1=p1,
            p2=p2,
            samples_per_pair=args.samples_per_pair,
            y_jitter_range=y_jitter_range,
            z_jitter_range=z_jitter_range,
            t_min=args.t_min,
            t_max=args.t_max,
        )

        if args.sort_by_pair:
            # Sort based on approximate distance from first camera.
            pair_samples.sort(
                key=lambda p: sum((p[i] - p1[i]) ** 2 for i in range(3))
            )

        all_samples.extend(pair_samples)

    output_path = Path(args.output)

    with output_path.open("w") as f:
        for p in all_samples:
            f.write(f"{p[0]:.6f},{p[1]:.6f},{p[2]:.6f}\n")

    print(f"Saved {len(all_samples)} positions to {output_path}")
    print(f"Y jitter range: {y_jitter_range}")
    print(f"Z jitter range: {z_jitter_range}")


if __name__ == "__main__":
    main()
