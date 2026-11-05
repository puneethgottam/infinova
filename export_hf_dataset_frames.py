#!/usr/bin/env python3
"""Download a LeRobot dataset from Hugging Face and export camera frames.

Run from the repository root with the LeRobot conda environment, for example:

    conda run -n infinova-lerobot python export_hf_dataset_frames.py lerobot/pusht_image

The output layout is:

    exported_frames/<repo_id>/
    ├── episode_000000/
    │   ├── camera1/
    │   │   ├── frame_000000.png
    │   │   └── ...
    │   └── ...
    └── ...
"""

from __future__ import annotations

import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lerobot.datasets import LeRobotDataset


def parse_episode_indices(raw: str | None) -> list[int] | None:
    if raw is None:
        return None

    episodes: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            episodes.update(range(int(start), int(end) + 1))
        else:
            episodes.add(int(part))

    return sorted(episodes)


def safe_path_component(value: str) -> str:
    value = value.replace("/", "__")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def camera_folder_name(camera_key: str, index: int) -> str:
    short_name = camera_key.rsplit(".", 1)[-1]
    if short_name:
        return safe_path_component(short_name)
    return f"camera{index + 1}"


def tensor_to_pil_image(frame: torch.Tensor) -> Image.Image:
    frame = frame.detach().cpu()

    if frame.ndim != 3:
        raise ValueError(f"Expected a 3D image tensor, got shape {tuple(frame.shape)}.")

    # LeRobot returns camera tensors as CHW. Keep HWC input working too.
    if frame.shape[0] in (1, 3, 4):
        frame = frame.permute(1, 2, 0)

    array = frame.numpy()
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, 0.0, 1.0) * 255.0
    else:
        array = np.clip(array, 0, 255)

    array = array.astype(np.uint8)
    if array.shape[-1] == 1:
        array = array[..., 0]
    return Image.fromarray(array)


def maybe_progress(iterable, *, total: int, desc: str):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


def export_dataset_frames(
    repo_id: str,
    output_dir: Path,
    dataset_root: Path | None,
    episodes: list[int] | None,
    revision: str | None,
    overwrite: bool,
) -> None:
    repo_output_dir = output_dir / safe_path_component(repo_id)
    if repo_output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{repo_output_dir} already exists. Pass --overwrite to replace the previous export."
            )
        shutil.rmtree(repo_output_dir)
    repo_output_dir.mkdir(parents=True, exist_ok=True)

    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_root,
        episodes=episodes,
        revision=revision,
        download_videos=True,
        return_uint8=True,
    )

    camera_keys = dataset.meta.camera_keys
    if not camera_keys:
        raise ValueError(f"Dataset {repo_id!r} has no image or video camera features.")

    frame_counts: dict[int, int] = defaultdict(int)
    camera_dirs = {key: camera_folder_name(key, index) for index, key in enumerate(camera_keys)}

    print(f"Dataset: {repo_id}")
    print(f"Episodes: {dataset.num_episodes}")
    print(f"Frames: {dataset.num_frames}")
    print(f"Cameras: {', '.join(camera_keys)}")
    print(f"Output: {repo_output_dir}")

    for rel_idx in maybe_progress(range(len(dataset)), total=len(dataset), desc="Exporting frames"):
        item = dataset[rel_idx]
        episode_index = int(item["episode_index"].item())
        frame_index = frame_counts[episode_index]
        episode_dir = repo_output_dir / f"episode_{episode_index:06d}"

        for camera_key in camera_keys:
            camera_dir = episode_dir / camera_dirs[camera_key]
            camera_dir.mkdir(parents=True, exist_ok=True)
            image = tensor_to_pil_image(item[camera_key])
            image.save(camera_dir / f"frame_{frame_index:06d}.png")

        frame_counts[episode_index] += 1

    print("Done.")
    for episode_index in sorted(frame_counts):
        print(f"episode_{episode_index:06d}: {frame_counts[episode_index]} frames per camera")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a LeRobot dataset from Hugging Face and export every camera frame as PNGs."
    )
    parser.add_argument("repo_id", help="Hugging Face dataset repo id, e.g. lerobot/pusht_image")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exported_frames"),
        help="Directory where the per-episode folders will be written.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Optional local LeRobot dataset cache/download directory.",
    )
    parser.add_argument(
        "--episodes",
        type=parse_episode_indices,
        default=None,
        help="Optional episode list/ranges, e.g. 0,2,5-9. Defaults to all episodes.",
    )
    parser.add_argument("--revision", default=None, help="Optional Hugging Face revision, branch, tag, or SHA.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing export directory.")
    args = parser.parse_args()

    export_dataset_frames(
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        dataset_root=args.dataset_root,
        episodes=args.episodes,
        revision=args.revision,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
