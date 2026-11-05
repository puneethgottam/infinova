#!/usr/bin/env python

"""Materialize folder-backed camera-view virtual episodes into a real
LeRobotDataset and publish it to the Hugging Face Hub.

``smolvla_folder_view_dataloader.py`` / ``pi0_folder_view_dataloader.py`` build
virtual episodes on the fly, at training time, by pairing rows from a base
LeRobot dataset (actions/states/tasks) with camera images loaded from a local
folder tree (``--image-root``). That requires the image folder to be present
on whatever machine trains the policy.

This script runs the exact same virtual-episode / camera-combination logic
once, up front, and writes the result out as a standalone LeRobotDataset
(3 image slots per frame, one dataset episode per camera combination). Once
pushed to the Hub, training can load it like any other repo_id from any
machine, with no local image folder and no custom dataloader.

Example, publishing only original-camera combinations (no novel views):

    python publish_folder_view_dataset.py \\
        --repo-id your-username/base_dataset \\
        --image-root /path/to/episode_folders \\
        --output-repo-id your-username/folder_view_dataset
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from tqdm import tqdm

from lerobot.datasets import LeRobotDataset
from lerobot.utils.constants import DEFAULT_FEATURES

from smolvla_folder_view_dataloader import DEFAULT_SLOT_IMAGE_KEYS, SmolVLAFolderViewDataset


def _probe_camera_shape(source: SmolVLAFolderViewDataset, episode_index: int, camera_dir: Path) -> tuple:
    # CHW as returned by _load_image -> HWC, matching how frames are written (see VirtualFrameSource).
    image = source._load_image(episode_index, 0, camera_dir)
    channels, height, width = image.shape
    return (height, width, channels)


def build_output_features(source: SmolVLAFolderViewDataset, slot_image_keys: tuple, use_videos: bool) -> dict:
    base_meta = source.base.meta
    features = {
        key: value.copy()
        for key, value in base_meta.features.items()
        if key not in base_meta.camera_keys and key not in DEFAULT_FEATURES
    }

    # Each slot always draws from the same camera "role" across every combination (the fixed
    # camera for fixed_slot_index, one of the original cameras otherwise), so its resolution is
    # stable dataset-wide even though which original camera lands there varies per combination.
    probe_episode_index = source._source_episode_indices()[0]
    fixed_camera_dir = Path(source.fixed_camera)
    variable_camera_dir = source.original_sources[0].relative_dir
    for slot_index, slot_key in enumerate(slot_image_keys):
        camera_dir = fixed_camera_dir if slot_index == source.fixed_slot_index else variable_camera_dir
        shape = _probe_camera_shape(source, probe_episode_index, camera_dir)
        features[slot_key] = {
            "dtype": "video" if use_videos else "image",
            "shape": shape,
            "names": ["height", "width", "channels"],
        }
    return features


class VirtualFrameSource(torch.utils.data.Dataset):
    """Loads one materialized (non-chunked) frame per virtual-episode index.

    Unlike ``SmolVLAFolderViewDataset.__getitem__``, this does not expand
    delta-timestamp action/state windows: it copies the base dataset's raw
    per-frame values as-is, which is what a from-scratch published dataset
    should contain.
    """

    def __init__(self, source: SmolVLAFolderViewDataset, non_camera_feature_keys: list[str]):
        self.source = source
        self.non_camera_feature_keys = non_camera_feature_keys

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, idx: int) -> dict:
        source = self.source
        virtual_episode = source._virtual_episode_for_index(idx)
        frame_offset = idx - virtual_episode.dataset_from_index
        source_abs_idx = virtual_episode.source_dataset_from_index + frame_offset
        source_rel_idx = source._absolute_to_relative_idx[source_abs_idx]
        row = source._hf_dataset[source_rel_idx]

        frame = {key: row[key] for key in self.non_camera_feature_keys}
        task_idx = int(row["task_index"])
        frame["task"] = source.base.meta.tasks.iloc[task_idx].name

        source_frame_index = int(row["frame_index"])
        for slot_key, camera_dir in zip(source.slot_image_keys, virtual_episode.camera_dirs, strict=True):
            image_chw = source._load_image(virtual_episode.source_episode_index, source_frame_index, camera_dir)
            frame[slot_key] = image_chw.permute(1, 2, 0).contiguous()

        return {
            "frame": frame,
            "episode_length": virtual_episode.dataset_to_index - virtual_episode.dataset_from_index,
        }


def publish(args: argparse.Namespace) -> None:
    source = SmolVLAFolderViewDataset(
        repo_id=args.repo_id,
        image_root=args.image_root,
        root=args.root,
        original_cameras=args.original_cameras,
        fixed_camera=args.fixed_camera,
        fixed_slot_index=args.fixed_slot_index,
        include_original_camera_pairs=not args.no_original_camera_pairs,
        include_real_novel_pairs=args.use_novel_views and not args.no_real_novel_pairs,
        max_orig_novel_pairs=args.max_orig_novel_pairs,
        target_pairs_per_episode=args.target_pairs_per_episode,
        max_combinations_per_episode=args.max_combinations_per_episode,
        combination_seed=args.combination_seed,
        slot_image_keys=tuple(args.slot_image_keys),
        episodes=args.episodes,
    )
    logging.info("source virtual episodes: %d", source.num_episodes)
    logging.info("source virtual frames: %d", source.num_frames)
    logging.info("camera combinations per source episode: %d", len(source.combinations))

    non_camera_feature_keys = [
        key
        for key in source.base.meta.features
        if key not in source.base.meta.camera_keys and key not in DEFAULT_FEATURES
    ]
    features = build_output_features(source, tuple(args.slot_image_keys), args.use_videos)

    if args.output_root is not None and args.overwrite and Path(args.output_root).exists():
        import shutil

        logging.warning("Removing existing output root: %s", args.output_root)
        shutil.rmtree(args.output_root)

    output = LeRobotDataset.create(
        repo_id=args.output_repo_id,
        fps=source.base.meta.fps,
        features=features,
        root=args.output_root,
        robot_type=source.base.meta.robot_type,
        use_videos=args.use_videos,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
        batch_encoding_size=args.batch_encoding_size,
    )

    frame_source = VirtualFrameSource(source, non_camera_feature_keys)
    loader = torch.utils.data.DataLoader(
        frame_source,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: batch[0],
    )

    try:
        episode_frame_count = 0
        for sample in tqdm(loader, total=len(frame_source), desc="frames"):
            output.add_frame(sample["frame"])
            episode_frame_count += 1
            if episode_frame_count == sample["episode_length"]:
                output.save_episode()
                episode_frame_count = 0
    finally:
        output.finalize()

    logging.info("published %d episodes / %d frames locally at %s", output.meta.total_episodes, output.meta.total_frames, output.root)

    if args.push:
        logging.info("pushing to hub: %s", args.output_repo_id)
        output.push_to_hub(
            branch=args.branch,
            tags=args.tags,
            license=args.license,
            private=args.private,
        )
        logging.info("done: https://huggingface.co/datasets/%s", args.output_repo_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", required=True, help="Base LeRobot dataset repo id providing actions/states/tasks, e.g. 'user/name'.")
    parser.add_argument(
        "--image-root",
        required=True,
        help="Folder containing episode_XXXXXX/cam*, episode_XXXXXX/camera3, and optionally novel_views_new/* images.",
    )
    parser.add_argument("--root", default=None, help="Optional local LeRobot dataset root/cache for --repo-id.")
    parser.add_argument("--output-repo-id", required=True, help="Hugging Face repo id to create and publish, e.g. 'user/name'.")
    parser.add_argument("--output-root", default=None, help="Local dir to build the dataset in. Defaults to $HF_LEROBOT_HOME/{output-repo-id}.")
    parser.add_argument("--overwrite", action="store_true", help="Remove --output-root first if it already exists.")

    parser.add_argument("--fixed-camera", default="camera3")
    parser.add_argument("--fixed-slot-index", type=int, default=2)
    parser.add_argument("--original-cameras", nargs="*", default=None)
    parser.add_argument("--use-novel-views", action="store_true")
    parser.add_argument(
        "--max-orig-novel-pairs",
        type=int,
        default=None,
        help="Cap on original<->novel pairs (both directions combined) per source episode. "
        "Sampled down if more exist; used as-is if fewer.",
    )
    parser.add_argument(
        "--target-pairs-per-episode",
        type=int,
        default=None,
        help="Total pairs per source episode: original-original pairs are always included in full, "
        "then novel-novel pairs are sampled to fill (target - orig_orig - orig_novel). "
        "Only affects exhaustive mode (i.e. when --max-combinations-per-episode is not set).",
    )
    parser.add_argument("--max-combinations-per-episode", type=int, default=None)
    parser.add_argument("--combination-seed", type=int, default=0)
    parser.add_argument("--no-original-camera-pairs", action="store_true")
    parser.add_argument("--no-real-novel-pairs", action="store_true")
    parser.add_argument("--episodes", type=int, nargs="*", default=None, help="Restrict to these source episode indices.")
    parser.add_argument("--slot-image-keys", nargs=3, default=list(DEFAULT_SLOT_IMAGE_KEYS))

    parser.add_argument("--use-videos", dest="use_videos", action="store_true", default=True)
    parser.add_argument("--no-videos", dest="use_videos", action="store_false", help="Store per-frame PNGs instead of encoding mp4s.")
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--batch-encoding-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4, help="Parallel workers for reading/decoding source frame images.")

    parser.add_argument("--push", dest="push", action="store_true", default=True)
    parser.add_argument("--no-push", dest="push", action="store_false", help="Build the dataset locally without pushing to the Hub.")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--branch", default=None)
    parser.add_argument("--tags", nargs="*", default=None)
    parser.add_argument("--license", default="apache-2.0")

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    publish(parse_args())


if __name__ == "__main__":
    main()
