#!/usr/bin/env python

"""Folder-backed view augmentation dataset for SmolVLA training.

This keeps actions, states, tasks, episode boundaries, fps, features, and
normalization statistics in the LeRobot/Hugging Face dataset, while loading the
three SmolVLA image slots from an external folder tree.

Expected image layout by default:

    /path/to/folder/
      episode_000000/
        cam0/frame_000000.png
        cam1/frame_000000.png
        camera3/frame_000000.png
        ...
        novel_views_new/view_00/frame_000000.png
        novel_views_new/view_01/frame_000000.png

Each original episode is exposed as many virtual episodes. Original-camera
permutations are created for every episode. When novel views are enabled,
real-novel combinations are added only for episodes that contain the
``novel_views_new`` folder. The fixed camera is always placed in slot 3 by
default:

    (real_camera, novel_view, camera3)
    (novel_view, real_camera, camera3)
"""

from __future__ import annotations

import argparse
import logging
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from PIL import Image

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.utils.constants import ACTION


DEFAULT_SLOT_IMAGE_KEYS = (
    "observation.images.slot_1",
    "observation.images.slot_2",
    "observation.images.slot_3",
)


@dataclass(frozen=True)
class CameraSource:
    """A named image source relative to an episode folder."""

    name: str
    relative_dir: Path


@dataclass(frozen=True)
class VirtualEpisode:
    virtual_episode_index: int
    source_episode_index: int
    combination_index: int
    camera_dirs: tuple[Path, Path, Path]
    dataset_from_index: int
    dataset_to_index: int
    source_dataset_from_index: int


class SmolVLAFolderViewDataset(torch.utils.data.Dataset):
    """Expose folder camera combinations as virtual LeRobot episodes.

    Args:
        repo_id: Hugging Face dataset id containing actions/states/tasks.
        image_root: Folder containing ``episode_XXXXXX`` image folders.
        root: Optional local LeRobot dataset root/cache path.
        original_cameras: Original camera folder names available under each
            episode. If omitted, uses ``cam0`` through ``cam4``.
        novel_views: Novel view folder names under ``novel_views_dir``. If
            omitted, discovers directories from the first episode folder.
        fixed_camera: Original camera folder kept fixed in one SmolVLA slot.
        fixed_slot_index: Zero-based image slot for the fixed camera. Use ``2``
            for "slot3".
        slot_image_keys: LeRobot feature keys returned to SmolVLA.
        frame_pattern: Filename pattern receiving ``frame_index``.
        image_transforms: Optional transform applied to loaded image tensors.
        episodes: Optional source episode indices to use.
        revision: Optional HF dataset revision.
        video_backend: Kept for LeRobotDataset compatibility.
        tolerance_s: Timestamp tolerance for LeRobot action chunks.
    """

    def __init__(
        self,
        repo_id: str,
        image_root: str | Path,
        root: str | Path | None = None,
        original_cameras: Sequence[str] | None = None,
        novel_views: Sequence[str] | None = None,
        fixed_camera: str = "camera3",
        fixed_slot_index: int = 2,
        include_original_camera_pairs: bool = True,
        include_real_novel_pairs: bool = False,
        max_orig_novel_pairs: int | None = None,
        target_pairs_per_episode: int | None = None,
        max_combinations_per_episode: int | None = None,
        combination_seed: int = 0,
        slot_image_keys: Sequence[str] = DEFAULT_SLOT_IMAGE_KEYS,
        novel_views_dir: str = "novel_views_new",
        episode_dir_pattern: str = "episode_{episode_index:06d}",
        frame_pattern: str = "frame_{frame_index:06d}.png",
        novel_frame_pattern: str = "frame_{frame_index:06d}_gaussian.png",
        image_transforms: Callable | None = None,
        episodes: list[int] | None = None,
        revision: str | None = None,
        video_backend: str | None = None,
        tolerance_s: float = 1e-4,
    ) -> None:
        super().__init__()
        if len(slot_image_keys) != 3:
            raise ValueError(f"SmolVLA expects exactly 3 image slot keys, got {slot_image_keys}.")
        if fixed_slot_index not in {0, 1, 2}:
            raise ValueError(f"fixed_slot_index must be 0, 1, or 2. Got {fixed_slot_index}.")
        if max_combinations_per_episode is not None and max_combinations_per_episode <= 0:
            raise ValueError(
                f"max_combinations_per_episode must be positive or None. Got {max_combinations_per_episode}."
            )
        if max_orig_novel_pairs is not None and max_orig_novel_pairs <= 0:
            raise ValueError(f"max_orig_novel_pairs must be positive or None. Got {max_orig_novel_pairs}.")
        if target_pairs_per_episode is not None and target_pairs_per_episode <= 0:
            raise ValueError(f"target_pairs_per_episode must be positive or None. Got {target_pairs_per_episode}.")

        self.image_root = Path(image_root)
        self.fixed_camera = fixed_camera
        self.fixed_slot_index = fixed_slot_index
        self.include_original_camera_pairs = include_original_camera_pairs
        self.include_real_novel_pairs = include_real_novel_pairs
        self.max_orig_novel_pairs = max_orig_novel_pairs
        self.target_pairs_per_episode = target_pairs_per_episode
        self.max_combinations_per_episode = max_combinations_per_episode
        self.combination_seed = combination_seed
        self.slot_image_keys = tuple(slot_image_keys)
        self.novel_views_dir = novel_views_dir
        self.episode_dir_pattern = episode_dir_pattern
        self.frame_pattern = frame_pattern
        self.novel_frame_pattern = novel_frame_pattern
        self.image_transforms = image_transforms

        # SmolVLA needs future action chunks; the base HF table remains the
        # source of truth for action/state/task alignment.
        base_meta = LeRobotDatasetMetadata(repo_id=repo_id, root=root, revision=revision)
        delta_timestamps = resolve_delta_timestamps(SmolVLAConfig(), base_meta)
        self.base = LeRobotDataset(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
            revision=revision,
            download_videos=False,
            video_backend=video_backend,
            return_uint8=True,
            tolerance_s=tolerance_s,
        )

        self._hf_dataset = self.base.hf_dataset
        self._absolute_to_relative_idx = self._build_absolute_to_relative_index()

        if original_cameras is None:
            original_cameras = [f"cam{idx}" for idx in range(5)]
        self.original_sources = tuple(CameraSource(name=cam, relative_dir=Path(cam)) for cam in original_cameras)
        if include_real_novel_pairs:
            self._novel_views_by_episode = self._discover_novel_views_by_episode(novel_views)
            novel_views = sorted({view for views in self._novel_views_by_episode.values() for view in views})
        else:
            self._novel_views_by_episode = {episode_idx: [] for episode_idx in self._source_episode_indices()}
            novel_views = []

        self.novel_sources = tuple(
            CameraSource(name=view, relative_dir=Path(novel_views_dir) / view) for view in novel_views
        )
        self.combinations = self._build_camera_combinations(self.novel_sources)
        if not self.combinations:
            raise ValueError(
                "No camera combinations were created. Check original_cameras, novel_views, "
                "include_original_camera_pairs, and include_real_novel_pairs."
            )
        self.virtual_episodes = self._build_virtual_episodes()
        if not self.virtual_episodes:
            raise ValueError("No virtual episodes were created from the selected source episodes.")
        self.meta = self._build_metadata_proxy()

    @property
    def num_frames(self) -> int:
        return len(self)

    @property
    def num_episodes(self) -> int:
        return len(self.virtual_episodes)

    def __len__(self) -> int:
        return self.virtual_episodes[-1].dataset_to_index if self.virtual_episodes else 0

    def __getitem__(self, idx: int) -> dict[str, Any]:
        virtual_episode = self._virtual_episode_for_index(idx)
        frame_offset = idx - virtual_episode.dataset_from_index
        source_abs_idx = virtual_episode.source_dataset_from_index + frame_offset
        source_rel_idx = self._absolute_to_relative_idx[source_abs_idx]
        item = self._get_non_camera_item(source_rel_idx)

        source_episode_index = virtual_episode.source_episode_index
        source_frame_index = int(_as_python_scalar(item["frame_index"]))

        for slot_key, camera_dir in zip(self.slot_image_keys, virtual_episode.camera_dirs, strict=True):
            item[slot_key] = self._load_image(source_episode_index, source_frame_index, camera_dir)

        # Remove base image/video keys so the policy only sees the three chosen slots.
        for camera_key in self.base.meta.camera_keys:
            item.pop(camera_key, None)

        item["source_episode_index"] = torch.tensor(source_episode_index, dtype=torch.int64)
        item["camera_combination_index"] = torch.tensor(virtual_episode.combination_index, dtype=torch.int64)
        item["episode_index"] = torch.tensor(self._virtual_episode_number(virtual_episode), dtype=torch.int64)
        return item

    def _get_non_camera_item(self, source_rel_idx: int) -> dict[str, Any]:
        item = self._hf_dataset[source_rel_idx]
        source_episode_index = int(_as_python_scalar(item["episode_index"]))
        source_abs_idx = int(_as_python_scalar(item["index"]))

        if self.base.reader.delta_indices is not None:
            query_indices, padding = self.base.reader._get_query_indices(source_abs_idx, source_episode_index)
            item = {**item, **padding, **self._query_non_camera_hf_dataset(query_indices)}

        for camera_key in self.base.meta.camera_keys:
            item.pop(camera_key, None)

        task_idx = int(_as_python_scalar(item["task_index"]))
        item["task"] = self.base.meta.tasks.iloc[task_idx].name

        subtasks = getattr(self.base.meta, "subtasks", None)
        if "subtask_index" in self.base.meta.features and subtasks is not None:
            subtask_idx = int(_as_python_scalar(item["subtask_index"]))
            item["subtask"] = subtasks.iloc[subtask_idx].name

        return item

    def _query_non_camera_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict[str, torch.Tensor]:
        result = {}
        for key, q_idx in query_indices.items():
            if key in self.base.meta.camera_keys:
                continue

            relative_indices = [self._absolute_to_relative_idx[idx] for idx in q_idx]
            try:
                result[key] = torch.stack(self._hf_dataset[key][relative_indices])
            except (KeyError, TypeError, IndexError):
                result[key] = torch.stack(self._hf_dataset[relative_indices][key])
        return result

    def _build_absolute_to_relative_index(self) -> dict[int, int]:
        indices = self._hf_dataset.data.column("index").to_numpy().tolist()
        return {int(abs_idx): rel_idx for rel_idx, abs_idx in enumerate(indices)}

    def _discover_novel_views_by_episode(
        self, requested_views: Sequence[str] | None = None
    ) -> dict[int, list[str]]:
        views_by_episode = {}
        missing_episode_count = 0
        empty_episode_count = 0

        for episode_index in self._source_episode_indices():
            novel_root = self.image_root / self._episode_dir(episode_index) / self.novel_views_dir
            if not novel_root.is_dir():
                views_by_episode[episode_index] = []
                missing_episode_count += 1
                continue

            available_views = {path.name for path in novel_root.iterdir() if path.is_dir()}
            if requested_views is None:
                views = sorted(available_views)
            else:
                views = sorted(view for view in requested_views if view in available_views)

            if not views:
                empty_episode_count += 1
            views_by_episode[episode_index] = views

        if missing_episode_count:
            logging.warning(
                "%d source episodes are missing %s; they will use original-camera permutations only.",
                missing_episode_count,
                self.novel_views_dir,
            )
        if empty_episode_count:
            logging.warning(
                "%d source episodes have no usable novel views; they will use original-camera permutations only.",
                empty_episode_count,
            )

        return views_by_episode

    def _build_camera_combinations(
        self, novel_sources: Sequence[CameraSource], source_episode_index: int | None = None
    ) -> list[tuple[Path, Path, Path]]:
        if self.max_combinations_per_episode is not None:
            return self._build_random_camera_combinations(novel_sources, source_episode_index)
        return self._build_exhaustive_camera_combinations(novel_sources, source_episode_index)

    def _sample_pairs(
        self, pairs: list[tuple[Path, Path]], count: int, seed_offset: int
    ) -> list[tuple[Path, Path]]:
        if count >= len(pairs):
            return list(pairs)
        rng = random.Random(self.combination_seed + seed_offset)
        return rng.sample(pairs, count)

    def _build_exhaustive_camera_combinations(
        self, novel_sources: Sequence[CameraSource], source_episode_index: int | None = None
    ) -> list[tuple[Path, Path, Path]]:
        fixed = Path(self.fixed_camera)
        seed_episode = 0 if source_episode_index is None else source_episode_index

        orig_orig_pairs: list[tuple[Path, Path]] = []
        if self.include_original_camera_pairs:
            orig_orig_pairs = [
                (first.relative_dir, second.relative_dir)
                for first in self.original_sources
                for second in self.original_sources
                if first.relative_dir != second.relative_dir
            ]

        orig_novel_pairs: list[tuple[Path, Path]] = []
        novel_novel_pairs: list[tuple[Path, Path]] = []
        if self.include_real_novel_pairs:
            for original in self.original_sources:
                for novel in novel_sources:
                    orig_novel_pairs.append((original.relative_dir, novel.relative_dir))
                    orig_novel_pairs.append((novel.relative_dir, original.relative_dir))
            if self.max_orig_novel_pairs is not None:
                orig_novel_pairs = self._sample_pairs(orig_novel_pairs, self.max_orig_novel_pairs, seed_episode)

            novel_novel_pairs = [
                (first.relative_dir, second.relative_dir)
                for first in novel_sources
                for second in novel_sources
                if first.relative_dir != second.relative_dir
            ]
            if self.target_pairs_per_episode is not None:
                remaining = self.target_pairs_per_episode - len(orig_orig_pairs) - len(orig_novel_pairs)
                remaining = max(0, remaining)
                novel_novel_pairs = self._sample_pairs(novel_novel_pairs, remaining, seed_episode + 1)

        all_pairs = orig_orig_pairs + orig_novel_pairs + novel_novel_pairs
        rng = random.Random(self.combination_seed + seed_episode + 2)
        rng.shuffle(all_pairs)
        return [self._make_camera_slots(first, second, fixed) for first, second in all_pairs]

    def _build_random_camera_combinations(
        self, novel_sources: Sequence[CameraSource], source_episode_index: int | None
    ) -> list[tuple[Path, Path, Path]]:
        fixed = Path(self.fixed_camera)
        camera_pool = [source.relative_dir for source in self.original_sources]
        if self.include_real_novel_pairs:
            camera_pool.extend(source.relative_dir for source in novel_sources)

        ordered_pairs = [
            (first, second) for first in camera_pool for second in camera_pool if first != second
        ]
        if not ordered_pairs:
            return []

        seed_episode = 0 if source_episode_index is None else source_episode_index
        rng = random.Random(self.combination_seed + seed_episode)
        rng.shuffle(ordered_pairs)
        selected_pairs = ordered_pairs[: self.max_combinations_per_episode]
        return [self._make_camera_slots(first, second, fixed) for first, second in selected_pairs]

    def _make_camera_slots(self, first: Path, second: Path, fixed: Path) -> tuple[Path, Path, Path]:
        slots: list[Path | None] = [None, None, None]
        slots[self.fixed_slot_index] = fixed
        variable_slot_indices = [slot for slot in range(3) if slot != self.fixed_slot_index]
        slots[variable_slot_indices[0]] = first
        slots[variable_slot_indices[1]] = second
        return tuple(slot for slot in slots if slot is not None)

    def _build_virtual_episodes(self) -> list[VirtualEpisode]:
        virtual_episodes = []
        virtual_from = 0
        for source_episode_index in self._source_episode_indices():
            source_ep = self.base.meta.episodes[source_episode_index]
            source_from = int(source_ep["dataset_from_index"])
            source_to = int(source_ep["dataset_to_index"])
            episode_length = source_to - source_from
            novel_sources = tuple(
                CameraSource(name=view, relative_dir=Path(self.novel_views_dir) / view)
                for view in self._novel_views_by_episode.get(source_episode_index, [])
            )
            combinations = self._build_camera_combinations(novel_sources, source_episode_index)
            for combination_index, camera_dirs in enumerate(combinations):
                virtual_to = virtual_from + episode_length
                virtual_episodes.append(
                    VirtualEpisode(
                        virtual_episode_index=len(virtual_episodes),
                        source_episode_index=source_episode_index,
                        combination_index=combination_index,
                        camera_dirs=camera_dirs,
                        dataset_from_index=virtual_from,
                        dataset_to_index=virtual_to,
                        source_dataset_from_index=source_from,
                    )
                )
                virtual_from = virtual_to
        return virtual_episodes

    def _build_metadata_proxy(self) -> SimpleNamespace:
        features = {
            key: value.copy()
            for key, value in self.base.meta.features.items()
            if key not in self.base.meta.camera_keys
        }

        camera_feature_template = self._camera_feature_template()
        for slot_key in self.slot_image_keys:
            features[slot_key] = camera_feature_template.copy()

        stats = {
            key: value
            for key, value in self.base.meta.stats.items()
            if key not in self.base.meta.camera_keys
        }
        camera_stats_template = self._camera_stats_template()
        for slot_key in self.slot_image_keys:
            stats[slot_key] = camera_stats_template

        episodes = {
            "episode_index": list(range(len(self.virtual_episodes))),
            "dataset_from_index": [ep.dataset_from_index for ep in self.virtual_episodes],
            "dataset_to_index": [ep.dataset_to_index for ep in self.virtual_episodes],
            "length": [ep.dataset_to_index - ep.dataset_from_index for ep in self.virtual_episodes],
        }

        return SimpleNamespace(
            fps=self.base.meta.fps,
            features=features,
            stats=stats,
            episodes=episodes,
            camera_keys=list(self.slot_image_keys),
            image_keys=list(self.slot_image_keys),
            video_keys=[],
            total_frames=len(self),
            total_episodes=len(self.virtual_episodes),
            tasks=self.base.meta.tasks,
        )

    def _camera_feature_template(self) -> dict[str, Any]:
        if self.base.meta.camera_keys:
            feature = self.base.meta.features[self.base.meta.camera_keys[0]].copy()
            feature["dtype"] = "image"
            return feature
        return {"dtype": "image", "shape": (480, 640, 3), "names": ["height", "width", "channels"]}

    def _camera_stats_template(self) -> dict[str, torch.Tensor]:
        if self.base.meta.camera_keys:
            return self.base.meta.stats.get(self.base.meta.camera_keys[0], {})
        return {}

    def _source_episode_indices(self) -> list[int]:
        return self.base.episodes if self.base.episodes is not None else list(range(self.base.meta.total_episodes))

    def _virtual_episode_for_index(self, idx: int) -> VirtualEpisode:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        # Episodes are contiguous and uniformly grouped by source episode but
        # source episode lengths may differ, so binary search keeps this cheap.
        lo = 0
        hi = len(self.virtual_episodes)
        while lo < hi:
            mid = (lo + hi) // 2
            if idx < self.virtual_episodes[mid].dataset_to_index:
                hi = mid
            else:
                lo = mid + 1
        return self.virtual_episodes[lo]

    def _virtual_episode_number(self, virtual_episode: VirtualEpisode) -> int:
        return virtual_episode.virtual_episode_index

    def _episode_dir(self, episode_index: int) -> str:
        return self.episode_dir_pattern.format(episode_index=episode_index)

    def _frame_file(self, frame_index: int) -> str:
        return self.frame_pattern.format(frame_index=frame_index)

    def _novel_frame_file(self, frame_index: int) -> str:
        return self.novel_frame_pattern.format(frame_index=frame_index)

    def _load_image(self, episode_index: int, frame_index: int, camera_dir: Path) -> torch.Tensor:
        frame_file = (
            self._novel_frame_file(frame_index)
            if camera_dir.parts and camera_dir.parts[0] == self.novel_views_dir
            else self._frame_file(frame_index)
        )
        path = self.image_root / self._episode_dir(episode_index) / camera_dir / frame_file
        if not path.exists():
            raise FileNotFoundError(f"Missing image frame: {path}")

        with Image.open(path) as image:
            image = image.convert("RGB")
            array = np.array(image, copy=True)

        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        if self.image_transforms is not None:
            tensor = self.image_transforms(tensor)
        return tensor


def make_smolvla_folder_view_dataloader(
    dataset: SmolVLAFolderViewDataset,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = True,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def _as_python_scalar(value: Any) -> int | float:
    if isinstance(value, torch.Tensor):
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Base LeRobot dataset repo id providing actions/states/tasks, e.g. 'user/name'.")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--fixed-camera", default="camera3")
    parser.add_argument("--fixed-slot-index", type=int, default=2)
    parser.add_argument("--original-cameras", nargs="*", default=None)
    parser.add_argument("--use-novel-views", action="store_true")
    parser.add_argument("--max-combinations-per-episode", type=int, default=None)
    parser.add_argument("--combination-seed", type=int, default=0)
    parser.add_argument("--no-original-camera-pairs", action="store_true")
    parser.add_argument("--no-real-novel-pairs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset = SmolVLAFolderViewDataset(
        repo_id=args.repo_id,
        image_root=args.image_root,
        root=args.root,
        original_cameras=args.original_cameras,
        fixed_camera=args.fixed_camera,
        fixed_slot_index=args.fixed_slot_index,
        include_original_camera_pairs=not args.no_original_camera_pairs,
        include_real_novel_pairs=args.use_novel_views and not args.no_real_novel_pairs,
        max_combinations_per_episode=args.max_combinations_per_episode,
        combination_seed=args.combination_seed,
    )
    dataloader = make_smolvla_folder_view_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    batch = next(iter(dataloader))
    print(f"virtual episodes: {dataset.num_episodes}")
    print(f"virtual frames: {dataset.num_frames}")
    print(f"max camera combinations per source episode: {len(dataset.combinations)}")
    print(f"batch keys: {sorted(batch)}")
    for key in dataset.meta.camera_keys:
        print(f"{key}: shape={tuple(batch[key].shape)} dtype={batch[key].dtype}")
    print(f"{ACTION}: shape={tuple(batch[ACTION].shape)} dtype={batch[ACTION].dtype}")


if __name__ == "__main__":
    main()
