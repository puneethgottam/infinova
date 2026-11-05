#!/usr/bin/env python

"""Train SmolVLA with folder-backed camera-view virtual episodes.

This is a deliberately small PyTorch training script. It avoids the standard
LeRobot training entrypoint while still reusing the SmolVLA policy,
preprocessor/postprocessor, normalization, and checkpoint format.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
from tqdm import tqdm

from lerobot.configs import FeatureType
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAConfig, SmolVLAPolicy
from lerobot.utils.constants import ACTION
from lerobot.utils.feature_utils import dataset_to_policy_features
from lerobot.utils.random_utils import set_seed

from smolvla_folder_view_dataloader import (
    SmolVLAFolderViewDataset,
    make_smolvla_folder_view_dataloader,
)


def build_policy_config(
    dataset: SmolVLAFolderViewDataset,
    device: str,
    pretrained_path: str | None,
    learning_rate: float,
) -> SmolVLAConfig:
    features = dataset_to_policy_features(dataset.meta.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    if pretrained_path is not None:
        cfg = SmolVLAConfig.from_pretrained(pretrained_path)
    else:
        cfg = SmolVLAConfig()

    cfg.device = device
    cfg.input_features = input_features
    cfg.output_features = output_features
    cfg.pretrained_path = Path(pretrained_path) if pretrained_path is not None else None
    cfg.optimizer_lr = learning_rate
    return cfg


def make_policy(cfg: SmolVLAConfig, pretrained_path: str | None) -> SmolVLAPolicy:
    if pretrained_path is None:
        policy = SmolVLAPolicy(cfg)
    else:
        policy = SmolVLAPolicy.from_pretrained(pretrained_path, config=cfg, strict=False)
    policy.train()
    policy.to(cfg.device)
    return policy


def save_training_artifacts(
    output_dir: Path,
    policy: SmolVLAPolicy,
    preprocessor,
    postprocessor,
    optimizer: torch.optim.Optimizer,
    step: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(output_dir)
    preprocessor.save_pretrained(output_dir)
    postprocessor.save_pretrained(output_dir)
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
        },
        output_dir / "training_state.pt",
    )


def train(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    if args.seed is not None:
        set_seed(args.seed)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

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
        episodes=args.episodes,
    )
    dataloader = make_smolvla_folder_view_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=device.startswith("cuda"),
        drop_last=args.drop_last,
    )

    pretrained_path = None if args.pretrained_path.lower() == "none" else args.pretrained_path
    cfg = build_policy_config(dataset, device, pretrained_path, args.lr)
    policy = make_policy(cfg, pretrained_path)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset.meta.stats)

    optimizer = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.startswith("cuda"))

    logging.info("source episodes: %d", dataset.base.num_episodes)
    logging.info("max camera combinations per source episode: %d", len(dataset.combinations))
    logging.info("virtual episodes: %d", dataset.num_episodes)
    logging.info("virtual frames: %d", dataset.num_frames)
    logging.info("image keys: %s", dataset.meta.camera_keys)
    logging.info("device: %s", device)

    step = 0
    policy.train()
    progress = tqdm(total=args.steps, desc="Training", unit="step")

    while step < args.steps:
        for batch in dataloader:
            start_time = time.perf_counter()
            for cam_key in dataset.meta.camera_keys:
                if batch[cam_key].dtype == torch.uint8:
                    batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0

            batch = preprocessor(batch)

            with torch.autocast(device_type="cuda", enabled=args.amp and device.startswith("cuda")):
                loss, output_dict = policy.forward(batch)

            scaler.scale(loss).backward()
            if args.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            progress.update(1)

            if step % args.log_freq == 0:
                data_s = time.perf_counter() - start_time
                extra = ""
                if output_dict:
                    extra = " ".join(
                        f"{key}={value:.4f}" for key, value in output_dict.items() if isinstance(value, float)
                    )
                logging.info("step=%d loss=%.4f data+train_s=%.3f %s", step, loss.item(), data_s, extra)

            if args.save_freq > 0 and step % args.save_freq == 0:
                checkpoint_dir = args.output_dir / f"checkpoint-{step:06d}"
                save_training_artifacts(checkpoint_dir, policy, preprocessor, postprocessor, optimizer, step)
                logging.info("saved checkpoint: %s", checkpoint_dir)

            if step >= args.steps:
                break

    progress.close()
    save_training_artifacts(args.output_dir, policy, preprocessor, postprocessor, optimizer, step)
    logging.info("saved final model: %s", args.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Base LeRobot dataset repo id providing actions/states/tasks, e.g. 'user/name'.")
    parser.add_argument(
        "--image-root",
        required=True,
        help="Folder containing episode_XXXXXX/cam*, episode_XXXXXX/camera3, and optionally novel_views_new/* images.",
    )
    parser.add_argument("--root", default=None, help="Optional local LeRobot dataset root/cache.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/train/smolvla_folder_views"))
    parser.add_argument("--pretrained-path", default="lerobot/smolvla_base", help="Use 'none' for scratch.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-10)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--grad-clip-norm", type=float, default=10.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument("--log-freq", type=int, default=10)
    parser.add_argument("--save-freq", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--fixed-camera", default="camera3")
    parser.add_argument("--fixed-slot-index", type=int, default=2)
    parser.add_argument("--original-cameras", nargs="*", default=None)
    parser.add_argument("--use-novel-views", action="store_true")
    parser.add_argument("--max-combinations-per-episode", type=int, default=None)
    parser.add_argument("--combination-seed", type=int, default=0)
    parser.add_argument("--no-original-camera-pairs", action="store_true")
    parser.add_argument("--no-real-novel-pairs", action="store_true")
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
