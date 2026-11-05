#!/usr/bin/env python3

import argparse
import re
import shutil
from pathlib import Path


FRAME_PATTERN = re.compile(r"^frame_(\d+)_gaussian\.png$")


def count_camera_images(camera_folder: Path) -> int:
    """
    Count image files in the cam0 folder.

    Add or remove extensions here if necessary.
    """
    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

    return sum(
        1
        for path in camera_folder.iterdir()
        if path.is_file() and path.suffix.lower() in valid_extensions
    )


def get_gaussian_frames(view_folder: Path) -> list[tuple[int, Path]]:
    """
    Return Gaussian frames as:
        [(frame_number, image_path), ...]

    Sorted by frame number.
    """
    frames = []

    for path in view_folder.iterdir():
        if not path.is_file():
            continue

        match = FRAME_PATTERN.match(path.name)
        if match:
            frame_number = int(match.group(1))
            frames.append((frame_number, path))

    return sorted(frames, key=lambda item: item[0])


def copy_frame(
    source: Path,
    destination: Path,
    overwrite: bool,
    dry_run: bool,
) -> bool:
    """
    Copy one frame.

    Returns True when a copy is performed or would be performed.
    """
    if destination.exists() and not overwrite:
        return False

    if dry_run:
        print(f"    Would copy: {source.name} -> {destination.name}")
    else:
        shutil.copy2(source, destination)
        print(f"    Copied: {source.name} -> {destination.name}")

    return True


def process_view_folder(
    view_folder: Path,
    target_count: int,
    overwrite: bool,
    dry_run: bool,
) -> None:
    """
    Extend a view folder so that its frame numbering covers:

        frame_000000_gaussian.png
        ...
        frame_{target_count - 1:06d}_gaussian.png

    Missing leading frames use the first available image.
    Missing trailing frames use the last available image.
    """
    frames = get_gaussian_frames(view_folder)

    if not frames:
        print(f"  Warning: no Gaussian frames found in {view_folder}")
        return

    first_frame_number, first_frame_path = frames[0]
    last_frame_number, last_frame_path = frames[-1]

    target_last_frame = target_count - 1

    print(f"  Processing {view_folder.name}")
    print(f"    Existing first frame: {first_frame_number:06d}")
    print(f"    Existing last frame:  {last_frame_number:06d}")
    print(f"    Required range:       000000 to {target_last_frame:06d}")

    if first_frame_number < 0:
        print("    Warning: invalid negative first frame number.")
        return

    if first_frame_number > target_last_frame:
        print(
            "    Warning: the first view frame is beyond the required "
            "cam0 frame range."
        )

    copied_before = 0
    copied_after = 0

    # Duplicate the first image backwards.
    #
    # Example:
    # First existing image = frame_000066_gaussian.png
    #
    # Creates:
    # frame_000000_gaussian.png
    # ...
    # frame_000065_gaussian.png
    for frame_number in range(0, min(first_frame_number, target_count)):
        destination = view_folder / (
            f"frame_{frame_number:06d}_gaussian.png"
        )

        if copy_frame(
            source=first_frame_path,
            destination=destination,
            overwrite=overwrite,
            dry_run=dry_run,
        ):
            copied_before += 1

    # Duplicate the last image forwards.
    #
    # Example:
    # Last existing image = frame_000416_gaussian.png
    # cam0 count = 509
    #
    # Creates:
    # frame_000417_gaussian.png
    # ...
    # frame_000508_gaussian.png
    for frame_number in range(last_frame_number + 1, target_count):
        destination = view_folder / (
            f"frame_{frame_number:06d}_gaussian.png"
        )

        if copy_frame(
            source=last_frame_path,
            destination=destination,
            overwrite=overwrite,
            dry_run=dry_run,
        ):
            copied_after += 1

    print(f"    Leading frames added:  {copied_before}")
    print(f"    Trailing frames added: {copied_after}")


def process_episode(
    episode_folder: Path,
    view_start: int,
    view_end: int,
    overwrite: bool,
    dry_run: bool,
) -> None:
    cam0_folder = episode_folder / "cam0"
    novel_views_folder = episode_folder / "novel_views_new"

    print(f"\n{'=' * 70}")
    print(f"Episode: {episode_folder.name}")
    print(f"{'=' * 70}")

    if not cam0_folder.is_dir():
        print(f"Warning: cam0 folder not found: {cam0_folder}")
        return

    if not novel_views_folder.is_dir():
        print(f"Warning: novel_views folder not found: {novel_views_folder}")
        return

    cam0_count = count_camera_images(cam0_folder)

    if cam0_count == 0:
        print(f"Warning: no images found in {cam0_folder}")
        return

    print(f"cam0 image count: {cam0_count}")
    print(f"Expected final frame: frame_{cam0_count - 1:06d}_gaussian.png")

    for view_number in range(view_start, view_end + 1):
        view_name = f"view_{view_number:05d}"
        view_folder = novel_views_folder / view_name

        if not view_folder.is_dir():
            print(f"  Skipping missing folder: {view_name}")
            continue

        process_view_folder(
            view_folder=view_folder,
            target_count=cam0_count,
            overwrite=overwrite,
            dry_run=dry_run,
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extend novel-view Gaussian image sequences to match the "
            "number of images in each episode's cam0 folder."
        )
    )

    parser.add_argument(
        "root_folder",
        type=Path,
        help="Root folder containing episode_000000, episode_000001, etc.",
    )

    parser.add_argument(
        "episode_start",
        type=int,
        help="First episode number to process, inclusive.",
    )

    parser.add_argument(
        "episode_end",
        type=int,
        help="Last episode number to process, inclusive.",
    )

    parser.add_argument(
        "--view-start",
        type=int,
        default=0,
        help="First view number to process. Default: 0",
    )

    parser.add_argument(
        "--view-end",
        type=int,
        default=11,
        help="Last view number to process. Default: 11",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination frames if they already exist.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the operations without copying any files.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    root_folder = args.root_folder.expanduser().resolve()

    if not root_folder.is_dir():
        raise FileNotFoundError(f"Root folder does not exist: {root_folder}")

    if args.episode_start < 0:
        raise ValueError("episode_start cannot be negative.")

    if args.episode_end < args.episode_start:
        raise ValueError(
            "episode_end must be greater than or equal to episode_start."
        )

    if args.view_start < 0:
        raise ValueError("view_start cannot be negative.")

    if args.view_end < args.view_start:
        raise ValueError("view_end must be greater than or equal to view_start.")

    print(f"Root folder:   {root_folder}")
    print(
        f"Episode range: episode_{args.episode_start:06d} "
        f"to episode_{args.episode_end:06d}"
    )
    print(
        f"View range:    view_{args.view_start:05d} "
        f"to view_{args.view_end:05d}"
    )
    print(f"Dry run:       {args.dry_run}")
    print(f"Overwrite:     {args.overwrite}")

    for episode_number in range(
        args.episode_start,
        args.episode_end + 1,
    ):
        episode_name = f"episode_{episode_number:06d}"
        episode_folder = root_folder / episode_name

        if not episode_folder.is_dir():
            print(f"\nSkipping missing episode: {episode_name}")
            continue

        process_episode(
            episode_folder=episode_folder,
            view_start=args.view_start,
            view_end=args.view_end,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )

    print("\nProcessing completed.")


if __name__ == "__main__":
    main()