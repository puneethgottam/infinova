#!/usr/bin/env python3
"""Render Gaussian Splat PLY files from custom camera poses.

Pose input can be:
  - .npy with shape (N, 4, 4), (N, 3, 4), (4, 4), or (3, 4)
  - .json containing either a list of matrices, {"poses": [...]}, or
    NeRF-style {"frames": [{"transform_matrix": ...}, ...]}

By default poses are interpreted as world-to-camera matrices. Use
--pose_type c2w when your input matrices are camera-to-world transforms.

You can also render from explicit camera XYZ positions by passing
--lookat_cameras_json and --camera_positions. In that mode, the script estimates
one shared look-at point from the camera rays in cameras.json, builds camera
poses at the requested XYZ positions, and renders each pose looking at that
shared point.

Single PLY mode writes:
  out_dir/00000.png

Folder mode writes one folder per pose:
  out_dir/view_00000/<ply_name>.png
  out_dir/view_00001/<ply_name>.png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torchvision

INSTANTSPLAT_DIR = Path(__file__).resolve().parent / "InstantSplatPP"
if str(INSTANTSPLAT_DIR) not in sys.path:
    sys.path.insert(0, str(INSTANTSPLAT_DIR))

from gaussian_renderer import GaussianModel, render
from utils.graphics_utils import focal2fov, getProjectionMatrix
from utils.pose_utils import get_tensor_from_camera


class RenderCamera:
    def __init__(
        self,
        width: int,
        height: int,
        fovx: float,
        fovy: float,
        znear: float,
        zfar: float,
    ) -> None:
        self.image_width = width
        self.image_height = height
        self.FoVx = fovx
        self.FoVy = fovy
        self.znear = znear
        self.zfar = zfar

        world_view = torch.eye(4, dtype=torch.float32, device="cuda")
        projection = getProjectionMatrix(
            znear=znear,
            zfar=zfar,
            fovX=fovx,
            fovY=fovy,
        ).transpose(0, 1).cuda()

        self.world_view_transform = world_view
        self.projection_matrix = projection
        self.full_proj_transform = (
            world_view.unsqueeze(0).bmm(projection.unsqueeze(0))
        ).squeeze(0)
        self.camera_center = world_view.inverse()[3, :3]


def load_poses(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        poses = np.load(path)
    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "frames" in payload:
            poses = [frame["transform_matrix"] for frame in payload["frames"]]
        elif isinstance(payload, dict) and "poses" in payload:
            poses = payload["poses"]
        else:
            poses = payload
        poses = np.asarray(poses, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported pose file extension: {path.suffix}")

    poses = np.asarray(poses, dtype=np.float32)
    if poses.shape == (4, 4) or poses.shape == (3, 4):
        poses = poses[None, ...]
    if poses.ndim != 3 or poses.shape[1:] not in ((4, 4), (3, 4)):
        raise ValueError(
            "Expected poses with shape (N,4,4), (N,3,4), (4,4), or (3,4); "
            f"got {poses.shape}"
        )

    if poses.shape[1:] == (3, 4):
        full = np.tile(np.eye(4, dtype=np.float32), (poses.shape[0], 1, 1))
        full[:, :3, :4] = poses
        poses = full

    return poses


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        raise ValueError(f"Cannot normalize near-zero vector: {vector}")
    return vector / norm


def viewmatrix(lookdir: np.ndarray, up: np.ndarray, position: np.ndarray) -> np.ndarray:
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    return np.stack([vec0, vec1, vec2, position], axis=1)


def load_cameras_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    cameras = payload["cameras"] if isinstance(payload, dict) and "cameras" in payload else payload
    if not isinstance(cameras, list):
        raise ValueError(f"Expected {path} to contain a camera list or a cameras key.")
    return cameras


def camera_json_to_c2w(camera: dict) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(camera["rotation"], dtype=np.float64)
    pose[:3, 3] = np.asarray(camera["position"], dtype=np.float64)
    return pose


def estimate_lookat_point(cameras: list[dict]) -> np.ndarray:
    poses = np.stack([camera_json_to_c2w(camera) for camera in cameras], axis=0)
    origins = poses[:, :3, 3]
    directions = -poses[:, :3, 2]
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

    lhs = np.zeros((3, 3), dtype=np.float64)
    rhs = np.zeros(3, dtype=np.float64)
    eye = np.eye(3, dtype=np.float64)
    for origin, direction in zip(origins, directions):
        projector = eye - np.outer(direction, direction)
        lhs += projector
        rhs += projector @ origin

    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def parse_xyz(value: str) -> np.ndarray:
    parts = [float(part) for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected x,y,z but got {value}")
    return np.asarray(parts, dtype=np.float64)


def load_positions(args: argparse.Namespace) -> np.ndarray:
    if args.camera_positions_file is not None:
        path = args.camera_positions_file
        if path.suffix.lower() == ".npy":
            positions = np.load(path)
        elif path.suffix.lower() == ".txt":
            positions = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        positions.append(parse_xyz(line))
            positions = np.stack(positions, axis=0)
        else:
            with path.open("r", encoding="utf-8") as f:
                positions = json.load(f)
        positions = np.asarray(positions, dtype=np.float64)
    else:
        positions = np.stack([parse_xyz(item) for item in args.camera_positions], axis=0)

    if positions.ndim == 1 and positions.shape[0] == 3:
        positions = positions[None, :]
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"Expected camera positions with shape (N,3), got {positions.shape}")
    return positions


def make_lookat_poses(args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    cameras = load_cameras_json(args.lookat_cameras_json)
    positions = load_positions(args)
    lookat = parse_xyz(args.lookat_point) if args.lookat_point else estimate_lookat_point(cameras)
    up = parse_xyz(args.lookat_up) if args.lookat_up else np.mean(
        [camera_json_to_c2w(camera)[:3, 1] for camera in cameras],
        axis=0,
    )

    c2w_poses = []
    for position in positions:
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :4] = viewmatrix(lookat - position, up, position).astype(np.float32)
        c2w_poses.append(c2w)

    metadata = {
        "lookat": lookat.tolist(),
        "up": up.tolist(),
        "camera_positions": positions.tolist(),
    }
    return np.linalg.inv(np.stack(c2w_poses, axis=0)), metadata


def parse_background(value: str) -> torch.Tensor:
    presets = {
        "black": (0.0, 0.0, 0.0),
        "white": (1.0, 1.0, 1.0),
    }
    if value.lower() in presets:
        rgb = presets[value.lower()]
    else:
        parts = [float(part) for part in value.split(",")]
        if len(parts) != 3:
            raise ValueError("--background must be black, white, or r,g,b")
        rgb = tuple(parts)
    return torch.tensor(rgb, dtype=torch.float32, device="cuda")


def intrinsics_to_fov(args: argparse.Namespace) -> tuple[float, float]:
    if args.fx is not None or args.fy is not None:
        if args.fx is None or args.fy is None:
            raise ValueError("Pass both --fx and --fy, or neither.")
        return focal2fov(args.fx, args.width), focal2fov(args.fy, args.height)

    if args.fovx is not None or args.fovy is not None:
        if args.fovx is None and args.fovy is None:
            raise ValueError("Internal FoV parsing error.")
        fovx = math.radians(args.fovx) if args.fovx is not None else None
        fovy = math.radians(args.fovy) if args.fovy is not None else None
        if fovx is None:
            fx = args.width / (2.0 * math.tan(fovy * 0.5))
            fovx = focal2fov(fx, args.width)
        if fovy is None:
            fy = args.height / (2.0 * math.tan(fovx * 0.5))
            fovy = focal2fov(fy, args.height)
        return fovx, fovy

    fovy = math.radians(args.fovy_default)
    fy = args.height / (2.0 * math.tan(fovy * 0.5))
    fovx = focal2fov(fy, args.width)
    return fovx, fovy


def find_ply_files(root: Path, pattern: str) -> list[Path]:
    ply_files = sorted(path for path in root.rglob(pattern) if path.is_file())
    if not ply_files:
        raise FileNotFoundError(f"No PLY files found under {root} with pattern {pattern}")
    return ply_files


def unique_output_name(ply_path: Path, root: Path | None, seen: set[str]) -> str:
    if root is None:
        base = ply_path.stem
    else:
        rel = ply_path.relative_to(root)
        base = "__".join(rel.with_suffix("").parts)

    name = base
    counter = 1
    while name in seen:
        name = f"{base}_{counter}"
        counter += 1
    seen.add(name)
    return name


def render_one_ply(
    ply_path: Path,
    output_name: str,
    poses: np.ndarray,
    camera: RenderCamera,
    pipeline: SimpleNamespace,
    background: torch.Tensor,
    args: argparse.Namespace,
) -> None:
    gaussians = GaussianModel(args.sh_degree)
    gaussians.load_ply(str(ply_path))

    for idx, w2c in enumerate(poses):
        camera_pose = get_tensor_from_camera(
            torch.tensor(w2c, dtype=torch.float32, device="cuda")
        )
        with torch.no_grad():
            image = render(
                camera,
                gaussians,
                pipeline,
                background,
                camera_pose=camera_pose,
            )["render"]

        if args.ply_root is None:
            output_path = args.out_dir / f"{idx:05d}.png"
        else:
            view_dir = args.out_dir / f"view_{idx:05d}"
            view_dir.mkdir(parents=True, exist_ok=True)
            output_path = view_dir / f"{output_name}.png"
        torchvision.utils.save_image(image, output_path)

    del gaussians
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render PNGs from a Gaussian Splat PLY and custom camera poses."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--ply", type=Path, help="Path to one 3DGS .ply file.")
    input_group.add_argument(
        "--ply_root",
        type=Path,
        help="Root folder containing subfolders with 3DGS .ply files.",
    )
    parser.add_argument(
        "--ply_pattern",
        default="*.ply",
        help="Pattern used with recursive search under --ply_root. Default: *.ply",
    )
    pose_group = parser.add_mutually_exclusive_group(required=True)
    pose_group.add_argument("--poses", type=Path, help="Path to .npy or .json poses.")
    pose_group.add_argument(
        "--camera_positions",
        nargs="+",
        help="Camera XYZ positions, e.g. --camera_positions 0,0,1 0.2,0,1.",
    )
    pose_group.add_argument(
        "--camera_positions_file",
        type=Path,
        help="TXT, JSON, or NPY array of camera XYZ positions. TXT format: one position per line with 3 comma-separated values (x,y,z). JSON/NPY: shape (N,3).",
    )
    parser.add_argument("--out_dir", required=True, type=Path, help="Directory for rendered PNGs.")
    parser.add_argument("--pose_type", choices=("w2c", "c2w"), default="w2c")
    parser.add_argument(
        "--lookat_cameras_json",
        type=Path,
        help="cameras.json used to estimate the shared look-at point for --camera_positions.",
    )
    parser.add_argument(
        "--lookat_point",
        default=None,
        help="Optional explicit look-at point as x,y,z. If omitted, estimate from cameras.json.",
    )
    parser.add_argument(
        "--lookat_up",
        default=None,
        help="Optional up vector as x,y,z. If omitted, use mean up direction from cameras.json.",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--fovx", type=float, default=None, help="Horizontal FoV in degrees.")
    parser.add_argument("--fovy", type=float, default=None, help="Vertical FoV in degrees.")
    parser.add_argument(
        "--fovy_default",
        type=float,
        default=60.0,
        help="Vertical FoV in degrees when neither focal length nor FoV is passed.",
    )
    parser.add_argument("--znear", type=float, default=0.01)
    parser.add_argument("--zfar", type=float, default=100.0)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--background", default="black", help="black, white, or r,g,b.")
    parser.add_argument("--convert_SHs_python", action="store_true")
    parser.add_argument("--compute_cov3D_python", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by this renderer.")

    if args.camera_positions or args.camera_positions_file:
        if args.lookat_cameras_json is None:
            raise ValueError("--lookat_cameras_json is required with camera positions.")
        cameras = load_cameras_json(args.lookat_cameras_json)
        ref_camera = cameras[0]
        if args.width == parser.get_default("width"):
            args.width = int(ref_camera["width"])
        if args.height == parser.get_default("height"):
            args.height = int(ref_camera["height"])
        if args.fx is None and args.fy is None and "fx" in ref_camera and "fy" in ref_camera:
            args.fx = float(ref_camera["fx"])
            args.fy = float(ref_camera["fy"])

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.poses is not None:
        poses = load_poses(args.poses)
        if args.pose_type == "c2w":
            poses = np.linalg.inv(poses)
        metadata = None
    else:
        poses, metadata = make_lookat_poses(args)
        with (args.out_dir / "lookat_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"Estimated look-at point: {metadata['lookat']}")

    fovx, fovy = intrinsics_to_fov(args)
    camera = RenderCamera(
        width=args.width,
        height=args.height,
        fovx=fovx,
        fovy=fovy,
        znear=args.znear,
        zfar=args.zfar,
    )
    pipeline = SimpleNamespace(
        convert_SHs_python=args.convert_SHs_python,
        compute_cov3D_python=args.compute_cov3D_python,
        debug=args.debug,
    )
    background = parse_background(args.background)

    if args.ply is not None:
        ply_jobs = [(args.ply, args.ply.stem)]
    else:
        ply_files = find_ply_files(args.ply_root, args.ply_pattern)
        seen_names: set[str] = set()
        ply_jobs = [
            (ply_path, unique_output_name(ply_path, args.ply_root, seen_names))
            for ply_path in ply_files
        ]

    for ply_idx, (ply_path, output_name) in enumerate(ply_jobs, start=1):
        print(f"[{ply_idx}/{len(ply_jobs)}] Rendering {ply_path}")
        render_one_ply(
            ply_path=ply_path,
            output_name=output_name,
            poses=poses,
            camera=camera,
            pipeline=pipeline,
            background=background,
            args=args,
        )

    print(f"Rendered {len(poses)} views for {len(ply_jobs)} PLY files to {args.out_dir}")


if __name__ == "__main__":
    main()
