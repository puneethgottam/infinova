# Infinova

Infinova augments robot demonstration data with novel-view renders synthesized
from sparse-view Gaussian Splatting, then packages the result into a
[LeRobot](https://github.com/huggingface/lerobot)-format dataset for VLA
policy training. It uses [InstantSplatPP](https://github.com/phai-lab/InstantSplatPP)
(included as a git submodule) to reconstruct a Gaussian Splat per frame from a
handful of fixed robot-mounted cameras, then renders additional synthetic
camera views for data augmentation.

## Pipeline

1. **`run_instantsplat_episode.sh`** — splats every frame in an episode, saving a
   Gaussian PLY and the `cameras.json` estimated during splatting for each frame.
2. **`select_best_cameras_json.py`** — the 5 cameras are physically static during
   an episode, but each frame's `cameras.json` is an independent pose estimate.
   This picks the frame whose estimate is most consistent with the others and
   saves it as `cameras.json` in the episode's output folder.
3. **`render_novel_views.sh`** — samples new camera positions between pairs of the
   real cameras (using the selected `cameras.json`) and renders every frame's
   Gaussian PLY from those novel positions.
4. **`publish_folder_view_dataset.py`** — combines the rendered novel views with
   a base [LeRobot](https://github.com/huggingface/lerobot) dataset's
   actions/states/tasks (included as a git submodule) into a standalone,
   publishable `LeRobotDataset`.
5. **`train_smolvla_folder_views.py`** — trains a SmolVLA policy directly on the
   folder-backed camera views (no publish step required).

## Setup

Infinova wraps two independent codebases with incompatible dependencies
(InstantSplatPP requires Python 3.10 + CUDA-compiled extensions; LeRobot
requires Python >= 3.12), so set up **two separate environments**.

### 1) Clone with both submodules

```bash
git clone --recursive https://github.com/puneethgottam/infinova.git
cd infinova
```

If you already cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

### 2) InstantSplatPP environment (splatting + rendering, steps 1-3)

Follow the installation steps in [`InstantSplatPP/README.md`](InstantSplatPP/README.md)
(conda environment, CUDA submodules, pre-trained MASt3R checkpoint, etc.) from
inside the `InstantSplatPP/` folder. `run_instantsplat_episode.sh` and
`render_novel_views.sh` call into this environment via an explicit Python
interpreter path, so once it's set up you can run them from the `infinova`
repo root.

### 3) LeRobot environment (dataset publishing + training, steps 4-5)

```bash
conda create -n infinova-lerobot python=3.12 -y
conda activate infinova-lerobot
pip install -e ./lerobot
pip install -e './lerobot[smolvla]'
```

See [`lerobot/README.md`](lerobot/README.md) for hardware-specific extras and
the full installation documentation.

## Usage

Each script is self-contained and documented with `--help`. All of them accept
either command-line arguments or a `DEFAULT_*` block you can fill in at the top
of the script if you'd rather not repeat paths on every run.

### 1) Splat an episode

```bash
./run_instantsplat_episode.sh INPUT_FOLDER OUTPUT_FOLDER PYTHON_PATH START_FRAME END_FRAME \
    --cameras cam0 cam1 cam2 cam3 cam4
```

- `INPUT_FOLDER` contains one subfolder per camera, each with `frame_######.png`
  images. Images must already be rectified — this script does not rectify.
- Results are written to `OUTPUT_FOLDER/instantsplat_gaussians/`:
  `frame_######_gaussian.ply` and `frame_######_cameras.json` per frame.
- Omit `OUTPUT_FOLDER` (pass `""`) to write results under `INPUT_FOLDER` instead.

```bash
./run_instantsplat_episode.sh --help
```

### 2) Pick the most consistent cameras.json

```bash
python select_best_cameras_json.py OUTPUT_FOLDER/instantsplat_gaussians
```

Ranks every `frame_*_cameras.json` in that folder by how far its per-camera
positions/rotations deviate from the across-frame consensus, and copies the
best one to `OUTPUT_FOLDER/instantsplat_gaussians/cameras.json`.

### 3) Render novel views

```bash
./render_novel_views.sh OUTPUT_FOLDER/instantsplat_gaussians NOVEL_VIEWS_OUT PYTHON_PATH \
    --pairs cam0,cam4 cam4,cam2 cam2,cam3 cam3,cam1 \
    --samples-per-pair 20 \
    --y-jitter-range -0.05 0.40 \
    --z-jitter-range -0.10 0.20
```

Samples novel camera positions between the given camera pairs (reading
positions from `cameras.json` produced in step 2), then renders every
`frame_*_gaussian.ply` in the gaussians folder from those positions into
`NOVEL_VIEWS_OUT`.

```bash
./render_novel_views.sh --help
```

### 4) Publish a LeRobot dataset from the rendered views

Requires the LeRobot environment (see Setup, step 3) and a base LeRobot
dataset (`--repo-id`) that provides the actions/states/tasks for each episode.
`--image-root` should contain one folder per episode with the original camera
subfolders plus the novel views rendered in step 3.

```bash
python publish_folder_view_dataset.py \
    --repo-id your-username/base_dataset \
    --image-root /path/to/episode_folders \
    --output-repo-id your-username/folder_view_dataset \
    --use-novel-views
```

This writes a standalone `LeRobotDataset` (optionally pushed to the Hugging
Face Hub) that can be loaded like any other `repo_id`, with no dependency on
the local image folder at training time.

### 5) Train SmolVLA directly on the folder views

Alternatively, skip publishing and train directly against the image folder:

```bash
python train_smolvla_folder_views.py \
    --repo-id your-username/base_dataset \
    --image-root /path/to/episode_folders \
    --use-novel-views
```

## Other scripts

- **`create_proper_dataset.py`** — dataset preparation helper for turning raw
  episode recordings into the per-camera frame-folder layout expected by
  `run_instantsplat_episode.sh`.
- **`sample_camera_positions_with_yz_jitter.py`** — standalone camera-position
  sampler used internally by `render_novel_views.sh`; can also be run directly
  for custom sampling (see `--help`).
- **`render_ply_custom_pose.py`** — standalone renderer used internally by
  `render_novel_views.sh`; also supports rendering from explicit pose matrices
  (`.npy`/`.json`) instead of sampled positions (see `--help`).
- **`smolvla_folder_view_dataloader.py`** — the dataloader used internally by
  both `publish_folder_view_dataset.py` and `train_smolvla_folder_views.py`;
  pairs a base LeRobot dataset's actions/states with camera images loaded
  from a local folder tree.

## Acknowledgement

This project builds on [InstantSplatPP](https://github.com/phai-lab/InstantSplatPP),
itself an extension of [InstantSplat](https://github.com/NVlabs/InstantSplat),
and on [LeRobot](https://github.com/huggingface/lerobot) for dataset format
and policy training. Please also cite the original InstantSplat paper and
codebase if you use this project in your research — see
[`InstantSplatPP/README.md`](InstantSplatPP/README.md) for the citation.
