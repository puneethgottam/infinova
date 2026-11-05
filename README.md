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

| Argument | Meaning |
| --- | --- |
| `INPUT_FOLDER` | Episode folder containing one subfolder per camera, each with `frame_######.png` images (e.g. `INPUT_FOLDER/cam0/frame_000010.png`). Images must already be rectified — this script does not rectify. |
| `OUTPUT_FOLDER` | Where results are written, under `OUTPUT_FOLDER/instantsplat_gaussians/`. Pass `""` to write under `INPUT_FOLDER` instead. |
| `PYTHON_PATH` | Path to the python interpreter with InstantSplatPP's dependencies installed (the conda env from Setup step 2). |
| `START_FRAME`, `END_FRAME` | Inclusive frame-number range to process. |
| `--cameras CAM1 CAM2 CAM3 CAM4 CAM5` | Exactly 5 camera subfolder names under `INPUT_FOLDER`, in the order the images should be fed to InstantSplat. |

Output per frame: `frame_######_gaussian.ply` (the trained Gaussian Splat) and
`frame_######_cameras.json` (the camera poses estimated during splatting).

All five positional arguments and `--cameras` can also be filled in as
`DEFAULT_*` variables at the top of the script, letting you run it with no
command-line arguments at all. Env vars (`WORK_ROOT`, `INSTANTSPLAT_DIR`,
`GPU_ID`, `N_VIEWS`, `TRAIN_ITERATIONS`, `RESOLUTION`, `MODEL_TYPE`) control
the InstantSplat run itself — see `--help` for defaults.

```bash
./run_instantsplat_episode.sh --help
```

### 2) Pick the most consistent cameras.json

```bash
python select_best_cameras_json.py OUTPUT_FOLDER/instantsplat_gaussians --out OUTPUT_FOLDER/instantsplat_gaussians/cameras.json
```

| Argument | Meaning |
| --- | --- |
| `gaussians_dir` (positional) | The `instantsplat_gaussians` folder produced by step 1, containing one `cameras.json` per frame. |
| `--pattern` | Glob pattern for per-frame camera files. Default: `frame_*_cameras.json`. |
| `--out` | Where to save a copy of the winning `cameras.json`. Default: `GAUSSIANS_DIR/cameras.json`. |

Ranks every matching file by how far its per-camera positions/rotations
deviate from the across-frame consensus (median position, mean rotation per
camera — the 5 physical cameras don't move during an episode, so deviation
indicates a noisier pose estimate for that frame), and copies the best one to
the `--out` path.

### 3) Render novel views

```bash
./render_novel_views.sh OUTPUT_FOLDER/instantsplat_gaussians NOVEL_VIEWS_OUT PYTHON_PATH \
    --pairs cam0,cam4 cam4,cam2 cam2,cam3 cam3,cam1 \
    --samples-per-pair 20 \
    --y-jitter-range -0.05 0.40 \
    --z-jitter-range -0.10 0.20
```

| Argument | Meaning |
| --- | --- |
| `GAUSSIANS_DIR` | The `instantsplat_gaussians` folder from step 1, containing `cameras.json` (from step 2) and the `frame_*_gaussian.ply` files. |
| `OUT_DIR` | Folder to write rendered novel-view PNGs to. |
| `PYTHON_PATH` | Path to the python interpreter with InstantSplatPP's dependencies installed. |
| `--pairs CAM_A,CAM_B ...` | One or more real-camera pairs to sample novel positions between. Names must match the `img_name` fields in `cameras.json` (e.g. `cam0,cam4`). |
| `--samples-per-pair N` | Number of novel positions to sample along each pair (default: 20). |
| `--y-jitter-range MIN MAX` | Random offset added to the Y coordinate of each sampled position. |
| `--z-jitter-range MIN MAX` | Random offset added to the Z coordinate of each sampled position. |
| `--seed N` | Random seed, for reproducible sampling. |

For each pair, a position is first linearly interpolated between the two real
cameras, then jittered in Y/Z. Every sampled position is pointed at the
episode's shared look-at point (estimated from the real cameras) and rendered
against every `frame_*_gaussian.ply` in `GAUSSIANS_DIR`.

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

| Argument | Meaning |
| --- | --- |
| `--repo-id` | Base LeRobot dataset repo id providing actions/states/tasks per episode. |
| `--image-root` | Folder containing `episode_XXXXXX/cam*`, `episode_XXXXXX/camera3`, and optionally `episode_XXXXXX/novel_views_new/*` images (the output of step 3, laid out per episode). |
| `--root` | Optional local cache/root for `--repo-id`, if it's not already cached locally. |
| `--output-repo-id` | Hugging Face repo id to create and publish the resulting dataset under, e.g. `user/name`. |
| `--output-root` | Local directory to build the dataset in. Defaults to `$HF_LEROBOT_HOME/{output-repo-id}`. |
| `--overwrite` | Remove `--output-root` first if it already exists. |
| `--fixed-camera` | Original camera folder name always kept in one fixed SmolVLA image slot (default: `camera3`). |
| `--fixed-slot-index` | Which of the 3 SmolVLA image slots (0, 1, or 2) the fixed camera occupies (default: `2`, i.e. slot 3). |
| `--original-cameras` | Original camera folder names available per episode. Defaults to `cam0`..`cam4` if omitted. |
| `--use-novel-views` | Include the rendered novel views (from step 3) as additional image-slot combinations, not just the original cameras. |
| `--max-orig-novel-pairs` | Cap on original↔novel camera pairs (both directions) per episode; sampled down if more exist. |
| `--target-pairs-per-episode` | Total combinations per episode: original-original pairs are always included in full, then novel-novel pairs are sampled to fill up to this target. Only applies when `--max-combinations-per-episode` is not set. |
| `--max-combinations-per-episode` | Hard cap on combinations generated per episode (exhaustive mode is used instead when this is unset). |
| `--combination-seed` | Random seed for sampling which camera combinations to include. |
| `--no-original-camera-pairs` | Exclude original-camera-only combinations. |
| `--no-real-novel-pairs` | Exclude combinations pairing a real camera with a novel view. |
| `--episodes` | Restrict to specific source episode indices instead of all of them. |
| `--slot-image-keys` | The 3 LeRobot feature key names used for the image slots (default: `observation.images.slot_1/2/3`). |
| `--use-videos` / `--no-videos` | Encode each camera's frames as an mp4 (default) or store per-frame PNGs instead. |
| `--image-writer-processes`, `--image-writer-threads`, `--batch-encoding-size` | Parallelism/batching for writing the output dataset. |
| `--num-workers` | Parallel workers for reading/decoding source frame images. |
| `--push` / `--no-push` | Push the finished dataset to the Hugging Face Hub (default) or only build it locally. |
| `--private` | Create the Hub repo as private. |
| `--branch` | Optional Hub branch/revision to push to. |
| `--tags` | Optional tags to attach to the Hub dataset card. |
| `--license` | License identifier for the Hub dataset card (default: `apache-2.0`). |

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

| Argument | Meaning |
| --- | --- |
| `--repo-id` | Base LeRobot dataset repo id providing actions/states/tasks per episode. |
| `--image-root` | Folder containing per-episode camera images, same layout as step 4. |
| `--root` | Optional local cache/root for `--repo-id`. |
| `--output-dir` | Where to save checkpoints and logs (default: `outputs/train/smolvla_folder_views`). |
| `--pretrained-path` | SmolVLA checkpoint to start from (default: `lerobot/smolvla_base`); pass `none` to train from scratch. |
| `--device` | Training device, e.g. `cuda`, `cpu`, or `auto` (default) to auto-detect. |
| `--steps` | Number of training steps (default: `10000`). |
| `--batch-size` | Training batch size (default: `8`). |
| `--num-workers` | Dataloader worker processes (default: `4`). |
| `--lr` | Learning rate (default: `1e-4`). |
| `--weight-decay`, `--adam-beta1`, `--adam-beta2`, `--adam-eps` | AdamW optimizer hyperparameters. |
| `--grad-clip-norm` | Gradient clipping norm (default: `10.0`). |
| `--amp` | Enable automatic mixed precision training. |
| `--drop-last` | Drop the last incomplete batch of each epoch. |
| `--log-freq` | Steps between logging training metrics (default: `10`). |
| `--save-freq` | Steps between checkpoint saves (default: `1000`). |
| `--seed` | Random seed for reproducibility (default: `1000`). |
| `--fixed-camera`, `--fixed-slot-index`, `--original-cameras` | Same meaning as in step 4: which camera stays fixed, which SmolVLA slot it occupies, and the available original camera names. |
| `--use-novel-views` | Include rendered novel views as additional image-slot combinations. |
| `--max-combinations-per-episode`, `--combination-seed` | Same meaning as in step 4: cap and seed for sampled camera combinations. |
| `--no-original-camera-pairs`, `--no-real-novel-pairs` | Same meaning as in step 4: exclude those combination types. |
| `--episodes` | Restrict training to specific source episode indices. |

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
