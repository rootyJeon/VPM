# 🚀 Inference (DROID)

Given the last frames of the three DROID cameras, the robot state, the actions that led to the current frame and an
instruction, [VPM-DROID](https://huggingface.co/rooty2020/VPM-DROID) generates the next **16 frames of every camera**,
a **16-step action chunk** and, optionally, **point tracks** on the generated clip.

One GPU with about 40 GB of memory is enough. The checkpoint, VAE and tokenizer are fetched from Hugging Face on
first use, or prepared with `bash scripts/download_assets.sh --no-nano --release`.

## Quick start

VPM-DROID sees the three DROID cameras in a fixed order (`exterior_1_left`, `exterior_2_left`, `wrist_left`) at
192×320 and needs no camera calibration.

**Your own observation**: the last 9 frames of each camera, the 8 actions between them, the state and the
instruction. `past.npz` holds `frames` uint8 `[3, 9, H, W, 3]` ending at the current frame; `past_actions.npy` the
8 DROID joint actions of the past frames, oldest first.

```bash
bash inference/infer_droid.sh \
    --past-frames past.npz --past-actions past_actions.npy \
    --state q0,q1,q2,q3,q4,q5,q6,gripper --prompt "put the marker in the cup" --name demo
```

`--images ext1.png,ext2.png,wrist.png` instead uses the current frame only (`--past-latents 1`). The video is fine
either way, but the actions are much more accurate with the 9-frame context.

**An episode of [DROID-VP](https://huggingface.co/datasets/rooty2020/DROID-VP)** (`--episode` is a row of
`tables/droid/episodes.npz`, `--frame` the current frame):

```bash
bash inference/infer_droid.sh --data-root data/DROID-VP --list 5           # usable episode rows
bash inference/infer_droid.sh --data-root data/DROID-VP --episode 42 --frame 40
```

The sample is built exactly as in training (without augmentation). The ground-truth frames and actions after the
current frame are saved for comparison, and `meta.json` reports the action error.

The state is 7 joint angles (rad) and the gripper (0 open … 1 closed), as recorded by DROID; the model sees the
same convention (`"gripper": "droid"` in its `release.json`). Frames of any 16:9 size work; DROID-VP stores them at
320×180.

**Point tracks**: `--tracks grid` also tracks a `--grid-size`² grid of points per camera on the generated clip with
the model's point-track head. The head was trained with the camera calibration as input. Episodes of DROID-VP bring
their own calibration (past frames at their true poses, the generated frames at the current pose, as in training),
and the tracks are lifted to 3D. For your own observation, pass `--cameras cameras.json` (per view
`{"intrinsics": 3x3 K in pixels at the input size, "extrinsics": 4x4 world-to-camera in metres}`, keys
`exterior_1_left`, `exterior_2_left`, `wrist_left`).

```bash
bash inference/infer_droid.sh --data-root data/DROID-VP --episode 42 --frame 40 --tracks grid
```

## Options

| option | default | meaning |
|:--|:--|:--|
| `--ckpt` | `rooty2020/VPM-DROID` | Hugging Face id, release dir or training `iter_*` dir (env `VPM_CHECKPOINT`) |
| `--gripper` | `auto` | gripper convention inside the checkpoint: `droid` or `flipped` (`1 - g`); `auto` reads its `release.json` |
| `--past-latents` | `3` (`1` with `--images`) | clean past: `1` = current frame, `2` / `3` = last 5 / 9 frames and the actions between them |
| `--num-steps` · `--guidance` · `--shift` · `--seed` | `35` · `3.0` · `5.0` · `0` | sampler |
| `--tracks` | `none` | `grid`: point tracks on the generated clip, `--grid-size`² queries per camera |
| `--cameras` | none (episodes: their own) | calibration json; with `--tracks`, conditions the head and adds 3D tracks |
| `--prompt` | episode instruction | `''` for unconditional generation |
| `--out-dir` | `outputs/droid_inference` | output root |

`NPROC_PER_NODE=N` shards the model over N GPUs; several episodes (`--episode a,b,c`) then run in parallel.
`python inference/inference_droid.py --help` lists every option.

## Outputs

`<out-dir>/<name>/` (`row<episode>_t<frame>` for episodes):

| file | content |
|:--|:--|
| `video/<view>.mp4` | clean past + 16 generated frames (15 fps) |
| `video/<view>_gt.mp4` | episodes only: the same frames from DROID-VP |
| `actions.npz` | `actions` [16, 8] DROID joint actions of the current frame and the next 15 (DROID convention), `state`, `dt`, episodes: `gt_actions` |
| `tracks.npz` | with `--tracks`: 2D tracks (320×192 and input pixels), `vis`, `conf`, `depth` (m); with `--cameras` also `xyz_cam`, `xyz_world` |
| `video/<view>_tracks.mp4` | with `--tracks`: tracks drawn on the generated video |
| `meta.json` | inputs, sampling settings and, for episodes, `joint_mae_rad` / `gripper_mae` |
