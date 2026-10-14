# 🚀 Inference (DROID)

Given the last frame of the two exterior DROID cameras, their calibration, the robot state and an instruction,
VPM generates the next **32 frames of both views**, a **32-step action chunk** and, optionally, **3D point tracks**
on the generated clip.

One GPU with about 40 GB of memory is enough. The checkpoint, VAE and tokenizer are fetched from Hugging Face on
first use, or prepared with `bash scripts/download_assets.sh --no-nano --release`.

## Quick start

**Your own observation**: one image per view, cameras, state and instruction.

```bash
bash inference/infer_droid.sh \
    --images view1.png,view2.png --cameras cameras.json \
    --state q0,q1,q2,q3,q4,q5,q6,gripper --prompt "put the marker in the cup" --name demo
```

**An episode of VPM-Data**: the clean past ends at frame `--frame`.

```bash
bash scripts/download_assets.sh --no-nano --no-vae --no-tokenizer --data droid
bash inference/infer_droid.sh --episode <scene> --frame 60 --tracks grid
```

`cameras.json` maps each view to `{"intrinsics": 3x3 K in pixels, "extrinsics": 4x4 world-to-camera in metres}`.
List the views in training order (`view1`, `view2`); the state is 7 joint angles (rad) and the gripper
(0 open … 1 closed).

## Options

| option | default | meaning |
|:--|:--|:--|
| `--ckpt` | `vpm-project/VPM-DROID` | Hugging Face id, release dir or training `iter_*` dir (env `VPM_CHECKPOINT`) |
| `--past-latents` | `1` | clean past: `1` = last frame, `p` = last `4(p-1)+1` frames (`--past-frames <npz>` for input B) |
| `--num-steps` · `--guidance` · `--shift` · `--seed` | `35` · `6.0` · `5.0` · `0` | sampler |
| `--tracks` | `none` | `grid` (`--grid-size`² queries per view) or `dataset` (episode track points) |
| `--prompt` | episode caption | `''` for unconditional generation |
| `--out-dir` | `outputs/droid_inference` | output root |

`NPROC_PER_NODE=N` shards the model over N GPUs; several episodes (`--episode a,b,c`) then run in parallel.
`python inference/inference_droid.py --help` lists every option.

## Outputs

`<out-dir>/<name>/` (`<scene>_t<frame>` for episodes):

| file | content |
|:--|:--|
| `video/<view>.mp4` | clean past + 32 generated frames (15 fps) |
| `actions.npz` | `actions` [32, 8] joint positions + gripper (DROID convention), `state`, `dt` |
| `tracks.npz` | with `--tracks`: 2D tracks, `conf`, `depth` (m), `xyz_cam`, `xyz_world` |
| `video/<view>_tracks.mp4` | with `--tracks`: tracks drawn on the generated video |
| `meta.json` | inputs and sampling settings |

Use `conf` as the per-point reliability of the tracks.

## VPM-DROID-v2

[VPM-DROID-v2](https://huggingface.co/rooty2020/VPM-DROID-v2) sees the three DROID cameras in a fixed order
(`exterior_1_left`, `exterior_2_left`, `wrist_left`) at 192×320 and needs no camera calibration. From the last 9 frames
of every camera with the 8 actions between them (or only the current frame), the robot state and the instruction, it
generates the next **16 frames of every camera** and a **16-step action chunk**. Use `inference/infer_droid_v2.sh`; the checkpoint
is fetched from Hugging Face on first use. It has no camera-conditioning weights, so `infer_droid.sh` does not load it.

**Your own observation**: the last 9 frames of each camera, the 8 actions between them, the state and the
instruction. `past.npz` holds `frames` uint8 `[3, 9, H, W, 3]` ending at the current frame; `past_actions.npy` the
8 DROID joint actions of the past frames, oldest first.

```bash
bash inference/infer_droid_v2.sh \
    --past-frames past.npz --past-actions past_actions.npy \
    --state q0,q1,q2,q3,q4,q5,q6,gripper --prompt "put the marker in the cup" --name demo
```

`--images ext1.png,ext2.png,wrist.png` instead uses the current frame only (`--past-latents 1`). The video is fine
either way, but the actions are much more accurate with the 9-frame context.

**An episode of [DROID-Raw](https://huggingface.co/datasets/rooty2020/DROID-Raw)** (`--episode` is a row of
`oxe_policy_cache/droid/episodes.npz`, `--frame` the current frame):

```bash
bash inference/infer_droid_v2.sh --data-root data/DROID-Raw --list 5           # usable episode rows
bash inference/infer_droid_v2.sh --data-root data/DROID-Raw --episode 42 --frame 40
```

The sample is built exactly as in training (without augmentation). The ground-truth frames and actions after the
current frame are saved for comparison, and `meta.json` reports the action error.

The state is 7 joint angles (rad) and the gripper (0 open … 1 closed), as recorded by DROID; the model sees the
gripper as `1 - g` internally. Frames of any 16:9 size work; DROID-Raw stores them at 320×180.

**Point tracks (experimental)**: `--tracks grid` also tracks a `--grid-size`² grid of points per camera on the
generated clip with the model's point-track head, as for VPM-DROID. Add `--cameras cameras.json` (per view
`{"intrinsics": 3x3 K in pixels at the input size, "extrinsics": 4x4 world-to-camera in metres}`, keys
`exterior_1_left`, `exterior_2_left`, `wrist_left`) to condition the head on the calibration and lift the tracks to 3D.

```bash
bash inference/infer_droid_v2.sh --data-root data/DROID-Raw --episode 42 --frame 40 --tracks grid
```

| option | default | meaning |
|:--|:--|:--|
| `--ckpt` | `rooty2020/VPM-DROID-v2` | Hugging Face id, release dir or training `iter_*` dir (env `VPM_CHECKPOINT`) |
| `--past-latents` | `3` (`1` with `--images`) | clean past: `1` = current frame, `2` / `3` = last 5 / 9 frames and the actions between them |
| `--num-steps` · `--guidance` · `--shift` · `--seed` | `35` · `3.0` · `5.0` · `0` | sampler |
| `--tracks` | `none` | `grid` (experimental): point tracks on the generated clip, `--grid-size`² queries per camera |
| `--cameras` | none | calibration json; with `--tracks`, conditions the head and adds 3D tracks |
| `--prompt` | episode instruction | `''` for unconditional generation |
| `--out-dir` | `outputs/droid_v2_inference` | output root |

Outputs in `<out-dir>/<name>/` (`row<episode>_t<frame>` for episodes):

| file | content |
|:--|:--|
| `video/<view>.mp4` | clean past + 16 generated frames (15 fps) |
| `video/<view>_gt.mp4` | episodes only: the same frames from DROID-Raw |
| `actions.npz` | `actions` [16, 8] DROID joint actions of the current frame and the next 15 (DROID convention), `state`, `dt`, episodes: `gt_actions` |
| `tracks.npz` | with `--tracks`: 2D tracks (320×192 and input pixels), `vis`, `conf`, `depth` (m); with `--cameras` also `xyz_cam`, `xyz_world` |
| `video/<view>_tracks.mp4` | with `--tracks`: tracks drawn on the generated video |
| `meta.json` | inputs, sampling settings and, for episodes, `joint_mae_rad` / `gripper_mae` |
