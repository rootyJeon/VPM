# VPM data format

VPM trains on **slots**: one directory per data source under a shared root, `VPM_DATA_ROOT` (default `data/`,
relative to the repository root). Every slot uses the same on-disk layout. The loaders are
`cosmos_framework/data/vfm/local_datasets/multiview_track_dataset.py` (video, cameras and 3D point tracks) and
`droid_action_dataset.py` (the same, plus robot actions). This page describes exactly what those loaders read.

- [Slots and stages](#slots-and-stages)
- [Downloading the prepared data](#downloading-the-prepared-data)
- [Directory layout](#directory-layout)
- [Episode index (`train/video_dataset_file.jsonl`)](#episode-index-trainvideo_dataset_filejsonl)
- [Videos](#videos)
- [Cameras (`camera/<scene>_cameras.json`)](#cameras-camerascene_camerasjson)
- [3D point tracks (`tracks/`)](#3d-point-tracks-tracks)
- [DROID robot actions (`actions/<scene>.npz`)](#droid-robot-actions-actionsscenenpz)
- [Preparing your own data](#preparing-your-own-data)

## Slots and stages

| slot | stages | views per sample | resolution bucket | camera over generated frames | min. window | default ratio |
|---|---|---|---|---|---|---|
| `psl4d` | 1, 2 | up to 4 | `480sq` | full trajectory | 33 frames | 3 |
| `droid` | 1, 2, 3 | up to 2 | `480` | held at the last clean frame | 33 frames | 3 (stage 3: the only slot) |
| `hrdexdb` | 1, 2 | 1 (a random view of the scene each sample) | `480` | held at the last clean frame | 33 frames | 1 |
| `molmo` | 1, 2 | 1 | `480sq` | full trajectory | 21 frames | 1 |

The slots hold, respectively: synthetic multi-view clips with four moving cameras (`psl4d`); DROID robot
episodes seen by the two exterior cameras (`droid`); multi-view hand-object manipulation recordings (`hrdexdb`);
and single-view clips with 3D point tracks from MolmoMotion-1M (`molmo`). The dataset card of
`vpm-project/VPM-Data` lists the sources and their licenses.

- **Views per sample.** The anchor view (the video the jsonl row points to) plus up to `num_views - 1` sibling
  views of the same scene, drawn at random among the views that have a video and both track files. With one view
  and a random anchor (`hrdexdb`), the single view is drawn among all of the scene's views.
- **Resolution bucket.** Each video is resized to cover the bucket size of its aspect ratio and center-cropped.
  The aspect ratio comes from the jsonl `width / height`: below 0.65 is 9:16, below 0.88 is 3:4, below 1.16 is 1:1,
  below 1.55 is 4:3, otherwise 16:9. Bucket `480`: 1:1 = 640x640, 4:3 = 736x544, 3:4 = 544x736,
  16:9 = 832x480, 9:16 = 480x832. Bucket `480sq`: the same, except 1:1 = 480x480.
- **Camera over generated frames.** All stages condition every view on its camera (Plucker rays plus camera
  rotary embeddings). With *held*, the generated frames get the camera pose of the last clean frame (static
  cameras). With *full*, they get the true per-frame trajectory (camera-controlled generation).
- **Min. window.** `t2w_windows` shorter than this are dropped when the index is loaded.
- **Default ratio.** `RATIOS` splits the GPUs between slots: every GPU loads exactly one slot, and a slot gets a
  share of the GPUs proportional to its ratio (at least one). A slot set to `0` is not loaded at all, and its
  data need not exist. Override it with, for example, `RATIOS="droid=1 psl4d=1"`.

**Clip and past/future split.** In stages 1 and 2 a sample is one whole window
(`start_frame..end_frame`, step `temporal_interval`), truncated to `4k+1` frames. The clean (conditioning)
past is a random 15-30% of its latent frames, and the rest is generated. In stage 3 the window is rebuilt
from the action labels (see [DROID robot actions](#droid-robot-actions-actionsscenenpz)).

## Downloading the prepared data

The prepared data is published as the Hugging Face dataset `vpm-project/VPM-Data`. The repository mirrors the
layout below, one top-level directory per slot:

```bash
# selected slots into $VPM_DATA_ROOT (default data/)
bash scripts/download_assets.sh --no-nano --no-vae --no-tokenizer --data droid
bash scripts/download_assets.sh --no-nano --no-vae --no-tokenizer --data all

# or with the Hugging Face CLI
hf download vpm-project/VPM-Data --repo-type dataset --include "droid/*" --local-dir data
```

Use `--local-dir` (real files). Do not point the loader at per-file symlinks, such as the blobs of a Hugging Face
cache. The loader resolves symlinks in the anchor video path and takes `<video dir>/..` as the slot root. A
symlink to a whole slot directory (`data/droid -> /fast/disk/droid`) is fine.

## Directory layout

```
<VPM_DATA_ROOT>/
  <slot>/
    train/video_dataset_file.jsonl      episode index: one row per (scene, anchor view)
    videos/<scene>__<view>.mp4          one video per view
    camera/<scene>_cameras.json         intrinsics + world-to-camera extrinsics of every view
    tracks/<scene>__<view>_3d.npz       3D point tracks in that view's camera frame
    tracks/<scene>__<view>_2d.npz       per-point visibility in that view
    actions/<scene>.npz                 droid only (stage 3): robot joint states and actions
```

- File names are split at the **last** `__`: `<scene>` may contain `__`, but `<view>` must not.
- All files of a scene share one frame index `t`. Row `t` of the tracks, the per-frame extrinsics and the action
  arrays describes video frame `t` of every view.
- `JSONL_<SLOT>` (for example `JSONL_DROID=a.jsonl,b.jsonl`) replaces a slot's index with one or more other
  jsonl files. The videos they reference must still sit in a `videos/` directory next to `camera/` and `tracks/`.

## Episode index (`train/video_dataset_file.jsonl`)

One JSON object per line:

| key | type | meaning |
|---|---|---|
| `uuid` | str | unique id of the row |
| `vision_path` | str | the anchor video, `.../videos/<scene>__<view>.mp4`. Absolute, or relative to the directory of the jsonl file (e.g. `../videos/<scene>__view1.mp4`) |
| `width`, `height` | int | pixel size of the videos. It is used for every view of the scene and is the size the intrinsics refer to |
| `duration` | float | seconds; rows longer than 61 s are skipped |
| `t2w_windows` | list | training windows (below); one is drawn at random per sample |
| `nb_frames`, `framerate` | optional | carried along, not required |

Each window is `{"start_frame": int, "end_frame": int, "temporal_interval": int, "caption": str}`, with
`end_frame` inclusive and frames counted from 0. Each window needs a caption: the loader reads `caption`, or else
`qwen3_32b_rewrite-dense`, or one of the other caption keys listed in `CAPTION_TYPES_AND_WEIGHTS` in
`cosmos_framework/data/vfm/local_datasets/sft_dataset.py`. A sample drawn from a window without a caption is
skipped. During training 10% of the captions are dropped for classifier-free guidance.

```json
{"uuid": "ep_000123", "vision_path": "../videos/ep_000123__view1.mp4", "width": 1280, "height": 720, "duration": 10.4,
 "t2w_windows": [{"start_frame": 0, "end_frame": 156, "temporal_interval": 1, "caption": "Put the marker in the cup"}]}
```

## Videos

- MP4 files that `ffmpeg` can decode (`ffmpeg` and `ffprobe` must be on `PATH`).
- All views of a scene have the same resolution (`width` x `height` of the row), the same frame count and
  synchronized frames.
- **Token budget.** A sample costs about `V * ((T - 1) / 4 + 1) * (H / 32) * (W / 32)` vision tokens, where
  `T` is the clip length in frames and `H x W` is the bucket size: 390 tokens per latent frame at 832x480 and 225
  at 480x480. Text and action tokens come on top. The packer drops a sample larger than `MAX_TOKENS`, with a
  warning. Keep windows short enough for the token budget of the stage: 50k in stages 1 and 3, 55k in stage 2.
  For example, a 2-view 832x480 clip of 157 frames is about 31k vision tokens.

## Cameras (`camera/<scene>_cameras.json`)

```json
{
  "view1": {
    "measured_intrinsics": [[535.0, 0.0, 636.5], [0.0, 535.0, 352.2], [0.0, 0.0, 1.0]],
    "optimized_extrinsics": [[r11, r12, r13, tx], [r21, r22, r23, ty], [r31, r32, r33, tz], [0, 0, 0, 1]]
  },
  "view2": { "...": "..." }
}
```

- Top-level keys are the view names (`<view>` in the file names). An optional `optimization_summary` entry is
  ignored, as are extra keys inside a view.
- `measured_intrinsics`: 3x3 pinhole matrix `K` in pixels, for the video at the row's `width` x `height`
  (before any resize or crop). No distortion model is used.
- `optimized_extrinsics`: **world-to-camera**, `x_cam = E @ x_world`. Give a 4x4 matrix for a static camera, or
  `[T, 4, 4]` (one matrix per video frame) for a moving camera. Translations are in **metres**. Camera axes
  follow OpenCV (x right, y down, z forward), and all views of a scene share one world frame. If
  `optimized_extrinsics` is absent, `vggt_extrinsics` is read instead.

## 3D point tracks (`tracks/`)

Every view that can be sampled needs both files, even if it has no points of its own (`N = 0`):

| file | key | shape / dtype | meaning |
|---|---|---|---|
| `<scene>__<view>_3d.npz` | `points_3d` | `[T, N, 3]` float | point positions in **this view's camera frame**, metres |
| | `valid_3d` | `[T, N]` bool | the 3D position is valid at frame `t` |
| `<scene>__<view>_2d.npz` | `visibility` | `[T, N]` bool | the point is visible in this view at frame `t` |

- Row `t` is video frame `t`. If the arrays have fewer rows than a window needs, their last row is repeated.
- `N` may differ between views. Other keys in the files are ignored.
- **How supervision is built.** A point counts as visible in its own view where `valid_3d & visibility`. Every
  view's points are lifted to world coordinates with that view's extrinsics and projected into all views of the
  sample, so each point gets a 2D position and a depth in every view. Visibility in another view is estimated in
  three tiers:
  1. A mutual nearest neighbour among that view's own points, closer than 5 cm in world space, passes on its
     visibility.
  2. Otherwise, a point of that view lying within 25 px and at least 10 cm closer to the camera marks it occluded.
  3. Otherwise, the point is visible when it projects inside the image with positive depth, and the label is
     flagged as untrusted.

  If you store the same physical points in every view, tier 1 gives exact cross-view labels. Each sample then
  draws 128 points among those valid somewhere in its window.

## DROID robot actions (`actions/<scene>.npz`)

Used by stage 3 (and by `inference/inference_droid_old.py` for episode inputs):

| key | shape / dtype | meaning |
|---|---|---|
| `joint_states` | `[n, 8]` float32 | robot state at video frame `t`: 7 joint positions (rad), then the gripper position |
| `joint_actions` | `[n, 8]` float32 | the joint-position action commanded at frame `t`: 7 joint targets (rad), then the gripper target |

- The gripper is in `[0, 1]`, with **0 = open and 1 = closed** (the DROID convention). By default
  (`gripper_flip=True`) the model sees `1 - g`. `inference/inference_droid_old.py` converts its outputs back to the
  DROID convention.
- Row `t` is video frame `t` (DROID videos run at 15 fps). Other keys in the file are ignored.
- **Sample construction (stage 3).** The number of clean past latents `p` is drawn from
  `[1]*16 + [2, 3, 4, 5]`: 80% image-to-video (one clean frame), 20% video-to-video. Then:
  - The clean past covers `p_px = 4 (p - 1) + 1` frames, and the window is `p_px + 32` frames, starting uniformly
    in `[0, n - window]`.
  - Action row 0 is `joint_states[t0]`, where `t0` is the last clean frame. Rows 1..32 are
    `joint_actions[t0 .. t0 + 31]`, one per generated frame.
  - The stream uses the `oxe/droid` action domain (8-D, zero-padded to 64). It replaces the per-view camera
    action of stages 1 and 2. The camera rays still condition every view.
- An episode needs `n >= 33` labelled frames, and its videos must be at least as long as the labels. The jsonl
  row still needs a valid `t2w_windows` entry (at least 33 frames, with the task caption): its caption is the
  instruction, and the window bounds are replaced.
- In the prepared DROID data the two exterior cameras are `view1` and `view2`, and the jsonl rows use `view1`
  as the anchor. The first view is the one the action stream is attached to.

## Preparing your own data

The repository ships the loaders, not converters. To train on new data, write the layout above:

1. Cut the videos per view into `videos/<scene>__<view>.mp4`, synchronized and at a common resolution.
2. Write the cameras with metric world-to-camera extrinsics and pixel intrinsics at that resolution.
3. Write the per-view 3D tracks (camera frame, metres) and visibility, indexed by video frame.
4. Write the jsonl index with captioned windows of at least the slot's minimum length, short enough for the token
   budget.
5. For robot data in stage 3, add `actions/<scene>.npz` in the DROID 8-D joint-position convention.

A new source can reuse a slot whose sampling settings fit (number of views, bucket, held or full camera).
Point the slot at your data with `VPM_DATA_ROOT` or `JSONL_<SLOT>`, and set `RATIOS`. To add a new slot, add it
to `SLOTS` and `RATIOS` in `cosmos_framework/configs/base/experiment/sft/vpm.py` and to `VPM_SLOTS` in the
stage script.

You are responsible for complying with the license of every dataset you download, convert or train on.
