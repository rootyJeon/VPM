# VP-WorldBench data format

VP-WorldBench is a multi-view 4D benchmark of 1,200 synchronized multi-view videos in four subsets. Every sample has
multi-view videos, per-frame camera poses and intrinsics, ground-truth 3D point tracks with per-view visibility, and
language captions; most subsets add dense depth and further labels.

| Subset | Content | Samples | Views | fps | Video | Frames | Ground truth |
|---|---|---|---|---|---|---|---|
| **Simul-Robot** | Unreal Engine humanoid (GR1T2) tabletop pick & place, outdoor city / indoor kitchen | 500 | 3 | 20 | 1280×1280 | 81–680 | exact (renderer): object + robot-link tracks, dense depth |
| **Simul-General** | CARLA pedestrians (332) and vehicles (168) seen by 4 target-following cameras | 500 | 4 | 15 | 1280×1280 | 81–181 | exact (simulator): target surface tracks (skinned), dense depth |
| **Real-Robot** | RB-Y1 two-armed robot with dexterous hands, teleoperated tabletop manipulation, ego + exo cameras | 100 | 2 | 15 | 1280×720 | 139–325 | forward kinematics of measured joints (arm + hand surface points) |
| **Real-General** | One person moving in a lab, 4 views chosen from a 48-camera rig | 100 | 4 | 15 | 1024×768 (undistorted) | 150 | pseudo-GT: multi-view triangulated tracks (mm-level) + body-model tracks, reconstruction depth, SAM2 masks, 2D/3D keypoints |

All samples are test samples.

## 1. Conventions (all subsets)

- **World frame**: right-handed, metres, `+z` up (gravity along `-z`). Simul-* worlds are recentred near the scene so that
  float32 coordinates keep sub-millimetre precision (Simul-Robot: table-top centre; Simul-General: target position at frame 0,
  ground height unchanged). Real-Robot uses the robot base frame, Real-General the rig/floor frame. `sample.json`
  `world.to_source` maps back to the source frame: `X_source = A @ X + b`.
- **Cameras**: OpenCV axes (x right, y down, z forward). Extrinsics are **world-to-camera** 4×4 matrices `w2c`, one per view
  **and frame**, shape `(V, T, 4, 4)`, also for static cameras. Rotations are proper (det = +1).
- **Intrinsics**: `K` (V, 3, 3) in pixels of the stored video; integer pixel coordinates are pixel **centres**:
  `u = fx·x/z + cx`, `v = fy·y/z + cy`. Stored videos have no lens distortion (Real-General is undistorted; the raw
  distortion model is kept). Simul-* vendors deliver `cx = cy = 640` in the Unreal/CARLA pixel-corner convention; the released
  `K` is converted to OpenCV centres (`cx = cy = 639.5`), the delivered matrix is kept as `K_delivered`.
- **Time**: integer frame index `t = 0 … T-1` at the subset's fps, shared by videos, cameras, tracks, depth and labels
  (exception: Real-Robot robot-log labels, see §6.3). Seconds = `t / fps`.
- **Images**: RGB uint8. Videos are H.264 mp4 without colour tags (BT.601 limited range). The toolkit converts decoded YUV
  to RGB with a fixed formula so results do not depend on the FFmpeg version (`vpm_worldbench.io.video`).
- **Depth**: planar camera z in metres after decoding (`value × unit_scale`); invalid pixels decode to 0.
- **Missing values**: NaN in float arrays, plus boolean masks.

## 2. Folder layout

```
VP-WorldBench/
  index.json                         one record per sample (subset, id, views, frames, fps, sizes, attributes, flags, ...)
  <Subset>/<id>/
    sample.json                      metadata (§3)
    videos/view{k}.mp4               k = 0..V-1
    cameras.npz                      §4
    tracks3d.npz                     §5
    captions.json                    §7
    depth/depth.zip                  Simul-*, Real-General (§6)
    labels/...                       subset-specific labels (§6)
    source/...                       original per-sample source files (provenance only)
  <Subset>/_source/                  subset-level source documentation (provenance only)
  _reproduce/Real-General/<id>/      lossless 480x480 frames 16..80 (real_480sq/view{k}.mp4) and frame-16 depth
                                     (depth_cond.npz) of the pre-release evaluation (legacy reproduction only)
```

## 3. sample.json

| key | content |
|---|---|
| `schema` | `vpm-worldbench/sample/1.0` |
| `uid`, `subset`, `id` | `"<Subset>/<id>"`, subset name, source id |
| `source` | source dataset, repository, revision, original id, licence notes |
| `n_views`, `n_frames`, `fps`, `duration_s` | |
| `views[k]` | `name` (`view{k}`), `video`, `width`, `height`, `source_name` (camera name/serial in the source), `static`, `hfov_deg`, subset-specific camera info (role, model, motion kind, ...) |
| `world` | frame description, `units`, `handedness`, `up`, `to_source` (A, b) |
| `files` | relative paths of `cameras`, `tracks3d`, `captions`, `depth` (or null), `labels` (name → path) |
| `tracks` | `n_points`, `groups`, `default_report_group` (a group name or `"all"`), `visibility_rule`, `accuracy`, `extras` (per-subset arrays), and alignment notes |
| `depth` | null, or the archive spec: `file`, `member` pattern, `view_member_names`, `dtype`, `unit_scale`, `invalid_values` (+`invalid_ge`), `width`, `height`, `planar_z`, `kind` (`rendered (synthetic GT)` / `reconstruction (pseudo-GT)`), intrinsics rule |
| `caption` | the global caption |
| `attributes` | filterable attributes (difficulty, category, scene, task, hand, session, ...) and statistics; `n_default_queries_at_ref_frame` = number of GT points usable as queries at frame 16 |
| `flags` | machine-readable quality caveats (§8) |
| `files_sha256` | checksums of every file of the sample |

## 4. cameras.npz

| key | shape | meaning |
|---|---|---|
| `view_names` | (V,) | `view0`, … |
| `source_view_names` | (V,) | camera names / serials in the source data |
| `image_size` | (V, 2) | `[W, H]` of the stored video |
| `K` | (V, 3, 3) | intrinsics at `image_size`, OpenCV pixel centres |
| `w2c` | (V, T, 4, 4) | world-to-camera per frame (float64, metres) |
| `static` | (V,) | camera pose constant over the clip |
| Simul-* | `K_delivered` (V, 3, 3) | intrinsics as delivered (pixel-corner convention) |
| Simul-General | `frame_abs` (T,) | absolute simulator frame numbers |
| Real-General | `K_raw`, `dist` (radtan5), `raw_size`, `K_undistort_raw` | raw distorted 2048×1536 capture |

## 5. tracks3d.npz (ground-truth 3D point tracks)

| key | shape / dtype | meaning |
|---|---|---|
| `xyz` | (T, N, 3) float32 | world position in metres; NaN where not `valid` |
| `valid` | (T, N) bool | position known at frame t |
| `in_frustum` | (T, N, V) bool | valid, in front of the camera and inside the image |
| `visible` | (T, N, V) bool | in_frustum and not occluded (subset-specific test, `sample.json tracks.visibility_rule`) |
| `default_gt` | (T, N) bool | entries scored by default (equals `valid` except Real-General, §6.4) |
| `group`, `group_names` | (N,) int16, (G,) | coarse group (e.g. object/robot, arm/hand) |
| `part`, `part_names` | (N,) int16, (P,) | fine part (object id, robot link, body region, material section) |
| `kind`, `kind_names` | (N,) int8, (8,) | unified taxonomy, identical codes in every subset: `object_moving`, `object_static`, `robot_arm`, `robot_hand`, `pedestrian`, `vehicle`, `human_body`, `dynamic_surface` |
| `side` | (N,) int8 | -1 none, 0 left, 1 right (the agent's own side) |
| extras | per subset | listed in `sample.json tracks.extras` (e.g. `is_primary`, `object_id`, `link_block`, `dup_of`, `err_mm`, `source`, `query_t`, `clothing`) |

Invariants (checked by `vpmwb validate`): `visible ⊆ in_frustum ⊆ valid`, `default_gt ⊆ valid`, `xyz` finite iff valid,
in-frustum points project inside the image.

Projected 2D tracks for any view and image size: `Sample.tracks2d(view, size=(W, H))`.

## 6. Subset details

### 6.1 Simul-Robot
- Points: 256 area-weighted surface samples per replayed object (1–4 objects; `object_moving` = targets, `object_static` =
  placers/receptacles) followed by 512 robot points on 36 GR1T2 arm/hand links. Objects are rigid, links are rigid per link.
  `is_primary` marks the main manipulated object; `default_report_group = "object"`.
- Visibility: the renderer's depth test (3 cm tolerance) and the 10 cm near clip plane.
- Depth: `depth/depth.zip`, 16-bit PNG per view and frame, centimetres, 65504 (≥ 65000) = sky/far.
- Labels: `labels/objects.json` (objects, kinds, meshes, bounds, motion, distractors that were pushed/fell, robot links),
  `labels/visibility_states.json` (per-object visible/occluded/out_of_frustum per frame and view; difficulty evidence).
- Cameras: static, except 27 `hard` clips where two cameras rotate in place (relay schedule in the visibility states).
  Camera motion is given by `cameras.npz static` and `attributes.moving_cameras`, not by a relay entry: 5 `normal`
  clips carry a planned relay schedule that was not applied in the render.

### 6.2 Simul-General
- Points: 256 surface samples of the single target (pedestrian or vehicle), moved by linear blend skinning; part = mesh
  section/material slot; other traffic is not tracked.
- Visibility: in-frustum AND depth test against the rendered depth at the nearest pixel, `|d − z| ≤ clip(0.02 z, 5 cm, 15 cm)`.
- All cameras move (they follow the target); `hard` clips (95) contain deliberate out-of-sight relays
  (`labels/oos_segments.json`).
- Depth: `depth/depth.zip`, 16-bit PNG, centimetres, 65535 = sky.
- Labels: `labels/target.npz` (target state per frame and view: visible/occluded/out_of_frustum/absent, mask pixel counts,
  world bounding box), `labels/oos_segments.json`.

### 6.3 Real-Robot
- Points: 512 arm points + 256 hand points (both arms/hands) from forward kinematics of the measured joint positions; part =
  rigid arm link block or hand side; `dup_of` marks 3 duplicated points.
- **Time alignment**: the videos trail the logged joint states by 2.3 frames (measured on all 200 episode-views); the released
  tracks are resampled onto the video clock (`tracks.time_offset_frames = -2.3`). Labels (`labels/action.parquet` 66-D joint
  targets, `labels/proprio_30hz.parquet` raw 30 Hz state incl. forces and wrist poses, `labels/ego_fk_c2w.npy`) stay on the
  robot-log clock: video frame t ↔ label time t − 2.3 (`attributes.labels_time_offset_frames`).
- Visibility: in-frustum and a point z-buffer of the robot's own points (self-occlusion only; scene objects do not occlude).
- No depth. Static cameras: view0 = exo (external), view1 = ego (head, fixed pitch; manual extrinsic fit, flagged).

### 6.4 Real-General
- Points (union of two pseudo-GT products, `source`): `orhtrack` (`dynamic_surface`: triangulated from up to 46 views with
  cycle-consistency checks; 6-fold camera cross-validation error median 1.6 mm, p90 4.5 mm) and whole-body body-model tracks (`human_body`, 14 body regions, per-frame
  error estimate `err_mm`). `default_gt` = orhtrack valid OR (whole-body valid AND `err_mm ≤ 20`).
- Depth: `depth/depth.zip`, 512×384, millimetres, 0 = hole; rendered per frame from the multi-view surface reconstruction
  (pseudo-GT). Its intrinsics follow the pixel-centre rule from the 1024×768 video grid (`depth.intrinsics_rule`).
- Labels: `labels/human/masks.npz` (SAM2.1 person masks, actor and other people, bit-packed on the 1024×768 grid, with
  per-frame quality flags; pseudo-GT), `labels/human/keypoints2d.npz` (133 COCO-WholeBody keypoints per view),
  `labels/human/keypoints3d.npz` (triangulated 3D keypoints + body-model keypoints), `labels/human/mhr_fit.npz` (MHR body
  model parameters), `labels/motion_events.json` (walk/turn/bend/... events with frame ranges).
- Static cameras, one calibration for all clips.

## 7. captions.json

| key | subsets | content |
|---|---|---|
| `caption` | all | one global description |
| `instruction` | Real-Robot | the teleoperation task instruction (verbatim) |
| `segments` | Real-General | contiguous motion segments: `start_frame` / `end_frame` (inclusive), `start_s` / `end_s` (half-open [start, end) seconds of the source recording), caption and body-part fields; `clipped: true` marks the last segment when it was cut at the clip end (its caption and `end_s` describe the longer source recording) |
| `windows` | Real-General | captions of fixed frame windows (e.g. 16–80) |
| `structured` | all | parsed attributes (objects, hands, task family, target class, actor, ...) |
| `legacy_prompts` | all | the exact per-view prompts used by the pre-release baseline runs |
| `camera_phrases_corrected` | Simul-General | camera-motion sentences computed with the correct heading (legacy prompts had a wrong pan direction) |

## 8. Quality flags

`sample.json flags` lists machine-readable caveats; nothing was silently dropped. Examples:
`caption_mentions_both_hands_single_hand_used`, `caption_claims_unfinished_placement`, `clip_ends_mid_motion`,
`no_target_motion_in_default_window`, `target_not_visible_in_view_at_ref_frame:<view>`, `target_invisible_all_views_some_frames`,
`quality_ok_false`, `few_default_queries_at_ref_frame:<n>` (fewer than 20 usable query points at frame 16),
`calibration_step_exo`, `ego_extrinsic_manual_fit`, `time_offset_residual_gt_0.5:<view>`, `view_without_wholebody_labels:<view>`,
`subject_closer_than_0.8m:<view>`, `wholebody_qa:<level>`, `people_identifiable`. The exact rule of every flag of a
sample is in its `attributes.flag_rules`.

## 9. Reading the data

```python
from vpm_worldbench import Benchmark
bench = Benchmark("/path/to/VP-WorldBench")
s = bench.sample("Real-General/C001")
frames = s.frames(0, [16, 17], size=(640, 480))        # RGB (2, 480, 640, 3)
K, w2c = s.K(0, size=(640, 480)), s.w2c(0, [16, 17])
tr = s.tracks                                          # GT 3D tracks
uv, vis, in_frame = s.tracks2d(0, size=(640, 480))     # projected 2D tracks
depth = s.depth(0, [16], size=(640, 480))              # metres, 0 = invalid
```
