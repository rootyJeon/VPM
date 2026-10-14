# Tasks beyond video generation

The same clips, cameras and 3D ground truth support seven further tasks. Each task is one module in
`vpm_worldbench/tasks/`; its docstring is the complete specification. This page summarises the tasks.

## Common conventions

- **Commands.** Every task has `export` (model inputs), `baseline` (reference predictions), `evaluate` and
  `summarize`:

  ```bash
  vpmwb <task> export   --root /data/VP-WorldBench --out inputs/<task>
  vpmwb <task> baseline --root /data/VP-WorldBench --pred preds/<task>/<baseline> --kind <baseline>
  vpmwb <task> evaluate --root /data/VP-WorldBench --pred preds/<task>/my_method --out runs/<task>/my_method
  vpmwb <task> summarize --out runs/<task>/my_method
  ```

  | task | baseline kinds (default first) |
  |---|---|
  | track3d | `frozen`, `oracle`, `lifter` |
  | track2d | `frozen`, `oracle` |
  | depth | `oracle`, `constant` |
  | recon4d | `oracle`, `frozen` |
  | nvs | `nearest_view`, `oracle` |
  | segmentation | `prompt_static`, `oracle`, `label_raster`, `empty`, `full` |
  | understanding | `prior`, `oracle`, `random`, `option_prior`, `full` |

- **Shared options.**
  - `--protocol` takes the task's preset name or a JSON/YAML file of its protocol fields. A file without `name` is
    named after the file.
  - `--subsets` and `--ids` accept names or aliases, space- or comma-separated (`--subsets rr rg`,
    `--ids rr/episode_000`). Ids that match no sample are named in a warning.
  - `--jobs` sets the number of worker processes.
  - In Python every task has `export_inputs`, `run_baseline(bench, pred_root, kind=..., protocol=...)`, `evaluate` and
    `summarize`.
- **Window.**
  - Most tasks evaluate the frames `ref_frame + j·frame_step`, j = 0 … horizon. The default is frames 16 … 80, the
    window of the generation task. Position 0 is the query, prompt or anchor frame.
  - `--ref-frame`, `--horizon` and `--frame-step` override the window and add `+field=value` to the protocol name.
  - `--horizon ≤ 0` means "to the end of the clip" for track3d, track2d and segmentation. Depth, recon4d, nvs and
    understanding need a fixed window (`--horizon ≥ 1`). In understanding the window options set the Real-General window
    templates (motion, turn direction, body part); give the same values to export, baseline and evaluate.
  - A sample is skipped (and listed) when the window does not fit. With the defaults every sample is evaluable.
- **No GT in the inputs.** Exports contain only what a method may use: frames, cameras where the task allows them,
  and the queries or prompts at position 0. QA answer keys and correspondence answers are never exported.
- **One run per output folder.** An `--out` folder belongs to one task, protocol and resolved prediction root
  (`<out>/run.json`). Evaluating another prediction root or protocol into it is refused unless `--overwrite` is
  given, which deletes the previous run's records. `summary.json` records the prediction root. Export folders of
  track3d, track2d, segmentation and understanding are tied the same way to their export options (`export.json`).
- **Results.** Per sample, `<out>/<Subset>/<id>.json` with a status:

  | status | meaning |
  |---|---|
  | `ok` | scored |
  | `partial` | track2d only: part of the sample scored |
  | `skipped` | not evaluable under the protocol |
  | `missing` | no prediction file |
  | `invalid` | unreadable or malformed prediction; the reason names the file |
  | `error` | any other failure, with reason and traceback |

  Records of earlier versions with `not_evaluable` / `missing_prediction` are read as `skipped` / `missing`. Failed
  samples are counted per subset (`n_invalid`, `n_error`; understanding counts failures per question or query: QA
  `n_missing` / `n_unparsed`, grounding `n_missing` / `n_invalid`), the run continues, and the next `evaluate` scores them again;
  `--strict` raises instead. `summary.json` and `summary.md` give means ± s.e. over samples per subset. In the JSON
  files NaN is written as `null`.
- **Failures and aggregation.** Non-finite predicted coordinates are failures: never within a threshold, and infinite
  errors. The aggregation of a failed sample's error depends on the task:
  - track3d, track2d and the generation report keep ∞ in a subset mean (the column shows ∞) and count the samples in
    `n_inf`;
  - depth, recon4d, nvs and segmentation average the finite values and count failures separately.
- **Exit status.**
  - 0: the command succeeded.
  - 1: the selection matches no sample, or nothing evaluable was scored (every sample missing / invalid / error;
    understanding: no prediction file under `--pred`). The message is one line `error: <task>: ...`.
  - 2: a usage or setup problem. An output folder (or export folder) of another run used without `--overwrite`, and a
    missing optional package, print one line `vpmwb: error: ...`. Malformed options print the command's usage, then
    one line `vpmwb <task> <cmd>: error: ...`. Malformed options include an unknown subset, protocol or lifter, an
    invalid override, a malformed `--size` / `--dense-size`, a missing `--root`, or a missing `--answer-key` file.
    `VPMWB_TRACEBACK=1` shows the traceback instead.
- **Ground-truth quality.** Simul-* GT is exact. Real-Robot GT is forward kinematics; its visibility models robot
  self-occlusion only, and the calibration is good to about 2 px. Real-General GT is pseudo-GT from multi-view
  reconstruction. Per-subset caveats are in each module docstring and in the `flags` of every sample.

## track3d: multi-view 3D point tracking and 3D motion estimation

Given the window frames and cameras, plus query points at t_q = 16 (world xyz, and their pixels in every view),
predict every query's 3D position in every frame.

| | |
|---|---|
| modes | `world` (all views or a chosen subset; world coordinates). `monocular` (one view; that camera's coordinates, as in TAPVid-3D) |
| inputs | `queries.npz` (`query_index`, `xyz`, per-view `uv` / `visible`), K and w2c per view and frame; optional depth of frame t_q (`--depth query`) |
| prediction | `<pred>/<Subset>/<id>.npz`: `xyz` (n, N, 3) world metres (monocular: `xyz_cam`); optional `vis` (n, N) |
| metrics | δ_avg, δ@d, AJ, OA, survival, ADE, MTE, FDE, split into all / static / dynamic tracks, per point group. Scene flow is computed over about 0.5 s strides on the moving entries (`flow/moving/*`: EPE3D, Acc); the all-entry values are in the JSON only |
| baselines | `oracle`, `frozen` (points stay at the query), `lifter` (MVTracker + MoGe-2 on the real frames; `--device auto`) |
| generation link | `Track3DProtocol.from_protocol(p)` reproduces the generation δ queries and GT visibility for `resize` / `letterbox` grids; centre-cropping protocols give a warning |

The distance ladders are those of δ ([METRICS.md](METRICS.md#delta)). The main point group is the sample's
`default_report_group`.

## track2d: 2D point tracking per view (TAP-Vid) and cross-view correspondence

| | |
|---|---|
| tracking | per view, on a 256×256 grid (configurable). Query modes `first` (default), `strided` and `fixed`. The queries carry opaque `track_id`s, not GT point indices, so tracks of the same point cannot be linked. Prediction `<pred>/<Subset>/<id>/view{k}.npz`: `uv` (n, N, 2), optional `vis`. Every view must be predicted: a missing view makes the sample's tracking result `missing` (status `partial` when only the correspondence part is present) |
| metrics | TAP-Vid AJ, <δ^x_avg>, OA (port of tapnet's `compute_tapvid_metrics`) |
| correspondence | on by default, at frame 8 (before the tracking window, so the tracking queries do not reveal the answers). For a point seen in view a, predict its pixel in every other view b that sees it at the same frame. Each point is queried from one source view only (`corr/queries.npz` with `query_id`). Prediction `corr.npz` `uv_b`. Metrics: % within 1–16 px and mean / median error |
| baselines | `oracle`, `frozen` (zero motion; identity correspondence) |

## depth: video depth

Predict the depth of every window frame of each view.

| | |
|---|---|
| prediction | `<pred>/<Subset>/<id>/view{k}.npz` with `depth` (n, H, W) or `disparity`, at any resolution. Maps whose aspect is the transposed video aspect are scored as given but flagged (`warnings` in the record, a line in `summary.md`) |
| GT | dense: rendered depth (Simul-*) and reconstruction depth (Real-General). Sparse: depth of visible GT track points, for all subsets including Real-Robot |
| metrics | AbsRel, SqRel, RMSE, RMSE-log, δ<1.25^k, coverage, after alignment `none` / `median` / `scale` / `scale_shift` (disparity, per sequence: the video-depth headline) |
| baselines | `oracle`, `constant` (shows what alignment alone achieves) |

## recon4d: 4D reconstruction and camera estimation

From the multi-view window (default frames 16, 20, …, 80), reconstruct world point maps and/or cameras in one
consistent world frame. Any similarity transform of the GT world is allowed.

| | |
|---|---|
| prediction | `<pred>/<Subset>/<id>/recon4d.npz`: `points` (n, V, H, W, 3) and/or `depth`, optional `w2c` and `K`. Transposed maps are flagged as in `depth` |
| settings | calibrated (K given, default), uncalibrated, known poses |
| metrics | cameras: RRA/RTA/AUC, RPE, ATE after Sim(3). Points: accuracy, completeness, Chamfer, F-score@τ. Multi-view depth consistency. Track error of the reconstructed points at the GT track pixels (δ ladder, dynamic / static) |
| baselines | `oracle`, `frozen` |

## nvs: held-out view synthesis

Render a held-out camera of the rig (default: the last view) for every window frame from the other views and the
target camera.

| | |
|---|---|
| prediction | `<pred>/<Subset>/<id>/view{target}.mp4` on the window grid (or with `meta.json` fps, checked as in generation; `first_frame` must be `ref`). Use the same `--target-view` for export, baseline and evaluate |
| metrics | PSNR, SSIM, LPIPS on the eval grid. Real-General's undistortion border is masked. SSIM / LPIPS need the `[metrics]` extra; `--no-ssim` / `--no-lpips` turn them off, and a missing package stops `evaluate` before the first sample |
| baselines | `nearest_view` (the source camera closest to the target), `oracle` |

## segmentation: promptable video object segmentation

Given point clicks, a box, the mask (Real-General) and a text name of one target at frame 16, segment it in every
window frame of each view.

| | |
|---|---|
| targets | Simul-Robot: the primary object (or the robot). Simul-General: the followed pedestrian or vehicle. Real-Robot: the active arm and hand. Real-General: the person |
| prediction | `<pred>/<Subset>/<id>/view{k}.npz` `masks` (n, H, W) bool (or bit-packed). Any resolution with the video's aspect; a wrong aspect, duplicate frame indices or an unreadable file make the sample `invalid`. Every prompted view must be predicted |
| metrics | dense J, F, J&F against SAM2.1 actor masks (Real-General, pseudo-GT). Sparse point-label precision / recall / IoU / F1 from the projected GT tracks and depth (all subsets) |
| baselines | `oracle`, `label_raster`, `prompt_static` (the prompt held fixed), `empty`, `full` |

## understanding: video understanding

| sub-task | content | prediction | metric |
|---|---|---|---|
| `qa` | multiple-choice questions generated only from GT geometry and metadata, never from a model: camera motion, which hand, which object moves first, object or vehicle type, how far the target travels, which body part moves most, motion order, … The answer positions are balanced. | `<pred>/qa.jsonl` `{"qid", "response"}` or `{"qid", "choice"}` | accuracy per template and subset |
| `retrieval` | text ↔ video retrieval with the clip captions. Segment-level retrieval on Real-General is opt-in (`--tasks retrieval_segment`, its own export folder), because its inputs would reveal the grounding answers | `<pred>/retrieval/<Subset>.npy` similarity matrix | R@1/5/10, median rank, both directions |
| `grounding` | Real-General: find the frame range of a segment caption | `<pred>/grounding.jsonl` with `start_frame` / `end_frame` (inclusive), or `start_s` / `end_s` as [start, end) seconds | mIoU, R@IoU 0.3/0.5/0.7 |

The QA headline numbers leave out the `num_views` sanity template, which has one answer per subset. Some templates
have two options, so chance differs by template: read the per-template and macro numbers next to the pooled one.

Baselines: `oracle`, `random`, `prior` (most frequent answer / mean-length interval), `option_prior` (answers from the
option texts alone; measures text priors) and `full` (grounding: the whole clip).

The QA answer key is not written into the export folder, because a model may read that folder.
`vpmwb understanding export --answer-key-out FILE` writes it to a separate file, and `evaluate --answer-key FILE`
scores a stale export with it. By default the key is regenerated from the data. A `qa/answer_key.jsonl` left by an
earlier version is removed on re-export.

The Real-General captions were written by vision-language-model annotators and checked by model verifiers. See
`Real-General/_source/CAPTIONS.md` in the data for their measured accuracy.
