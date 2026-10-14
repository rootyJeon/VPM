# Generation protocol

The main VP-WorldBench task: given the first frame(s) of every view of a multi-view clip (and a text prompt,
optionally the cameras), generate the future of **every view**. The generated views are scored for appearance
(PSNR / SSIM / LPIPS, FVD), cross-view consistency (cross-view MEt3R) and 3D motion accuracy (δ). See
[METRICS.md](METRICS.md) for the metrics.

Everything below is a field of `vpm_worldbench.Protocol` and can be changed. The defaults define the
VP-WorldBench v1 numbers (`vpm-v1`).

## 1. Time

```
          context (c frames)        t_ref            scored frames (horizon H, step s)
 ... | t_ref-c+1 ... t_ref-1 |   t_ref   | t_ref+s  t_ref+2s  ...  t_ref+H*s | ...
                               ^ conditioning frame = window position 0
                                 GT tracks are queried here
```

| field | default | meaning |
|---|---|---|
| `ref_frame` | 16 | t_ref: the last given frame. The 3D tracks are queried at t_ref. |
| `context_frames` | 1 | the model gets frames `t_ref-c+1 … t_ref` of every view. 1 = image-to-video (I2V), > 1 = video-to-video (V2V). |
| `horizon` | 64 | number of scored future frames |
| `frame_step` | 1 | scored frames are `t_ref + k·frame_step`, k = 1 … horizon |
| `views` | all | view indices to generate and score (e.g. `[0, 1]`) |
| `clip_filter` | none | filters on the `index.json` records (top-level keys, then attributes), e.g. `{"difficulty": "hard"}`; needs `index.json` |

The **window** is `[t_ref, t_ref+s, …, t_ref+H·s]` (`Protocol.window_indices`), so H+1 frames. Position 0 is the
conditioning frame; positions 1 … H are scored.

A sample is **evaluable** iff its context starts at frame ≥ 0, the window ends at frame ≤ T−1 and every requested view
exists. The other samples are skipped and listed with the reason in the outputs. With the defaults every sample of the benchmark is evaluable:
the shortest clips have 81 frames, and the window covers frames 16 … 80.

Examples:

```python
from vpm_worldbench import Protocol
Protocol()                                     # I2V: frame 16 given, frames 17..80 scored
Protocol(context_frames=9)                     # V2V: frames 8..16 given, frames 17..80 scored
Protocol(ref_frame=0, horizon=80)              # first-frame I2V over 81 frames
Protocol(ref_frame=32, horizon=32, frame_step=2)   # frames 34, 36, ..., 96
Protocol(views=[0, 1], clip_filter={"difficulty": "hard"})
```

Protocols can be stored as YAML or JSON (`Protocol.save`, `Protocol.load`; examples in `configs/`). Unknown keys are
rejected.

A file without `name` is named after its file stem. A stem that is a preset name becomes `custom-<stem>`.

A file named `legacy-prerelease` (the one name that switches on behaviour) is that preset only when it evaluates like
it: every field except `text`, `camera_conditioning`, `context_frames` and `clip_filter` equal to the preset's.
Otherwise it loads as `custom-legacy-prerelease`, with a warning. `name: vpm-v1` is kept as given; it switches
nothing on. The name is not part of the reference key (§5).

Every field is checked when a protocol is built or loaded:
- integers and booleans have the right type;
- `eval_size` is pairs of positive integers;
- `resize_mode`, `met3r_aspect`, `fvd_cov`, `gt_visibility` and `reference_frames` take their allowed values;
- `depth_anchor` is `gt_depth`, `gt_points` or `none`;
- `views` is distinct indices ≥ 0;
- `clip_filter` is a mapping;
- `lifter` is registered or `module:attr`.

So a `--protocol` that is neither a preset nor a readable, valid file stops with exit status 2 before anything is
exported or scored. `vpmwb protocol`, `generation export` and `summarize` print one line `vpmwb: error: ...`.
`generation evaluate` / `reference` print their usage first, then `vpmwb generation <cmd>: error: --protocol ...`.

## 2. Pixels

| field | default | meaning |
|---|---|---|
| `eval_size` | Simul-* 480×480, Real-Robot 848×480, Real-General 640×480 (W×H) | per-subset grid on which every metric compares frames |
| `resize_mode` | `resize` | how frames of any size reach the eval grid: `resize` (stretch), `center_crop` (scale, then crop to the aspect), `letterbox` (scale and pad with black) |

Predictions may have any resolution. They are mapped to the eval grid with the same resize mode as the reference
frames. Intrinsics follow every resize with the pixel-centre rule (`geometry.ImageMap.apply_K`). With `center_crop`,
the δ queries and the GT visibility also require a point to project inside the cropped grid of a view.

## 3. Model inputs

`vpmwb generation export` (alias `vpmwb export-inputs`) writes what a generator needs:

```
<inputs>/manifest.jsonl                                  one line per evaluable sample
<inputs>/protocol.json                                   the protocol used
<inputs>/skipped.json                                    non-evaluable samples and why
<inputs>/<Subset>/<id>/context/view{k}/frame_{t:04d}.png context frames (native size, or --size WxH)
<inputs>/<Subset>/<id>/inputs.json                       prompts, frame indices, fps, number of frames to generate,
                                                         K (at the exported size) and w2c of the context frames
                                                         (and of the window frames if camera_conditioning is set)
```

| field | default | meaning |
|---|---|---|
| `text` | `caption` | prompt per view: `caption` (global caption), `caption+camera` (+ a camera-motion sentence of the view over the window), `window` (caption of the evaluated window where available, Real-General), `legacy` (the exact pre-release prompts), `none` |
| `camera_conditioning` | false | also export the cameras of the window frames (camera-controlled generation) |

These fields affect only the exported inputs, not the metrics. State them when you report numbers.

With `--size WxH` the context frames reach that size with the protocol's `resize_mode` of each subset, so they are
framed like the scored reference frames. `--resize-mode` overrides it with a warning, and a size that cannot be framed
like the reference also warns. Each view's camera in `inputs.json` records the mode used (`resize_mode`, null at
native size). `--subsets` / `--ids` accept aliases (`rr`, `rr/episode_000`). Unmatched ids are named in a warning; an
empty selection exits with status 1, as in every other command.

## 4. Predictions

```
<pred_root>/<Subset>/<id>/view{k}.mp4       one video per scored view, k = view index (view0, view1, ...)
<pred_root>/<Subset>/<id>/meta.json         optional
```

By default a video must already be on the window grid. Frame j is GT frame `window_indices[j]`. Frame 0 is the
conditioning frame, and at least H+1 frames are needed; extra frames are ignored.

The optional `meta.json` covers other generators:

| key | meaning |
|---|---|
| `"first_frame": "future"` | the video starts at the first scored frame; the GT conditioning frame is used as position 0 |
| `"fps": 24.0` | the video's own frame rate. Frames are taken at the times of the window frames, x = (t − t_ref)·fps/fps_gt, in exact rational arithmetic (each rate read as the nearest fraction with denominator ≤ 10⁶, so 29.97 = 2997/100). When fps ≥ 12 the nearest frame floor(x + ½) is used, so a tie takes the later frame. Below 12 fps the two neighbours are linearly blended. With `first_frame: future`, video frame i is at time (i + 1)/fps after t_ref; below 12 fps the first scored position can then blend in the given conditioning frame |

`meta.json` must be a JSON object with `first_frame` exactly `"ref"` or `"future"` and `fps` a finite JSON number > 0
(`"24"` or `true` are rejected);
otherwise the sample is `invalid`, in generation and in nvs.

`vpmwb generation export` writes `fps_window` = clip fps / `frame_step` into `inputs.json`. A video generated at that
rate is on the window grid without a `meta.json`.

Any codec PyAV can decode is accepted. Real and generated videos are decoded the same way. The toolkit's YUV→RGB
conversion is fixed (`vpm_worldbench.io.video`), so results do not depend on the FFmpeg build.

**Encode predictions losslessly** (H.264 `-qp 0`, FFV1, or PNG frames converted losslessly), or at least near-losslessly
(CRF ≤ 4). PSNR barely notices compression: a CRF-18 re-encode of the baselines changes it by ≤ 0.03 dB. δ and MEt3R
are much more sensitive. The same CRF-18 re-encode moves per-clip δ by up to ~7 points, and a CRF-4 re-encode moves
per-clip MEt3R by up to ~0.04.

## 5. Running the evaluation

```bash
vpmwb generation evaluate --root /data/VP-WorldBench --pred preds/my_model --out runs/my_model \
      [--protocol vpm-v1 | configs/v2v_9.yaml] [--subsets sr rr] [--metrics appearance fvd met3r delta] [--shard 0/8]
vpmwb generation evaluate --root /data/VP-WorldBench --pred a=preds/a b=preds/b --out runs/     # several models, one pass
vpmwb generation summarize --run my_model=runs/my_model --run other=runs/other --out results/ [--split difficulty=hard]
```

- **Per-sample results.** They go to `<run>/<Subset>/<id>.json`. Each record stores the predictions it was scored
  from: `pred_root`, the resolved prediction folder, and `pred_stamp`, a digest of the names, sizes and modification
  times of `view*.mp4` and `meta.json`.
- **Resuming and sharding.** Runs are resumable. A complete record is reused only while its predictions are unchanged;
  predictions regenerated in place are scored again by the same command. Incomplete or failed records are redone, and
  `--overwrite` redoes everything. `--shard k/n` splits the samples across processes or jobs.
- **One prediction root per run.** A run directory holds one prediction root: another root is refused (exit 2) unless
  `--overwrite` is given. `summarize` marks records of a previous root `stale` and leaves them out. Records written
  before these fields existed (such as the published baselines) are reused as they are.
- **Never-evaluated samples.** Every call records its selected samples, so `summarize` lists samples that were never
  evaluated, and `--require-complete` makes it exit with status 3 in that case.
- **Devices and exit status.** `--device` defaults to `auto` (CUDA when PyTorch sees a GPU, else CPU). MEt3R needs a
  CUDA GPU with the official stack. `evaluate` /
  `reference` exit with:
  - 0 when the run succeeds;
  - 1 when the selection matches no sample, every evaluated sample of a model failed, or every computed reference row
    failed;
  - 2 on a setup problem: missing optional packages, weights that cannot be loaded, a run directory of another
    protocol or prediction root, or bad arguments such as a malformed `--pred` / `--run`. The packages each metric
    needs are checked before anything is written and listed at once: torch, pytorch3d, featup and met3r for MEt3R;
    torch and moge for δ. Weights, MASt3R and the MVTracker repository are checked when the models load.

  `VPMWB_TRACEBACK=1` shows the traceback of a setup error; `--raise-errors` re-raises per-sample failures.

  `summarize` exits 2 for an unusable run or reference directory. Under `--require-complete` it exits 3 when a
  selected sample has no record, or when the summary has warnings: records that disagree in weights, options or data
  version, a missing reference directory, or a subset without reference rows.
- **Reference rows.** These are the real-video FVD features and MEt3R, the real video through the δ lifter, and the
  frozen-points baseline. `--metrics appearance` (PSNR / SSIM / LPIPS) needs none and writes no reference directory.
  The rows are computed once per sample and cached under a key: a digest of the resolved per-subset metric settings. Equivalent spellings of a protocol share the key. Model inputs (`text`,
  `camera_conditioning`, `context_frames`) and `clip_filter` do not enter it, so `configs/vpm-v1.yaml`,
  `v2v_9.yaml` and `camera_control.yaml` share the reference rows of `vpm-v1`. `vpmwb protocol` prints the key and
  every resolved setting.
- **Where references are stored.** By default under `_reference/` next to the run directory, so runs in one parent
  folder share them. Pass `--reference-dir` explicitly when runs live elsewhere. `vpmwb generation reference`
  computes only the reference rows.
- **MVTracker version.** Without `--mvtracker-repo`, δ fetches the reference MVTracker commit (ceea8ad) through
  torch.hub. On offline nodes, pass a local clone with `--mvtracker-repo`. Records store the repository and the
  commit used (`models.delta.paths.hub_repo` / `hub_commit`).
- **Weights and data version.** Every record also stores the weights it was computed with (I3D, MEt3R, MVTracker,
  MoGe-2) and the data version. `summarize` warns when the records of one table disagree.
- **Tables.** `summarize` builds paper-style tables per subset over the **paired clip set**, i.e. the clips scored
  for every model. It reports n and lists excluded and not-evaluated clips. Columns computed on fewer clips (δ_avg
  dyn., δ mov.) give their n in a footnote (`columns_n` in `summary.json`). Every mean ± s.e. also carries `n_inf`,
  the number of failed clips with an infinite value. The output folder receives
  `summary.json` and `summary.md`; `summary.json` also lists each run's prediction root (`pred_roots`). The shortcuts
  `vpmwb eval-generation` and `vpmwb report` are aliases.

## 6. Presets

| preset | use |
|---|---|
| `vpm-v1` (default) | the official setting described above |
| `legacy-prerelease` | approximately reproduces internal numbers computed before the release (§7). Not for new results. |

## 7. Legacy (pre-release) numbers

Numbers computed before the release are **not comparable** to `vpm-v1`. The release fixes or changes:

- **MVTracker input range:** the pre-release δ passed RGB/255 to MVTracker, which rescales by 1/255 itself, so its
  feature encoder saw nearly constant images. `vpm-v1` passes 0…255.
- **World frame for MVTracker:** Simul-Robot δ was computed in Unreal's left-handed centimetre world. MVTracker is not
  mirror-invariant: the per-clip δ of the real video moves by about 5 points on average between the two handednesses.
  `vpm-v1` uses the released right-handed world.
- **Real-Robot time alignment:** the RB-Y1 videos trail the logged joint states by 2.3 frames. The released tracks are
  aligned to the video.
- **Simul intrinsics:** the delivered intrinsics use the pixel-corner convention (cx = 640). The released `K` uses pixel
  centres (cx = 639.5).
- **Scored frames:** the pre-release appearance metrics used in-memory frames before lossy encoding, decoded with
  OpenCV. The release decodes the stored videos with the fixed converter.
- **Other changes:** query selection, per-subset FVD window counts and covariance estimators, and Real-General's
  squashed 480×480 frames.

`legacy-prerelease` switches these back where possible (`presets.LEGACY_PRERELEASE`). It also uses the lossless frames in
`_reproduce/` for Real-General (lossless 480×480 frames and the frame-16 depth). The recipes and the residual differences measured on 28 samples are documented in
`vpm_worldbench/legacy.py`.
