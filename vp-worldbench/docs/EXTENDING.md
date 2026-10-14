# Extending VP-WorldBench

## 1. Build on the data API

Everything in the toolkit reads the data through `Benchmark` and `Sample`.

```python
from vpm_worldbench import Benchmark
bench = Benchmark("/data/VP-WorldBench")
for s in bench.samples("Simul-Robot", difficulty="hard"):       # filters on index.json attributes
    frames = s.frames(0, range(16, 81), size=(480, 480))        # RGB, any size / resize mode
    K, w2c = s.K(0, size=(480, 480)), s.w2c(0, range(16, 81))   # intrinsics follow the resize
    tr = s.tracks                                               # xyz, valid, in_frustum, visible, default_gt
    obj = tr.group_mask("object")                               # or tr.kind == KIND_NAMES.index("robot_hand")
    uv, vis, inside = s.tracks2d(0, size=(480, 480))            # projected GT 2D tracks
    d = s.depth(0, [16], size=(480, 480))                       # metres, 0 = invalid (None without depth)
    caps, labels = s.captions, s.label_path("objects")          # subset-specific labels: docs/DATA_FORMAT.md §6
```

`Benchmark.samples(subset, ids=...)` takes a subset name, an alias (`rr`) or a list of them; ids may be uids with
aliases (`rr/episode_000`). A call over the whole selection warns once about ids that match no sample, and
`Benchmark.unmatched_ids(...)` lists them. `Benchmark(root)` on a missing folder raises `DataRootError` (a
`FileNotFoundError`).

`vpm_worldbench.geometry` has the conventions in code: `project`, `unproject_depth`, `image_map` and `ImageMap.apply_K`
(pixel-centre resizing, cropping and letterboxing), `umeyama`, `camera_center`.

## 2. A new task

Use a task module as the template. [`tasks/nvs.py`](../vpm_worldbench/tasks/nvs.py) is the shortest. A task has:

1. **A protocol dataclass.** It holds the window (`ref_frame`, `horizon`, `frame_step`), the views, the eval grid
   and the task options, with `to_dict` / `from_dict`, so it can be loaded from YAML and recorded in every result.
2. **`export_inputs(bench, out_dir, protocol)`.** It writes everything a method may use, and nothing it may not:
   never GT beyond the prompt or query. It also writes `manifest.jsonl`, `protocol.json` and `skipped.json`.
3. **A prediction format.** Use one file per sample (or per sample and view) under
   `<pred_root>/<Subset>/<id>/`, documented in the module docstring. Treat non-finite values as failures, not as
   missing data.
4. **`evaluate(bench, pred_root, out_dir, protocol)`.** It writes `<out>/<Subset>/<id>.json` per sample with the status
   (`ok` / `skipped` / `missing` / `invalid` / `error`, plus `partial` for partly scored samples; `tasks/common.py`
   `STATUSES`), then `summarize(out_dir)`, which writes `summary.json` / `summary.md` with means ± s.e. over samples per
   subset. Tie the output folder to its prediction root and protocol (`run.json`), as the shared driver does.
5. **Baselines** that bracket the metric, e.g. `oracle` (the GT, so 100% or zero error) and a trivial one (frozen,
   constant, nearest view). They catch protocol bugs early.
6. **`main(argv)`** with the subcommands `export`, `baseline`, `evaluate` and `summarize`. End the module with
   `if __name__ == "__main__": raise SystemExit(cli.main([TASK] + sys.argv[1:]))`, so that
   `python -m vpm_worldbench.tasks.<task>` reports usage and setup errors like `vpmwb`.
7. **Registration.** Add the module name to `cli.TASKS` to get `vpmwb <task> ...`.

The shared helpers for windows, the GT on an eval grid, the per-sample driver, JSON writing, summaries and the CLI are
in [`tasks/common.py`](../vpm_worldbench/tasks/common.py): `TaskProtocol`, `track_observations`, `select_samples`,
`evaluate_task`, `summarize_records`, `task_cli`, `MissingPrediction` / `InvalidPrediction`, `RunDirError` (refuses
another run in an output folder; `vpmwb` prints one line, exit 2), `check_out_stamp` (`run.json` / `export.json`
identity), `run_pred_root` (the prediction root in `summary.json`) and `cli_size` (WxH options). Metric functions live in
`vpm_worldbench/metrics/` as pure numpy functions on arrays.

## 3. A new 3D lifter for δ

δ lifts a generated multi-view video to 3D tracks. To use another tracker or depth model, implement `Lifter.track`:

```python
from vpm_worldbench.lifters.base import Lifter, register_lifter

class MyLifter(Lifter):
    name = "my_lifter"
    def __init__(self, device="cuda", **kw): ...
    def track(self, frames, K, w2c, queries, anchor, **kw):
        # frames (V, n, H, W, 3) uint8; K (V, n, 3, 3); w2c (V, n, 4, 4) world->camera, metres;
        # queries (N, 4) [t, x, y, z] world; anchor: GT depth / points at position 0 for metric scale
        return xyz, vis            # (n, N, 3) world metres, (n, N) bool

register_lifter("my_lifter", MyLifter)
```

Then set `lifter: my_lifter` in the protocol. Register the lifter before the protocol that names it is built or
loaded: protocols check `lifter` when they are constructed. A protocol YAML can also name an importable class directly with
`lifter: my_package.lifters:MyLifter`, so no registration is needed. The evaluator constructs the lifter with
`device=` and the `lifter_kwargs`. Its `prepare_anchor` gets the real video once per sample to fix the scene
normalisation.

For depth + tracker pipelines, subclass `DepthAnchoredLifter` and implement `estimate_depth` and `run_tracker`. This
reuses the anchoring, scene normalisation and world mapping of the reference lifter.

Report the lifter next to δ: values from different lifters are not comparable.

## 4. A new protocol preset

Ship a YAML file (see `configs/`) or add a function to `protocol.PRESETS`. Per-subset values (eval size, δ ladders,
depth anchor, …) are dicts keyed by subset name. Missing subsets fall back to `presets.SUBSET_DEFAULTS`.

## 5. A new subset or dataset

1. Convert the data to the per-sample format of [DATA_FORMAT.md](DATA_FORMAT.md):
   - world frame: right-handed, metres, +z up;
   - OpenCV world-to-camera matrices for every frame;
   - pixel-centre intrinsics;
   - tracks with `valid ⊇ in_frustum ⊇ visible` and the unified `kind` codes;
   - `default_gt`, captions, and `flags` for known caveats.
2. Add the subset to `constants.SUBSETS` and its defaults to `presets.SUBSET_DEFAULTS`: eval size, δ ladder,
   survival threshold, report group, depth anchor, far clip.
3. Run `vpmwb validate --root R --deep --hash` and `vpmwb index --root R`. Every sample must pass before any
   evaluation. `validate` exits with 0 when every selected sample is valid, 1 when a sample fails, and 2 when the
   selection is empty, a requested subset folder is missing, an id matches no sample or `--root` is not a directory.
   `vpmwb index` exits 2 (one line) for a missing root or a root without samples, and then writes no `index.json`.
