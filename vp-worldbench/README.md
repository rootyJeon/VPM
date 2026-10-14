<h1 align="center">📊 VP-WorldBench</h1>

<p align="center">
  <a href="https://huggingface.co/datasets/rooty2020/VP-WorldBench"><img src="https://img.shields.io/badge/VP--WorldBench-Dataset-FFD21E.svg?logo=huggingface&logoColor=000" alt="VP-WorldBench"></a>
</p>

**VP-WorldBench** is a multi-view 4D benchmark for video world models: 1,200 synchronized multi-view clips. Every clip
has camera intrinsics and per-frame poses, ground-truth 3D point tracks with per-view visibility, and captions. Most
also have dense depth and further labels. This folder holds its toolkit, `vpm_worldbench` (CLI `vpmwb`), which reads
the data, exports model inputs and scores predictions. The data is on Hugging Face:
[rooty2020/VP-WorldBench](https://huggingface.co/datasets/rooty2020/VP-WorldBench) (CC BY-NC 4.0; see its dataset card).

| subset | clips | views | content | 3D ground truth | depth |
|:--|--:|--:|:--|:--|:--|
| **Simul-Robot** | 500 | 3 | humanoid tabletop pick & place (Unreal Engine) | exact (objects + robot links) | rendered |
| **Simul-General** | 500 | 4 | pedestrians and vehicles with moving cameras (CARLA) | exact (skinned target) | rendered |
| **Real-Robot** | 100 | 2 | dual-arm robot with dexterous hands, ego + exo cameras | forward kinematics | – |
| **Real-General** | 100 | 4 | a person in a 47-camera studio | multi-view pseudo-GT | reconstruction |

## Install

```bash
cd vp-worldbench
pip install -e .                    # data access, input export, PSNR and the numpy metrics
pip install -e ".[metrics,fvd]"     # + SSIM, LPIPS, FVD
```

MEt3R and δ need their reference models (not on PyPI). Their setup is in the docstrings of
[`metrics/met3r.py`](vpm_worldbench/metrics/met3r.py) and [`lifters/mvtracker.py`](vpm_worldbench/lifters/mvtracker.py).

## Data

```bash
hf download rooty2020/VP-WorldBench --repo-type dataset --local-dir data/VP-WorldBench
# without depth (~77 GB instead of ~274 GB): add --exclude "*/depth/*"
vpmwb validate --root data/VP-WorldBench
```

```python
from vpm_worldbench import Benchmark
s = Benchmark("data/VP-WorldBench").sample("Real-Robot/episode_000")
frames = s.frames(view=0, indices=[16, 17, 18], size=(848, 480))   # RGB uint8
K, w2c = s.K(0, size=(848, 480)), s.w2c(0, [16])                   # intrinsics, world-to-camera
tracks = s.tracks                                                   # GT 3D tracks: xyz, valid, visible, ...
```

The format is described in [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md).

## Evaluate a video generator

Given the first frame of every view, generate 64 future frames per view. Predictions are scored on
**appearance** (PSNR, SSIM, LPIPS, FVD), **cross-view consistency** (cross-view MEt3R) and **3D motion accuracy** (δ: the
generated views are lifted to 3D tracks and compared with the GT).

```bash
vpmwb generation export    --root data/VP-WorldBench --out inputs/                  # context frames, prompts, cameras
# generate preds/my_model/<Subset>/<id>/view{k}.mp4 for every view
vpmwb generation evaluate  --root data/VP-WorldBench --pred preds/my_model --out runs/my_model
vpmwb generation summarize --run my_model=runs/my_model --out results/
```

Protocol variants (V2V context, horizon, resolution, camera control, ...) are YAML files in [configs/](configs/),
passed with `--protocol`; see [docs/PROTOCOL.md](docs/PROTOCOL.md) and [docs/METRICS.md](docs/METRICS.md).

## Other tasks

Every task has `export`, `baseline`, `evaluate` and `summarize` commands, e.g.
`vpmwb track3d evaluate --root data/VP-WorldBench --pred preds/track3d/my_tracker --out runs/track3d/my_tracker`.

| task | input → prediction | metrics |
|:--|:--|:--|
| `track3d` | multi-view video + queries → 3D tracks | δ_avg, AJ, OA, ADE |
| `track2d` | one view + 2D queries → 2D tracks | TAP-Vid AJ, δ^x_avg, OA |
| `depth` | one view → per-frame depth | AbsRel, δ<1.25, RMSE |
| `recon4d` | multi-view video → cameras + point maps | ATE, RPE, Chamfer |
| `nvs` | other views + target camera → held-out view | PSNR, SSIM, LPIPS |
| `segmentation` | video + point/box prompt → masks | J&F, point IoU |
| `understanding` | video(s) + question → answer | QA accuracy, caption retrieval |

Formats are in [docs/TASKS.md](docs/TASKS.md); new tasks, metrics and lifters in [docs/EXTENDING.md](docs/EXTENDING.md).

## Baselines

Image-to-video, one conditioning frame per view, 64 generated frames per view, every view generated independently.
Means over clips (± s.e.). *Real video* scores the real future; *Frozen points* is a video in which nothing moves.
δ mov. counts only frames where the GT point moves.

**Simul-Robot** (500 clips)

| method | FVD ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Cross-view MEt3R ↓ | δ_avg ↑ | δ mov. ↑ |
|:--|--:|--:|--:|--:|--:|--:|--:|
| *Real video* | – | – | – | – | (0.428 raw) | 60.7 ± 0.6 | 22.0 ± 0.8 |
| *Frozen points* | – | – | – | – | – | 84.5 ± 0.4 | 0.0 ± 0.0 |
| HunyuanVideo-1.5 (480p I2V, step-distilled) | 98.3 | 20.19 | 0.779 | 0.161 | 0.0225 | 45.3 ± 0.5 | 4.8 ± 0.4 |
| Wan2.2-TI2V-5B | 314.8 | 15.79 | 0.492 | 0.320 | 0.1080 | 32.2 ± 0.6 | 2.9 ± 0.3 |
| Cosmos3-Nano | 104.6 | 20.02 | 0.724 | 0.159 | 0.0404 | 43.3 ± 0.6 | 6.4 ± 0.4 |

Cross-view MEt3R floor 0.0044; FVD over 6000 windows per method; δ mov. over the 425 clips with moving points.

**Simul-General** (500 clips)

| method | FVD ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Cross-view MEt3R ↓ | δ_avg ↑ | δ mov. ↑ |
|:--|--:|--:|--:|--:|--:|--:|--:|
| *Real video* | – | – | – | – | (0.482 raw) | 60.1 ± 0.8 | 55.7 ± 0.9 |
| *Frozen points* | – | – | – | – | – | 7.3 ± 0.2 | 0.0 ± 0.0 |
| HunyuanVideo-1.5 (480p I2V, step-distilled) | 115.8 | 11.45 | 0.205 | 0.609 | 0.0607 | 25.2 ± 0.8 | 17.8 ± 0.8 |
| Wan2.2-TI2V-5B | 206.1 | 10.43 | 0.174 | 0.629 | 0.0762 | 19.9 ± 0.5 | 12.2 ± 0.5 |
| Cosmos3-Nano | 167.4 | 10.99 | 0.177 | 0.605 | 0.0622 | 24.3 ± 0.6 | 17.0 ± 0.6 |

Cross-view MEt3R floor 0.0287; FVD over 8000 windows per method; δ mov. over all 500 clips.

**Real-Robot** (100 clips)

| method | FVD ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Cross-view MEt3R ↓ | δ_avg ↑ | δ mov. ↑ |
|:--|--:|--:|--:|--:|--:|--:|--:|
| *Real video* | – | – | – | – | (0.499 raw) | 75.1 ± 0.5 | 48.2 ± 2.3 |
| *Frozen points* | – | – | – | – | – | 61.9 ± 0.6 | 0.0 ± 0.0 |
| HunyuanVideo-1.5 (480p I2V, step-distilled) | 426.4 | 16.03 | 0.768 | 0.237 | 0.1021 | 34.6 ± 0.6 | 24.3 ± 1.4 |
| Wan2.2-TI2V-5B | 550.5 | 14.58 | 0.719 | 0.291 | 0.1593 | 26.5 ± 0.6 | 13.6 ± 1.3 |
| Cosmos3-Nano | 326.1 | 16.36 | 0.753 | 0.232 | 0.0994 | 37.2 ± 0.8 | 16.1 ± 1.2 |

Cross-view MEt3R floor 0.0093; FVD over 800 windows per method; δ mov. over the 92 clips with moving points.

**Real-General** (100 clips)

| method | FVD ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Cross-view MEt3R ↓ | δ_avg ↑ | δ mov. ↑ |
|:--|--:|--:|--:|--:|--:|--:|--:|
| *Real video* | – | – | – | – | (0.591 raw) | 68.0 ± 1.6 | 48.0 ± 2.4 |
| *Frozen points* | – | – | – | – | – | 37.6 ± 2.5 | 0.0 ± 0.0 |
| HunyuanVideo-1.5 (480p I2V, step-distilled) | 94.7 | 20.75 | 0.845 | 0.136 | 0.0135 | 29.9 ± 1.7 | 3.6 ± 0.6 |
| Wan2.2-TI2V-5B | 122.6 | 17.46 | 0.608 | 0.230 | 0.0299 | 28.2 ± 1.5 | 4.0 ± 0.7 |
| Cosmos3-Nano | 56.2 | 21.80 | 0.803 | 0.127 | 0.0145 | 31.9 ± 1.8 | 3.9 ± 0.7 |

Cross-view MEt3R floor 0.0038; FVD over 1600 windows per method; δ mov. over the 90 clips with moving points.

## Tests

```bash
pip install -e ".[dev]" && pytest tests/                                # synthetic data, CPU
VPMWB_RELEASED_ROOT=data/VP-WorldBench pytest tests/                    # + checks on the released data
```

## License

The toolkit code is released under Apache-2.0. Scoring uses third-party models under their own licenses
(MASt3R, FeatUp, MEt3R, MVTracker, MoGe-2, I3D), which are not redistributed. The data follows its dataset card.
