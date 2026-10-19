<h1 align="center">📊 VP-WorldBench</h1>

<p align="center">
  <a href="https://huggingface.co/datasets/rooty2020/VP-WorldBench"><img src="https://img.shields.io/badge/VP--WorldBench-Dataset-FFD21E.svg?logo=data:image/svg%2Bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI5NSIgaGVpZ2h0PSI4OCIgZmlsbD0ibm9uZSI%2BPHBhdGggZmlsbD0iI0ZGRDIxRSIgZD0iTTQ3LjIxIDc2LjVhMzQuNzUgMzQuNzUgMCAxIDAgMC02OS41IDM0Ljc1IDM0Ljc1IDAgMCAwIDAgNjkuNVoiLz48cGF0aCBmaWxsPSIjRkY5RDBCIiBkPSJNODEuOTYgNDEuNzVhMzQuNzUgMzQuNzUgMCAxIDAtNjkuNSAwIDM0Ljc1IDM0Ljc1IDAgMCAwIDY5LjUgMFptLTczLjUgMGEzOC43NSAzOC43NSAwIDEgMSA3Ny41IDAgMzguNzUgMzguNzUgMCAwIDEtNzcuNSAwWiIvPjxwYXRoIGZpbGw9IiMzQTNCNDUiIGQ9Ik01OC41IDMyLjNjMS4yOC40NCAxLjc4IDMuMDYgMy4wNyAyLjM4YTUgNSAwIDEgMC02Ljc2LTIuMDdjLjYxIDEuMTUgMi41NS0uNzIgMy43LS4zMlpNMzQuOTUgMzIuM2MtMS4yOC40NC0xLjc5IDMuMDYtMy4wNyAyLjM4YTUgNSAwIDEgMSA2Ljc2LTIuMDdjLS42MSAxLjE1LTIuNTYtLjcyLTMuNy0uMzJaIi8%2BPHBhdGggZmlsbD0iI0ZGMzIzRCIgZD0iTTQ2Ljk2IDU2LjI5YzkuODMgMCAxMy04Ljc2IDEzLTEzLjI2IDAtMi4zNC0xLjU3LTEuNi00LjA5LS4zNi0yLjMzIDEuMTUtNS40NiAyLjc0LTguOSAyLjc0LTcuMTkgMC0xMy02Ljg4LTEzLTIuMzhzMy4xNiAxMy4yNiAxMyAxMy4yNloiLz48cGF0aCBmaWxsPSIjM0EzQjQ1IiBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik0zOS40MyA1NGE4LjcgOC43IDAgMCAxIDUuMy00LjQ5Yy40LS4xMi44MS41NyAxLjI0IDEuMjguNC42OC44MiAxLjM3IDEuMjQgMS4zNy40NSAwIC45LS42OCAxLjMzLTEuMzUuNDUtLjcuODktMS4zOCAxLjMyLTEuMjVhOC42MSA4LjYxIDAgMCAxIDUgNC4xN2MzLjczLTIuOTQgNS4xLTcuNzQgNS4xLTEwLjcgMC0yLjM0LTEuNTctMS42LTQuMDktLjM2bC0uMTQuMDdjLTIuMzEgMS4xNS01LjM5IDIuNjctOC43NyAyLjY3cy02LjQ1LTEuNTItOC43Ny0yLjY3Yy0yLjYtMS4yOS00LjIzLTIuMS00LjIzLjI5IDAgMy4wNSAxLjQ2IDguMDYgNS40NyAxMC45N1oiIGNsaXAtcnVsZT0iZXZlbm9kZCIvPjxwYXRoIGZpbGw9IiNGRjlEMEIiIGQ9Ik03MC43MSAzN2EzLjI1IDMuMjUgMCAxIDAgMC02LjUgMy4yNSAzLjI1IDAgMCAwIDAgNi41Wk0yNC4yMSAzN2EzLjI1IDMuMjUgMCAxIDAgMC02LjUgMy4yNSAzLjI1IDAgMCAwIDAgNi41Wk0xNy41MiA0OGMtMS42MiAwLTMuMDYuNjYtNC4wNyAxLjg3YTUuOTcgNS45NyAwIDAgMC0xLjMzIDMuNzYgNy4xIDcuMSAwIDAgMC0xLjk0LS4zYy0xLjU1IDAtMi45NS41OS0zLjk0IDEuNjZhNS44IDUuOCAwIDAgMC0uOCA3IDUuMyA1LjMgMCAwIDAtMS43OSAyLjgyYy0uMjQuOS0uNDggMi44LjggNC43NGE1LjIyIDUuMjIgMCAwIDAtLjM3IDUuMDJjMS4wMiAyLjMyIDMuNTcgNC4xNCA4LjUyIDYuMSAzLjA3IDEuMjIgNS44OSAyIDUuOTEgMi4wMWE0NC4zMyA0NC4zMyAwIDAgMCAxMC45MyAxLjZjNS44NiAwIDEwLjA1LTEuOCAxMi40Ni01LjM0IDMuODgtNS42OSAzLjMzLTEwLjktMS43LTE1LjkyLTIuNzctMi43OC00LjYyLTYuODctNS03Ljc3LS43OC0yLjY2LTIuODQtNS42Mi02LjI1LTUuNjJhNS43IDUuNyAwIDAgMC00LjYgMi40NmMtMS0xLjI2LTEuOTgtMi4yNS0yLjg2LTIuODJBNy40IDcuNCAwIDAgMCAxNy41MiA0OFptMCA0Yy41MSAwIDEuMTQuMjIgMS44Mi42NSAyLjE0IDEuMzYgNi4yNSA4LjQzIDcuNzYgMTEuMTguNS45MiAxLjM3IDEuMzEgMi4xNCAxLjMxIDEuNTUgMCAyLjc1LTEuNTMuMTUtMy40OC0zLjkyLTIuOTMtMi41NS03LjcyLS42OC04LjAxLjA4LS4wMi4xNy0uMDIuMjQtLjAyIDEuNyAwIDIuNDUgMi45MyAyLjQ1IDIuOTNzMi4yIDUuNTIgNS45OCA5LjNjMy43NyAzLjc3IDMuOTcgNi44IDEuMjIgMTAuODMtMS44OCAyLjc1LTUuNDcgMy41OC05LjE2IDMuNTgtMy44MSAwLTcuNzMtLjktOS45Mi0xLjQ2LS4xMS0uMDMtMTMuNDUtMy44LTExLjc2LTcgLjI4LS41NC43NS0uNzYgMS4zNC0uNzYgMi4zOCAwIDYuNyAzLjU0IDguNTcgMy41NC40MSAwIC43LS4xNy44My0uNi43OS0yLjg1LTEyLjA2LTQuMDUtMTAuOTgtOC4xNy4yLS43My43MS0xLjAyIDEuNDQtMS4wMiAzLjE0IDAgMTAuMiA1LjUzIDExLjY4IDUuNTMuMTEgMCAuMi0uMDMuMjQtLjEuNzQtMS4yLjMzLTIuMDQtNC45LTUuMi01LjIxLTMuMTYtOC44OC01LjA2LTYuOC03LjMzLjI0LS4yNi41OC0uMzggMS0uMzggMy4xNyAwIDEwLjY2IDYuODIgMTAuNjYgNi44MnMyLjAyIDIuMSAzLjI1IDIuMWMuMjggMCAuNTItLjEuNjgtLjM4Ljg2LTEuNDYtOC4wNi04LjIyLTguNTYtMTEuMDEtLjM0LTEuOS4yNC0yLjg1IDEuMzEtMi44NVoiLz48cGF0aCBmaWxsPSIjRkZEMjFFIiBkPSJNMzguNiA3Ni42OWMyLjc1LTQuMDQgMi41NS03LjA3LTEuMjItMTAuODQtMy43OC0zLjc3LTUuOTgtOS4zLTUuOTgtOS4zcy0uODItMy4yLTIuNjktMi45Yy0xLjg3LjMtMy4yNCA1LjA4LjY4IDguMDEgMy45MSAyLjkzLS43OCA0LjkyLTIuMjkgMi4xNy0xLjUtMi43NS01LjYyLTkuODItNy43Ni0xMS4xOC0yLjEzLTEuMzUtMy42My0uNi0zLjEzIDIuMi41IDIuNzkgOS40MyA5LjU1IDguNTYgMTEtLjg3IDEuNDctMy45My0xLjcxLTMuOTMtMS43MXMtOS41Ny04LjcxLTExLjY2LTYuNDRjLTIuMDggMi4yNyAxLjU5IDQuMTcgNi44IDcuMzMgNS4yMyAzLjE2IDUuNjQgNCA0LjkgNS4yLS43NSAxLjItMTIuMjgtOC41My0xMy4zNi00LjQtMS4wOCA0LjExIDExLjc3IDUuMyAxMC45OCA4LjE1LS44IDIuODUtOS4wNi01LjM4LTEwLjc0LTIuMTgtMS43IDMuMjEgMTEuNjUgNi45OCAxMS43NiA3LjAxIDQuMyAxLjEyIDE1LjI1IDMuNDkgMTkuMDgtMi4xMloiLz48cGF0aCBmaWxsPSIjRkY5RDBCIiBkPSJNNzcuNCA0OGMxLjYyIDAgMy4wNy42NiA0LjA3IDEuODdhNS45NyA1Ljk3IDAgMCAxIDEuMzMgMy43NiA3LjEgNy4xIDAgMCAxIDEuOTUtLjNjMS41NSAwIDIuOTUuNTkgMy45NCAxLjY2YTUuOCA1LjggMCAwIDEgLjggNyA1LjMgNS4zIDAgMCAxIDEuNzggMi44MmMuMjQuOS40OCAyLjgtLjggNC43NGE1LjIyIDUuMjIgMCAwIDEgLjM3IDUuMDJjLTEuMDIgMi4zMi0zLjU3IDQuMTQtOC41MSA2LjEtMy4wOCAxLjIyLTUuOSAyLTUuOTIgMi4wMWE0NC4zMyA0NC4zMyAwIDAgMS0xMC45MyAxLjZjLTUuODYgMC0xMC4wNS0xLjgtMTIuNDYtNS4zNC0zLjg4LTUuNjktMy4zMy0xMC45IDEuNy0xNS45MiAyLjc4LTIuNzggNC42My02Ljg3IDUuMDEtNy43Ny43OC0yLjY2IDIuODMtNS42MiA2LjI0LTUuNjJhNS43IDUuNyAwIDAgMSA0LjYgMi40NmMxLTEuMjYgMS45OC0yLjI1IDIuODctMi44MkE3LjQgNy40IDAgMCAxIDc3LjQgNDhabTAgNGMtLjUxIDAtMS4xMy4yMi0xLjgyLjY1LTIuMTMgMS4zNi02LjI1IDguNDMtNy43NiAxMS4xOGEyLjQzIDIuNDMgMCAwIDEtMi4xNCAxLjMxYy0xLjU0IDAtMi43NS0xLjUzLS4xNC0zLjQ4IDMuOTEtMi45MyAyLjU0LTcuNzIuNjctOC4wMWExLjU0IDEuNTQgMCAwIDAtLjI0LS4wMmMtMS43IDAtMi40NSAyLjkzLTIuNDUgMi45M3MtMi4yIDUuNTItNS45NyA5LjNjLTMuNzggMy43Ny0zLjk4IDYuOC0xLjIyIDEwLjgzIDEuODcgMi43NSA1LjQ3IDMuNTggOS4xNSAzLjU4IDMuODIgMCA3LjczLS45IDkuOTMtMS40Ni4xLS4wMyAxMy40NS0zLjggMTEuNzYtNy0uMjktLjU0LS43NS0uNzYtMS4zNC0uNzYtMi4zOCAwLTYuNzEgMy41NC04LjU3IDMuNTQtLjQyIDAtLjcxLS4xNy0uODMtLjYtLjgtMi44NSAxMi4wNS00LjA1IDEwLjk3LTguMTctLjE5LS43My0uNy0xLjAyLTEuNDQtMS4wMi0zLjE0IDAtMTAuMiA1LjUzLTExLjY4IDUuNTMtLjEgMC0uMTktLjAzLS4yMy0uMS0uNzQtMS4yLS4zNC0yLjA0IDQuODgtNS4yIDUuMjMtMy4xNiA4LjktNS4wNiA2LjgtNy4zMy0uMjMtLjI2LS41Ny0uMzgtLjk4LS4zOC0zLjE4IDAtMTAuNjcgNi44Mi0xMC42NyA2Ljgycy0yLjAyIDIuMS0zLjI0IDIuMWEuNzQuNzQgMCAwIDEtLjY4LS4zOGMtLjg3LTEuNDYgOC4wNS04LjIyIDguNTUtMTEuMDEuMzQtMS45LS4yNC0yLjg1LTEuMzEtMi44NVoiLz48cGF0aCBmaWxsPSIjRkZEMjFFIiBkPSJNNTYuMzMgNzYuNjljLTIuNzUtNC4wNC0yLjU2LTcuMDcgMS4yMi0xMC44NCAzLjc3LTMuNzcgNS45Ny05LjMgNS45Ny05LjNzLjgyLTMuMiAyLjctMi45YzEuODYuMyAzLjIzIDUuMDgtLjY4IDguMDEtMy45MiAyLjkzLjc4IDQuOTIgMi4yOCAyLjE3IDEuNTEtMi43NSA1LjYzLTkuODIgNy43Ni0xMS4xOCAyLjEzLTEuMzUgMy42NC0uNiAzLjEzIDIuMi0uNSAyLjc5LTkuNDIgOS41NS04LjU1IDExIC44NiAxLjQ3IDMuOTItMS43MSAzLjkyLTEuNzFzOS41OC04LjcxIDExLjY2LTYuNDRjMi4wOCAyLjI3LTEuNTggNC4xNy02LjggNy4zMy01LjIzIDMuMTYtNS42MyA0LTQuOSA1LjIuNzUgMS4yIDEyLjI4LTguNTMgMTMuMzYtNC40IDEuMDggNC4xMS0xMS43NiA1LjMtMTAuOTcgOC4xNS44IDIuODUgOS4wNS01LjM4IDEwLjc0LTIuMTggMS42OSAzLjIxLTExLjY1IDYuOTgtMTEuNzYgNy4wMS00LjMxIDEuMTItMTUuMjYgMy40OS0xOS4wOC0yLjEyWiIvPjwvc3ZnPiA%3D" alt="VP-WorldBench"></a>
</p>

**VP-WorldBench** is a multi-view 4D benchmark for video world models: 1,200 synchronized multi-view clips. Every clip
has camera intrinsics and per-frame poses, ground-truth 3D point tracks with per-view visibility, and captions. Most
also have dense depth and further labels. This folder holds its toolkit, `vpm_worldbench` (CLI `vpmwb`), which reads
the data, exports model inputs and scores predictions. The data is on Hugging Face:
[rooty2020/VP-WorldBench](https://huggingface.co/datasets/rooty2020/VP-WorldBench) (CC BY-NC 4.0; see its dataset card).

| subset | clips | views | content | 3D ground truth | depth |
|:--|--:|--:|:--|:--|:--|
| **Simul-Robot** | 500 | 3 | humanoid tabletop pick & place (Unreal Engine): 436 at new locations in the training city, 64 in two unseen kitchens; 27 hard | exact (objects + robot links) | rendered |
| **Simul-General** | 500 | 4 | pedestrians (332) and vehicles (168) with moving cameras (CARLA), in a different city from the urban training data; 95 hard | exact (skinned target) | rendered |
| **Real-Robot** | 100 | 2 | dual-arm robot with dexterous hands, teleoperated, ego + exo cameras | forward kinematics | – |
| **Real-General** | 100 | 4 | a person in a 48-camera studio (4 views released) | multi-view pseudo-GT | reconstruction |

In a **hard** clip (`difficulty: hard` in `index.json`) individual cameras lose sight of the target for at least 30
consecutive frames while at least one camera sees it at every frame.

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
**appearance** (PSNR, SSIM, LPIPS, FVD), **cross-view consistency** (cross-view MEt3R) and **3D motion accuracy** (δ3D avg:
the generated views are lifted to 3D tracks and compared with the GT).

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

## Results

The results of the paper (Table 2). Image-to-video: every model receives one frame of each view and generates the
next 64 frames, resized to 480p. Single-view models generate each view independently. FVD and PSNR are computed per
view and averaged over the views. **MV**: the model generates all views jointly. VPM (V) is trained on multi-view
videos only, VPM (V+T) also on 3D point trajectories. X-WAM is evaluated on the robot subsets only.

| subset | method | MV | FVD ↓ | PSNR ↑ | Cross-view MEt3R ↓ | δ3D avg ↑ |
|:--|:--|:-:|--:|--:|--:|--:|
| **Simul-Robot** (500) | HunyuanVideo-1.5 | ✗ | 98.34 | 20.21 | 0.023 | 41.62 |
| | Wan2.2-5B | ✗ | 315.13 | 15.84 | 0.108 | 29.51 |
| | X-WAM | ✓ | 203.83 | 19.97 | 0.038 | 38.80 |
| | Cosmos3-Nano | ✗ | 104.74 | 20.05 | 0.040 | 39.03 |
| | VPM (V) | ✓ | 78.52 | 21.41 | 0.022 | 43.62 |
| | **VPM (V+T)** | ✓ | **67.35** | **22.08** | **0.016** | **50.05** |
| **Simul-General** (500) | HunyuanVideo-1.5 | ✗ | 115.63 | 11.45 | 0.061 | 23.10 |
| | Wan2.2-5B | ✗ | 205.73 | 10.43 | 0.076 | 18.00 |
| | Cosmos3-Nano | ✗ | 167.02 | 10.99 | 0.062 | 22.12 |
| | VPM (V) | ✓ | 101.37 | 16.95 | 0.057 | 28.71 |
| | **VPM (V+T)** | ✓ | **81.92** | **19.16** | **0.050** | **36.80** |
| **Real-Robot** (100) | HunyuanVideo-1.5 | ✗ | 426.76 | 16.04 | 0.102 | 36.17 |
| | Wan2.2-5B | ✗ | 550.55 | 14.58 | 0.159 | 29.65 |
| | X-WAM | ✓ | 492.31 | 17.63 | 0.091 | 50.86 |
| | Cosmos3-Nano | ✗ | 326.11 | 16.37 | 0.099 | 41.50 |
| | VPM (V) | ✓ | 296.83 | 18.42 | 0.074 | 50.83 |
| | **VPM (V+T)** | ✓ | **255.19** | **20.57** | **0.064** | **64.75** |
| **Real-General** (100) | HunyuanVideo-1.5 | ✗ | 105.83 | 20.00 | 0.014 | 21.35 |
| | Wan2.2-5B | ✗ | 115.52 | 16.97 | 0.030 | 16.82 |
| | Cosmos3-Nano | ✗ | 66.24 | 20.89 | 0.015 | 20.47 |
| | VPM (V) | ✓ | 62.85 | 20.96 | 0.010 | 26.54 |
| | **VPM (V+T)** | ✓ | **56.48** | **22.10** | **0.007** | **33.92** |

δ3D avg is reported as `delta_avg` by the toolkit. The toolkit also reports SSIM, LPIPS, δ mov., standard errors and
the Cross-view MEt3R floors.

## Tests

```bash
pip install -e ".[dev]" && pytest tests/                                # synthetic data, CPU
VPMWB_RELEASED_ROOT=data/VP-WorldBench pytest tests/                    # + checks on the released data
```

## License

The toolkit code is released under Apache-2.0. Scoring uses third-party models under their own licenses
(MASt3R, FeatUp, MEt3R, MVTracker, MoGe-2, I3D), which are not redistributed. The data follows its dataset card.
