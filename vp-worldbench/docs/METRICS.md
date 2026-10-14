# Metrics of the generation task

| metric | measures | better | unit | aggregation |
|---|---|---|---|---|
| PSNR / SSIM / LPIPS | per-pixel fidelity to the real future | ↑ / ↑ / ↓ | dB / – / – | per clip (frames, views), then mean ± s.e. over clips |
| FVD | distribution of motion and appearance | ↓ | – | pooled over all windows of a subset |
| Cross-view MEt3R | multi-view consistency, relative to the real video | ↓ | – | per clip, then mean ± s.e. over clips; floor reported |
| δ (δ_avg, δ mov.) | 3D motion accuracy of the generated views (δ mov.: moving points only) | ↑ | % | per clip, then mean ± s.e. over clips |

All per-clip quantities are computed on the protocol window ([PROTOCOL.md](PROTOCOL.md)) at the subset's eval grid.
Window position 0 is the conditioning frame: appearance, FVD and MEt3R never score it. δ uses it only for the query
positions and track bookkeeping (see [δ](#delta)). "s.e." is std(ddof=1)/√n over clips. Tables compare
models on the **paired** clip set, i.e. the clips scored for every model in the table.

Reference rows (computed with the same code, cached per protocol):

- **Real video**: the real future through the same pipeline. It is the reference for MEt3R and gives the δ of the
  lifter on real footage, the practical ceiling of δ. No FVD is computed for it; the table shows –.
- **Frozen points** (δ only): every point stays where it was at t_ref. This is the δ of a video with no motion.

<a id="appearance"></a>
## PSNR, SSIM, LPIPS

Per view and scored frame (positions 1 … H), on uint8 RGB at the eval grid:

- PSNR = 10·log10(255² / MSE), with MSE in float64 and floored at 1e-8. The clip value is the mean of the per-frame
  PSNRs over frames and views, not a PSNR of the pooled MSE.
- SSIM: `skimage.metrics.structural_similarity(channel_axis=2, data_range=255)` with default settings.
- LPIPS: AlexNet, v0.1, on x/127.5 − 1.

The evaluator also stores the PSNR of position 0, prediction vs real conditioning frame, per view. It flags a clip when
the minimum over its views is below 25 dB, and `summary.md` counts the flagged clips per method.
This sanity check catches misaligned or re-encoded first frames.

<a id="fvd"></a>
## FVD

1. Per view, the window is cut into non-overlapping `fvd_window` = 16-frame windows starting at position 1. The
   default horizon of 64 gives 4 windows per view.
2. Every frame is resized to 224×224 with `cv2.INTER_AREA`; non-square grids are squashed.
3. Each window goes through the Kinetics-400 I3D (TATS / cd-fvd weights `i3d_pretrained_400.pt`) and is mapped to its
   400-d logits averaged over time.
4. Real and generated windows are taken at the same samples, views and positions.
5. The Fréchet distance is computed once per subset over the pooled window features:

   FVD = ‖μ_r − μ_g‖² + Tr(Σ_r + Σ_g − 2(Σ_r Σ_g)^½)

   The covariance is `np.cov` (unbiased) by default. `fvd_cov: biased` gives the 1/N estimate of cd-fvd.

FVD is strongly biased upwards for small window counts (400-d features). The window count n is given in the
per-subset footnote of `summary.md`, with a dagger when n ≤ 400, and as `fvd.n_real` / `n_fake` in `summary.json`.
Compare FVDs only at equal n.

<a id="met3r"></a>
## Cross-view MEt3R

[MEt3R](https://github.com/mohammadasim98/met3r) (Asim et al., CVPR 2025) scores an image pair (I₁, I₂):

1. MASt3R predicts both point maps in I₁'s camera.
2. FeatUp-upsampled DINO features of both images are attached to the points and rendered into I₁'s camera.
3. The score is the mean cosine dissimilarity over the pixels that both renderings cover. 0 means identical features;
   lower means more consistent.

VP-WorldBench uses the official implementation with its default settings: MASt3R, `dino16` + FeatUp JBU, cosine
distance, 256×256 inputs, naive rasterizer. Every pair is symmetrised: (S(I₁,I₂) + S(I₂,I₁))/2. Frames are centre-cropped
to square by default (`met3r_aspect`).

**Cross-view MEt3R of a clip.** All view pairs (a < b) are scored at the same timestep, at window positions
`t_ref + 4k` (k ≥ 1; `met3r_step`). With 64 frames this gives 16 timesteps × V(V−1)/2 pairs. The clip value m(c) is the
mean over pairs and timesteps. Position 0 is also computed and stored as row 0 of `cross`, but it enters neither m(c)
nor the score nor the floors.

**Why a distance to the real video and not raw MEt3R.** Raw MEt3R is not a quality ordering. Real multi-view pairs already score 0.4–0.6,
because wide baselines leave little overlap and DINO features change with viewpoint. Generated views that drift apart
overlap less, and low overlap is cheap, so a generator can score *below* the real video. The benchmark therefore
compares each generated clip with the real video of the same clip:

- per clip, the difference d(c) = m_gen(c) − m_real(c), computed after averaging inside the clip;
- **Cross-view MEt3R** = mean_c |d(c)|. This is the reported number; lower is better.

Also reported: the signed mean of d(c) (> 0 means less consistent than real), the share of clips with d(c) < 0, and
the raw means.

**Floor.** A generator exactly as consistent as the real video still scores above 0, because MEt3R of the same clip
fluctuates over time. Two floors are reported:

- **Floor:** take x(c, k), the real MEt3R averaged over pairs, and remove the subset-mean time profile. Per clip,
  compute |mean(first half) − mean(second half)| / √2. The floor is the mean over clips.
- **Noise floor:** the same with interleaved timesteps (even vs odd k) and no profile removal. This is the measurement
  noise alone.

Both floors depend on the clip set and are recomputed for every table. They measure how much the real video's own
MEt3R fluctuates over time on fixed frames. They do **not** measure input noise, which can be larger (next
paragraph). A score near the floor means "as consistent as the real video, up to its temporal fluctuation".

**Numerics and input sensitivity.** On identical input frames, clip means reproduce to about 1e-4 and single pairs to
about 1e-3, given the batching and TF32 settings. MEt3R is far more sensitive to its *input*:

- a 1–2 level change of the decoded pixels (a different YUV→RGB converter) moved per-clip means of generated clips by
  up to ~0.04;
- a near-lossless re-encode (H.264 CRF 4) moved them by about as much;
- single low-overlap pairs moved by up to ~0.45.

Per-clip shifts of this size are larger than the `vpm-v1` floors: floors 0.004–0.029 and noise floors
0.0008–0.008 on the full subsets. That is about 1.4× the floor on Simul-General and 4–10× on the other subsets.
Decode real and generated videos the same way, which the evaluator does, and submit lossless predictions
([PROTOCOL.md](PROTOCOL.md) §4). The installation recipe with pinned commits is
in `vpm_worldbench/metrics/met3r.py`.

<a id="delta"></a>
## δ: 3D motion accuracy of the generated video

The generated multi-view video is lifted to 3D point tracks. The tracks of the GT query points are compared with the
GT 3D tracks.

**Lifter.** The default lifter (`lifters/mvtracker.py`) works in four steps:

1. MoGe-2 ViT-L metric depth for every frame and view, with fov from the GT intrinsics.
2. **Anchoring** of the depth on window position 0 (t_ref) per view:
   - Simul-* and Real-General: scale + shift least-squares fit against the GT or reconstruction depth at t_ref.
   - Real-Robot, which has no dense depth: median depth ratio at the GT track points visible at t_ref.

   Only frame t_ref is used. Later frames carry no GT.
3. MVTracker-style scene normalisation computed from the *reference* depth, so it is identical for every model scored
   on a sample.
4. One [MVTracker](https://github.com/ethz-vlg/mvtracker) call (Rajić et al., ICCV 2025) with all queries, given the
   GT cameras of the window. The predictions are mapped back to the world frame.

The pinned installation recipe of the δ stack is in the docstring of `vpm_worldbench/lifters/mvtracker.py`: MVTracker
commit ceea8ad, MoGe 3.0.0 (commit 74fbce0), torch 2.10.0, Python ≥ 3.10. The default lifter fetches that MVTracker
commit through torch.hub (`--mvtracker-repo <clone>` on offline nodes). Every record stores fingerprints of the
weights it used (`models.delta.tracker_weights` / `moge_weights`; reference stack: 01943de6871bce32 / 2e283b2674fab4d4),
and `summarize` warns when the records of one table disagree.

**Queries.** All GT points with `default_gt` at t_ref that are visible in at least one view at t_ref, inside the eval
grid (`query_requires_visible`). The query position is the GT position at t_ref.

**GT visibility per frame.** `default_gt` AND visible in at least one scored view, inside its eval grid
(`gt_visibility: visible`, occlusion-aware; `in_frustum` ignores occlusion).

**Scores** follow MVTracker's `evaluate_predictions`, i.e. TAP-Vid metrics on metric 3D positions, averaged **per
track**. Tracks with fewer than 2 GT-visible frames from t_ref on (t_ref counts) are dropped. The scored entries of
a track are its GT-visible frames after t_ref.

| score | definition |
|---|---|
| δ@d | share of scored frames with 3D error < d |
| **δ_avg** | mean of δ@d over the subset's threshold ladder |
| **δ mov.** | δ_avg pooled over the **moving entries** only: scored frames where the GT point is at least the top ladder rung away from its position at t_ref (see below) |
| AJ | average Jaccard: TP / (scored + FP), where FP = predicted visible but GT occluded or error ≥ d, averaged over the ladder |
| OA | occlusion accuracy: share of frames after t_ref where predicted visibility = GT visibility |
| survival | share of the window until the first GT-visible frame (t_ref included) with error > `delta_survival_m` |
| ADE / MTE | mean / median 3D error over the scored entries of the kept tracks, pooled (metres; cm in `summary.md`) |
| FDE | mean over tracks of the error at the last scored frame (`summary.json` only) |

Ladders, report groups and survival thresholds per subset (`presets.py`):

| subset | ladder d (cm) | survival (m) | main group |
|---|---|---|---|
| Simul-Robot | 1, 2, 4, 8, 16 | 0.5 | `object` (the robot is reported separately) |
| Simul-General | 10, 20, 40, 80, 160 | 0.5 | `target` (the single followed agent) |
| Real-Robot | 1, 2, 5, 10, 20 | 0.10 | all (arms + hands) |
| Real-General | 5, 10, 20, 40 | 1.0 | all (person) |

Tracks are also split by the GT path length L, summed over the GT-visible positions from t_ref on: static (L < 1 cm)
and dynamic (L > 10 cm). The
**dynamic δ_avg** keeps the moving tracks, but still counts their frames before and after the motion, where a static
video is exact. δ mov. (below) isolates the motion.

**Moving δ.** δ_avg counts every scored frame of every query point. Many points barely move in the window, for
example objects that wait to be picked up. On those frames a video that does not move at all is exactly right, while
any real lifter makes errors of a few centimetres. δ_mov keeps only the entries where the GT point has moved at least
the top ladder rung (16 cm on Simul-Robot, 1.6 m on Simul-General, 20 cm on Real-Robot, 40 cm on Real-General) from
its query position, and pools them over the clip's tracks. There the frozen-points prediction is wrong at every
threshold, so it scores **0 by construction**. δ_mov therefore measures only the motion itself. Clips without such
entries have no δ_mov. The set is defined by the GT alone, so it is the same for every method.

On all 1,200 clips (`vpm-v1`, main point group; δ mov. on the 1,107 clips with moving points), the reference rows and
the best of the three baselines score:

| | Simul-Robot | Simul-General | Real-Robot | Real-General |
|---|---|---|---|---|
| δ_avg: real video / frozen points | 60.7 / **84.5** | 60.1 / 7.3 | 75.1 / 61.9 | 68.0 / 37.6 |
| δ_avg: best baseline | 45.3 | 25.2 | 37.2 | 31.9 |
| δ mov.: real video / frozen points | 22.0 / 0 | 55.7 / 0 | 48.2 / 0 | 48.0 / 0 |
| δ mov.: best baseline | 6.4 | 17.8 | 24.3 | 4.0 |

- Under δ_avg, frozen points beat the real video on Simul-Robot, and beat every model on Real-Robot and Real-General.
- Under δ mov., the real video is always the ceiling and frozen points the floor.

Report δ mov. next to δ_avg. Read δ_avg together with the frozen row.

**Reading δ.**

- δ measures whether the generated pixels move like the real scene when viewed by the real cameras. It needs
  view-consistent geometry and correct motion.
- The real-video row gives the lifter's own accuracy on real footage, which is the practical ceiling.
- The frozen row gives the δ_avg of a static video. A model near or below it has no usable motion as far as δ_avg can
  tell; δ mov. separates the two.
- The tracker amplifies tiny input differences. Per-clip δ moves by about 0.1–1.3 points under a decoder swap or a
  CRF-4 re-encode, and by up to ~7 points under a CRF-18 re-encode. Compare models on means over many clips, and
  submit lossless predictions.

## Using the metrics outside the evaluator

All metric functions take plain arrays:

```python
from vpm_worldbench.metrics.appearance import psnr_frames
from vpm_worldbench.metrics.fvd import I3D, fvd_windows, frechet
from vpm_worldbench.metrics.met3r import MEt3R, cross_view, delta_table
from vpm_worldbench.metrics.tracking import track3d_metrics, track2d_metrics
from vpm_worldbench.lifters.base import get_lifter
```

Their docstrings give shapes, units and the exact correspondence to the reference implementations.
