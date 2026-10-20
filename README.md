<p align="center">
  <img src="docs/vpm-icon.svg" alt="VPM" width="120">
</p>

<p align="center">
  <a href="https://rootyjeon.github.io/video-point-model/"><img src="https://img.shields.io/badge/Project-Page-1f6feb.svg?logo=googlechrome&logoColor=white" alt="Project page"></a>
  <a href=""><img src="https://img.shields.io/badge/VPM-Paper-red?logo=arxiv&logoColor=red" alt="VPM Paper on arXiv"/></a>
  <a href="https://huggingface.co/datasets/rooty2020/VPM-Data"><img src="https://img.shields.io/badge/VPM--Data-Dataset-yellow?style=flat&logo=huggingface" alt="VPM-Data"></a>
  <a href="https://huggingface.co/datasets/rooty2020/VP-WorldBench"><img src="https://img.shields.io/badge/VP--WorldBench-Benchmark-yellow?style=flat&logo=huggingface" alt="VP-WorldBench"></a>
</p>

# Learning the 4D World through Point Trajectory Forecasting

> [Byungwoo Jeon](https://rootyjeon.github.io/)<sup>1\*</sup>, [Sunghwan Hong](https://sunghwanhong.github.io/)<sup>2\*</sup>, [Dohyeon Kim](https://openreview.net/profile?id=~Dohyeon_Kim5)<sup>3</sup>, [Jonghoon Lee](https://jhoonjwa.github.io/)<sup>1</sup>, [Jungwoo Park](https://scholar.google.com/citations?user=k2e82kcAAAAJ&hl=en)<sup>1</sup>, [Inhee Lee](https://blog.sulwon.com/)<sup>3</sup>, [Doosan Baek](https://pslab.ai/en/)<sup>6</sup>, [Jinil Kim](https://pslab.ai/en/)<sup>6</sup>, [Taehyun Lim](https://pslab.ai/en/)<sup>6</sup>, [Hanbyul Joo](https://jhugestar.github.io/)<sup>3,5</sup>, [Marc Pollefeys](https://people.inf.ethz.ch/pomarc/)<sup>2</sup>, [Saining Xie](https://www.sainingxie.com/)<sup>4†</sup>, [Jinwoo Shin](https://alinlab.kaist.ac.kr/shin.html)<sup>1,5†</sup>
>
> <sup>1</sup>KAIST, <sup>2</sup>ETH Zürich, <sup>3</sup>Seoul National University, <sup>4</sup>New York University, <sup>5</sup>RLWRLD, <sup>6</sup>Physics Sim Lab<br>
> <sup>\*</sup>Equal contribution, <sup>†</sup>Equal advising
>
> We present **Video-Point Model (VPM)**, a multi-view video world model built on [NVIDIA Cosmos3](https://huggingface.co/nvidia/Cosmos3-Nano).
> - 🎥 **Multi-view Video**: Generating synchronized videos from multi-cameras.
> - 📍 **3D Point Tracks**: Consistent 3D points across views, read from its own attention features.
> - 🦾 **Robot action**: Predicting a robot action chunk together with the future frames.

## TODO
- [X] Release VPM-Data and VP-WorldBench
- [X] Release inference and training code
- [ ] Release VPM-Data with 1M samples (expected: Dec. 21)
- [ ] Release VPM trained on the Cosmos3-Super backbone (expected: Dec. 21)

## 🔧 Installation

```bash
git clone https://github.com/rootyJeon/VPM.git && cd VPM
sudo apt-get install -y ffmpeg
uv sync --extra train --group cu130-train      # CUDA 12.8: --group cu128-train
source .venv/bin/activate
```

## 📊 VP-WorldBench

VP-WorldBench is a **4D world modeling benchmark** for video world models: 1,200 synchronized multi-view clips with cameras,
3D point tracks and captions, in four subsets (simulated / real × robot / general). Its main task is multi-view
video generation, scored on appearance, cross-view consistency and 3D motion accuracy. VP-WorldBench can be used to evaluate diverse tasks such as 3D/4D reconstruction, human motion, and video understanding.

```bash
pip install -e "vp-worldbench[metrics,fvd]"
vpmwb generation export   --root data/VP-WorldBench --out inputs/
vpmwb generation evaluate --root data/VP-WorldBench --pred preds/my_model --out runs/my_model
```

Details for download, results, and the evaluation on other tasks are in [vp-worldbench/README.md](vp-worldbench/README.md).

## 🗂️ VPM-Data

VPM-Data is the training set of VPM: **10K simulated clips** that pair synchronized multi-view videos with persistent
3D point trajectories. It has 8,000 tabletop manipulation clips, in which a Fourier GR-1 T2 humanoid picks and places
objects among clutter, and 2,000 urban clips of pedestrians and vehicles filmed by moving cameras. 
We will release an extended version of VPM-Data with 1M samples, including real-world clips.

Every clip comes with

- 🎥 **multi-view videos**, time-synchronized across the cameras;
- 📷 **camera intrinsics and extrinsics** for every view and frame;
- 📍 **persistent 3D point trajectories** on the robot, the manipulated objects, people and vehicles, with per-view visibility;
- 🌊 **depth maps** rendered for every view and frame;
- 📝 **language captions**;
- 🏷️ **metadata**, such as object and actor labels, target visibility states (visible, occluded, out of frame),
  camera motion types, and scene and task attributes.

All annotations come from the simulator, so they are exact and aligned across views and time. This makes VPM-Data
useful well beyond world modeling, for tasks such as video generation, 3D/4D reconstruction, novel view synthesis,
depth estimation, 2D/3D point tracking, and video understanding.

The files and their formats are described on the [🤗 dataset card](https://huggingface.co/datasets/rooty2020/VPM-Data).

## 🦾 Robot Data with 3D Point Trajectories

We also release the robot manipulation data with 3D point trajectories that we built for VPM. Every frame comes with
persistent, metric 3D points on the robot arm and gripper (and the manipulated objects in simulation), computed from
the robot CAD model and the recorded robot or simulator states instead of a visual point tracker, together with
calibrated multi-view videos and language. DROID-VP keeps whole DROID episodes with the calibration and per-frame
poses of all three cameras, and RoboCasa episodes are cut into language-annotated subtask clips.

| Dataset | Source | Samples | Views | 3D points per frame | Size | Download |
|:--|:--|:--|:--|:--|:--|:--|
| DROID-VP | [DROID](https://droid-dataset.github.io/) | 50,430 episodes, 17,982 with point tracks | 3 × 320×180, 15 fps | 512 arm + 256 gripper | 1.03 TB | [🤗 DROID-VP](https://huggingface.co/datasets/rooty2020/DROID-VP) |
| LIBERO-VP | [LIBERO](https://libero-project.github.io/) | 6,500 episodes, 130 tasks | 3 × 256×256, 20 fps | 512 arm + 256 gripper + 256 object | 13.8 GB | [🤗 LIBERO-VP](https://huggingface.co/datasets/rooty2020/LIBERO-VP) |
| RoboCasa-VP | [RoboCasa365](https://robocasa.ai/) | 217,594 subtask clips from 24,504 episodes, 235 tasks | 3 × 256×256, 20 fps | 512 arm + 256 gripper + 256 per object | 487 GB | [🤗 RoboCasa-VP](https://huggingface.co/datasets/rooty2020/RoboCasa-VP) |

```bash
hf download rooty2020/DROID-VP    --repo-type dataset --local-dir data/DROID-VP
hf download rooty2020/LIBERO-VP   --repo-type dataset --local-dir data/LIBERO-VP
hf download rooty2020/RoboCasa-VP --repo-type dataset --local-dir data/RoboCasa-VP
```

LIBERO-VP and RoboCasa-VP ship as tar shards of self-contained samples (videos, `robot.npz` / `gripper.npz` /
`object*.npz` point tracks and a `meta.json` with cameras and language). DROID-VP keeps the layout of the VPM-DROID
training run (JPEG frame caches, joint / pose / camera tables and point caches) and comes with a reader,
`tools/read_episode.py`. To train on them, convert them to the slot layout in [training/DATA.md](training/DATA.md#preparing-your-own-data).

## 📦 Checkpoints

```bash
bash scripts/download_assets.sh             # Cosmos3-Nano, Wan2.2 VAE and the Qwen3-VL tokenizer
bash scripts/download_assets.sh --release   # + the VPM-DROID checkpoint
```

Cosmos3-Nano is converted to a DCP checkpoint for training, as in
[Cosmos-Framework](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/training.md#step-2--prepare-checkpoint).

| Model | Setup | Download |
|:--|:--|:--|
| VPM-DROID | Video + Action + Point tracks | [🤗 VPM-DROID](https://huggingface.co/rooty2020/VPM-DROID) |

## 🚀 Inference

VPM-DROID takes all three DROID cameras (exterior left, exterior right, wrist) and generates the next 16 frames of
every camera, a 16-step action chunk and, with `--tracks grid`, point tracks, from the last 9 frames and the actions
between them:

```bash
bash inference/infer_droid.sh \
    --past-frames past.npz --past-actions past_actions.npy \
    --state q0,q1,q2,q3,q4,q5,q6,gripper --prompt "put the marker in the cup"
```

Outputs (videos, `actions.npz`, optional point tracks) are written to `outputs/droid_inference/`.

Input formats, options and multi-GPU use are in [inference/README.md](inference/README.md).

## 🏋️ Training

VPM is trained from Cosmos3-Nano in three stages:

| Stage | Initialization | Trains |
|:--|:--|:--|
| **1**  | Cosmos3-Nano | video generator, camera conditioning |
| **2**  | stage 1 | tracking head |
| **3**  | stage 2 | video generator, action generator, tracking head |

```bash
bash scripts/download_assets.sh --data all   # VPM-Data
bash training/train_stage1.sh
bash training/train_stage2.sh
bash training/train_stage3.sh
```

Settings, multi-node runs, resuming and checkpoint export are in [training/README.md](training/README.md);
the data format is in [training/DATA.md](training/DATA.md).

## 🔗 Related Projects

- [Cosmos-Framework](https://github.com/NVIDIA/cosmos-framework) and [Cosmos 3](https://huggingface.co/collections/nvidia/cosmos3): the training framework and base model VPM builds on.
- [MVTracker](https://github.com/ethz-vlg/mvtracker): multi-view 3D point tracking, used by VP-WorldBench.
- [Wan2.2](https://github.com/Wan-Video/Wan2.2): the video VAE.
- [DROID](https://droid-dataset.github.io/): the robot manipulation dataset.

## 📜 License

The code and the VPM model weights are released under [CC BY-NC 4.0](LICENSE) (attribution: **Byungwoo Jeon**).
Non-commercial use only. For commercial licensing contact imbw2024@kaist.ac.kr.

**Built on NVIDIA Cosmos.** VPM is a derivative of [cosmos-framework](https://github.com/NVIDIA/cosmos-framework) and
the base model [nvidia/Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano) (Copyright (c) 2025-2026 NVIDIA
CORPORATION & AFFILIATES), which NVIDIA distributes under the OpenMDW License Agreement 1.1. As that agreement
requires, a copy is included as [LICENSE-OpenMDW-1.1.txt](LICENSE-OpenMDW-1.1.txt) and NVIDIA's copyright notices are
retained. Its terms apply to the Cosmos-derived portions, and the CC BY-NC 4.0 terms apply to VPM as released.
Other third-party code is listed in [NOTICE](NOTICE). Datasets follow the licenses on their Hugging Face cards.

## 📝 Citation

If you find **VPM**, its dataset, and the benchmark useful for your project, please consider giving this repo a 🌟 and cite our work using the following BibTeX:
```bibtex
@misc{vpm2026,
  title  = {Learning the 4D World through Point Trajectory Forecasting},
  author = {Jeon, Byungwoo and Hong, Sunghwan and Kim, Dohyeon and Lee, Jonghoon and Park, Jungwoo and Lee, Inhee and
            Baek, Doosan and Kim, Jinil and Lim, Taehyun and Joo, Hanbyul and Pollefeys, Marc and Xie, Saining and
            Shin, Jinwoo},
  year   = {2026},
  note   = {Preprint},
  url    = {https://rootyjeon.github.io/video-point-model/}
}
```

## 📞 Contact

For questions, issues, or collaborations, please contact [Byungwoo Jeon](https://rootyjeon.github.io/).<br>
If you are interested in receiving robot data similar to VPM, please contact [Physics Sim Lab](https://pslab.ai/en/).
