<h1 align="center">Learning the 4D World through Point Trajectory Forecasting</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2610.06940"><img src="https://img.shields.io/badge/arXiv-2610.06940-b31b1b.svg?logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://rootyjeon.github.io/video-point-model/"><img src="https://img.shields.io/badge/Project-Page-1f6feb.svg?logo=googlechrome&logoColor=white" alt="Project page"></a>
  <a href="https://huggingface.co/datasets/rooty2020/VPM-Data"><img src="https://img.shields.io/badge/VPM--Data-Dataset-FFD21E.svg?logo=huggingface&logoColor=000" alt="VPM-Data"></a>
  <a href="https://huggingface.co/datasets/rooty2020/VP-WorldBench"><img src="https://img.shields.io/badge/VP--WorldBench-Benchmark-FFD21E.svg?logo=huggingface&logoColor=000" alt="VP-WorldBench"></a>
</p>

<div align="center">
  <a href="https://rootyjeon.github.io/">Byungwoo&nbsp;Jeon</a><sup>1*</sup> &ensp;<b>&middot;</b>&ensp;
  <a href="https://sunghwanhong.github.io/">Sunghwan&nbsp;Hong</a><sup>2*</sup> &ensp;<b>&middot;</b>&ensp;
  <a href="https://openreview.net/profile?id=~Dohyeon_Kim5">Dohyeon&nbsp;Kim</a><sup>3</sup> &ensp;<b>&middot;</b>&ensp;
  <a href="https://jhoonjwa.github.io/">Jonghoon&nbsp;Lee</a><sup>1</sup> &ensp;<b>&middot;</b>&ensp;
  <a href="https://scholar.google.com/citations?user=k2e82kcAAAAJ&hl=en">Jungwoo&nbsp;Park</a><sup>1</sup> &ensp;<b>&middot;</b>&ensp;
  <a href="https://blog.sulwon.com/">Inhee&nbsp;Lee</a><sup>3</sup> &ensp;<b>&middot;</b>&ensp;
  <a href="https://pslab.ai/en/">Doosan&nbsp;Baek</a><sup>5</sup> &ensp;<b>&middot;</b>&ensp;
  <a href="https://pslab.ai/en/">Jinil&nbsp;Kim</a><sup>5</sup> &ensp;<b>&middot;</b>&ensp;
  <a href="https://pslab.ai/en/">Taehyun&nbsp;Lim</a><sup>5</sup> &ensp;<b>&middot;</b>&ensp;
  <a href="https://jhugestar.github.io/">Hanbyul&nbsp;Joo</a><sup>3,4</sup> &ensp;<b>&middot;</b>&ensp;
  <a href="https://people.inf.ethz.ch/pomarc/">Marc&nbsp;Pollefeys</a><sup>2</sup> &ensp;<b>&middot;</b>&ensp;
  <a href="https://www.sainingxie.com/">Saining&nbsp;Xie</a><sup>6&dagger;</sup> &ensp;<b>&middot;</b>&ensp;
  <a href="https://alinlab.kaist.ac.kr/shin.html">Jinwoo&nbsp;Shin</a><sup>1,4&dagger;</sup>
  <br>
  <br>
  <sup>1</sup>KAIST &emsp; <sup>2</sup>ETH Zürich &emsp; <sup>3</sup> Seoul National University &emsp; <sup>4</sup>RLWRLD &emsp; <sup>5</sup>Physics Sim Lab &emsp; <sup>6</sup>New York University
  <br>
  <sup>*</sup>Equal Contribution &emsp; <sup>&dagger;</sup>Equal Advising &emsp;<br>
</div>

<h3 align="center">[<a href="https://rootyjeon.github.io/video-point-model/">project page</a>]&emsp;[<a href="https://arxiv.org/abs/2410.06940">arXiv</a>]</h3>

<br>

**Video-Point Model (VPM)** is a multi-view video world model built on [NVIDIA Cosmos3](https://huggingface.co/nvidia/Cosmos3-Nano).

- 🎥 **Multi-view Video**: Generating synchronized videos from multi-cameras.
- 📍 **3D Point Tracks**: Consistent 3D points across views, read from its own attention features.
- 🦾 **Robot action**: Predicting a robot action chunk together with the future frames.

## TODO
- [X] Release VPM-Data and VP-WorldBench
- [X] Release inference code
- [X] Release training code
- [ ] Release extended VPM-Data with 1M samples (expected: 12/21)

## 🔧 Installation

```bash
git clone https://github.com/vpm-project/VPM.git && cd VPM
sudo apt-get install -y ffmpeg
uv sync --extra train --group cu130-train      # CUDA 12.8: --group cu128-train
source .venv/bin/activate
```

## 📊 VP-WorldBench

VP-WorldBench is a 4D world modeling benchmark for video world models: 1,200 synchronized multi-view clips with cameras,
3D point tracks and captions, in four subsets (simulated / real × robot / general). Its main task is multi-view
video generation, scored on appearance, cross-view consistency and 3D motion accuracy. VP-WorldBench can be used to evaluate diverse tasks such as 3D/4D reconstruction, human motion, and video understanding.

```bash
pip install -e "vp-worldbench[metrics,fvd]"
vpmwb generation export   --root data/VP-WorldBench --out inputs/
vpmwb generation evaluate --root data/VP-WorldBench --pred preds/my_model --out runs/my_model
```

Details for download, results, and the evaluation on other tasks are in [vp-worldbench/README.md](vp-worldbench/README.md).

## 🗂️ VPM-Data

VPM-Data is the training set of VPM: **10K simulated clips** that pair synchronized multi-view videos with persistent 3D point trajectories. It has 8,000 tabletop manipulation clips and 2,000 urban clips of pedestrians and vehicles captured by moving cameras. Each views are captured from different initial frames.

Every clip comes with 

- 🎥 **Multi-view videos** are time-synchronized across the cameras
- 📷 **Camera intrinsic / extrinsic** for every view and frame
- 📍 **3D point tracks** on the robot, the manipulated objects, people and vehicles
- 🌊 **Depth maps** rendered for every view and frame
- 📝 **Language caption** for each samples

You can use VPM-Data for diverse tasks such as video generation, 3D/4D reconstruction, depth estimation, human motion & pose estimation, and video reasoning.
Data and their formats are described on the [🤗 dataset card](https://huggingface.co/datasets/rooty2020/VPM-Data).
We plan to release extended version of VPM-Data (1M) captured from both simulation and real-world.

## 🦾 Robot Data with 3D Point Trajectories

We also release the robot manipulation data with 3D point trajectories that we built for VPM. Every frame comes with persistent, metric 3D points on the robot arm and gripper (and the manipulated objects in simulation), computed from the robot CAD model and the recorded robot or simulator states, together with calibrated multi-view videos and language. DROID and RoboCasa episodes are cut into language-annotated subtask clips, and real DROID episodes are kept only when the projected robot model matches the robot in the image.

| dataset | source | samples | views | 3D points per frame | size | download |
|:--|:--|:--|:--|:--|:--|:--|
| DROID-VP | [DROID](https://droid-dataset.github.io/) | 89,312 subtask clips from 28,308 episodes | 2 × 1280×720, 15 fps | 512 arm + 256 gripper | 327 GB | [🤗 DROID-VP](https://huggingface.co/datasets/rooty2020/DROID-VP) |
| LIBERO-VP | [LIBERO](https://libero-project.github.io/) | 6,500 episodes, 130 tasks | 3 × 256×256, 20 fps | 512 arm + 256 gripper + 256 object | 13.8 GB | [🤗 LIBERO-VP](https://huggingface.co/datasets/rooty2020/LIBERO-VP) |
| RoboCasa-VP | [RoboCasa365](https://robocasa.ai/) | 217,594 subtask clips from 24,504 episodes, 235 tasks | 3 × 256×256, 20 fps | 512 arm + 256 gripper + 256 per object | 487 GB | [🤗 RoboCasa-VP](https://huggingface.co/datasets/rooty2020/RoboCasa-VP) |

```bash
hf download rooty2020/DROID-VP    --repo-type dataset --local-dir data/DROID-VP
hf download rooty2020/LIBERO-VP   --repo-type dataset --local-dir data/LIBERO-VP
hf download rooty2020/RoboCasa-VP --repo-type dataset --local-dir data/RoboCasa-VP
```

To train on the robot data, convert them to the slot layout in [training/DATA.md](training/DATA.md#preparing-your-own-data).

## 📦 Checkpoints

```bash
bash scripts/download_assets.sh             # Cosmos3-Nano, Wan2.2 VAE and the Qwen3-VL tokenizer
bash scripts/download_assets.sh --release   # + the VPM-DROID checkpoint
```

Cosmos3-Nano is converted to a DCP checkpoint for training, as in
[Cosmos-Framework](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/training.md#step-2--prepare-checkpoint).

| model | setup | download |
|:--|:--|:--|
| VPM-DROID | Video + Trajectory + Action | [🤗 VPM-DROID](https://huggingface.co/rooty2020/VPM-DROID) |

## 🚀 Inference

Generate the next 32 frames of both DROID views and the action chunk from the current observation:

```bash
bash inference/infer_droid.sh \
    --images view1.png,view2.png --cameras cameras.json \
    --state q0,q1,q2,q3,q4,q5,q6,gripper --prompt "put the marker in the cup" \
    --tracks grid
```

Outputs (videos, `actions.npz`, optional 3D point tracks) are written to `outputs/droid_inference/`.
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
@article{vpm2026,
  title   = {Learning the 4D World through Point Trajectory Forecasting},
  author  = {Author One and Author Two and Author Three},
  journal = {arXiv preprint arXiv:2610.06940},
  year    = {2026}
}
```

## 📞 Contact

For questions, issues, or collaborations, please contact [Byungwoo Jeon](https://rootyjeon.github.io/).<br>
If you are interested in receiving robot data similar to VPM, please contact [Physics Sim Lab](https://pslab.ai/en/).
