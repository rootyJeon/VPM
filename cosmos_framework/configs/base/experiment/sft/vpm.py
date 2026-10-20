# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""VPM training recipes.

All stages share the Cosmos3-Nano backbone with multi-view view-axis packing, diffusion forcing over latent frames,
Plucker/RoCE camera conditioning and a multi-view 3D point-tracking head tapped from layer 18.

- ``vpm_stage1``: multi-view world model. Generation tower, camera conditioning and view axis train with the
  video flow-matching loss; the tracking head is attached but not trained. Warm start: Cosmos3-Nano.
- ``vpm_stage2``: point-track head. Only the tracking head trains, with the 3D track loss on the frozen backbone.
  Warm start: stage 1.
- ``vpm_stage3``: full fine-tuning of the video stream (generation tower), the action stream (action encoder/decoder
  and action embedding) and the point-track head on DROID, with the video flow-matching, action flow-matching, track
  and correspondence losses. Warm start: stage 2.
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.sft.vision_sft_nano import vision_sft_nano
from cosmos_framework.data.vfm.local_datasets.droid_action_dataset import get_droid_action_dataset
from cosmos_framework.data.vfm.local_datasets.multiview_track_dataset import get_multiview_track_dataset

cs = ConfigStore.instance()

DATA_ROOT = "${oc.env:VPM_DATA_ROOT,data}"
PAST_LATENTS_I2V_V2V = [1] * 16 + [2, 3, 4, 5]
NUM_TRACK_POINTS = 128

VIDEO_STREAM = [
    "moe_gen",
    "time_embedder",
    "vae2llm",
    "llm2vae",
    "action2llm",
    "action_modality_embed",
    "view_inv_freq",
    "camera_conditioner",
    "camera_roce",
]
ACTION_STREAM = ["action2llm", "llm2action", "action_modality_embed"]

SLOTS = {
    "vpm_data": dict(num_views=4, resolution="480sq", future_camera="full", random_anchor=False, min_window_frames=33),
    "droid": dict(num_views=2, resolution="480", future_camera="hold", random_anchor=False, min_window_frames=33),
    "hrdexdb": dict(num_views=1, resolution="480", future_camera="hold", random_anchor=True, min_window_frames=33),
    "molmo": dict(num_views=1, resolution="480sq", future_camera="full", random_anchor=False, min_window_frames=21),
}
RATIOS = {"vpm_data": 3, "droid": 3, "hrdexdb": 1, "molmo": 1}


def _jsonl(slot: str) -> list:
    return [f"{DATA_ROOT}/{slot}/train/video_dataset_file.jsonl"]


def _slot(base_dataset: dict, slot: str, **dataset_overrides) -> dict:
    ds = copy.deepcopy(base_dataset)
    ds["_target_"] = get_multiview_track_dataset
    ds["jsonl_paths"] = _jsonl(slot)
    ds.update(SLOTS[slot])
    ds.update(dataset_overrides)
    return dict(ratio=RATIOS[slot], dataset=ds)


vpm_base = copy.deepcopy(vision_sft_nano)
vpm_base["job"]["name"] = "vpm_base"

_mc = vpm_base["model"]["config"]
_mc["causal_training_strategy"] = "diffusion_forcing"
_mc["camera_conditioning_enabled"] = True
_mc["camera_roce_mode"] = "qk_vo"
_mc["max_num_tokens_after_packing"] = 50000
_mc["diffusion_expert_config"]["max_vae_latent_side_after_patchify"] = 26
_mc["tokenizer"]["vae_path"] = "${oc.env:WAN_VAE_PATH,checkpoints/wan22_vae/Wan2.2_VAE.pth}"

_rf = _mc["rectified_flow_training_config"]
_rf["normalize_loss_by_active"] = True
_rf["cond_noise_aug_std"] = 0.0
_rf["student_forcing_enabled"] = False
_rf["variable_future_enabled"] = False
_rf["tracking_enabled"] = True
_rf["tracking_layer"] = 18
_rf["tracking_weight"] = 0.1
_rf["tracking_corr_weight"] = 0.01
_rf["tracking_seq_cap"] = 50.0
_rf["tracking_seq_sigma_weight"] = False
_rf["tracking_min_valid_points"] = 8
_rf["tracking_camera_enabled"] = True
_rf["tracking_depth_enabled"] = True
_rf["tracking_upsample"] = 4
_rf["tracking_num_points"] = NUM_TRACK_POINTS
_rf["tracking_iters"] = 4
_rf["tracking_corr_radius"] = 3
_rf["tracking_corr_levels"] = 4
_rf["tracking_num_virtual_tracks"] = 64
_rf["tracking_use_checkpoint"] = True
_rf["tracking_motion_weight_enabled"] = False
_rf["tracking_vel_weight"] = 0.1
_rf["tracking_feedback_enabled"] = False
_rf["tracking_freeze_backbone"] = True
_rf["tracking_trainable_substrings"] = list(VIDEO_STREAM)

vpm_base["optimizer"]["keys_to_select"] = list(VIDEO_STREAM)
vpm_base["optimizer"]["lr"] = 1.0e-4
vpm_base["checkpoint"]["strict_resume"] = False
vpm_base["checkpoint"]["load_path"] = "${oc.env:BASE_CHECKPOINT_PATH}"
vpm_base["checkpoint"]["keys_to_skip_loading"] = ["net_ema."]
vpm_base["checkpoint"]["save_iter"] = 250
vpm_base["trainer"]["max_iter"] = 2000
vpm_base["scheduler"]["cycle_lengths"] = [2000]
vpm_base["scheduler"]["warm_up_steps"] = [50]

_dl = vpm_base["dataloader_train"]
_dl["max_sequence_length"] = 50000
_dl["max_samples_per_batch"] = 6
_dl["max_samples_per_batch_by_dataset"] = {}
_dl["dataloader"]["num_workers"] = 2
_dl["dataloader"]["prefetch_factor"] = 2

_video = _dl["dataloader"]["datasets"]["video"]["dataset"]
_video["conditioning_config"] = {1: 1.0}
_video["num_video_frames"] = -1
_video["frame_selection_mode"] = "random"
_video["append_duration_fps_timestamps"] = False
_video["append_resolution_info"] = False
_video["num_views"] = 2
_video["num_future_latents"] = 4
_video["translation_scale"] = 1.0
_video["ref_translation_scale"] = 1.0
_video["max_action_dim"] = 64
_video["num_track_points"] = NUM_TRACK_POINTS
_video["variable_past_latents"] = None
_video["variable_past_frac"] = [0.15, 0.30]
_video["future_camera"] = "hold"
_video["random_anchor"] = False
_video["min_window_frames"] = 33
_BASE_DATASET = copy.deepcopy(_video)
_dl["dataloader"]["datasets"] = {slot: _slot(_BASE_DATASET, slot) for slot in SLOTS}


vpm_stage1 = copy.deepcopy(vpm_base)
vpm_stage1["job"]["name"] = "vpm_stage1"
_rf = vpm_stage1["model"]["config"]["rectified_flow_training_config"]
_rf["tracking_weight"] = 0.0
_rf["tracking_corr_weight"] = 0.0


vpm_stage2 = copy.deepcopy(vpm_base)
vpm_stage2["job"]["name"] = "vpm_stage2"
vpm_stage2["model"]["config"]["max_num_tokens_after_packing"] = 55000
vpm_stage2["dataloader_train"]["max_sequence_length"] = 55000
vpm_stage2["optimizer"]["keys_to_select"] = ["tracking_head"]
vpm_stage2["optimizer"]["lr_multipliers"] = {"tracking_head": 5.0}
vpm_stage2["optimizer"]["lr"] = 2.0e-5
_rf = vpm_stage2["model"]["config"]["rectified_flow_training_config"]
_rf["loss_scale"] = 0.0
_rf["action_loss_weight"] = 0.0
_rf["tracking_weight"] = 1.0
_rf["tracking_corr_weight"] = 0.0
_rf["tracking_seq_cap"] = 30.0
_rf["tracking_seq_sigma_weight"] = True
_rf["tracking_trainable_substrings"] = []


vpm_stage3 = copy.deepcopy(vpm_base)
vpm_stage3["job"]["name"] = "vpm_stage3"
_rf = vpm_stage3["model"]["config"]["rectified_flow_training_config"]
_rf["independent_action_schedule"] = True
_rf["shift_action"] = 5
_rf["use_high_sigma_strategy_action"] = False
_rf["action_loss_weight"] = 10.0
_rf["tracking_trainable_substrings"] = list(VIDEO_STREAM) + ["llm2action"]
vpm_stage3["optimizer"]["keys_to_select"] = list(VIDEO_STREAM) + ["llm2action", "tracking_head"]
vpm_stage3["optimizer"]["lr_multipliers"] = {key: 5.0 for key in ACTION_STREAM}
vpm_stage3["optimizer"]["lr"] = 2.0e-5
vpm_stage3["trainer"]["max_iter"] = 5000
vpm_stage3["scheduler"]["cycle_lengths"] = [5000]
vpm_stage3["scheduler"]["warm_up_steps"] = [100]
vpm_stage3["checkpoint"]["save_iter"] = 200
_droid = _slot(
    _BASE_DATASET,
    "droid",
    variable_past_latents=list(PAST_LATENTS_I2V_V2V),
    variable_past_frac=None,
    action_chunk=32,
    gripper_flip=True,
)
_droid["dataset"]["_target_"] = get_droid_action_dataset
_droid["ratio"] = 1
vpm_stage3["dataloader_train"]["dataloader"]["datasets"] = {"droid": _droid}
vpm_stage3["dataloader_train"]["dataloader"]["num_workers"] = 6


for _name, _node in (("vpm_stage1", vpm_stage1), ("vpm_stage2", vpm_stage2), ("vpm_stage3", vpm_stage3)):
    cs.store(group="experiment", package="_global_", name=_name, node=_node)
