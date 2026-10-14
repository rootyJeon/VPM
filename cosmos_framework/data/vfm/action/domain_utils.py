# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Domain ID helpers for cross-embodiment action datasets."""

EMBODIMENT_TO_DOMAIN_ID: dict[str, int] = {
    "no_action": 0,
    "av": 1,
    "camera_pose": 2,
    "hand_pose": 3,
    "pusht": 4,
    "libero": 5,
    "umi": 6,
    "bridge_orig_lerobot": 7,
    "droid_lerobot": 8,
    "robomind-franka": 8,
    "robomind-franka-dual": 12,
    "robomind-ur": 13,
    "agibotworld": 15,
    "fractal": 20,
    "oxe/droid": 8,
    "oxe/teleop_london_tower": 8,
    "oxe/teleop_solve_equation": 8,
    "oxe/bridge": 7,
    "oxe/fractal20220817_data": 20,
    "oxe/taco_play": 9,
    "oxe/utaustin_mutex": 10,
    "oxe/stanford_hydra_dataset": 11,
    "oxe/berkeley_autolab_ur5": 14,
    "oxe/austin_sailor_dataset": 16,
    "oxe/austin_sirius_dataset": 17,
    "oxe/berkeley_fanuc_manipulation": 18,
    "oxe/jaco_play": 19,
    "oxe/fmb_dataset": 21,
    "oxe/kuka": 22,
    "oxe/berkeley_cable_routing": 23,
    "oxe/roboturk": 24,
    "oxe/dlr_edan_shared_control": 25,
    "oxe/austin_buds_dataset": 26,
    "oxe/nyu_franka_play_dataset": 27,
    "oxe/nyu_door_opening_surprising_effectiveness": 28,
    "oxe/cmu_stretch": 29,
    "oxe/furniture_bench_dataset": 30,
    "oxe/bc_z": 31,
    "oxe/language_table": 32,
    "oxe/shared": 33,
}


EMBODIMENT_TO_RAW_ACTION_DIM: dict[str, int] = {
    "av": 9,
    "camera_pose": 9,
    "pusht": 2,
    "umi": 10,
    "bridge_orig_lerobot": 10,
    "droid_lerobot": 10,
    "robomind-franka": 10,
    "robomind-franka-dual": 20,
    "robomind-ur": 10,
    "agibotworld": 29,
    "fractal": 10,
    "oxe": 7,
}


def get_domain_id(embodiment_type: str) -> int:
    """Get the domain ID for a given embodiment type."""
    key = embodiment_type.lower().strip()
    if key not in EMBODIMENT_TO_DOMAIN_ID:
        raise KeyError(
            f"Unknown embodiment type: {embodiment_type!r}. "
            f"Available embodiments: {sorted(EMBODIMENT_TO_DOMAIN_ID.keys())}"
        )
    return EMBODIMENT_TO_DOMAIN_ID[key]
