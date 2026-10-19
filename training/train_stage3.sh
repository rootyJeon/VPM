#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# VPM stage 3: joint video + robot action + point track on DROID. Warm start from stage 2; fully fine-tunes the
# generation tower, the action stream (action encoder/decoder, action embedding) and the tracking head with video
# flow matching, action flow matching and the track losses.
# Data slot: droid (needs droid/actions/<scene>.npz).
#
#   bash training/train_stage3.sh [hydra.key=value ...]
#
# Stage defaults (any exported value wins; full contract in training/_launch_common.sh):
#   BASE_CHECKPOINT_PATH=<latest vpm_stage2 checkpoint under OUTPUT_ROOT>
#   MAX_TOKENS=50000   MAX_SAMPLES_PER_PACK=6   LR=2.0e-5   ACTION_LR_MULT=5 (LR multiplier of the action stream)
#   MAX_ITER=5000   WARMUP_STEPS=100   SAVE_ITER=200

set -euo pipefail
# shellcheck source=training/_launch_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_launch_common.sh"

VPM_STAGE=3
VPM_EXP=vpm_stage3
VPM_PREV_EXP=vpm_stage2
VPM_SLOTS=(droid)

: "${RATIOS:=droid=1}"
: "${MAX_TOKENS:=50000}"
: "${MAX_SAMPLES_PER_PACK:=6}"
: "${LR:=2.0e-5}"
: "${ACTION_LR_MULT:=5}"
: "${MAX_ITER:=5000}"
: "${WARMUP_STEPS:=100}"
: "${SAVE_ITER:=200}"

[[ "$ACTION_LR_MULT" =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]] || vpm_die "ACTION_LR_MULT must be a number"
M="$ACTION_LR_MULT"
VPM_STAGE_OVERRIDES=("optimizer.lr_multipliers={action2llm:$M,llm2action:$M,action_modality_embed:$M}")

vpm_launch "$@"
