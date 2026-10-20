#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# VPM stage 2: multi-view 3D point-track head. Warm start from stage 1; only the tracking head trains, with the
# track loss on the frozen backbone (the x5 head LR multiplier is part of the experiment).
# Data slots: vpm_data, droid, hrdexdb, molmo.
#
#   bash training/train_stage2.sh [hydra.key=value ...]
#
# Stage defaults (any exported value wins; full contract in training/_launch_common.sh):
#   BASE_CHECKPOINT_PATH=<latest vpm_stage1 checkpoint under OUTPUT_ROOT>
#   RATIOS="vpm_data=3 droid=3 hrdexdb=1 molmo=1"      MAX_TOKENS=55000   LR=2.0e-5
#   MAX_ITER=2000   WARMUP_STEPS=50   SAVE_ITER=250

set -euo pipefail
# shellcheck source=training/_launch_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_launch_common.sh"

VPM_STAGE=2
VPM_EXP=vpm_stage2
VPM_PREV_EXP=vpm_stage1
VPM_SLOTS=(vpm_data droid hrdexdb molmo)
VPM_STAGE_OVERRIDES=()

: "${RATIOS:=vpm_data=3 droid=3 hrdexdb=1 molmo=1}"
: "${MAX_TOKENS:=55000}"
: "${LR:=2.0e-5}"
: "${MAX_ITER:=2000}"
: "${WARMUP_STEPS:=50}"
: "${SAVE_ITER:=250}"

vpm_launch "$@"
