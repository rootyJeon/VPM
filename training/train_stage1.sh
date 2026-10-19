#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# VPM stage 1: multi-view world model. Warm start from Cosmos3-Nano; trains the generation tower, camera
# conditioning and view axis with the video flow-matching loss (the tracking head is attached with random weights, not trained).
# Data slots: psl4d, droid, hrdexdb, molmo.
#
#   bash training/train_stage1.sh [hydra.key=value ...]
#
# Stage defaults (any exported value wins; full contract in training/_launch_common.sh):
#   BASE_CHECKPOINT_PATH=checkpoints/Cosmos3-Nano
#   RATIOS="psl4d=3 droid=3 hrdexdb=1 molmo=1"      MAX_TOKENS=50000   LR=1.0e-4
#   MAX_ITER=2000   WARMUP_STEPS=50   SAVE_ITER=250

set -euo pipefail
# shellcheck source=training/_launch_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_launch_common.sh"

VPM_STAGE=1
VPM_EXP=vpm_stage1
VPM_SLOTS=(psl4d droid hrdexdb molmo)
VPM_STAGE_OVERRIDES=()

: "${BASE_CHECKPOINT_PATH:=checkpoints/Cosmos3-Nano}"
: "${RATIOS:=psl4d=3 droid=3 hrdexdb=1 molmo=1}"
: "${MAX_TOKENS:=50000}"
: "${LR:=1.0e-4}"
: "${MAX_ITER:=2000}"
: "${WARMUP_STEPS:=50}"
: "${SAVE_ITER:=250}"

vpm_launch "$@"
