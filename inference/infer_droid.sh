#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# DROID inference with a VPM stage-3 checkpoint (torchrun wrapper of inference/inference_droid.py).
#
#   # an episode in the VPM data layout (clean past ends at frame 120), with point tracks
#   bash inference/infer_droid.sh --episode <scene> --frame 120 --tracks grid
#   # raw arrays
#   bash inference/infer_droid.sh --images view1.png,view2.png --cameras cameras.json \
#       --state q0,q1,q2,q3,q4,q5,q6,gripper --prompt "put the marker in the cup"
#
# All arguments go to inference/inference_droid.py (see --help); relative paths in them are resolved against the
# current directory, relative paths in the environment variables against the repository root. Environment:
#   VPM_CHECKPOINT   release dir, training iter_* dir or Hugging Face repo id (default: vpm-project/VPM-DROID)
#   VPM_DATA_ROOT    dataset root; episodes are read from $VPM_DATA_ROOT/droid (default: data)
#   WAN_VAE_PATH     Wan2.2 VAE weights (default: checkpoints/wan22_vae/Wan2.2_VAE.pth, else downloaded)
#   OUTPUT_ROOT      outputs go to $OUTPUT_ROOT/droid_inference/<sample> (default: outputs)
#   NPROC_PER_NODE   GPUs; the model is sharded over them and episodes run one per GPU (default: 1)
#   HF_HOME          Hugging Face cache (Qwen3-VL tokenizer, downloaded checkpoint and VAE)
#   VPM_PYTHON       interpreter (default: $VIRTUAL_ENV/bin/python, else .venv/bin/python, else python)

set -euo pipefail
# shellcheck source=training/_launch_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/../training/_launch_common.sh"

: "${VPM_CHECKPOINT:=vpm-project/VPM-DROID}"
: "${VPM_DATA_ROOT:=data}"
: "${OUTPUT_ROOT:=outputs}"
: "${NPROC_PER_NODE:=1}"

VPM_DATA_ROOT="$(vpm_abspath "$VPM_DATA_ROOT")"
OUTPUT_ROOT="$(vpm_abspath "$OUTPUT_ROOT")"
if [[ -n "${WAN_VAE_PATH:-}" ]]; then
    WAN_VAE_PATH="$(vpm_abspath "$WAN_VAE_PATH")"
    [[ -f "$WAN_VAE_PATH" ]] || vpm_die "WAN_VAE_PATH=$WAN_VAE_PATH does not exist"
elif [[ -f "$VPM_ROOT/checkpoints/wan22_vae/Wan2.2_VAE.pth" ]]; then
    WAN_VAE_PATH="$VPM_ROOT/checkpoints/wan22_vae/Wan2.2_VAE.pth"
else
    WAN_VAE_PATH=""
fi
if [[ "$VPM_CHECKPOINT" == /* || "$VPM_CHECKPOINT" == .* || -e "$VPM_ROOT/$VPM_CHECKPOINT" ]]; then
    VPM_CHECKPOINT="$(vpm_abspath "$VPM_CHECKPOINT")"
fi
export VPM_CHECKPOINT VPM_DATA_ROOT WAN_VAE_PATH OUTPUT_ROOT
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$VPM_ROOT${PYTHONPATH:+:$PYTHONPATH}"

python="${VPM_PYTHON:-}"
if [[ -z "$python" ]]; then
    if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
        python="$VIRTUAL_ENV/bin/python"
    elif [[ -x "$VPM_ROOT/.venv/bin/python" ]]; then
        python="$VPM_ROOT/.venv/bin/python"
    else
        python="python"
    fi
fi
venv_dir="$(cd "$(dirname "$(command -v "$python" || echo "$python")")/.." 2>/dev/null && pwd || true)"
for cu_lib in "$venv_dir"/lib/python3*/site-packages/nvidia/cu13/lib; do
    [[ -d "$cu_lib" ]] && export LD_LIBRARY_PATH="$cu_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    break
done

if [[ " $* " == *" --episode "* || " $* " == *" --episode="* ]]; then
    for tool in ffmpeg ffprobe; do
        command -v "$tool" >/dev/null 2>&1 || vpm_die "$tool not on PATH (needed to decode the episode videos)"
    done
fi

vpm_log "DROID inference: checkpoint $VPM_CHECKPOINT, $NPROC_PER_NODE GPU(s), output $OUTPUT_ROOT/droid_inference"
exec "$python" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
    "$VPM_ROOT/inference/inference_droid.py" --ckpt "$VPM_CHECKPOINT" "$@"
