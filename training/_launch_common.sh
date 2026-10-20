# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# shellcheck shell=bash
#
# Shared launcher for training/train_stage{1,2,3}.sh. Source it, do not run it.
#
# The stage script sets VPM_STAGE, VPM_EXP (Hydra experiment = job name), VPM_SLOTS (array of data slots),
# optionally VPM_PREV_EXP (experiment whose latest checkpoint is the default warm start), VPM_STAGE_OVERRIDES
# (array of Hydra overrides), and the stage
# defaults of the variables below with the ${VAR:=default} idiom, so values exported by the caller win.
# Then it calls `vpm_launch "$@"`; extra arguments are appended as Hydra overrides.
#
# Environment contract (relative paths are resolved against the repository root):
#   VPM_DATA_ROOT          dataset root holding one directory per slot (default: data)
#   BASE_CHECKPOINT_PATH   warm-start checkpoint, a directory with a model/ DCP subdirectory (ignored on resume)
#   WAN_VAE_PATH           Wan2.2 VAE weights (default: checkpoints/wan22_vae/Wan2.2_VAE.pth)
#   OUTPUT_ROOT            run directory root (default: outputs); exported as IMAGINAIRE_OUTPUT_ROOT
#   NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT   torchrun topology
#   MAX_TOKENS MAX_SAMPLES_PER_PACK LR MAX_ITER SAVE_ITER WARMUP_STEPS CYCLE_LEN   training knobs
#   RATIOS                 per-slot GPU ratios, e.g. "vpm_data=3 droid=3 hrdexdb=1 molmo=1"; a slot that is
#                          omitted or set to 0 is not trained on
#   JSONL_<SLOT>           optional comma-separated jsonl list replacing <slot>/train/video_dataset_file.jsonl
#   WANDB_MODE             online | offline | disabled (default: disabled)
#   HF_HOME                Hugging Face cache (Qwen3-VL tokenizer)
#   KEEP_CHECKPOINTS       keep only the newest N iter_* directories (default: 0 = keep all)
#   VPM_PYTHON             interpreter (default: $VIRTUAL_ENV/bin/python, else .venv/bin/python, else python)
#   DRY_RUN=1              run the checks, print the torchrun command and exit

VPM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VPM_ROOT

vpm_log() { echo ">>> [$(date '+%F %T')] $*"; }
vpm_die() { echo "ERROR: $*" >&2; exit 1; }

vpm_abspath() {
    local p="$1"
    if [[ -z "$p" || "$p" == /* ]]; then
        printf '%s' "$p"
    else
        printf '%s/%s' "$VPM_ROOT" "${p#./}"
    fi
}

vpm_quote() {
    [[ "$1" != *"'"* ]] || vpm_die "single quotes are not supported in paths: $1"
    printf "'%s'" "$1"
}

vpm_latest_iter() {
    local ckpt_dir="$OUTPUT_ROOT/cosmos3/sft/$1/checkpoints"
    [[ -f "$ckpt_dir/latest_checkpoint.txt" ]] || return 0
    printf '%s/%s' "$ckpt_dir" "$(tr -d '[:space:]' <"$ckpt_dir/latest_checkpoint.txt")"
}

vpm_count_gpus() {
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        local -a ids
        IFS=',' read -r -a ids <<<"$CUDA_VISIBLE_DEVICES"
        echo "${#ids[@]}"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true
    else
        echo 0
    fi
}

vpm_prune_checkpoints() {
    local ckpt_dir="$1" keep="$2" latest d
    local -a all
    while :; do
        sleep 300
        [[ -d "$ckpt_dir" ]] || continue
        latest="$(tr -d '[:space:]' <"$ckpt_dir/latest_checkpoint.txt" 2>/dev/null || true)"
        mapfile -t all < <(find "$ckpt_dir" -maxdepth 1 -type d -name 'iter_*' | sort)
        (( ${#all[@]} > keep )) || continue
        for d in "${all[@]:0:${#all[@]}-keep}"; do
            [[ "$(basename "$d")" == "$latest" ]] && continue
            [[ -n "$(find "$d" -maxdepth 0 -mmin +10 2>/dev/null)" ]] || continue
            vpm_log "pruning old checkpoint $(basename "$d")"
            rm -rf "$d"
        done
    done
}

vpm_launch() {
    set -euo pipefail
    : "${VPM_STAGE:?}" "${VPM_EXP:?}"
    [[ ${VPM_SLOTS+x} ]] || vpm_die "VPM_SLOTS is not set"
    [[ ${VPM_STAGE_OVERRIDES+x} ]] || VPM_STAGE_OVERRIDES=()

    : "${VPM_DATA_ROOT:=data}"
    : "${WAN_VAE_PATH:=checkpoints/wan22_vae/Wan2.2_VAE.pth}"
    : "${OUTPUT_ROOT:=outputs}"
    : "${NNODES:=1}"
    : "${NODE_RANK:=0}"
    : "${MASTER_ADDR:=127.0.0.1}"
    : "${MASTER_PORT:=29500}"
    : "${WANDB_MODE:=disabled}"
    : "${KEEP_CHECKPOINTS:=0}"
    : "${DRY_RUN:=0}"
    : "${RATIOS:?}" "${MAX_TOKENS:?}" "${LR:?}" "${MAX_ITER:?}" "${SAVE_ITER:?}" "${WARMUP_STEPS:?}"
    : "${CYCLE_LEN:=$MAX_ITER}"

    VPM_DATA_ROOT="$(vpm_abspath "$VPM_DATA_ROOT")"
    WAN_VAE_PATH="$(vpm_abspath "$WAN_VAE_PATH")"
    OUTPUT_ROOT="$(vpm_abspath "$OUTPUT_ROOT")"
    if [[ -z "${BASE_CHECKPOINT_PATH:-}" && -n "${VPM_PREV_EXP:-}" ]]; then
        BASE_CHECKPOINT_PATH="$(vpm_latest_iter "$VPM_PREV_EXP")"
    fi
    BASE_CHECKPOINT_PATH="$(vpm_abspath "${BASE_CHECKPOINT_PATH:-}")"
    export VPM_DATA_ROOT WAN_VAE_PATH OUTPUT_ROOT BASE_CHECKPOINT_PATH WANDB_MODE
    export IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT"
    export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
    export PYTHONPATH="$VPM_ROOT${PYTHONPATH:+:$PYTHONPATH}"

    local python="${VPM_PYTHON:-}"
    if [[ -z "$python" ]]; then
        if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
            python="$VIRTUAL_ENV/bin/python"
        elif [[ -x "$VPM_ROOT/.venv/bin/python" ]]; then
            python="$VPM_ROOT/.venv/bin/python"
        else
            python="python"
        fi
    fi
    local venv_dir cu_lib
    venv_dir="$(cd "$(dirname "$(command -v "$python" || echo "$python")")/.." 2>/dev/null && pwd || true)"
    for cu_lib in "$venv_dir"/lib/python3*/site-packages/nvidia/cu13/lib; do
        [[ -d "$cu_lib" ]] && export LD_LIBRARY_PATH="$cu_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        break
    done

    if [[ -z "${NPROC_PER_NODE:-}" ]]; then
        NPROC_PER_NODE="$(vpm_count_gpus)"
        if (( NPROC_PER_NODE == 0 )); then
            [[ "$DRY_RUN" == "1" ]] || vpm_die "no GPU found; set NPROC_PER_NODE"
            NPROC_PER_NODE=8
        fi
    fi

    local toml="$VPM_ROOT/examples/toml/$VPM_EXP.toml"
    local ckpt_dir="$OUTPUT_ROOT/cosmos3/sft/$VPM_EXP/checkpoints"
    local resuming=0
    [[ -f "$ckpt_dir/latest_checkpoint.txt" ]] && resuming=1

    declare -A ratio=()
    local slot kv key val
    for slot in "${VPM_SLOTS[@]}"; do ratio[$slot]=0; done
    # shellcheck disable=SC2086
    for kv in ${RATIOS:-}; do
        key="${kv%%=*}"
        val="${kv#*=}"
        [[ -n "${ratio[$key]+x}" ]] || vpm_die "RATIOS: unknown slot '$key' (stage $VPM_STAGE slots: ${VPM_SLOTS[*]})"
        [[ "$val" =~ ^[0-9]+$ ]] || vpm_die "RATIOS: '$kv' must be <slot>=<non-negative integer>"
        ratio[$key]="$val"
    done

    local fail=0
    vpm_check() {
        if [[ ! -e "$2" ]]; then
            echo "MISSING $1: $2" >&2
            echo "        -> $3" >&2
            fail=1
        fi
    }
    vpm_check "experiment TOML" "$toml" "examples/toml/$VPM_EXP.toml ships with the repository"
    vpm_check "Wan2.2 VAE" "$WAN_VAE_PATH" "bash scripts/download_assets.sh, or export WAN_VAE_PATH=<file>"
    if (( resuming == 0 )); then
        if [[ -z "$BASE_CHECKPOINT_PATH" ]]; then
            echo "MISSING warm-start checkpoint: BASE_CHECKPOINT_PATH is not set and no ${VPM_PREV_EXP:-previous-stage}" \
                "checkpoint was found under $OUTPUT_ROOT/cosmos3/sft" >&2
            echo "        -> run the previous stage with the same OUTPUT_ROOT, or export BASE_CHECKPOINT_PATH=<iter dir>" >&2
            fail=1
        else
            vpm_check "warm-start checkpoint (DCP)" "$BASE_CHECKPOINT_PATH/model/.metadata" \
                "export BASE_CHECKPOINT_PATH=<dir whose model/ is a torch.distributed.checkpoint (DCP) directory> (see training/README.md)"
        fi
    fi
    local -a active=()
    local -a data_overrides=()
    local jsonl_var jsonl_list path quoted
    for slot in "${VPM_SLOTS[@]}"; do
        data_overrides+=("dataloader_train.dataloader.datasets.$slot.ratio=${ratio[$slot]}")
        (( ratio[$slot] > 0 )) || continue
        active+=("$slot")
        jsonl_var="JSONL_${slot^^}"
        if [[ -n "${!jsonl_var:-}" ]]; then
            quoted=""
            IFS=',' read -r -a jsonl_list <<<"${!jsonl_var}"
            for path in "${jsonl_list[@]}"; do
                path="$(vpm_abspath "$path")"
                vpm_check "$slot jsonl" "$path" "fix $jsonl_var"
                quoted+="${quoted:+,}$(vpm_quote "$path")"
            done
            data_overrides+=("dataloader_train.dataloader.datasets.$slot.dataset.jsonl_paths=[$quoted]")
        else
            vpm_check "$slot jsonl" "$VPM_DATA_ROOT/$slot/train/video_dataset_file.jsonl" \
                "prepare the data (training/DATA.md), set VPM_DATA_ROOT, or drop the slot with RATIOS=\"$slot=0 ...\""
        fi
    done
    if (( VPM_STAGE == 3 )); then
        vpm_check "DROID action labels" "$VPM_DATA_ROOT/droid/actions" "training/DATA.md, DROID"
    fi
    for key in ffmpeg ffprobe; do
        command -v "$key" >/dev/null 2>&1 || { echo "MISSING $key on PATH (needed by the video loaders)" >&2; fail=1; }
    done
    if [[ "$DRY_RUN" != "1" ]]; then
        "$python" -c "import torch" 2>/dev/null || {
            echo "MISSING a working Python environment: '$python' cannot import torch (see README, Installation)" >&2
            fail=1
        }
    fi
    (( ${#active[@]} > 0 )) || { echo "RATIOS leaves no active slot" >&2; fail=1; }
    if (( NNODES * NPROC_PER_NODE < ${#active[@]} )); then
        echo "need at least one GPU per active slot (${#active[@]} slots, $((NNODES * NPROC_PER_NODE)) GPUs)" >&2
        fail=1
    fi
    if (( MAX_ITER > CYCLE_LEN )); then
        echo "MAX_ITER=$MAX_ITER exceeds the LR schedule CYCLE_LEN=$CYCLE_LEN" >&2
        fail=1
    fi
    case "$WANDB_MODE" in online | offline | disabled) ;; *)
        echo "WANDB_MODE must be online, offline or disabled (got '$WANDB_MODE')" >&2
        fail=1
        ;;
    esac
    (( fail == 0 )) || vpm_die "preflight failed"

    local -a overrides=(
        "checkpoint.load_path=$(vpm_quote "$BASE_CHECKPOINT_PATH")"
        "trainer.max_iter=$MAX_ITER"
        "checkpoint.save_iter=$SAVE_ITER"
        "scheduler.cycle_lengths=[$CYCLE_LEN]"
        "scheduler.warm_up_steps=[$WARMUP_STEPS]"
        "optimizer.lr=$LR"
        "job.wandb_mode=$WANDB_MODE"
        "model.config.max_num_tokens_after_packing=$MAX_TOKENS"
        "dataloader_train.max_sequence_length=$MAX_TOKENS"
    )
    [[ -z "${MAX_SAMPLES_PER_PACK:-}" ]] || overrides+=("dataloader_train.max_samples_per_batch=$MAX_SAMPLES_PER_PACK")
    if (( resuming == 1 )); then
        overrides+=("checkpoint.keys_to_skip_loading=[]")
    else
        overrides+=("checkpoint.keys_to_skip_loading=['net_ema.']")
    fi
    overrides+=("${data_overrides[@]}")
    (( ${#VPM_STAGE_OVERRIDES[@]} == 0 )) || overrides+=("${VPM_STAGE_OVERRIDES[@]}")
    (( $# == 0 )) || overrides+=("$@")

    local -a dist=(--nproc_per_node="$NPROC_PER_NODE")
    if (( NNODES > 1 )); then
        dist+=(--nnodes="$NNODES" --node_rank="$NODE_RANK"
            --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" --rdzv_id="${RDZV_ID:-$VPM_EXP}")
    else
        dist+=(--nnodes=1 --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT")
    fi
    local -a cmd=("$python" -m torch.distributed.run "${dist[@]}"
        -m cosmos_framework.scripts.train --sft-toml="$toml" -- "${overrides[@]}")

    vpm_log "VPM stage $VPM_STAGE ($VPM_EXP), node $NODE_RANK/$NNODES x $NPROC_PER_NODE GPUs"
    if (( resuming == 1 )); then
        vpm_log "resuming from $ckpt_dir/$(tr -d '[:space:]' <"$ckpt_dir/latest_checkpoint.txt")"
    else
        vpm_log "warm start from $BASE_CHECKPOINT_PATH"
    fi
    local ratio_str=""
    for slot in "${VPM_SLOTS[@]}"; do ratio_str+=" $slot=${ratio[$slot]}"; done
    vpm_log "data: $VPM_DATA_ROOT, slot ratios:$ratio_str"
    vpm_log "lr=$LR max_iter=$MAX_ITER warmup=$WARMUP_STEPS save_iter=$SAVE_ITER max_tokens=$MAX_TOKENS" \
        "max_samples_per_pack=${MAX_SAMPLES_PER_PACK:-experiment default}"
    vpm_log "output: $OUTPUT_ROOT/cosmos3/sft/$VPM_EXP"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '%q ' "${cmd[@]}"
        echo
        return 0
    fi

    mkdir -p "$OUTPUT_ROOT/logs"
    local log_file="$OUTPUT_ROOT/logs/${VPM_EXP}_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"
    vpm_log "log: $log_file"

    if (( KEEP_CHECKPOINTS > 0 && NODE_RANK == 0 )); then
        vpm_prune_checkpoints "$ckpt_dir" "$KEEP_CHECKPOINTS" &
        VPM_PRUNER_PID=$!
        trap 'kill "$VPM_PRUNER_PID" 2>/dev/null || true' EXIT
    fi

    cd "$VPM_ROOT"
    set +e
    "${cmd[@]}" 2>&1 | tee -a "$log_file"
    local rc=${PIPESTATUS[0]}
    set -e
    vpm_log "done (exit $rc)"
    return "$rc"
}
