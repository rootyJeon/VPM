#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Download the third-party assets VPM needs, and optionally the released VPM checkpoint and the prepared data.
# Idempotent: finished steps are detected and skipped, interrupted downloads resume.
#
#   bash scripts/download_assets.sh                     Cosmos3-Nano (converted), Wan2.2 VAE, Qwen3-VL tokenizer
#   bash scripts/download_assets.sh --no-nano --release inference only: VAE, tokenizer and the released checkpoint
#   bash scripts/download_assets.sh --data droid        also the prepared data of the listed slots ("all" = every slot)
#
# What goes where (relative paths are resolved against the repository root):
#   $CHECKPOINT_DIR/Cosmos3-Nano/model         stage-1 warm start (DCP), converted by scripts/convert_cosmos3_nano.py
#   $CHECKPOINT_DIR/wan22_vae/Wan2.2_VAE.pth   video tokenizer (WAN_VAE_PATH default of the launchers)
#   $HF_HOME/hub                                Qwen3-VL-8B-Instruct text tokenizer files, and the Cosmos3-Nano download
#   $CHECKPOINT_DIR/VPM-DROID                   released checkpoint (--release)
#   $VPM_DATA_ROOT/<slot>                       prepared data (--data)
#
# Environment:
#   CHECKPOINT_DIR     default: checkpoints
#   VPM_DATA_ROOT      default: data
#   HF_HOME, HF_TOKEN  Hugging Face cache and token (log in with `hf auth login` for gated repositories)
#   VPM_RELEASE_REPO   Hugging Face repo of the released checkpoint (default: rooty2020/VPM-DROID)
#   VPM_DATA_REPO      Hugging Face dataset repo of the prepared data (default: vpm-project/VPM-Data)
#   VPM_PYTHON         interpreter with the VPM environment (default: $VIRTUAL_ENV/bin/python, .venv/bin/python, python3)

set -euo pipefail

VPM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

NANO_REPO="nvidia/Cosmos3-Nano"
NANO_REVISION="e59a53c25979a090fa8706c9acc0c254a6e89b92"
WAN_VAE_REPO="Wan-AI/Wan2.2-TI2V-5B"
WAN_VAE_FILE="Wan2.2_VAE.pth"
WAN_VAE_REVISION="921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
TOKENIZER_REPO="Qwen/Qwen3-VL-8B-Instruct"
ALL_SLOTS="psl4d,droid,hrdexdb,molmo"

RELEASE_REPO="${VPM_RELEASE_REPO:-rooty2020/VPM-DROID}"
DATA_REPO="${VPM_DATA_REPO:-vpm-project/VPM-Data}"

log() { echo ">>> $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

usage() {
    sed -n '5,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'
Options:
  --no-nano       skip Cosmos3-Nano (only needed to train stage 1)
  --no-vae        skip the Wan2.2 VAE
  --no-tokenizer  skip the Qwen3-VL tokenizer prefetch
  --release       also download the released checkpoint ($VPM_RELEASE_REPO)
  --data SLOTS    also download prepared data for SLOTS (comma separated: psl4d,droid,hrdexdb,molmo, or all)
  -h, --help      show this help
EOF
}

abspath() {
    if [[ "$1" == /* ]]; then printf '%s' "$1"; else printf '%s/%s' "$VPM_ROOT" "${1#./}"; fi
}

do_nano=1
do_vae=1
do_tokenizer=1
do_release=0
data_slots=""
while (($#)); do
    case "$1" in
        --no-nano) do_nano=0 ;;
        --no-vae) do_vae=0 ;;
        --no-tokenizer) do_tokenizer=0 ;;
        --release) do_release=1 ;;
        --data)
            (($# >= 2)) || die "--data needs a slot list"
            data_slots="$2"
            shift
            ;;
        --data=*) data_slots="${1#--data=}" ;;
        -h | --help)
            usage
            exit 0
            ;;
        *) die "unknown argument '$1' (see --help)" ;;
    esac
    shift
done

CHECKPOINT_DIR="$(abspath "${CHECKPOINT_DIR:-checkpoints}")"
VPM_DATA_ROOT="$(abspath "${VPM_DATA_ROOT:-data}")"

PY="${VPM_PYTHON:-}"
if [[ -z "$PY" ]]; then
    if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
        PY="$VIRTUAL_ENV/bin/python"
    elif [[ -x "$VPM_ROOT/.venv/bin/python" ]]; then
        PY="$VPM_ROOT/.venv/bin/python"
    else
        PY="python3"
    fi
fi
# This repository's cosmos_framework first (as in the training and inference launchers).
export PYTHONPATH="$VPM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -c "import huggingface_hub" 2>/dev/null ||
    die "'$PY' cannot import huggingface_hub; install the environment first (README, Installation) or set VPM_PYTHON"

hf_python() {
    "$PY" - "$@"
}

placeholder_guard() {
    [[ "$2" != *"<"* ]] || die "$1 is still the placeholder '$2'; export $3=<org>/<repo>"
}

((do_release == 0)) || placeholder_guard "the released checkpoint repo" "$RELEASE_REPO" VPM_RELEASE_REPO
slots=()
if [[ -n "$data_slots" ]]; then
    placeholder_guard "the prepared-data repo" "$DATA_REPO" VPM_DATA_REPO
    [[ "$data_slots" != "all" ]] || data_slots="$ALL_SLOTS"
    IFS=',' read -r -a slots <<<"$data_slots"
    for slot in "${slots[@]}"; do
        [[ ",$ALL_SLOTS," == *",$slot,"* ]] || die "--data: unknown slot '$slot' (slots: $ALL_SLOTS)"
    done
fi

if ((do_nano)); then
    nano_out="$CHECKPOINT_DIR/Cosmos3-Nano"
    if [[ -f "$nano_out/model/.metadata" ]]; then
        log "Cosmos3-Nano: $nano_out/model already converted, skipping"
    else
        log "Cosmos3-Nano: downloading $NANO_REPO@$NANO_REVISION and converting to $nano_out/model"
        "$PY" "$VPM_ROOT/scripts/convert_cosmos3_nano.py" --out "$nano_out" --repo-id "$NANO_REPO" \
            --revision "$NANO_REVISION"
        [[ -f "$nano_out/model/.metadata" ]] || die "Cosmos3-Nano conversion did not produce $nano_out/model/.metadata"
    fi
fi

if ((do_vae)); then
    vae_dir="$CHECKPOINT_DIR/wan22_vae"
    if [[ -s "$vae_dir/$WAN_VAE_FILE" ]]; then
        log "Wan2.2 VAE: $vae_dir/$WAN_VAE_FILE already present, skipping"
    else
        log "Wan2.2 VAE: downloading $WAN_VAE_REPO/$WAN_VAE_FILE@$WAN_VAE_REVISION to $vae_dir"
        mkdir -p "$vae_dir"
        hf_python "$WAN_VAE_REPO" "$WAN_VAE_FILE" "$WAN_VAE_REVISION" "$vae_dir" <<'EOF'
import sys
from huggingface_hub import hf_hub_download

repo, filename, revision, local_dir = sys.argv[1:5]
print(hf_hub_download(repo, filename, revision=revision, local_dir=local_dir))
EOF
    fi
fi

if ((do_tokenizer)); then
    log "Qwen3-VL tokenizer: caching the text-tokenizer files of $TOKENIZER_REPO in the Hugging Face cache"
    hf_python "$TOKENIZER_REPO" <<'EOF'
import sys
from huggingface_hub import snapshot_download

print(snapshot_download(sys.argv[1], allow_patterns=["*.json", "*.txt", "*.jinja"]))
EOF
fi

if ((do_release)); then
    release_dir="$CHECKPOINT_DIR/VPM-DROID"
    log "Released checkpoint: downloading $RELEASE_REPO to $release_dir"
    mkdir -p "$release_dir"
    hf_python "$RELEASE_REPO" "$release_dir" <<'EOF'
import sys
from huggingface_hub import snapshot_download

print(snapshot_download(sys.argv[1], local_dir=sys.argv[2]))
EOF
fi

if ((${#slots[@]})); then
    log "Prepared data: downloading slots ${slots[*]} of $DATA_REPO to $VPM_DATA_ROOT"
    mkdir -p "$VPM_DATA_ROOT"
    hf_python "$DATA_REPO" "$VPM_DATA_ROOT" "${slots[@]}" <<'EOF'
import sys
from huggingface_hub import snapshot_download

repo, local_dir, slots = sys.argv[1], sys.argv[2], sys.argv[3:]
print(snapshot_download(repo, repo_type="dataset", local_dir=local_dir, allow_patterns=[f"{s}/*" for s in slots]))
EOF
fi

log "done"
((do_nano)) && log "  BASE_CHECKPOINT_PATH for stage 1: $CHECKPOINT_DIR/Cosmos3-Nano"
((do_vae)) && log "  WAN_VAE_PATH: $CHECKPOINT_DIR/wan22_vae/$WAN_VAE_FILE"
((do_release)) && log "  released checkpoint: $CHECKPOINT_DIR/VPM-DROID"
((${#slots[@]} == 0)) || log "  VPM_DATA_ROOT: $VPM_DATA_ROOT"
exit 0
