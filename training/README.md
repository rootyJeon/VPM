# 🏋️ Training

VPM is trained from Cosmos3-Nano in three stages. Each stage warm-starts from the latest checkpoint of the previous
one under the same `OUTPUT_ROOT`.

| | stage 1 · world model | stage 2 · point-track head | stage 3 · video + action + track |
|:--|:--|:--|:--|
| launcher | `train_stage1.sh` | `train_stage2.sh` | `train_stage3.sh` |
| starts from | Cosmos3-Nano | stage 1 | stage 2 |
| trains | generation tower, camera conditioning | tracking head | generation tower, action stream, tracking head (full fine-tuning) |
| losses | video flow matching | 3D point tracks | video FM · action FM (×10) · tracks (×0.1) |
| data (GPU ratio) | psl4d 3 · droid 3 · hrdexdb 1 · molmo 1 | same | droid with actions |
| lr · iters | 1e-4 · 2k | 2e-5 (head ×5) · 2k | 2e-5 (action ×5) · 5k |

The language tower stays frozen throughout. The tracking head starts from random weights and is first trained in
stage 2.

## Data

```bash
bash scripts/download_assets.sh --data all      # or --data droid,molmo
```

The data goes to `data/<slot>/` (`VPM_DATA_ROOT`). The layout and file formats are described in [DATA.md](DATA.md).

## Run

```bash
bash training/train_stage1.sh
bash training/train_stage2.sh
bash training/train_stage3.sh
```

- Each launcher checks its inputs and then starts `torchrun`. `DRY_RUN=1` runs only the checks and prints the command.
- Extra arguments are passed on as Hydra overrides, e.g. `bash training/train_stage1.sh trainer.logging_iter=10`.
- Runs are written to `$OUTPUT_ROOT/cosmos3/sft/vpm_stage<N>/`. Rerunning the same command resumes from its
  latest checkpoint.

Main environment variables (the full list is in the header of [`_launch_common.sh`](_launch_common.sh)):

| variable | default | meaning |
|:--|:--|:--|
| `VPM_DATA_ROOT` · `OUTPUT_ROOT` | `data` · `outputs` | data and run roots |
| `BASE_CHECKPOINT_PATH` | previous stage | warm start (a DCP `model/` dir), stage 1: `checkpoints/Cosmos3-Nano` |
| `RATIOS` | see table | GPU share per slot, e.g. `"psl4d=3 droid=3 hrdexdb=0 molmo=1"` (0 = unused) |
| `MAX_TOKENS` · `MAX_SAMPLES_PER_PACK` | 50k (stage 2: 55k) · 6 | packed tokens / samples per GPU; lower for < 96 GB GPUs |
| `LR` · `MAX_ITER` · `SAVE_ITER` | see table | schedule |
| `NNODES` · `NODE_RANK` · `MASTER_ADDR` · `NPROC_PER_NODE` | single node | multi-node `torchrun` |
| `WANDB_MODE` | `disabled` | `online` / `offline` / `disabled` |

## Multi-node and Slurm

Run the same launcher on every node with a shared `OUTPUT_ROOT`. [`train.sbatch.example`](train.sbatch.example)
does this with one `srun` task per node. It also saves a checkpoint before the time limit, so chained jobs resume:

```bash
STAGE=1 sbatch training/train.sbatch.example
STAGE=2 sbatch --dependency=afterany:<job id> training/train.sbatch.example
```

## Export

`scripts/export_checkpoint.py` turns a training checkpoint into sharded bf16 safetensors that
`inference/infer_droid_old.sh --ckpt` loads:

```bash
python scripts/export_checkpoint.py \
    --src outputs/cosmos3/sft/vpm_stage3/checkpoints/iter_000005000 --out release/vpm_stage3
```

- `--ema` exports the EMA weights.
