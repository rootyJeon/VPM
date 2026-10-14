# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.distributed as dist
import torch.utils.data
import wandb

from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.callbacks.dataset_display import display_name
from cosmos_framework.utils import distributed, log
from cosmos_framework.utils.callback import Callback
from cosmos_framework.utils.easy_io import easy_io


@dataclass
class _LossRecord:
    loss: torch.Tensor | float = 0
    iter_count: int = 0
    name: str | None = None

    def reset(self) -> None:
        self.loss = 0
        self.iter_count = 0

    def get_stat(self, return_valid_mask_sum: bool = False) -> Tuple[float, float]:
        if self.iter_count == 0:
            self.loss = torch.tensor([float("nan")], device="cuda")
            self.iter_count = 1
        self.loss = self.loss.mean()
        msg_str = f"{self.name}: sum_loss={self.loss.item()}/iter_count={self.iter_count}="
        avg_loss_tensor = self.loss / self.iter_count
        valid_mask = torch.tensor([torch.isfinite(avg_loss_tensor).float()], device="cuda")
        msg_str += f"avg_loss={avg_loss_tensor.item()}, valid_mask={valid_mask.item()}, "

        avg_loss_tensor = torch.where(
            torch.isfinite(avg_loss_tensor), avg_loss_tensor, torch.tensor([0.0], device="cuda")
        )

        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_mask, op=dist.ReduceOp.SUM)
        msg_str += f" | all_reduce: avg_loss={avg_loss_tensor.item()}, valid_mask={valid_mask.item()}"
        if valid_mask.item() > 0:
            final_avg_loss = (avg_loss_tensor / valid_mask).item()
            valid_mask_sum = valid_mask.item()
        else:
            final_avg_loss = 0.0
            valid_mask_sum = 0

        avg_loss = final_avg_loss
        msg_str += f" | final: avg_loss={final_avg_loss}"
        if self.name is not None:
            log.debug(msg_str, rank0_only=False)
        self.reset()
        if return_valid_mask_sum:
            return avg_loss, valid_mask_sum
        else:
            return avg_loss


class WandbCallback(Callback):
    def __init__(
        self,
        logging_iter_multipler: int = 1,
        save_logging_iter_multipler: int = 1,
        save_s3: bool = False,
    ) -> None:
        super().__init__()
        self.final_loss_log = _LossRecord()
        self.final_loss_log_per_dataset = {}
        self.final_all_loss_log = {}
        self.final_all_loss_log_per_dataset = {}
        self.logging_iter_multipler = logging_iter_multipler
        self.save_logging_iter_multipler = save_logging_iter_multipler
        assert self.logging_iter_multipler > 0, "logging_iter_multipler should be greater than 0"
        self.save_s3 = save_s3
        self.wandb_extra_tag = f"@{logging_iter_multipler}" if logging_iter_multipler > 1 else ""
        self.name = "wandb_loss_log" + self.wandb_extra_tag
        self.unstable_count = torch.zeros(1, device="cuda")

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        if torch.isnan(loss) or torch.isinf(loss):
            log.critical(
                f"Unstable loss {loss} at iteration {iteration}",
                rank0_only=False,
            )
            self.unstable_count += 1

        dataset_name = display_name(data_batch.get("dataset_name", "default"))

        if isinstance(dataset_name, list):
            dataset_name = list(set(dataset_name))

        if dataset_name == "default" and "__url__" in data_batch:
            dataset_name = ["/".join(data_batch["__url__"][0].split("/")[:-1])]

        for single_dataset_name in dataset_name:
            if single_dataset_name not in self.final_loss_log_per_dataset:
                self.final_loss_log_per_dataset[single_dataset_name] = _LossRecord()
                self.final_loss_log_per_dataset[single_dataset_name].name = single_dataset_name
            self.final_loss_log_per_dataset[single_dataset_name].loss += loss.detach().float()
            self.final_loss_log_per_dataset[single_dataset_name].iter_count += 1

        if "avg_num_assistant_tokens" in output_batch:
            per_seq_loss = (
                loss
                * output_batch["avg_num_assistant_tokens"]
                * output_batch["batch_size_local"]
                / output_batch["current_num_assistant_tokens"]
            )
            per_seq_key = f"per_seq/{dataset_name}"
            if per_seq_key not in self.final_loss_log_per_dataset:
                self.final_loss_log_per_dataset[per_seq_key] = _LossRecord()
                self.final_loss_log_per_dataset[per_seq_key].name = per_seq_key
            self.final_loss_log_per_dataset[per_seq_key].loss += per_seq_loss
            self.final_loss_log_per_dataset[per_seq_key].iter_count += 1

        self.final_loss_log.loss += loss.detach().float()
        self.final_loss_log.iter_count += 1

        for key in output_batch.keys():
            if "loss" in key and "per_instance" not in key:
                if key not in self.final_all_loss_log:
                    self.final_all_loss_log[key] = _LossRecord()
                self.final_all_loss_log[key].loss += output_batch[key].detach().float()
                self.final_all_loss_log[key].iter_count += 1
                if "per_seq" in key:
                    continue
                for single_dataset_name in dataset_name:
                    dkey = f"{single_dataset_name}/{key}"
                    if dkey not in self.final_all_loss_log_per_dataset:
                        self.final_all_loss_log_per_dataset[dkey] = _LossRecord()
                        self.final_all_loss_log_per_dataset[dkey].name = dkey
                    self.final_all_loss_log_per_dataset[dkey].loss += output_batch[key].detach().float()
                    self.final_all_loss_log_per_dataset[dkey].iter_count += 1

        if iteration % (self.config.trainer.logging_iter * self.logging_iter_multipler) == 0:
            avg_final_loss = self.final_loss_log.get_stat()

            local_all_loss_keys = list(self.final_all_loss_log.keys())
            gathered_all_loss_keys = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered_all_loss_keys, local_all_loss_keys)
            union_all_loss_keys = set()
            for keys in gathered_all_loss_keys:
                union_all_loss_keys.update(keys)
            union_all_loss_keys = sorted(union_all_loss_keys)
            for key in union_all_loss_keys:
                if key not in self.final_all_loss_log:
                    dummy = _LossRecord()
                    dummy.loss += torch.tensor([float("nan")], device="cuda")
                    dummy.iter_count += 1
                    self.final_all_loss_log[key] = dummy

            avg_final_all_loss = {}
            for key in union_all_loss_keys:
                avg_final_all_loss[key] = self.final_all_loss_log[key].get_stat()

            local_dataset_names = list(self.final_loss_log_per_dataset.keys())
            all_dataset_names = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(all_dataset_names, local_dataset_names)

            union_dataset_names = set()
            for names in all_dataset_names:
                union_dataset_names.update(names)
            union_dataset_names = sorted(list(union_dataset_names))
            for dataset_name in union_dataset_names:
                if dataset_name not in self.final_loss_log_per_dataset:
                    dummy = _LossRecord()
                    dummy.loss += torch.tensor([float("nan")], device="cuda")
                    dummy.iter_count += 1
                    self.final_loss_log_per_dataset[dataset_name] = dummy

            avg_final_loss_per_dataset = {}
            for dataset_name in union_dataset_names:
                avg_loss, valid_mask_sum = self.final_loss_log_per_dataset[dataset_name].get_stat(
                    return_valid_mask_sum=True
                )
                if valid_mask_sum > 0:
                    avg_final_loss_per_dataset[dataset_name] = avg_loss

            local_detail_keys = list(self.final_all_loss_log_per_dataset.keys())
            gathered_detail_keys = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered_detail_keys, local_detail_keys)
            union_detail_keys = sorted({k for keys in gathered_detail_keys for k in keys})
            avg_final_all_loss_per_dataset = {}
            for dkey in union_detail_keys:
                if dkey not in self.final_all_loss_log_per_dataset:
                    dummy = _LossRecord()
                    dummy.loss += torch.tensor([float("nan")], device="cuda")
                    dummy.iter_count += 1
                    self.final_all_loss_log_per_dataset[dkey] = dummy
                avg_loss, valid_mask_sum = self.final_all_loss_log_per_dataset[dkey].get_stat(
                    return_valid_mask_sum=True
                )
                if valid_mask_sum > 0:
                    avg_final_all_loss_per_dataset[dkey] = avg_loss

            dist.all_reduce(self.unstable_count, op=dist.ReduceOp.SUM)

            if distributed.is_rank0() and wandb.run is not None:
                info = {}
                info.update(
                    {
                        f"train{self.wandb_extra_tag}/loss": avg_final_loss,
                        f"train{self.wandb_extra_tag}/unstable_count": self.unstable_count.item(),
                        "iteration": iteration,
                    }
                )
                for key, loss in avg_final_all_loss.items():
                    info.update(
                        {
                            f"train{self.wandb_extra_tag}_detail/{key}": loss,
                        }
                    )
                for dkey, loss in avg_final_all_loss_per_dataset.items():
                    info[f"train{self.wandb_extra_tag}_per_data_detail/{dkey}"] = loss
                for dataset_name, loss in avg_final_loss_per_dataset.items():
                    tag = ""
                    if "per_seq" in dataset_name:
                        tag = "_per_seq"
                        dataset_name = dataset_name.replace("per_seq/", "")
                    info.update(
                        {
                            f"train{self.wandb_extra_tag}_per_data{tag}/{dataset_name}": loss,
                        }
                    )
                if self.save_s3:
                    if (
                        iteration
                        % (
                            self.config.trainer.logging_iter
                            * self.logging_iter_multipler
                            * self.save_logging_iter_multipler
                        )
                        == 0
                    ):
                        easy_io.dump(
                            info,
                            f"s3://rundir/{self.name}/Train_Iter{iteration:09d}.json",
                        )

                if wandb:
                    wandb.log(info, step=iteration)

            self.unstable_count.zero_()
            self.final_loss_log_per_dataset = {}
