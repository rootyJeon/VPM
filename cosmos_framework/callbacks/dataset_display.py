# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Slot-key -> display names for logging only.

The joint dataloader stamps each batch with its config slot key. ``dataloader_state`` keys its resume state by that
name, so any renaming happens only at the reporting boundary. The map comes from ``COSMOS_DATASET_DISPLAY_MAP``
("slot:display,slot:display"); unset means identity.
"""

import os

_MAP: "dict[str, str] | None" = None


def display_name(name):
    """Remap one slot key, or a list of them (wandb_log batches names into lists)."""
    if isinstance(name, list):
        return [display_name(n) for n in name]
    global _MAP
    if _MAP is None:
        _MAP = {}
        for pair in os.environ.get("COSMOS_DATASET_DISPLAY_MAP", "").split(","):
            if ":" in pair:
                k, _, v = pair.partition(":")
                if k.strip() and v.strip():
                    _MAP[k.strip()] = v.strip()
    return _MAP.get(name, name)
