"""Per-subset defaults of the generation protocol and its metrics (docs/PROTOCOL.md, docs/METRICS.md)."""

SUBSET_DEFAULTS = {
    "Simul-Robot": {
        "eval_size": (480, 480),
        "resize_mode": "resize",
        "delta_ladder_m": (0.01, 0.02, 0.04, 0.08, 0.16),
        "delta_survival_m": 0.5,
        "delta_group": "object",
        "depth_anchor": "gt_depth",
        "far_clip_m": 80.0,
    },
    "Simul-General": {
        "eval_size": (480, 480),
        "resize_mode": "resize",
        "delta_ladder_m": (0.10, 0.20, 0.40, 0.80, 1.60),
        "delta_survival_m": 0.5,
        "delta_group": None,
        "depth_anchor": "gt_depth",
        "far_clip_m": 80.0,
    },
    "Real-Robot": {
        "eval_size": (848, 480),
        "resize_mode": "resize",
        "delta_ladder_m": (0.01, 0.02, 0.05, 0.10, 0.20),
        "delta_survival_m": 0.10,
        "delta_group": None,
        "depth_anchor": "gt_points",
        "far_clip_m": 20.0,
    },
    "Real-General": {
        "eval_size": (640, 480),
        "resize_mode": "resize",
        "delta_ladder_m": (0.05, 0.10, 0.20, 0.40),
        "delta_survival_m": 1.0,
        "delta_group": None,
        "depth_anchor": "gt_depth",
        "far_clip_m": 20.0,
    },
}

STATIC_MAX_M = 0.01
DYNAMIC_MIN_M = 0.10

LEGACY_PRERELEASE = {
    "Simul-Robot": {"fvd_windows": 4, "fvd_cov": "unbiased", "met3r_aspect": "center_crop", "query_requires_visible": False,
                    "gt_visibility": "visible", "rgb_scale_quirk": True, "plain_K_scaling": True},
    "Simul-General": {"fvd_windows": 4, "fvd_cov": "unbiased", "met3r_aspect": "center_crop", "query_requires_visible": False,
                      "gt_visibility": "in_frustum", "rgb_scale_quirk": True, "plain_K_scaling": True},
    "Real-Robot": {"fvd_windows": 4, "fvd_cov": "biased", "met3r_aspect": "center_crop", "query_requires_visible": True,
                   "gt_visibility": "visible", "rgb_scale_quirk": True, "plain_K_scaling": True},
    "Real-General": {"fvd_windows": 2, "fvd_cov": "biased", "met3r_aspect": "resize", "query_requires_visible": True,
                     "gt_visibility": "visible", "rgb_scale_quirk": True, "plain_K_scaling": True,
                     "eval_size": (480, 480), "reference_frames": "_reproduce"},
}
