"""Subset names, schema versions and the unified point taxonomy (see docs/DATA_FORMAT.md)."""

SCHEMA_SAMPLE = "vpm-worldbench/sample/1.0"
SUBSETS = ("Simul-Robot", "Simul-General", "Real-Robot", "Real-General")

KIND_NAMES = ("object_moving", "object_static", "robot_arm", "robot_hand", "pedestrian", "vehicle", "human_body",
              "dynamic_surface")
SIDE_NAMES = {-1: "none", 0: "left", 1: "right"}

SUBSET_ALIASES = {
    "simul-robot": "Simul-Robot", "simul_robot": "Simul-Robot", "sr": "Simul-Robot",
    "simul-general": "Simul-General", "simul_general": "Simul-General", "sg": "Simul-General",
    "real-robot": "Real-Robot", "real_robot": "Real-Robot", "rr": "Real-Robot",
    "real-general": "Real-General", "real_general": "Real-General", "rg": "Real-General",
}


def canonical_subset(name: str) -> str:
    if name in SUBSETS:
        return name
    key = name.strip().lower()
    if key in SUBSET_ALIASES:
        return SUBSET_ALIASES[key]
    raise KeyError(f"unknown subset {name!r}; expected one of {SUBSETS}")
