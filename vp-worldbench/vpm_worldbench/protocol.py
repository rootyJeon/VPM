"""The generation protocol: which frames condition the model, which frames are scored, and how.

Every field can be changed; the defaults reproduce the VP-WorldBench v1 setting (I2V from frame 16, 64 future frames).

    p = Protocol()                                     # I2V, condition on frame 16, score frames 17..80
    p = Protocol(ref_frame=32, context_frames=9)       # V2V: frames 24..32 given, score 33..96
    p = Protocol(horizon=32, frame_step=2)             # score every 2nd frame: 18, 20, ..., 80
    p.window_indices -> frame indices [t_ref, t_ref+step, ..., t_ref+horizon*step]
"""
from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass, field, fields, replace
from numbers import Integral, Real
from pathlib import Path
from typing import Optional

from .constants import SUBSETS, canonical_subset
from .presets import LEGACY_PRERELEASE, SUBSET_DEFAULTS
from .utils import UsageError

TEXT_MODES = ("caption", "caption+camera", "window", "legacy", "none")
PER_SUBSET_FIELDS = ("eval_size", "resize_mode", "delta_ladder_m", "delta_survival_m", "delta_group", "depth_anchor")
REFERENCE_FRAMES = ("video", "_reproduce")
NON_EVAL_FIELDS = ("name", "text", "camera_conditioning", "context_frames", "clip_filter")


def _is_int(x) -> bool:
    return isinstance(x, Integral) and not isinstance(x, bool)


def _is_number(x) -> bool:
    """A finite real number (bools are not numbers here)."""
    return isinstance(x, Real) and not isinstance(x, bool) and math.isfinite(x)


def _plain(x):
    """Tuples (also nested ones) -> lists, so values compare and serialise like their JSON / YAML spelling."""
    if isinstance(x, (list, tuple)):
        return [_plain(y) for y in x]
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    return x


def _module_attr(name: str) -> bool:
    """True for a 'package.module:attr' path, the form `lifters.base.get_lifter` imports (not imported here)."""
    mod, sep, attr = name.partition(":")
    return bool(sep) and attr.isidentifier() and all(p.isidentifier() for p in mod.split("."))


@dataclass
class Protocol:
    name: str = "vpm-v1"
    ref_frame: int = 16
    context_frames: int = 1
    horizon: int = 64
    frame_step: int = 1
    views: Optional[list] = None
    eval_size: dict = field(default_factory=dict)
    resize_mode: dict = field(default_factory=dict)
    text: str = "caption"
    camera_conditioning: bool = False
    fvd_window: int = 16
    fvd_windows: Optional[int] = None
    fvd_cov: str = "unbiased"
    met3r_step: int = 4
    met3r_aspect: str = "center_crop"
    met3r_size: int = 256
    delta_ladder_m: dict = field(default_factory=dict)
    delta_survival_m: dict = field(default_factory=dict)
    delta_group: dict = field(default_factory=dict)
    query_requires_visible: bool = True
    gt_visibility: str = "visible"
    depth_anchor: dict = field(default_factory=dict)
    lifter: str = "mvtracker"
    rgb_scale_quirk: bool = False
    plain_K_scaling: bool = False
    reference_frames: str = "video"
    clip_filter: dict = field(default_factory=dict)

    def __post_init__(self):
        """Checks every field (TypeError / ValueError / KeyError naming the field), so that an invalid protocol fails
        here, not in the middle of an export or evaluation; maps the subset keys of per-subset fields to canonical
        names."""
        from .geometry import RESIZE_MODES
        from .lifters.base import _LIFTERS, ANCHOR_MODES
        from .metrics.fvd import COV_MODES
        from .metrics.met3r import ASPECTS
        if not isinstance(self.name, str) or not self.name:
            raise TypeError(f"name must be a non-empty string, got {self.name!r}")
        for f in ("ref_frame", "context_frames", "horizon", "frame_step", "fvd_window", "met3r_step", "met3r_size"):
            if not _is_int(getattr(self, f)):
                raise TypeError(f"{f} must be an integer, got {getattr(self, f)!r}")
        if self.ref_frame < 0:
            raise ValueError("ref_frame must be >= 0")
        if self.context_frames < 1:
            raise ValueError("context_frames must be >= 1")
        if self.horizon < 1 or self.frame_step < 1:
            raise ValueError("horizon and frame_step must be >= 1")
        if self.fvd_window < 1 or self.met3r_step < 1 or self.met3r_size < 1:
            raise ValueError("fvd_window, met3r_step and met3r_size must be >= 1")
        if self.fvd_windows is not None and not (_is_int(self.fvd_windows) and self.fvd_windows >= 1):
            raise ValueError(f"fvd_windows must be null (as many windows as fit) or an integer >= 1, got "
                             f"{self.fvd_windows!r}")
        if self.views is not None and not (isinstance(self.views, (list, tuple)) and self.views
                                           and all(_is_int(v) and v >= 0 for v in self.views)
                                           and len(set(self.views)) == len(self.views)):
            raise ValueError(f"views must be null (all views) or a list of distinct view indices >= 0, got "
                             f"{self.views!r}")
        if self.text not in TEXT_MODES:
            raise ValueError(f"text must be one of {TEXT_MODES}")
        for f in ("camera_conditioning", "query_requires_visible", "rgb_scale_quirk", "plain_K_scaling"):
            if not isinstance(getattr(self, f), bool):
                raise TypeError(f"{f} must be true or false, got {getattr(self, f)!r}")
        for f, allowed in (("gt_visibility", ("visible", "in_frustum")), ("fvd_cov", COV_MODES),
                           ("met3r_aspect", ASPECTS), ("reference_frames", REFERENCE_FRAMES)):
            if getattr(self, f) not in allowed:
                raise ValueError(f"{f} must be one of {allowed}, got {getattr(self, f)!r}")
        if not isinstance(self.lifter, str) or (self.lifter not in _LIFTERS and not _module_attr(self.lifter)):
            raise ValueError(f"lifter {self.lifter!r} is neither registered ({', '.join(sorted(_LIFTERS))}; "
                             f"lifters.base.register_lifter) nor an importable 'module:attr' path")
        if not isinstance(self.clip_filter, dict) or not all(isinstance(k, str) for k in self.clip_filter):
            raise TypeError(f"clip_filter must be a mapping of index.json attribute -> accepted value(s), e.g. "
                            f"{{'difficulty': 'hard'}}, got {self.clip_filter!r}")
        valid = {
            "eval_size": (lambda v: isinstance(v, (list, tuple)) and len(v) == 2 and all(_is_int(x) and x >= 1
                                                                                         for x in v),
                          "a pair of positive integers [W, H]"),
            "resize_mode": (lambda v: v in RESIZE_MODES, f"one of {RESIZE_MODES}"),
            "delta_ladder_m": (lambda v: isinstance(v, (list, tuple)) and len(v) > 0 and all(_is_number(x) and x > 0
                                                                                             for x in v),
                               "a non-empty list of distances > 0 (metres)"),
            "delta_survival_m": (lambda v: _is_number(v) and v > 0, "a distance > 0 (metres)"),
            "delta_group": (lambda v: v is None or isinstance(v, str), "a point-group name or null (all points)"),
            "depth_anchor": (lambda v: v in ANCHOR_MODES, f"one of {ANCHOR_MODES}"),
        }
        for f in PER_SUBSET_FIELDS:
            d = getattr(self, f)
            if not isinstance(d, dict):
                raise TypeError(f"{f} must be a dict subset -> value, got {type(d).__name__}")
            canon = {}
            test, what = valid[f]
            for k, v in d.items():
                c = canonical_subset(k)
                if c in canon:
                    raise ValueError(f"{f}: subset {c} given twice ({k!r})")
                if not test(v):
                    raise ValueError(f"{f} of {c} must be {what}, got {v!r}")
                canon[c] = v
            setattr(self, f, canon)

    def get(self, key: str, subset: str):
        subset = canonical_subset(subset)
        own = getattr(self, key, None)
        if isinstance(own, dict) and subset in own:
            return own[subset]
        return SUBSET_DEFAULTS[subset][key]

    def eval_size_for(self, subset: str):
        return tuple(self.get("eval_size", subset))

    def resize_mode_for(self, subset: str) -> str:
        return self.get("resize_mode", subset)

    @property
    def context_indices(self) -> list:
        return list(range(self.ref_frame - self.context_frames + 1, self.ref_frame + 1))

    @property
    def eval_indices(self) -> list:
        return [self.ref_frame + k * self.frame_step for k in range(1, self.horizon + 1)]

    @property
    def window_indices(self) -> list:
        """[t_ref] + eval frames: the grid predictions are delivered on (prediction frame 0 = t_ref)."""
        return [self.ref_frame] + self.eval_indices

    @property
    def last_frame(self) -> int:
        return self.ref_frame + self.horizon * self.frame_step

    def evaluable(self, sample) -> tuple:
        """(ok, reason). A sample is evaluable iff the context and the whole window lie inside the clip."""
        if self.context_indices[0] < 0:
            return False, f"context starts at {self.context_indices[0]} < 0"
        if self.last_frame > sample.n_frames - 1:
            return False, f"window ends at {self.last_frame} > last frame {sample.n_frames - 1}"
        if self.views is not None and max(self.views) >= sample.n_views:
            return False, f"view {max(self.views)} >= n_views {sample.n_views}"
        return True, ""

    def views_for(self, sample) -> list:
        return list(range(sample.n_views)) if self.views is None else list(self.views)

    def met3r_positions(self) -> list:
        """Positions (indices into window_indices) scored by MEt3R: every met3r_step GT frames after t_ref."""
        out = []
        for j, t in enumerate(self.window_indices):
            if j > 0 and (t - self.ref_frame) % self.met3r_step == 0:
                out.append(j)
        return out

    def to_dict(self) -> dict:
        """JSON-ready dict (tuples, also nested ones, become lists), so to_dict() survives a save/load round trip
        (except for a protocol named legacy-prerelease that evaluates differently from that preset: see `load`)."""
        return _plain(asdict(self))

    def _eval_fields(self) -> dict:
        """to_dict() without NON_EVAL_FIELDS and with every per-subset field given for all four subsets (the presets
        fill in the missing ones): equal for protocols that evaluate alike, however they are spelled."""
        d = {k: v for k, v in self.to_dict().items() if k not in NON_EVAL_FIELDS}
        for f in PER_SUBSET_FIELDS:
            d[f] = {s: _plain(self.get(f, s)) for s in SUBSETS}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Protocol":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise KeyError(f"unknown protocol fields {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def load(cls, path) -> "Protocol":
        """A json / yaml file of Protocol fields (omitted fields take their defaults). A file without a 'name' is named
        after its file stem, so results and exports of a custom window are not labelled 'vpm-v1'; a stem that is a
        preset name becomes 'custom-<stem>' (the preset names select preset-only behaviour, see `legacy_option`).
        A file named legacy-prerelease (a preset whose name switches on recipes of its own) is that preset only when
        it evaluates like it: all fields but the model inputs (text, camera_conditioning, context_frames) and
        clip_filter equal the preset's, per-subset defaults filled in, so a saved preset loads back as the preset.
        Otherwise it is loaded as 'custom-legacy-prerelease' with a warning naming the differing fields: the name alone
        never switches the legacy recipes on. 'vpm-v1' switches nothing on and is the default name of every Protocol
        (so files saved from Python protocols carry it); it is kept as given."""
        path = Path(path)
        text = path.read_text()
        if path.suffix in (".yaml", ".yml"):
            import yaml
            d = yaml.safe_load(text)
        else:
            d = json.loads(text)
        if d is None:
            d = {}
        if not isinstance(d, dict):
            raise ValueError(f"{path}: a protocol file holds a mapping of Protocol fields, got {type(d).__name__}")
        if d.get("name") in (None, ""):
            d = {**d, "name": f"custom-{path.stem}" if path.stem in PRESETS else path.stem}
        p = cls.from_dict(d)
        if p.name in PRESETS and PRESETS[p.name] is not Protocol:
            mine, preset = p._eval_fields(), PRESETS[p.name]()._eval_fields()
            differ = [k for k in preset if mine[k] != preset[k]]
            if differ:
                warnings.warn(f"{path}: name {p.name!r} selects that preset's recipes, but the file evaluates "
                              f"differently ({', '.join(differ)}); loaded as 'custom-{p.name}', without them",
                              stacklevel=2)
                p = replace(p, name=f"custom-{p.name}")
        return p

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))


def legacy_prerelease() -> Protocol:
    """Reproduces the internal pre-release numbers (only meaningful with the _reproduce/ pack for Real-General).
    Per-subset legacy switches are applied by the metric code through `legacy_option(subset, key)`."""
    return Protocol(name="legacy-prerelease", rgb_scale_quirk=True, plain_K_scaling=True,
                    eval_size={"Real-General": (480, 480)}, reference_frames="_reproduce")


def legacy_option(protocol: Protocol, subset: str, key: str, default=None):
    if protocol.name != "legacy-prerelease":
        return default
    return LEGACY_PRERELEASE[canonical_subset(subset)].get(key, default)


PRESETS = {"vpm-v1": Protocol, "legacy-prerelease": legacy_prerelease}


class ProtocolNotFoundError(UsageError, FileNotFoundError):
    """`get_protocol` got neither a preset name nor an existing file."""


class ProtocolFileError(UsageError, ValueError):
    """`get_protocol` got a file that cannot be read or parsed, or whose fields are no valid Protocol."""


def get_protocol(name_or_path: str | None) -> Protocol:
    """None / '' / 'vpm-v1' -> the default; a preset name (PRESETS); else a json / yaml file (`Protocol.load`).
    Anything else raises ProtocolNotFoundError (a FileNotFoundError) naming the presets; a file that cannot be read or
    parsed, or with unknown or invalid fields, raises ProtocolFileError (a ValueError; `Protocol.load` itself raises
    the underlying error)."""
    if name_or_path in (None, "", "vpm-v1"):
        return Protocol()
    if isinstance(name_or_path, str) and name_or_path in PRESETS:
        return PRESETS[name_or_path]()
    if not Path(name_or_path).is_file():
        raise ProtocolNotFoundError(f"protocol {str(name_or_path)!r} is neither a preset ({', '.join(PRESETS)}) nor an "
                                    f"existing json / yaml file")
    import yaml
    try:
        return Protocol.load(name_or_path)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as e:
        msg = e.args[0] if isinstance(e, KeyError) and e.args else str(e)
        raise ProtocolFileError(f"protocol file {str(name_or_path)!r}: {type(e).__name__}: "
                                f"{' '.join(str(msg).split())}") from e
