"""The whole benchmark folder: enumerate and filter samples.

    bench = Benchmark("/data/VP-WorldBench")
    for s in bench.samples("Simul-General", difficulty="hard"): ...      # attribute filters need index.json
    for s in bench.samples(["sr", "rr"], ids=["rr/episode_000", "c001_episode_000000"]): ...
    bench.sample("Simul-Robot/c001_episode_000000")
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Iterator

from ..constants import SUBSETS, canonical_subset
from ..utils import UsageError, canonical_uid
from .sample import Sample


class DataRootError(UsageError, FileNotFoundError):
    """The data root given to `Benchmark` is not a directory (`vpmwb` prints it as a one-line error)."""


class Benchmark:
    def __init__(self, root):
        self.root = Path(root)
        if not self.root.is_dir():
            raise DataRootError(f"data root {str(self.root)!r} does not exist or is not a directory (expected a "
                                f"VP-WorldBench folder: {', '.join(SUBSETS)}, index.json)")
        p = self.root / "index.json"
        self.index = json.loads(p.read_text()) if p.exists() else None

    def subsets(self) -> list:
        return [s for s in SUBSETS if (self.root / s).is_dir()]

    def ids(self, subset: str) -> list:
        subset = canonical_subset(subset)
        if self.index is not None:
            return [r["id"] for r in self.index["samples"] if r["subset"] == subset]
        d = self.root / subset
        return sorted(p.name for p in d.iterdir() if (p / "sample.json").exists()) if d.is_dir() else []

    def sample(self, uid_or_subset: str, sid: str | None = None) -> Sample:
        if sid is None:
            subset, sid = uid_or_subset.split("/", 1)
        else:
            subset = uid_or_subset
        return Sample(self.root, canonical_subset(subset), sid)

    def records(self, subset: str | None = None, **filters) -> list:
        """Index records (dicts) matching attribute filters, e.g. difficulty='hard', category='vehicle'.
        A filter value may be a callable (record -> bool) or a list of accepted values."""
        if self.index is None:
            raise RuntimeError("index.json missing; use ids()/sample()")
        out = []
        for r in self.index["samples"]:
            if subset is not None and r["subset"] != canonical_subset(subset):
                continue
            ok = True
            for k, v in filters.items():
                x = r.get(k, r.get("attributes", {}).get(k))
                if callable(v):
                    ok &= bool(v(r))
                elif isinstance(v, (list, tuple, set)):
                    ok &= x in v
                else:
                    ok &= x == v
            if ok:
                out.append(r)
        return out

    def _chosen(self, sub: str, filters: dict) -> list:
        return [r["id"] for r in self.records(sub, **filters)] if filters else self.ids(sub)

    def _subset_list(self, subset) -> list:
        if not subset:
            return self.subsets()
        names = [subset] if isinstance(subset, str) else subset
        return list(dict.fromkeys(canonical_subset(s) for s in names))

    def samples(self, subset=None, ids=None, **filters) -> Iterator[Sample]:
        """Samples of `subset` (a name or alias, a list of them, or None = every subset present), optionally restricted
        to `ids` (sample ids or uids 'Subset/id', aliases such as 'rr/episode_000' accepted) and to attribute filters
        (`records`). When the call is the whole selection (`subset` None or a list), requested ids that match no sample
        give a warning after a complete iteration; a call with one subset name stays silent, since its caller may
        find the other ids in other subsets (`unmatched_ids` checks any selection)."""
        subsets = self._subset_list(subset)
        wanted = None if ids is None else [canonical_uid(i) for i in ([ids] if isinstance(ids, str) else ids)]
        keep = None if wanted is None else set(wanted)
        matched = set()
        for sub in subsets:
            chosen = self._chosen(sub, filters)
            if keep is not None:
                chosen = [i for i in chosen if i in keep or f"{sub}/{i}" in keep]
                matched.update(chosen)
                matched.update(f"{sub}/{i}" for i in chosen)
            for sid in chosen:
                yield Sample(self.root, sub, sid)
        if wanted and (not subset or not isinstance(subset, str)):
            missed = [i for i in dict.fromkeys(wanted) if i not in matched]
            if missed:
                where = ", ".join(subsets) or "no subset"
                warnings.warn(f"{len(missed)} requested id(s) match no sample of {where}"
                              f"{' after the attribute filters' if filters else ''}: {missed[:10]}", stacklevel=2)

    def unmatched_ids(self, ids, subsets=None, **filters) -> list:
        """The entries of `ids` (sample ids or uids, aliases accepted) that match no sample of `subsets` (a name, a
        list or None = every subset present) after the attribute filters, in the given order."""
        found = set()
        for sub in self._subset_list(subsets):
            chosen = self._chosen(sub, filters)
            found.update(chosen)
            found.update(f"{sub}/{i}" for i in chosen)
        return [i for i in ([ids] if isinstance(ids, str) else ids) if canonical_uid(i) not in found]
