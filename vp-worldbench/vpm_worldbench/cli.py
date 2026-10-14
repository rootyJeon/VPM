"""`vpmwb`: the VP-WorldBench command line.

    vpmwb info       --root DATA                       subsets, clip counts, frames, flags
    vpmwb validate   --root DATA [--deep] [--hash]     check every sample (docs/DATA_FORMAT.md)
    vpmwb index      --root DATA                       rebuild index.json
    vpmwb protocol   [--protocol vpm-v1|FILE]          print the resolved generation settings per subset + reference key

    vpmwb generation export|evaluate|reference|summarize ...   the main task (docs/PROTOCOL.md, docs/METRICS.md)
    vpmwb <task>     export|baseline|evaluate|summarize ...
          task = track3d | track2d | depth | recon4d | nvs | segmentation | understanding   (docs/TASKS.md)

Shortcuts: `export-inputs` = generation export, `eval-generation` = generation evaluate, `report` = generation summarize.
`vpmwb <command> --help` shows the options of every command.
A missing optional package, metric models that cannot be loaded and invalid input (an unknown command, subset or
protocol, an invalid protocol file or --size, a missing data root or index.json) end with a one-line
`vpmwb: error: ...` and exit status 2 (VPMWB_TRACEBACK=1 shows the traceback instead). Malformed options are reported
by the command's argument parser: its usage, then one error line, exit status 2.
"""
from __future__ import annotations

import importlib
import json
import sys

TASKS = ("track3d", "track2d", "depth", "recon4d", "nvs", "segmentation", "understanding")
SHORTCUTS = {"export-inputs": ("generation", "export"), "eval-generation": ("generation", "evaluate"),
             "report": ("generation", "summarize")}


def _info(argv) -> int:
    import argparse
    from collections import Counter

    from .data.benchmark import Benchmark, DataRootError
    ap = argparse.ArgumentParser(prog="vpmwb info")
    ap.add_argument("--root", required=True)
    a = ap.parse_args(argv)
    bench = Benchmark(a.root)
    if bench.index is None:
        raise DataRootError(f"data root {a.root!r} has no index.json: run `vpmwb index --root {a.root}` first")
    for sub in bench.subsets():
        recs = bench.records(sub)
        if not recs:
            continue
        frames = [r["n_frames"] for r in recs]
        sizes = Counter("x".join(map(str, r["image_size"][0])) for r in recs)
        flags = Counter(f.split(":")[0] for r in recs for f in r["flags"])
        diff = Counter(r.get("difficulty") for r in recs)
        print(f"{sub}: {len(recs)} clips, views {sorted({r['n_views'] for r in recs})}, fps {sorted({r['fps'] for r in recs})}, "
              f"frames {min(frames)}-{max(frames)}, size {dict(sizes)}, depth {sum(r['has_depth'] for r in recs)}, "
              f"difficulty {dict(diff)}")
        for f, n in flags.most_common():
            print(f"    flag {f}: {n}")
    return 0


def _protocol(argv) -> int:
    import argparse

    from .constants import SUBSETS
    from .tasks.generation import protocol_hash, resolve_protocol, subset_settings
    ap = argparse.ArgumentParser(prog="vpmwb protocol", description="Print a generation protocol with every per-subset "
                                 "setting resolved (presets, legacy switches) and its reference-cache key.")
    ap.add_argument("--protocol", default="vpm-v1", help="vpm-v1 | legacy-prerelease | <yaml/json>")
    a = ap.parse_args(argv)
    p = resolve_protocol(a.protocol)
    out = {"protocol": p.to_dict(), "reference_key": protocol_hash(p),
           "context_indices": p.context_indices, "window_indices": [p.window_indices[0], "...", p.window_indices[-1]],
           "n_scored_frames": p.horizon, "met3r_positions": p.met3r_positions(),
           "per_subset": {s: subset_settings(p, s) for s in SUBSETS}}
    print(json.dumps(out, indent=1, default=list))
    return 0


def _module_main(module: str, argv) -> int:
    mod = importlib.import_module(f"vpm_worldbench.{module}")
    rc = mod.main(argv)
    return 0 if rc is None else int(rc)


def _generation(argv) -> int:
    """export -> vpm_worldbench.inputs.main; evaluate (alias eval) / reference / summarize / --help ->
    vpm_worldbench.tasks.generation.main."""
    if argv and argv[0] == "export":
        return _module_main("inputs", list(argv[1:]))
    gen = importlib.import_module("vpm_worldbench.tasks.generation")
    return int(gen.main(list(argv)) or 0)


def _one_line_errors() -> tuple:
    """Exceptions that main() reports as one line instead of a traceback: missing optional packages (ImportError),
    invalid input or setup (utils.UsageError: unknown subsets / protocols, invalid protocol files or sizes, missing data
    roots, run directories of another run, ...) and, of tasks.generation, metric models that cannot be loaded
    (ModelLoadError) and run directories of another protocol or prediction root (RunDirError)."""
    from .utils import UsageError
    errs = [ImportError, UsageError]
    gen = sys.modules.get("vpm_worldbench.tasks.generation")
    for name in ("ModelLoadError", "RunDirError"):
        cls = getattr(gen, name, None)
        if isinstance(cls, type) and issubclass(cls, Exception):
            errs.append(cls)
    return tuple(errs)


def main(argv=None) -> int:
    """Run one `vpmwb` command line; returns its exit status. Usage and setup errors (`_one_line_errors`) end with
    the one line `utils.error_line` on stderr and status 2, as does a command that exits with a message
    (SystemExit('...')); with VPMWB_TRACEBACK set (to anything but '' / '0') they propagate with their traceback."""
    from .utils import error_line, traceback_requested
    try:
        return _main(argv)
    except SystemExit as e:
        if not isinstance(e.code, str) or traceback_requested():
            raise
        print(error_line(e.code), file=sys.stderr)
        return 2
    except Exception as e:
        if traceback_requested() or not isinstance(e, _one_line_errors()):
            raise
        line = error_line(e)
        if isinstance(e, ModuleNotFoundError) and "pip install" not in line:
            line += " (an optional package: see the extras in the README, section Install)"
        print(line, file=sys.stderr)
        return 2


def _main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd in SHORTCUTS:
        task, sub = SHORTCUTS[cmd]
        return _generation([sub, *rest])
    if cmd == "info":
        return _info(rest)
    if cmd == "protocol":
        return _protocol(rest)
    if cmd in ("validate", "index"):
        return _module_main(cmd, rest)
    if cmd == "generation":
        return _generation(rest)
    if cmd in TASKS:
        return _module_main(f"tasks.{cmd}", rest)
    from .utils import UsageError
    raise UsageError(f"unknown command {cmd!r}; commands: info, validate, index, protocol, generation, "
                     f"{', '.join(TASKS)}, {', '.join(SHORTCUTS)} (`vpmwb --help`)")


if __name__ == "__main__":
    raise SystemExit(main())
