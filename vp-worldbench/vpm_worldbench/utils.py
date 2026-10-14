"""Small helpers shared by the command line tools."""
from __future__ import annotations

import os
from typing import Optional, Sequence

from .constants import SUBSETS, canonical_subset

TRACEBACK_ENV = "VPMWB_TRACEBACK"


class UsageError(Exception):
    """Invalid command-line input or setup (an unknown subset, protocol, a missing data root, ...): `vpmwb` prints
    `error_line(e)` and exits with status 2 (`traceback_requested()`: the traceback instead). Subclasses also derive
    from the builtin error raised before (KeyError, FileNotFoundError, ValueError), so existing `except` clauses keep
    working."""


class UnknownSubsetError(UsageError, KeyError):
    """A subset name that is neither a subset nor an alias (constants.SUBSET_ALIASES)."""

    def __str__(self):
        return str(self.args[0]) if self.args else ""


class SizeError(UsageError, ValueError):
    """A WxH size option that is not two positive integers (`parse_size`)."""


def traceback_requested() -> bool:
    """True when VPMWB_TRACEBACK is set to anything but '' or '0': usage / setup errors then keep their traceback
    instead of the one-line report (`vpmwb`, and every command that reports such errors itself)."""
    return os.environ.get(TRACEBACK_ENV, "") not in ("", "0")


def error_line(e) -> str:
    """The one-line report of a usage / setup error (an exception or a message): 'vpmwb: error: <message>', with every
    run of whitespace (newlines too) collapsed to one space; a KeyError's message is not quoted."""
    if isinstance(e, BaseException):
        msg = e.args[0] if isinstance(e, KeyError) and len(e.args) == 1 else e
        msg = " ".join(str(msg).split()) or type(e).__name__
    else:
        msg = " ".join(str(e).split())
    return f"vpmwb: error: {msg}"


def parse_list(values: Optional[Sequence[str]]) -> Optional[list]:
    """['a,b', 'c'] or 'a,b' -> ['a', 'b', 'c']; None / empty -> None."""
    if values is None:
        return None
    if isinstance(values, str):
        values = [values]
    out = [x.strip() for v in values for x in str(v).split(",") if x.strip()]
    return out or None


def parse_subsets(values) -> Optional[list]:
    """Like parse_list, with subset aliases resolved (constants.canonical_subset) and repeats dropped; an unknown name
    raises UnknownSubsetError (a KeyError)."""
    out = parse_list(values)
    if out is None:
        return None
    try:
        return list(dict.fromkeys(canonical_subset(s) for s in out))
    except KeyError as e:
        raise UnknownSubsetError(e.args[0] if e.args else f"unknown subset; expected one of {SUBSETS}") from None


def canonical_uid(x: str) -> str:
    """'rr/episode_000' -> 'Real-Robot/episode_000' (subset aliases resolved); a bare sample id, or an entry whose
    subset part is no known subset, is returned unchanged."""
    x = str(x).strip()
    if "/" not in x:
        return x
    sub, sid = x.split("/", 1)
    try:
        return f"{canonical_subset(sub)}/{sid}"
    except KeyError:
        return x


def parse_size(s: Optional[str], flag: str = "--size") -> Optional[tuple]:
    """'640x480' -> (640, 480); None / '' / 'native' -> None. Anything else that is not two positive integers WxH raises
    SizeError (a UsageError and a ValueError) naming the option `flag`."""
    if s in (None, "", "native"):
        return None
    try:
        w, h = (int(v) for v in str(s).lower().split("x"))
    except ValueError:
        w = h = 0
    if w < 1 or h < 1:
        raise SizeError(f"{flag} must be WxH (e.g. 640x480), got {s!r}")
    return w, h
