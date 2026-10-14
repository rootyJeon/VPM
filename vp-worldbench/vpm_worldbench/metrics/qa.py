"""Video-understanding metrics (task `understanding`): multiple-choice and exact-match answer scoring with text
normalisation, per-category accuracy, and text-video retrieval (R@k, median / mean rank) from a similarity matrix.

    score_multiple_choice(responses, answers, choices=options, categories=cats)   # {"accuracy", "per_category", ...}
    score_exact_match(predictions, references)                                     # free-form short answers
    retrieval_metrics(sim)                     # sim[i, j] = score(text i, video j); text i matches video i
    bidirectional_retrieval(sim)               # {"text_to_video": ..., "video_to_text": ...}

Conventions
- Summary dicts give accuracies and recalls in percent (0-100); ranks are 1-based. Empty inputs give NaN with n = 0.
- Multiple choice: options are labelled A, B, C, ...; a gold answer is a letter ("B", "(b)") or a 0-based option index.
  `extract_choice` maps a free-form model response to a letter with fixed, ordered rules (below); a response it
  cannot map counts as wrong (reported as "n_unparsed"). Ask models to answer with the letter only.
- Exact match: prediction and every reference are normalised (`normalize_text`) and compared for equality; an item
  is correct if any reference matches (SQuAD-style EM, Rajpurkar et al., EMNLP 2016, with VQA-style number words).

`normalize_text`: lower case; dotted abbreviations joined ("U.S." -> "us"); thousands separators removed
("1,000" -> "1000"); leading-dot decimals completed (".5" -> "0.5"); a minus sign directly before a number kept
("-3"); other punctuation -> space, except decimal points inside numbers; number words zero..ninety -> digits,
including compounds ("twenty-one", "twenty one" -> "21"); articles a / an / the removed; whitespace collapsed.
Numbers are not canonicalised further ("3.50" != "3.5").

`extract_choice` rules, applied after removing markdown emphasis (`*` and backticks: "**B**" -> "B"); first match
wins (a letter outside the first n_choices letters is ignored). A *label* is a letter in brackets (any case: "(b)",
"[B]") or an upper-case letter standing alone; in rules 3 and 4 also a lower-case letter that ends the clause
("answer: b").
  1. the whole response is a label, optionally quoted: "B", "b", "(B)", "[b]", "B.", "B:", "'B'", "['B']";
  2. the response starts with a label followed by text: "B. a red cup", "(b) red cup", "B) red cup",
     "B, because ...", "B - the cup ...", or with an upper-case letter alone on its first line ("B" + explanation);
  3. "answer [is|was|would be|be] [:|=] [option|choice] X";
  4. "option [is|was] [:|=] X" / "choice X";
  5. "is X" / "it's X" closing a clause: "I think it's B.", "it is (C) because ...";
  6. with option texts: the normalised response equals exactly one normalised option, else exactly one option text
     occurs in it as whole words (options contained in a longer matching option are discounted);
  7. exactly one distinct upper-case bracketed letter "(X)" / "[X]" anywhere.
Hedges are not credited. A label directly followed by "or" / "and" / "," / "/" / "&" / "+" and another label
("(A) or (B)", "B, C", "B and C") does not count. Rule 2 does not count either when another upper-case bracketed
label occurs anywhere in the response ("B. (A) is also possible"); such a response is then decided by rules 3 to 6
only, so rule 7 is skipped. In rule 3, a bare "A" / "I" followed by a lower-case word is the article / pronoun
("the answer is A person walking"), unless the word is a connective such as "because".

Retrieval (MSR-VTT protocol as in CLIP4Clip): for query i the rank of its best-scoring correct item among all items
(1 = top); R@k = percent of queries with rank <= k; MdR / MnR = median / mean rank. Ties with incorrect items are
broken pessimistically by default (an incorrect item with an equal score is ranked above), so a constant similarity
matrix scores at the bottom instead of the top; NaN similarities count as -inf.
"""
from __future__ import annotations

import re
import string
from typing import Optional, Sequence

import numpy as np

__all__ = ["normalize_text", "exact_match", "extract_choice", "answer_letter", "multiple_choice_correct", "accuracy",
           "score_multiple_choice", "score_exact_match", "retrieval_ranks", "retrieval_metrics",
           "bidirectional_retrieval", "LETTERS", "NUMBER_WORDS"]

LETTERS = string.ascii_uppercase
_UNIT_WORDS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
               "seventeen eighteen nineteen").split()
_TENS_WORDS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
               "ninety": 90}
NUMBER_WORDS = {**{w: str(i) for i, w in enumerate(_UNIT_WORDS)}, **{w: str(v) for w, v in _TENS_WORDS.items()}}
_DIGIT_WORDS = {w: i for i, w in enumerate(_UNIT_WORDS) if 1 <= i <= 9}
_ARTICLES = frozenset(("a", "an", "the"))
_PUNCT_TABLE = str.maketrans({c: " " for c in (string.punctuation.replace(".", "") + "‘’“”"
                                                "–—…·−")})
_RE_ACRONYM = re.compile(r"(?<![A-Za-z0-9])[A-Za-z](?:\.[A-Za-z])+\.?(?![A-Za-z0-9])")
_RE_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_RE_LEADING_DOT = re.compile(r"(?<![\w.])\.(?=\d)")
_RE_MINUS = re.compile(r"(?<![\w.])[-−](?=\.?\d)")
_RE_NON_DECIMAL_DOT = re.compile(r"(?<!\d)\.|\.(?!\d)")
_SIGN = "\x00"
_CONNECTIVES = frozenset(("because", "since", "as", "so", "which", "but", "and", "then", "therefore", "given",
                          "due"))
_NAN = float("nan")


def _number_tokens(tokens):
    """Number words -> digits; "twenty one" / "twenty-one" -> "21" (compounds of two words only)."""
    out, i = [], 0
    while i < len(tokens):
        t = tokens[i].lower()
        m = re.fullmatch(r"([a-z]+)-([a-z]+)", t)
        if m and m.group(1) in _TENS_WORDS and m.group(2) in _DIGIT_WORDS:
            out.append(str(_TENS_WORDS[m.group(1)] + _DIGIT_WORDS[m.group(2)]))
            i += 1
        elif t in _TENS_WORDS and i + 1 < len(tokens) and tokens[i + 1].lower() in _DIGIT_WORDS:
            out.append(str(_TENS_WORDS[t] + _DIGIT_WORDS[tokens[i + 1].lower()]))
            i += 2
        else:
            out.append(NUMBER_WORDS.get(t, tokens[i]))
            i += 1
    return out


def normalize_text(text, *, lowercase: bool = True, remove_articles: bool = True, remove_punctuation: bool = True,
                   number_words: bool = True) -> str:
    """Normalise a short answer for exact matching (see the module docstring). None -> ""."""
    s = "" if text is None else str(text)
    if lowercase:
        s = s.lower()
    if remove_punctuation:
        s = _RE_ACRONYM.sub(lambda m: m.group(0).replace(".", ""), s)
        s = _RE_THOUSANDS.sub("", s)
        s = _RE_LEADING_DOT.sub("0.", s)
        s = _RE_MINUS.sub(_SIGN, s)
        s = _RE_NON_DECIMAL_DOT.sub(" ", s)
        s = s.translate(_PUNCT_TABLE).replace(_SIGN, "-")
    tokens = s.split()
    if number_words:
        tokens = _number_tokens(tokens)
    if remove_articles:
        tokens = [t for t in tokens if t.lower() not in _ARTICLES]
    return " ".join(tokens)


def exact_match(prediction, references, normalize: bool = True) -> bool:
    """True if `prediction` equals any of `references` (a string or a list of strings) after normalisation."""
    refs = [references] if isinstance(references, str) or references is None else list(references)
    f = normalize_text if normalize else (lambda x: "" if x is None else str(x).strip())
    p = f(prediction)
    return any(p == f(r) for r in refs)


def _label_tail(lower: bool) -> str:
    """Regex of a label with named groups: br (bracketed, any case), up (bare upper case), lo (bare lower case that
    ends the clause; only if `lower`)."""
    tail = r"(?:[\(\[]\s*(?P<br>[A-Za-z])\s*[\)\]]|(?P<up>[A-Z])(?![A-Za-z0-9])"
    if lower:
        tail += r"|(?P<lo>[a-z])(?=[ \t]*(?:[\r\n.,;:!?)\]]|$))"
    return tail + ")"


_RE_MARKUP = re.compile(r"[*`]+")
_QUOTE = "['\"‘’“”]?"
_RE_WHOLE = re.compile(r"^\s*[\(\[]?\s*" + _QUOTE + r"\s*([A-Za-z])\s*" + _QUOTE + r"\s*[\)\]]?\s*[.:]?\s*$")
_RE_LEADING = re.compile(
    r"^\s*(?:(?P<b1>[\(\[]\s*(?P<l1>[A-Za-z])\s*[\)\]])\s*\S"
    r"|(?P<l2>[A-Z])(?:[.:),]|[ \t]+[-–—]|[-–—][ \t]+)\s*\S"
    r"|(?P<l3>[A-Z])[ \t]*\r?\n)")
_RE_ANSWER = re.compile(r"(?i:answer)(?![A-Za-z])\s*(?:(?i:is|was|would\s+be|be)\s*)?(?:[:=]\s*)?"
                        r"(?P<opt>(?i:option|choice)(?![A-Za-z])\s*)?" + _label_tail(True))
_RE_OPTION = re.compile(r"(?i:option|choice)(?![A-Za-z])\s*(?:(?i:is|was)\s*)?(?:[:=]\s*)?" + _label_tail(True))
_RE_IS = re.compile(r"(?<![A-Za-z])(?i:is|it's|it’s)\s+" + _label_tail(False))
_RE_HEDGE = re.compile(r"[ \t]*[.:)\]]?[ \t]*(?:,|/|&|\+|(?i:or|and)(?![A-Za-z]))[ \t]*" + _label_tail(False))
_RE_NEXT_WORD = re.compile(r"[ \t]+([A-Za-z]+)")
_RE_CLAUSE_END = re.compile(r"[ \t]*(?:$|[\r\n.,;:!?)\]])")
_RE_BRACKETED = re.compile(r"[\(\[]\s*([A-Z])\s*[\)\]]")


def _n_letters(n_choices: Optional[int], choices: Optional[Sequence[str]]) -> str:
    n = n_choices if n_choices is not None else (len(choices) if choices is not None else len(LETTERS))
    if not 1 <= n <= len(LETTERS):
        raise ValueError(f"number of choices must be in 1..{len(LETTERS)}, got {n}")
    return LETTERS[:n]


def _contains(phrase: str, text: str) -> bool:
    return bool(phrase) and re.search(r"(?<!\S)" + re.escape(phrase) + r"(?!\S)", text) is not None


def _article_like(text: str, end: int, letter: str) -> bool:
    """A bare "A" / "I" ending at `end` that is followed by a lower-case non-connective word (article / pronoun)."""
    if letter not in ("A", "I"):
        return False
    w = _RE_NEXT_WORD.match(text, end)
    return w is not None and w.group(1)[0].islower() and w.group(1).lower() not in _CONNECTIVES


def _hedged(text: str, end: int, letter: str, allowed: str) -> bool:
    """The label ending at `end` is directly followed by "or" / "and" / "," / "/" / "&" / "+" and another label."""
    h = _RE_HEDGE.match(text, end)
    if h is None:
        return False
    other = (h.group("br") or h.group("up")).upper()
    if h.group("up") and _article_like(text, h.end(), other):
        return False
    return other != letter and other in allowed


def _clause_ends(text: str, end: int) -> bool:
    if _RE_CLAUSE_END.match(text, end):
        return True
    w = _RE_NEXT_WORD.match(text, end)
    return w is not None and w.group(1).lower() in _CONNECTIVES


def extract_choice(response, n_choices: Optional[int] = None, choices: Optional[Sequence[str]] = None) -> Optional[str]:
    """Map a model response to an option letter ("A", "B", ...) or None (rules in the module docstring).

    Args:
        response: the model's answer text.
        n_choices: number of options (default: len(choices), else 26).
        choices: optional option texts in label order (enables rule 6).
    """
    if response is None:
        return None
    text = _RE_MARKUP.sub("", str(response))
    allowed = _n_letters(n_choices, choices)
    m = _RE_WHOLE.match(text)
    if m is not None and m.group(1).upper() in allowed:
        return m.group(1).upper()
    ambiguous = False
    m = _RE_LEADING.match(text)
    if m is not None:
        name = next(k for k in ("l1", "l2", "l3") if m.group(k))
        letter = m.group(name).upper()
        if letter in allowed:
            end = m.end("b1") if name == "l1" else m.end(name)
            others = set(_RE_BRACKETED.findall(text)) & set(allowed) - {letter}
            if not others and not _hedged(text, end, letter, allowed):
                return letter
            ambiguous = True
    for rule, rx in ((3, _RE_ANSWER), (4, _RE_OPTION), (5, _RE_IS)):
        for m in rx.finditer(text):
            g = m.groupdict()
            letter = (g["br"] or g["up"] or g.get("lo") or "").upper()
            if letter not in allowed or _hedged(text, m.end(), letter, allowed):
                continue
            if g["up"] and rule == 3 and not g["opt"] and _article_like(text, m.end(), letter):
                continue
            if g["up"] and rule == 5 and not _clause_ends(text, m.end()):
                continue
            return letter
    if choices is not None:
        resp = normalize_text(text)
        opts = [normalize_text(c) for c in choices]
        exact = [k for k, o in enumerate(opts) if o and o == resp]
        if len(exact) == 1:
            return LETTERS[exact[0]]
        hits = [k for k, o in enumerate(opts) if _contains(o, resp)]
        hits = [k for k in hits if not any(k2 != k and opts[k2] != opts[k] and _contains(opts[k], opts[k2])
                                           for k2 in hits)]
        if len(hits) == 1:
            return LETTERS[hits[0]]
    if not ambiguous:
        found = set(_RE_BRACKETED.findall(text)) & set(allowed)
        if len(found) == 1:
            return found.pop()
    return None


def answer_letter(answer) -> str:
    """Gold answer -> letter: a 0-based option index (int) or a label string ("B", "b", "(B)")."""
    if isinstance(answer, (int, np.integer)) and not isinstance(answer, bool):
        if not 0 <= int(answer) < len(LETTERS):
            raise ValueError(f"option index {answer} out of range")
        return LETTERS[int(answer)]
    m = _RE_WHOLE.match(str(answer))
    if m is None:
        raise ValueError(f"cannot read an option label from gold answer {answer!r}")
    return m.group(1).upper()


def _per_item(x, n: int, name: str):
    if x is None or isinstance(x, (int, np.integer)):
        return [x] * n
    x = list(x)
    if len(x) != n:
        raise ValueError(f"{name} has {len(x)} entries, expected {n}")
    return x


def multiple_choice_correct(responses, answers, choices=None, n_choices=None):
    """Per-item correctness of multiple-choice responses.

    Args:
        responses: (N,) model responses (strings).
        answers: (N,) gold answers (letters or 0-based indices).
        choices: optional (N,) per-item lists of option texts (rule 6 of `extract_choice`; None entries allowed).
            Options shared by every item are passed as [options] * N; a flat list of strings is rejected.
        n_choices: optional int or (N,) ints (number of options per item).

    Returns: (correct (N,) bool array, extracted (N,) list of letters or None).
    """
    responses = list(responses)
    n = len(responses)
    answers = _per_item(answers, n, "answers")
    if isinstance(choices, str):
        raise TypeError("choices must be (N,) per-item lists of option texts, got a string")
    choices = [None] * n if choices is None else _per_item(choices, n, "choices")
    if any(isinstance(c, str) for c in choices):
        raise ValueError("choices must hold one list of option texts per item, got a flat list of strings; for "
                         "options shared by all items pass [options] * len(responses)")
    n_choices = _per_item(n_choices, n, "n_choices")
    extracted = [extract_choice(r, k, c) for r, k, c in zip(responses, n_choices, choices)]
    correct = np.array([e is not None and e == answer_letter(a) for e, a in zip(extracted, answers)], bool)
    return correct, extracted


def accuracy(correct, categories=None) -> dict:
    """{"accuracy" [percent], "n", "n_correct"} plus, with categories ((N,) labels), "per_category":
    {category: {"accuracy", "n"}} (sorted by name) and "macro_accuracy" (mean over categories)."""
    c = np.asarray(correct, bool).ravel()
    out = {"accuracy": 100.0 * float(c.mean()) if c.size else _NAN, "n": int(c.size), "n_correct": int(c.sum())}
    if categories is not None:
        cats = [str(x) for x in categories]
        if len(cats) != c.size:
            raise ValueError(f"categories has {len(cats)} entries, expected {c.size}")
        per = {}
        for cat in sorted(set(cats)):
            sel = np.array([x == cat for x in cats], bool)
            per[cat] = {"accuracy": 100.0 * float(c[sel].mean()), "n": int(sel.sum())}
        out["per_category"] = per
        out["macro_accuracy"] = float(np.mean([v["accuracy"] for v in per.values()])) if per else _NAN
    return out


def score_multiple_choice(responses, answers, *, choices=None, n_choices=None, categories=None) -> dict:
    """`multiple_choice_correct` + `accuracy`; adds "n_unparsed" (responses mapped to no option) and "extracted"."""
    correct, extracted = multiple_choice_correct(responses, answers, choices, n_choices)
    out = accuracy(correct, categories)
    out["n_unparsed"] = int(sum(e is None for e in extracted))
    out["extracted"] = extracted
    return out


def score_exact_match(predictions, references, *, categories=None, normalize: bool = True) -> dict:
    """Exact-match accuracy: references is (N,) of strings or lists of acceptable strings. Returns `accuracy` dict."""
    predictions = list(predictions)
    references = _per_item(references, len(predictions), "references")
    correct = [exact_match(p, r, normalize) for p, r in zip(predictions, references)]
    return accuracy(correct, categories)


def _gt_mask(shape, gt) -> np.ndarray:
    nq, nc = shape
    if gt is None:
        if nq != nc:
            raise ValueError(f"a non-square similarity matrix {shape} needs gt")
        return np.eye(nq, dtype=bool)
    g = np.asarray(gt)
    if g.dtype == bool:
        if g.shape != (nq, nc):
            raise ValueError(f"gt mask {g.shape} must be {(nq, nc)}")
        return g
    if g.size and not np.issubdtype(g.dtype, np.integer):
        raise TypeError(f"gt must be None, (Nq,) integer indices or an (Nq, Nc) bool mask, got dtype {g.dtype}")
    g = g.astype(np.int64).ravel()
    if g.size != nq or (g.size and (g.min() < 0 or g.max() >= nc)):
        raise ValueError(f"gt indices must be (Nq,) = ({nq},) in [0, {nc})")
    m = np.zeros((nq, nc), bool)
    m[np.arange(nq), g] = True
    return m


def retrieval_ranks(sim, gt=None, ties: str = "pessimistic") -> np.ndarray:
    """1-based rank of the best-scoring correct item of every query.

    Args:
        sim: (Nq, Nc) similarity, higher = more similar (NaN = -inf).
        gt: None (item i is the match of query i; square sim), (Nq,) int index of the match, or (Nq, Nc) bool
            (several correct items per query).
        ties: "pessimistic" (incorrect items with an equal score rank above) or "optimistic" (below).

    Returns: (Nq,) int64 ranks in [1, Nc] (empty for Nq = 0).
    """
    if ties not in ("pessimistic", "optimistic"):
        raise ValueError(f"ties={ties!r} not in ('pessimistic', 'optimistic')")
    S = np.asarray(sim, np.float64)
    if S.ndim != 2:
        raise ValueError(f"sim must be (Nq, Nc), got {S.shape}")
    S = np.where(np.isnan(S), -np.inf, S)
    M = _gt_mask(S.shape, gt)
    if S.shape[0] == 0:
        return np.zeros(0, np.int64)
    if not M.any(1).all():
        raise ValueError("every query needs at least one correct item")
    best = np.max(np.where(M, S, -np.inf), axis=1, keepdims=True)
    above = (S >= best) if ties == "pessimistic" else (S > best)
    return (np.sum(above & ~M, axis=1) + 1).astype(np.int64)


def retrieval_metrics(sim, gt=None, ks: Sequence[int] = (1, 5, 10), ties: str = "pessimistic") -> dict:
    """{"R@k" [percent] for k in ks, "median_rank", "mean_rank", "n_queries"} (see `retrieval_ranks`); NaN values
    for an empty query set."""
    r = retrieval_ranks(sim, gt, ties)
    if r.size == 0:
        out = {f"R@{int(k)}": _NAN for k in ks}
        out.update(median_rank=_NAN, mean_rank=_NAN, n_queries=0)
        return out
    out = {f"R@{int(k)}": 100.0 * float(np.mean(r <= k)) for k in ks}
    out.update(median_rank=float(np.median(r)), mean_rank=float(np.mean(r)), n_queries=int(r.size))
    return out


def bidirectional_retrieval(sim, gt=None, ks: Sequence[int] = (1, 5, 10), ties: str = "pessimistic") -> dict:
    """Text->video and video->text retrieval from one (N_text, N_video) similarity matrix (gt as in
    `retrieval_ranks`, from the text side). Videos without any matching text are not video->text queries.
    Returns {"text_to_video": ..., "video_to_text": ...}."""
    S = np.asarray(sim, np.float64)
    if S.ndim != 2:
        raise ValueError(f"sim must be (N_text, N_video), got {S.shape}")
    M = _gt_mask(S.shape, gt)
    has = M.any(0)
    return {"text_to_video": retrieval_metrics(S, M, ks, ties),
            "video_to_text": retrieval_metrics(S.T[has], M.T[has], ks, ties)}
