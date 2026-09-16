"""Reject evaluation prompts whose text is in the training corpus.

Selecting held-out data by index is not enough. The calibration pool is indices
0..95,999 of OpenThoughts3 shuffled with seed 42 and evaluation draws from
200,000 up, which makes the two ranges disjoint -- but OpenThoughts3 aggregates
many sources, so one problem can occupy several indices and a prompt from
200,000 can be a duplicate of one the model trained on. Disjoint indices, shared
text.

So the check has to be on content. This builds the set of problems the training
file actually contains and exposes a predicate for candidate prompts.

Matching notes, both learned the hard way:
  * The training file is JSONL, {"text": "<|im_start|>user ..."}, so backslashes
    are escaped on the raw line. Needles carrying LaTeX never match the line as
    bytes. Always decode the row first.
  * Normalisation drops chat-template control tokens and collapses whitespace,
    so the same problem under different line wrapping still matches.

Exact normalised equality catches verbatim duplicates. A shingle overlap catches
the near-duplicates -- same problem, one reworded sentence -- which exact
matching misses and which are just as contaminating.
"""
import json
import re

_CTRL = re.compile(r"<\|[^|]*\|>")
_WS = re.compile(r"\s+")


def norm(s):
    return _WS.sub(" ", _CTRL.sub(" ", s)).strip()


def user_body(text):
    """The user turn of a templated conversation, normalised."""
    t = text.split("<|im_start|>user", 1)[-1]
    t = t.split("<|im_start|>", 1)[0]
    return norm(t)


def shingles(text, n=48, step=24):
    """Overlapping character windows, used for near-duplicate detection."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(0, len(text) - n + 1, step)}


class TrainingIndex:
    """Normalised problems in the training file, plus their shingles."""

    def __init__(self, path, shingle_n=48, shingle_step=24, verbose=True):
        self.exact = set()
        self.shingles = set()
        self.n_rows = 0
        for line in open(path, errors="ignore"):
            try:
                text = json.loads(line)["text"]
            except Exception:
                continue
            self.n_rows += 1
            b = user_body(text)
            if not b:
                continue
            self.exact.add(b)
            self.shingles |= shingles(b, shingle_n, shingle_step)
        self._n, self._step = shingle_n, shingle_step
        if verbose:
            print(f"[heldout] indexed {self.n_rows} training rows, "
                  f"{len(self.exact)} problems, {len(self.shingles)} shingles",
                  flush=True)

    def verdict(self, problem_text, overlap=0.30):
        """(contaminated, reason). `problem_text` is a raw user problem."""
        b = norm(problem_text)
        if not b:
            return False, "empty"
        if b in self.exact:
            return True, "exact duplicate"
        sh = shingles(b, self._n, self._step)
        if not sh:
            return False, "too short to shingle"
        frac = len(sh & self.shingles) / len(sh)
        if frac >= overlap:
            return True, f"{frac:.0%} shingle overlap"
        return False, f"{frac:.0%} shingle overlap"

    def is_contaminated(self, problem_text, overlap=0.30):
        return self.verdict(problem_text, overlap)[0]
