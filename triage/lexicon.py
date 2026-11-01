"""Domain keyword features: one hit-count per category.

Forty tickets are too few for a text model to learn vocabulary such as
"invoice", "locked out" or "would be great" from scratch. These patterns
encode what a support lead would tell a new starter. The model still learns
how much to trust each count, so they are features, not hard rules.

Rule: every pattern must match at least one *labelled* ticket (enforced by
tests/test_lexicon.py). The new tickets are a blind test set and must never
shape this list. New patterns should come from newly labelled tickets.

Caveat: the patterns were written after reading the labelled tickets, so
cross-validated scores for models using them are optimistic. See the README.
Patterns are matched on lower-cased, redacted text.
"""
import re

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

from .config import CATEGORIES

PATTERNS: dict[str, list[str]] = {
    "Billing": [
        r"charg", r"invoice", r"refund", r"\bbill", r"payment", r"\bpay\b", r"\bpaid\b", r"\bcard\b",
        r"\[card\]", r"\bvat\b", r"\bcost", r"discount", r"bank transfer", r"downgrad", r"billing",
    ],
    "Account Access": [
        r"log ?in", r"password", r"locked", r"lock(ed)? out", r"two-factor", r"\bcode\b", r"reset",
        r"access", r"suspend", r"email address", r"get in\b",
    ],
    "Bug": [
        r"crash", r"blank", r"does nothing", r"wrong", r"disappear", r"\blate\b", r"cuts? off",
        r"after (your|the) update", r"shouldn.t happen",
    ],
    "Feature Request": [
        r"would love", r"would (be|help)", r"could (you|the app|we)", r"can we have", r"please add",
        r"\badd\b", r"feature", r"option to", r"\bsuggest",
    ],
    "Cancellation": [
        r"cancel", r"close (the|my|our) account", r"switching to", r"auto-?renew", r"we.re done",
    ],
}

def _compile(patterns: dict[str, list[str]]) -> dict[str, list[re.Pattern]]:
    return {cat: [re.compile(p) for p in patterns.get(cat, [])] for cat in CATEGORIES}


_COMPILED = _compile(PATTERNS)


def keyword_hits(text: str, compiled: dict[str, list[re.Pattern]] = _COMPILED) -> np.ndarray:
    t = text.lower()
    return np.array([sum(bool(p.search(t)) for p in compiled[c]) for c in CATEGORIES], dtype=float)


def pattern_usage(texts) -> list[tuple[str, str, int]]:
    """(category, pattern, number of texts it matches) for every pattern."""
    lowered = [t.lower() for t in texts]
    return [
        (cat, p.pattern, sum(bool(p.search(t)) for t in lowered))
        for cat, pats in _COMPILED.items()
        for p in pats
    ]


class KeywordFeatures(BaseEstimator, TransformerMixin):
    """Turns texts into a (n, 5) matrix of per-category keyword hit counts."""

    def __init__(self, patterns: dict[str, list[str]] | None = None):
        self.patterns = patterns

    def fit(self, X, y=None):
        self.compiled_ = _COMPILED if self.patterns is None else _compile(self.patterns)
        return self

    def transform(self, X):
        return np.vstack([keyword_hits(t, self.compiled_) for t in X])


class KeywordRules(BaseEstimator, ClassifierMixin):
    """Rules-only baseline: probability proportional to keyword hits plus smoothing."""

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def fit(self, X, y=None):
        self.classes_ = np.array(sorted(CATEGORIES))
        return self

    def predict_proba(self, X):
        hits = KeywordFeatures().fit(X).transform(X) + self.alpha
        proba = hits / hits.sum(axis=1, keepdims=True)
        order = [CATEGORIES.index(c) for c in self.classes_]
        return proba[:, order]

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
