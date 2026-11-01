"""Domain keyword features: one hit-count per category.

Forty tickets are too few for a text model to learn vocabulary such as
"invoice", "locked out" or "would be great" from scratch. These patterns
encode what a support lead would tell a new starter. The model still learns
how much to trust each count, so they are features, not hard rules.

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
        r"\[card\]", r"\bvat\b", r"receipt", r"pric", r"\bcost", r"discount", r"direct debit",
        r"bank transfer", r"downgrad", r"upgrad", r"\bfee", r"overcharg", r"billing",
    ],
    "Account Access": [
        r"log ?in", r"sign ?in", r"password", r"locked", r"lock(ed)? out", r"two-factor", r"\b2fa\b",
        r"verification", r"\bcode\b", r"reset", r"access", r"suspend", r"invite", r"email address",
        r"username", r"get in\b", r"can.t get into",
    ],
    "Bug": [
        r"crash", r"error", r"blank", r"broken", r"does ?n.t work", r"not working", r"does nothing",
        r"wrong", r"disappear", r"keep\w* coming back", r"n.t load", r"not load", r"\blate\b", r"slow",
        r"freez", r"glitch", r"\bbug", r"cuts? off", r"after (your|the) update", r"shouldn.t happen",
        r"stopped working", r"won.t (open|save|load)",
    ],
    "Feature Request": [
        r"would love", r"would (be|help)", r"it.d be", r"could (you|the app|we)", r"can we have",
        r"please add", r"\badd\b", r"feature", r"\bwish", r"option to", r"ability to", r"integrat",
        r"support for", r"\bsuggest", r"would like to be able",
    ],
    "Cancellation": [
        r"cancel", r"close (the|my|our) account", r"switching to", r"\bleav", r"auto-?renew",
        r"terminat", r"end (our|the|my) (plan|subscription|contract)", r"no longer need",
        r"we.re done", r"\bsold\b", r"closing down", r"delete (my|our) account", r"unsubscribe",
    ],
}

_COMPILED = {cat: [re.compile(p) for p in pats] for cat, pats in PATTERNS.items()}


def keyword_hits(text: str) -> np.ndarray:
    t = text.lower()
    return np.array([sum(bool(p.search(t)) for p in _COMPILED[c]) for c in CATEGORIES], dtype=float)


class KeywordFeatures(BaseEstimator, TransformerMixin):
    """Turns texts into a (n, 5) matrix of per-category keyword hit counts."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.vstack([keyword_hits(t) for t in X])


class KeywordRules(BaseEstimator, ClassifierMixin):
    """Rules-only baseline: probability proportional to keyword hits plus smoothing."""

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def fit(self, X, y=None):
        self.classes_ = np.array(sorted(CATEGORIES))
        return self

    def predict_proba(self, X):
        hits = KeywordFeatures().transform(X) + self.alpha
        proba = hits / hits.sum(axis=1, keepdims=True)
        order = [CATEGORIES.index(c) for c in self.classes_]
        return proba[:, order]

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
