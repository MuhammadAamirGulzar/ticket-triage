"""Guard against test-set leakage into the hand-written keyword list."""
from triage.data import load_clean
from triage.lexicon import pattern_usage


def test_every_keyword_pattern_is_grounded_in_labelled_data():
    ungrounded = [(cat, p) for cat, p, n in pattern_usage(load_clean()["text"]) if n == 0]
    assert ungrounded == [], (
        "Keyword patterns must come from labelled tickets only, never from the blind "
        f"test set. These match no labelled ticket: {ungrounded}"
    )
