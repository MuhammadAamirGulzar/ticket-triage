"""Candidate classifiers. Each is an sklearn Pipeline over redacted ticket text.

Vectorisers live inside the pipeline so they are re-fitted on each training
fold, which keeps test-fold vocabulary out of training.
"""
from typing import Callable

from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from .config import SEED

# Keep apostrophes and placeholder brackets: "can't" and "[CARD]" are useful signals.
TOKEN_PATTERN = r"(?u)\[?\b\w[\w']*\b\]?"


def _word_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, token_pattern=TOKEN_PATTERN)


def _word_char_tfidf() -> FeatureUnion:
    return FeatureUnion([
        ("word", _word_tfidf()),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
    ])


def majority() -> Pipeline:
    return Pipeline([("vec", CountVectorizer()), ("clf", DummyClassifier(strategy="most_frequent"))])


def nb_counts() -> Pipeline:
    return Pipeline([
        ("vec", CountVectorizer(ngram_range=(1, 2), token_pattern=TOKEN_PATTERN)),
        ("clf", MultinomialNB(alpha=0.5)),
    ])


def nb_tfidf() -> Pipeline:
    return Pipeline([("vec", _word_tfidf()), ("clf", MultinomialNB(alpha=0.1))])


def complement_nb() -> Pipeline:
    return Pipeline([("vec", _word_tfidf()), ("clf", ComplementNB(alpha=0.3))])


def logreg_word() -> Pipeline:
    return Pipeline([
        ("vec", _word_tfidf()),
        ("clf", LogisticRegression(C=10, max_iter=5000, class_weight="balanced", random_state=SEED)),
    ])


def logreg_word_char() -> Pipeline:
    return Pipeline([
        ("vec", _word_char_tfidf()),
        ("clf", LogisticRegression(C=10, max_iter=5000, class_weight="balanced", random_state=SEED)),
    ])


def svm_word_char() -> Pipeline:
    # Calibration gives the SVM probabilities; cv=3 is the most the smallest class allows.
    svm = LinearSVC(C=0.5, class_weight="balanced", random_state=SEED)
    return Pipeline([("vec", _word_char_tfidf()), ("clf", CalibratedClassifierCV(svm, cv=3))])


CANDIDATES: dict[str, Callable[[], Pipeline]] = {
    "majority_class": majority,
    "naive_bayes_counts": nb_counts,
    "naive_bayes_tfidf": nb_tfidf,
    "complement_nb_tfidf": complement_nb,
    "logreg_word_tfidf": logreg_word,
    "logreg_word_char_tfidf": logreg_word_char,
    "linear_svm_word_char_calibrated": svm_word_char,
}
