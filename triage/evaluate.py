"""Measure classifiers on the labelled tickets.

Everything here is out-of-fold: a ticket is only ever scored by a model that
did not see it during training.

- Model comparison: repeated stratified 5-fold CV (10 repeats), mean ± std.
- Selection bias check: nested CV, where the inner loop picks the best
  candidate and the outer loop scores that choice.
- Error analysis for the final model: leave-one-out CV, so every ticket gets
  exactly one out-of-fold prediction.
"""
import math
import warnings
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import (
    LeaveOneOut,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline

from .config import CATEGORIES, SEED

warnings.filterwarnings("ignore", category=ConvergenceWarning)

SCORING = {"accuracy": "accuracy", "macro_f1": "f1_macro", "log_loss": "neg_log_loss"}


def compare_models(
    texts: pd.Series, labels: pd.Series, candidates: dict[str, Callable[[], Pipeline]],
    n_splits: int = 5, n_repeats: int = 10,
) -> pd.DataFrame:
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    rows = []
    for name, factory in candidates.items():
        scores = cross_validate(factory(), texts, labels, cv=cv, scoring=SCORING)
        rows.append({
            "model": name,
            "accuracy_mean": scores["test_accuracy"].mean(),
            "accuracy_std": scores["test_accuracy"].std(),
            "macro_f1_mean": scores["test_macro_f1"].mean(),
            "macro_f1_std": scores["test_macro_f1"].std(),
            "log_loss_mean": -scores["test_log_loss"].mean(),
        })
    return pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)


def nested_cv(
    texts: pd.Series, labels: pd.Series, candidates: dict[str, Callable[[], Pipeline]],
    outer_repeats: int = 3,
) -> dict:
    """Score the whole 'pick the best candidate' procedure, not one lucky model."""
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=outer_repeats, random_state=SEED)
    inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
    accs, chosen = [], []
    for train_idx, test_idx in outer.split(texts, labels):
        x_tr, y_tr = texts.iloc[train_idx], labels.iloc[train_idx]
        inner_f1 = {
            name: cross_validate(f(), x_tr, y_tr, cv=inner, scoring="f1_macro")["test_score"].mean()
            for name, f in candidates.items()
        }
        best = max(inner_f1, key=inner_f1.get)
        model = candidates[best]().fit(x_tr, y_tr)
        accs.append(accuracy_score(labels.iloc[test_idx], model.predict(texts.iloc[test_idx])))
        chosen.append(best)
    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "chosen": pd.Series(chosen).value_counts().to_dict(),
    }


def out_of_fold_proba(model: Pipeline, texts: pd.Series, labels: pd.Series) -> pd.DataFrame:
    """Leave-one-out probabilities, one row per ticket, one column per category."""
    proba = cross_val_predict(clone(model), texts, labels, cv=LeaveOneOut(), method="predict_proba")
    classes = sorted(labels.unique())
    return pd.DataFrame(proba, columns=classes, index=texts.index)[CATEGORIES]


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (centre - half, centre + half)


def coverage_table(proba: pd.DataFrame, y_true: pd.Series, thresholds=None) -> pd.DataFrame:
    """For each confidence threshold: share of tickets auto-routed and accuracy on them."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.20, 0.96, 0.05), 2)
    conf = proba.max(axis=1).to_numpy()
    correct = (proba.idxmax(axis=1) == y_true).to_numpy()
    rows = []
    for t in thresholds:
        auto = conf >= t
        n_auto = int(auto.sum())
        rows.append({
            "threshold": t,
            "auto_routed": n_auto,
            "coverage": n_auto / len(conf),
            "auto_accuracy": correct[auto].mean() if n_auto else float("nan"),
            "sent_to_human": int((~auto).sum()),
            "errors_auto_routed": int((auto & ~correct).sum()),
        })
    return pd.DataFrame(rows)


def per_ticket_results(df: pd.DataFrame, proba: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    ranked = np.argsort(-proba.to_numpy(), axis=1)
    cats = np.array(CATEGORIES)
    out = df[["ticket_id", "text", label_col, "original_label", "secondary_label", "ambiguous"]].copy()
    out = out.rename(columns={label_col: "true_label"})
    out["predicted"] = cats[ranked[:, 0]]
    out["confidence"] = proba.max(axis=1).round(3).to_numpy()
    out["second_choice"] = cats[ranked[:, 1]]
    out["second_confidence"] = np.sort(proba.to_numpy(), axis=1)[:, -2].round(3)
    out["correct"] = out["predicted"] == out["true_label"]
    return out


def summary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    k, n = int((y_true == y_pred).sum()), len(y_true)
    lo, hi = wilson_interval(k, n)
    return {
        "n": n,
        "correct": k,
        "accuracy": k / n,
        "accuracy_ci95": (lo, hi),
        "macro_f1": f1_score(y_true, y_pred, labels=CATEGORIES, average="macro", zero_division=0),
        "per_class": classification_report(
            y_true, y_pred, labels=CATEGORIES, output_dict=True, zero_division=0
        ),
        "confusion": pd.DataFrame(
            confusion_matrix(y_true, y_pred, labels=CATEGORIES),
            index=[f"true: {c}" for c in CATEGORIES],
            columns=[f"pred: {c}" for c in CATEGORIES],
        ),
    }
