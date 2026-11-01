"""Train the final model and produce predictions for new tickets."""
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config
from .evaluate import coverage_table, out_of_fold_proba
from .models import CANDIDATES, FINAL_MODEL
from .policy import choose_threshold, decide
from .redact import redact

OUTPUT_COLUMNS = [
    "ticket_id", "predicted_category", "confidence", "second_category", "second_confidence",
    "needs_human", "flags", "reason",
]


def train_final(df: pd.DataFrame, model_name: str = FINAL_MODEL) -> dict:
    """Fit on all labelled tickets; pick the threshold from leave-one-out predictions."""
    factory = CANDIDATES[model_name]
    proba = out_of_fold_proba(factory(), df["text"], df["label"])
    threshold = choose_threshold(coverage_table(proba, df["label"]))
    model = factory().fit(df["text"], df["label"])
    return {
        "model": model,
        "model_name": model_name,
        "threshold": threshold,
        "categories": config.CATEGORIES,
        "trained_on_rows": len(df),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def save_bundle(bundle: dict, path: Path = config.MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: Path = config.MODEL_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Run: python -m triage train")
    return joblib.load(path)


def predict_tickets(bundle: dict, tickets: pd.DataFrame) -> pd.DataFrame:
    """Return one row per ticket. Ticket text is deliberately not included."""
    model, threshold = bundle["model"], bundle["threshold"]
    redactions = [redact(t) for t in tickets["text"]]
    texts = [r.text for r in redactions]

    proba = pd.DataFrame(model.predict_proba(texts), columns=model.classes_)[config.CATEGORIES]
    ranked = np.argsort(-proba.to_numpy(), axis=1)
    cats = np.array(config.CATEGORIES)

    rows = []
    for i, (ticket_id, r) in enumerate(zip(tickets["ticket_id"], redactions)):
        top, second = cats[ranked[i, 0]], cats[ranked[i, 1]]
        conf, second_conf = proba.iloc[i][top], proba.iloc[i][second]
        decision = decide(r.text, r.found, top, conf, threshold)
        rows.append({
            "ticket_id": ticket_id,
            "predicted_category": top,
            "confidence": round(float(conf), 3),
            "second_category": second,
            "second_confidence": round(float(second_conf), 3),
            "needs_human": decision.needs_human,
            "flags": ";".join(decision.flags),
            "reason": "; ".join(decision.reasons) if decision.reasons else "auto-routed",
        })
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
