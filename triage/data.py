"""Load tickets and build the cleaned labelled dataset.

The raw CSV is never edited. The cleaned file is regenerated from the raw file
plus dataset/label_audit.csv, so every label change is reviewable.
"""
from pathlib import Path

import pandas as pd

from . import config
from .redact import redact


def load_tickets(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = {"ticket_id", "text"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df["text"] = df["text"].str.strip()
    return df


def load_audit(path: Path = config.LABEL_AUDIT) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def build_clean_dataset(
    raw_path: Path = config.RAW_LABELLED, audit_path: Path = config.LABEL_AUDIT
) -> pd.DataFrame:
    raw = load_tickets(raw_path)
    audit = load_audit(audit_path).set_index("ticket_id")

    unknown = set(audit.index) - set(raw["ticket_id"])
    if unknown:
        raise ValueError(f"Audit refers to unknown tickets: {sorted(unknown)}")
    for _, row in audit.iterrows():
        actual = raw.loc[raw["ticket_id"] == row.name, "label"].item()
        if actual != row["original_label"]:
            raise ValueError(f"{row.name}: audit expects {row['original_label']!r}, raw has {actual!r}")

    out = raw.rename(columns={"label": "original_label"})
    out["label"] = out["ticket_id"].map(audit["corrected_label"]).fillna(out["original_label"])
    out["secondary_label"] = out["ticket_id"].map(audit["secondary_label"]).fillna("")
    out["ambiguous"] = out["ticket_id"].map(audit["ambiguous"]).fillna("0").astype(int)

    redactions = out["text"].map(redact)
    out["text"] = redactions.map(lambda r: r.text)
    out["pii_found"] = redactions.map(lambda r: ";".join(r.found))

    bad = set(out["label"]) - set(config.CATEGORIES)
    if bad:
        raise ValueError(f"Unknown labels after cleaning: {sorted(bad)}")
    return out[["ticket_id", "text", "label", "original_label", "secondary_label", "ambiguous", "pii_found"]]


def write_clean_dataset(path: Path = config.CLEAN_LABELLED) -> pd.DataFrame:
    df = build_clean_dataset()
    df.to_csv(path, index=False)
    return df


def load_clean(path: Path = config.CLEAN_LABELLED) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["ambiguous"] = df["ambiguous"].astype(int)
    return df
