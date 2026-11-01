"""Shared constants: categories, file paths and random seed."""
from pathlib import Path

CATEGORIES = ["Billing", "Bug", "Account Access", "Feature Request", "Cancellation"]

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
RAW_LABELLED = DATASET_DIR / "tickets_labelled.csv"
RAW_NEW = DATASET_DIR / "tickets_new.csv"
LABEL_AUDIT = DATASET_DIR / "label_audit.csv"
CLEAN_LABELLED = DATASET_DIR / "tickets_labelled_clean.csv"

REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "classifier.joblib"
PREDICTIONS_DIR = ROOT / "predictions"

SEED = 42
