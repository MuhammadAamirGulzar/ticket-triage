import re

import pandas as pd
import pytest

from triage import config
from triage.data import load_clean, load_tickets
from triage.predict import OUTPUT_COLUMNS, predict_tickets, train_final


@pytest.fixture(scope="module")
def bundle():
    return train_final(load_clean())


@pytest.fixture(scope="module")
def predictions(bundle):
    return predict_tickets(bundle, load_tickets(config.RAW_NEW))


def test_one_row_per_ticket_with_expected_columns(predictions):
    assert list(predictions.columns) == OUTPUT_COLUMNS
    assert list(predictions["ticket_id"]) == [f"N{i:02d}" for i in range(1, 11)]
    assert set(predictions["predicted_category"]) <= set(config.CATEGORIES)
    assert predictions["confidence"].between(0, 1).all()
    assert (predictions["confidence"] >= predictions["second_confidence"]).all()


def test_output_contains_no_personal_data(predictions):
    blob = predictions.to_csv(index=False)
    assert "@" not in blob
    assert not re.search(r"\d{5,}", blob.replace(".", ""))


def test_personal_data_is_flagged(predictions):
    n06 = predictions.set_index("ticket_id").loc["N06"]
    assert "PERSONAL_DATA" in n06["flags"]


def test_card_number_forces_human(bundle):
    tickets = pd.DataFrame({"ticket_id": ["X1"], "text": ["My new card is 4111 1111 1111 1111, please update."]})
    row = predict_tickets(bundle, tickets).iloc[0]
    assert row["needs_human"]
    assert "PAYMENT_DATA" in row["flags"]
    assert "4111" not in row.to_string()


def test_threshold_is_within_policy_bounds(bundle):
    assert 0.5 <= bundle["threshold"] <= 0.95
