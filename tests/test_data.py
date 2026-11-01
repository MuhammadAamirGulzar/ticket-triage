from triage import config
from triage.data import build_clean_dataset


def test_clean_dataset_shape_and_labels():
    df = build_clean_dataset()
    assert len(df) == 40
    assert df["ticket_id"].is_unique
    assert set(df["label"]) <= set(config.CATEGORIES)


def test_corrections_are_applied_and_originals_kept():
    df = build_clean_dataset().set_index("ticket_id")
    assert df.loc["T007", "label"] == "Account Access"
    assert df.loc["T007", "original_label"] == "Billing"
    assert df.loc["T019", "label"] == "Feature Request"
    assert df.loc["T033", "label"] == "Bug"
    assert (df["label"] != df["original_label"]).sum() == 3


def test_ambiguous_tickets_are_marked():
    df = build_clean_dataset().set_index("ticket_id")
    assert set(df.index[df["ambiguous"] == 1]) == {"T006", "T015", "T030", "T039", "T040"}
    assert df.loc["T015", "secondary_label"] == "Billing"


def test_no_personal_data_left_in_clean_text():
    df = build_clean_dataset()
    text = " ".join(df["text"])
    assert "4111" not in text
    assert "@" not in text
    assert "07700" not in text
    assert "Mill Lane" not in text
    assert df.set_index("ticket_id").loc["T005", "pii_found"] == "CARD;EXPIRY"
