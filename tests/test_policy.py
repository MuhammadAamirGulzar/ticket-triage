import pandas as pd

from triage.policy import choose_threshold, decide


def test_confident_plain_ticket_is_auto_routed():
    d = decide("App crashes on Android when I swap a shift.", [], "Bug", 0.97, threshold=0.85)
    assert not d.needs_human
    assert d.flags == []


def test_low_confidence_needs_human():
    d = decide("Our account was suspended but we paid on time.", [], "Account Access", 0.51, threshold=0.85)
    assert d.needs_human
    assert "low confidence" in d.reasons[0]


def test_card_details_always_need_human_even_when_confident():
    d = decide("Please update my card, new one is [CARD] [EXPIRY].", ["CARD", "EXPIRY"], "Billing", 0.99, 0.85)
    assert d.needs_human
    assert "PAYMENT_DATA" in d.flags
    assert "PERSONAL_DATA" not in d.flags


def test_security_issue_is_escalated():
    d = decide("Staff can see each other's pay rates, that shouldn't happen.", [], "Bug", 0.95, 0.85)
    assert d.needs_human
    assert "SECURITY" in d.flags


def test_outage_is_escalated():
    d = decide("After your update yesterday nobody at our Bristol site can log in.", [], "Bug", 0.9, 0.85)
    assert d.needs_human
    assert "OUTAGE" in d.flags


def test_personal_data_and_retention_are_informational():
    d = decide("Cancel please, reach me on [EMAIL].", ["EMAIL"], "Cancellation", 0.95, 0.85)
    assert not d.needs_human
    assert set(d.flags) == {"PERSONAL_DATA", "RETENTION"}


def test_choose_threshold_respects_target_and_floor():
    cov = pd.DataFrame({
        "threshold": [0.3, 0.5, 0.7, 0.85, 0.9],
        "auto_accuracy": [1.0, 0.95, 0.97, 1.0, 1.0],
    })
    assert choose_threshold(cov, target_accuracy=0.98) == 0.85
    assert choose_threshold(cov, target_accuracy=0.96) == 0.7
    # 0.3 meets the target but is below the safety floor.
    assert choose_threshold(cov, target_accuracy=0.5) == 0.5
