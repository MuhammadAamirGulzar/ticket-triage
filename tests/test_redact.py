from triage.redact import luhn_valid, redact


def test_card_and_expiry_from_t005_are_redacted():
    r = redact("Please update my card, new one is 4111 1111 1111 1111 exp 09/28.")
    assert r.text == "Please update my card, new one is [CARD] [EXPIRY]."
    assert r.has_card
    assert "EXPIRY" in r.found


def test_number_failing_luhn_is_not_treated_as_card():
    assert not luhn_valid("1234567812345678")
    r = redact("Order ref 1234 5678 1234 5678 is wrong.")
    assert "[CARD]" not in r.text
    assert not r.has_card


def test_email_and_uk_mobile_are_redacted():
    r = redact("Reach me on sarah.jones@gmail.com or 07700 900123.")
    assert r.text == "Reach me on [EMAIL] or [PHONE]."
    assert r.found == ["EMAIL", "PHONE"]


def test_international_phone_format():
    assert redact("Call +44 7700 900456 please").text == "Call [PHONE] please"


def test_street_address_and_postcode_are_redacted():
    r = redact("Please send the refund cheque to 14 Mill Lane, Leeds LS1 4AB.")
    assert r.text == "Please send the refund cheque to [ADDRESS], Leeds [POSTCODE]."


def test_ordinary_ticket_text_is_untouched():
    for text in [
        "Invoice shows 14 locations but we only have 12.",
        "Locked out after 3 attempts.",
        "Shift reminders are being sent an hour late since the clocks changed.",
        "We were billed for a location we closed in August.",
    ]:
        r = redact(text)
        assert r.text == text
        assert r.found == []
