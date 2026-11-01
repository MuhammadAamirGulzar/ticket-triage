"""Remove personal and payment data from ticket text.

Runs on every ticket before it is stored, used for training or passed to a
model, so training and prediction see the same placeholders. Patterns are
tuned for UK customers (phone numbers and postcodes).
"""
import re
from dataclasses import dataclass, field

# 13-19 digits, optionally separated by single spaces or dashes.
_CARD_CANDIDATE = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
_EXPIRY = re.compile(r"\bexp(?:iry|ires|\.)?\s*:?\s*(?:0?[1-9]|1[0-2])\s*/\s*(?:\d{4}|\d{2})\b", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
# UK numbers: 0 or +44 followed by 9-10 more digits, spaces allowed.
_PHONE = re.compile(r"(?<![\w+])(?:\+44\s?|0)(?:\d\s?){8,9}\d\b")
_POSTCODE = re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b")
_STREET = re.compile(
    r"\b\d{1,4}[a-zA-Z]?\s+(?:[A-Z][a-z]+\s){1,3}"
    r"(?:Lane|Street|St|Road|Rd|Avenue|Ave|Close|Drive|Way|Court|Place|Terrace|Crescent|Hill|Gardens|Row|Square)\b"
)


@dataclass
class Redaction:
    text: str
    found: list[str] = field(default_factory=list)

    @property
    def has_card(self) -> bool:
        return "CARD" in self.found


def luhn_valid(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def redact(text: str) -> Redaction:
    found: list[str] = []

    def _card(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 19 and luhn_valid(digits):
            found.append("CARD")
            return "[CARD]"
        return m.group()

    text = _CARD_CANDIDATE.sub(_card, text)
    for name, pattern in (
        ("EXPIRY", _EXPIRY),
        ("EMAIL", _EMAIL),
        ("PHONE", _PHONE),
        ("ADDRESS", _STREET),
        ("POSTCODE", _POSTCODE),
    ):
        text, n = pattern.subn(f"[{name}]", text)
        found.extend([name] * n)
    return Redaction(text=text, found=sorted(set(found)))
