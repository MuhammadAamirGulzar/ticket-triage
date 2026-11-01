"""Decide which tickets a person must see, and why.

A ticket goes to a human when the model is not confident enough, or when its
content needs a person whatever the category (payment data, a possible data
breach, an outage, a legal threat, or too little text to judge).
"""
import re
from dataclasses import dataclass

import pandas as pd

# Never auto-route below this, even if the (small) evaluation set suggests it is safe.
MIN_THRESHOLD = 0.5
DEFAULT_TARGET_ACCURACY = 0.98

_SECURITY = re.compile(
    r"can see (each other|other (people|staff)|someone else)|shouldn.t (be able to )?see|"
    r"breach|hack|unauthori[sz]ed|privacy|personal data|gdpr|data (leak|protection)"
)
_OUTAGE = re.compile(r"\bnobody\b|\bno ?one\b|\beveryone\b|\ball (our|my|the) (staff|sites|users|locations)\b|\boutage\b|\bis down\b")
_LEGAL = re.compile(r"solicitor|lawyer|legal action|\bsue\b|chargeback|ombudsman|trading standards|small claims")
_RETENTION = re.compile(r"downgrad|instead of cancel")


@dataclass
class Decision:
    needs_human: bool
    flags: list[str]
    reasons: list[str]


def choose_threshold(coverage: pd.DataFrame, target_accuracy: float = DEFAULT_TARGET_ACCURACY) -> float:
    """Lowest confidence threshold whose auto-routed accuracy meets the target."""
    ok = coverage[(coverage["auto_accuracy"] >= target_accuracy) & (coverage["threshold"] >= MIN_THRESHOLD)]
    if ok.empty:
        return float(coverage["threshold"].max())
    return float(ok["threshold"].min())


def decide(
    redacted_text: str, pii_found: list[str], predicted: str, confidence: float, threshold: float
) -> Decision:
    text = redacted_text.lower()
    flags: list[str] = []
    reasons: list[str] = []

    if confidence < threshold:
        reasons.append(f"low confidence ({confidence:.2f} < {threshold:.2f})")
    if len(text.split()) < 3:
        flags.append("TOO_SHORT")
        reasons.append("too little text to classify")
    if "CARD" in pii_found:
        flags.append("PAYMENT_DATA")
        reasons.append("contains card details: remove from ticket system and ask customer not to send them")
    if _SECURITY.search(text):
        flags.append("SECURITY")
        reasons.append("possible data exposure: escalate to security")
    if _OUTAGE.search(text):
        flags.append("OUTAGE")
        reasons.append("may affect many users: check for an incident")
    if _LEGAL.search(text):
        flags.append("LEGAL")
        reasons.append("mentions legal or chargeback action")

    # Informational flags: shown to the team but do not by themselves need a person.
    if set(pii_found) - {"CARD", "EXPIRY"}:
        flags.append("PERSONAL_DATA")
    if predicted == "Cancellation" or _RETENTION.search(text):
        flags.append("RETENTION")

    return Decision(needs_human=bool(reasons), flags=flags, reasons=reasons)
