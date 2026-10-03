"""
Central configuration and controlled vocabularies for the Buy-or-Wait agent.

Every "magic" constant used more than once lives here so the rest of the
codebase reads declaratively and the rules in problem_statement.md map
one-to-one onto named constants instead of being scattered as literals.
"""

from __future__ import annotations
from decimal import Decimal, ROUND_HALF_UP
import os

# ---------------------------------------------------------------------------
# Forecast window
# ---------------------------------------------------------------------------
# problem_statement.md, "90-Day Safety Check": forecast is fixed at 90 days,
# anchored at request_date (not re-anchored to a later candidate payment date).
FORECAST_DAYS = 90

# Default assumed spacing (days) for a recurring series that has only a
# single historical occurrence and therefore no observable interval yet.
# Used only to size *one* forward projection so the series still exists in
# the forecast; never used to invent amounts.
DEFAULT_RECURRENCE_INTERVAL_DAYS = 30

# A series is treated as "regular" (safe to project on a fixed cadence) when
# the coefficient of variation of its historical inter-event gaps is below
# this threshold. Irregular series still get *one* forward projection (see
# recurrence.py) but are not repeated every N days inside the window.
RECURRENCE_REGULARITY_CV_MAX = Decimal("0.35")

# ---------------------------------------------------------------------------
# Output enums (verbatim from problem_statement.md "Allowed values")
# ---------------------------------------------------------------------------
AFFORDABILITY_STATUSES = {
    "affordable_now",
    "affordable_with_plan",
    "affordable_later",
    "not_affordable",
}

PAYMENT_METHODS = {
    "full_payment",
    "partial_payment",
    "installments",
    "wait",
    "not_recommended",
}

REQUEST_TYPES = {
    "purchase", "travel", "education", "family_transfer", "debt_repayment",
    "investment", "housing", "emergency_expense", "other",
}

# financial_events.csv controlled vocab (verified against the actual data)
EVENT_TYPES = {
    "expense", "refund", "income", "investment_purchase", "investment_sale",
    "investment_valuation", "debt_payment", "subscription",
}
DIRECTIONS = {"credit", "debit", "non_cash"}
STATUSES = {"settled", "pending", "scheduled", "cancelled", "failed", "unrealized"}
FLEXIBILITIES = {"fixed", "reducible", "stoppable", "reducible_or_stoppable"}

REDUCIBLE_FLEX = {"reducible", "reducible_or_stoppable"}
STOPPABLE_FLEX = {"stoppable", "reducible_or_stoppable"}

# Cash states to include in the forward-looking balance simulation.
# problem_statement.md: "Ignore pending credits, failed or cancelled
# transactions, duplicate records, and unrealized investments."
# -> settled and scheduled are always included; pending is included only
#    for debits (a pending charge is a real, conservative liability); failed
#    is excluded UNLESS a message states the obligation will be retried
#    (handled in reconciliation.py, not here).
INCLUDED_STATUSES_ALWAYS = {"settled", "scheduled"}
INCLUDED_STATUS_DEBIT_ONLY = {"pending"}
EXCLUDED_STATUSES = {"cancelled", "failed", "unrealized"}

CURRENCIES = {"INR", "ZAR", "IDR", "USD", "EUR"}

# ---------------------------------------------------------------------------
# Money handling
# ---------------------------------------------------------------------------
# All monetary arithmetic uses Decimal, never float, to avoid binary
# floating point rounding drift over a 90-day, multi-event simulation.
MONEY_QUANT = Decimal("0.01")


def money(x) -> Decimal:
    """Coerce any numeric-ish input to a 2dp Decimal. Never routes through float."""
    if x is None or x == "":
        raise ValueError("money() received an empty value; blank amounts must be "
                          "resolved via evidence extraction before reaching money().")
    if isinstance(x, Decimal):
        d = x
    else:
        d = Decimal(str(x))
    return d.quantize(MONEY_QUANT, rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Paths (resolved relative to the repo root so the tool runs from anywhere)
# ---------------------------------------------------------------------------
CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = CODE_DIR  # project root is where main.py and agent/ live
DATASET_DIR = os.path.join(REPO_ROOT, "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "media", "images")
OUTPUT_CSV = os.path.join(REPO_ROOT, "output.csv")

# ---------------------------------------------------------------------------
# Optional LLM backend (extraction-only — see agent/llm_client.py).
# Disabled gracefully when no key is present; the rule-based extractor
# handles the full known template catalog on its own.
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
LLM_EXTRACTION_ENABLED = bool(ANTHROPIC_API_KEY)
# Right-sized model for short, structured extraction calls (not the final
# financial decision, which is always made by deterministic code).
LLM_TEXT_MODEL = os.environ.get("AGENT_LLM_TEXT_MODEL", "claude-haiku-4-5-20251001")
LLM_VISION_MODEL = os.environ.get("AGENT_LLM_VISION_MODEL", "claude-haiku-4-5-20251001")
