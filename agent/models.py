"""
Typed domain models. Keeping these as explicit dataclasses (rather than
passing raw dicts / DataFrame rows around) is what lets validators.py and
the planner reason about the data with actual type safety, and is what
makes the pipeline readable file-to-file.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Optional, List, Dict


@dataclass
class FinancialEvent:
    event_id: str
    user_id: str
    event_type: str
    description: str
    category: str
    direction: str          # credit | debit | non_cash
    amount: Optional[Decimal]   # None until resolved (blank-amount rows)
    currency: str
    event_date: date
    settlement_date: Optional[date]
    status: str
    linked_event_id: Optional[str]
    flexibility: str
    minimum_allowed_amount: Optional[Decimal]

    # --- fields populated during reconciliation, not from the raw CSV ---
    amount_home: Optional[Decimal] = None   # amount converted to user's home_currency
    resolved_by: Optional[str] = None       # 'csv' | 'image:<image_id>' | 'message:<id>'
    excluded: bool = False                  # True if reconciliation drops this event
    exclusion_reason: str = ""

    def cash_date(self) -> date:
        """The date this event actually moves cash: settlement_date if present."""
        return self.settlement_date or self.event_date


@dataclass
class UserProfile:
    user_id: str
    home_currency: str
    current_available_balance: Decimal
    minimum_balance_to_keep: Decimal
    financial_priorities: List[str]
    expense_categories_to_protect: List[str]
    expense_categories_user_is_willing_to_reduce: List[str]
    expense_categories_user_is_willing_to_stop: List[str]
    payment_methods_user_will_consider: List[str]
    max_installment_months: Optional[int]


@dataclass
class PaymentOption:
    payment_option_id: str
    request_id: str
    payment_method: str          # full_payment | installments
    payment_amount: Decimal      # amount per installment (or full amount)
    number_of_payments: int
    first_payment_date: date
    payment_frequency_days: Optional[int]
    financing_fee: Decimal
    total_payable_amount: Decimal

    def schedule(self) -> List[tuple]:
        """List of (date, amount) tuples in chronological order."""
        out = []
        d = self.first_payment_date
        freq = self.payment_frequency_days or 0
        for i in range(self.number_of_payments):
            dt = d if i == 0 else _add_days(self.first_payment_date, freq * i)
            out.append((dt, self.payment_amount))
        return out


def _add_days(d: date, n: int) -> date:
    from datetime import timedelta
    return d + timedelta(days=n)


@dataclass
class Message:
    message_id: str
    user_id: str
    request_id: Optional[str]
    related_event_id: Optional[str]
    sent_at: str
    source_type: str
    message_text: str


@dataclass
class ImageRef:
    image_id: str
    user_id: str
    request_id: Optional[str]
    related_event_id: Optional[str]

    @property
    def path_suffix(self) -> str:
        return f"media/images/{self.image_id}.png"


@dataclass
class Request:
    request_id: str
    user_id: str
    request_date: date
    request_type: str
    requested_amount: Decimal
    desired_completion_date: date
    allows_partial_payment: bool
    request_text: str
    # ground truth, only present for sample_requests.csv rows
    expected: Optional[Dict[str, str]] = None


@dataclass
class Fact:
    """
    A single structured claim extracted from a message or image, ready to be
    applied to the raw event timeline by reconciliation.py.

    `kind` is one of:
      amount_fill          - resolves a blank event.amount (image evidence)
      income_change         - future recurring income changes to new_amount from new_date
      income_stop            - future recurring income for this series stops entirely
      income_new              - a new income series starts at new_amount from new_date
      income_unconfirmed     - the next payout in a variable earnings stream (gig/platform
                                  work) is pending and not withdrawable, so FUTURE occurrences
                                  of that stream must not be projected as cash
      income_exclude_component - a named sub-component (bonus/commission) is NOT confirmed;
                                  do not add it to projected income
      expense_rate_change    - future recurring expense (e.g. rent) changes by a multiplier
                                  or to a new absolute amount from new_date
      obligation_persists    - a failed debit will be retried; keep it as a future liability
      exclude_pending_credit  - an explicit reminder that a credit is not yet real (no-op,
                                  since pending credits are excluded by default, but logged
                                  for auditability)
      confirm_settled          - a specific event is confirmed final/settled cash (no-op
                                  beyond audit trail; status-based inclusion already covers it)
      date_shift                - a scheduled event's date moves to new_date
      no_op                     - message carried no actionable financial fact (logged only)
    """
    kind: str
    user_id: str
    source: str                      # "message:<id>" or "image:<id>"
    related_event_id: Optional[str] = None
    request_id: Optional[str] = None
    new_amount: Optional[Decimal] = None
    new_currency: Optional[str] = None
    new_date: Optional[date] = None
    multiplier: Optional[Decimal] = None
    category_hint: Optional[str] = None
    description_hint: Optional[str] = None
    confidence: str = "confirmed"    # confirmed | estimated | low
    note: str = ""


@dataclass
class OutputRow:
    request_id: str
    amount_safe_to_pay: Decimal
    affordability_status: str
    recommended_payment_method: str
    payment_plan: str
    earliest_date_for_full_payment: str   # "" or YYYY-MM-DD
    spending_changes_needed: str
    decision_explanation: str

    def as_csv_row(self) -> list:
        return [
            self.request_id,
            format(self.amount_safe_to_pay, "f"),
            self.affordability_status,
            self.recommended_payment_method,
            self.payment_plan,
            self.earliest_date_for_full_payment,
            self.spending_changes_needed,
            self.decision_explanation,
        ]
