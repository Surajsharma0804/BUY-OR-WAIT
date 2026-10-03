"""
Decision explanations.

These are graded on "usefulness and consistency", so they are generated
deterministically from the SAME numbers that produced the decision -- never
narrated separately. That guarantees the explanation can never contradict
the structured columns, which is the usual failure mode when an LLM writes
the justification independently of the computation.

The phrasing follows the style of the worked examples in
dataset/sample_requests.csv, e.g.
    "Pay ZAR 25,256 today. This leaves at least ZAR 18,000 available over
     the next 90 days."
    "Use 3 installments of IDR 15,952,906.67, starting 8 August 2025."
    "Do not make this payment by 12 January 2026. None of the available
     options keeps the ZAR 13,100 minimum protected."
"""

from __future__ import annotations
from datetime import date
from decimal import Decimal
from typing import Optional

from .models import Request, UserProfile
from .config import money


def _money_str(ccy: str, amount: Decimal) -> str:
    q = money(amount)
    whole = q == q.to_integral_value()
    s = f"{q:,.0f}" if whole else f"{q:,.2f}"
    return f"{ccy} {s}"


def _date_str(d: date) -> str:
    # "8 August 2025" -- matches the sample explanations' style
    return f"{d.day} {d.strftime('%B')} {d.year}"


def build_explanation(req: Request, profile: UserProfile, plan, status: str,
                       amount_safe: Decimal, requested: Decimal,
                       earliest: Optional[date], tl) -> str:
    ccy = profile.home_currency
    minimum = money(profile.minimum_balance_to_keep)
    parts = []

    changes_txt = _changes_sentence(plan, ccy)

    if plan.method == "full_payment":
        if changes_txt:
            parts.append(f"{changes_txt}, then pay {_money_str(ccy, requested)} today.")
        else:
            parts.append(f"Pay {_money_str(ccy, requested)} today.")
        parts.append(f"This leaves at least {_money_str(ccy, minimum)} available "
                      f"over the next 90 days.")

    elif plan.method == "installments":
        n = plan.num_payments
        per = plan.schedule[0][1]
        start = plan.schedule[0][0]
        if changes_txt:
            parts.append(f"{changes_txt}, then use {n} installments of "
                          f"{_money_str(ccy, per)}, starting {_date_str(start)}.")
        else:
            parts.append(f"Use {n} installments of {_money_str(ccy, per)}, "
                          f"starting {_date_str(start)}.")
        parts.append(f"This leaves at least {_money_str(ccy, minimum)} available.")

    elif plan.method == "partial_payment":
        first_d, first_a = plan.schedule[0]
        second_d, second_a = plan.schedule[1]
        parts.append(f"Pay {_money_str(ccy, first_a)} today and the remaining "
                      f"{_money_str(ccy, second_a)} on {_date_str(second_d)}.")
        parts.append(f"This completes the full request and keeps the "
                      f"{_money_str(ccy, minimum)} minimum protected.")

    elif plan.method == "wait":
        pay_date = plan.schedule[0][0]
        parts.append(f"Pay {_money_str(ccy, requested)} in full on {_date_str(pay_date)}.")
        parts.append(f"Paying earlier would take the balance below the "
                      f"{_money_str(ccy, minimum)} minimum.")

    else:  # not_recommended
        capacity_exists = amount_safe > 0 and (earliest is not None or amount_safe >= requested)
        if capacity_exists:
            # Real capacity exists (now, in full, or by `earliest`) but no
            # payment method the user accepts (or, for a specific request,
            # allows_partial_payment=False) can capture it -- distinct from
            # genuinely never being safe, and must not be described the same
            # way, or the explanation contradicts earliest_date_for_full_payment
            # / amount_safe_to_pay in the very same row.
            parts.append(f"Do not proceed with the {_money_str(ccy, requested)} request "
                         f"under a method you currently accept.")
            if earliest is not None and earliest > req.request_date:
                parts.append(f"{_money_str(ccy, amount_safe)} is available today, and the full amount "
                             f"is projected to be safe by {_date_str(earliest)}, but no payment method "
                             f"you accept can complete this request on those terms.")
            else:
                parts.append(f"{_money_str(ccy, amount_safe)} is available today, but no payment "
                             f"method you accept can complete this request safely.")
        elif amount_safe > 0:
            parts.append(f"Do not proceed with the {_money_str(ccy, requested)} request.")
            parts.append(f"Although {_money_str(ccy, amount_safe)} is available today, "
                          f"the full amount cannot be completed safely within 90 days.")
        else:
            parts.append(f"Do not make this payment by "
                          f"{_date_str(req.desired_completion_date)}.")
            parts.append(f"None of the available options keeps the "
                          f"{_money_str(ccy, minimum)} minimum protected.")

    text = " ".join(parts)
    # Keep explanations short and single-line for CSV cleanliness.
    return " ".join(text.split())


def _changes_sentence(plan, ccy: str) -> str:
    if not plan.spending_changes:
        return ""
    bits = []
    for c in plan.spending_changes:
        label = (c.label or c.category or "flexible expense").strip()
        label = label[0].lower() + label[1:] if label else "flexible expense"
        if c.action == "stop":
            bits.append(f"stop the {label}")
        else:
            bits.append(f"reduce the {label} to {_money_str(ccy, c.new_amount)}")
    if len(bits) == 1:
        s = bits[0]
    elif len(bits) == 2:
        s = f"{bits[0]} and {bits[1]}"
    else:
        s = ", ".join(bits[:-1]) + f", and {bits[-1]}"
    return s[0].upper() + s[1:]
