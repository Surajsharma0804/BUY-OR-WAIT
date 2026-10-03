"""
Deterministic output validation.

The README's pre-submission checklist and the problem statement's schema
rules are encoded here as executable assertions rather than things to
eyeball. Nothing reaches output.csv without passing.

Checks:
  * affordability_status / recommended_payment_method in the allowed sets
  * 0 <= amount_safe_to_pay <= requested_amount
  * payment_plan syntax: chronological "<YYYY-MM-DD>:<amount>" joined by "|", or "none"
  * installments plans EXACTLY match a supplied payment option
  * partial_payment: exactly 2 payments, sums to requested_amount, first on
    request_date, second on earliest_date_for_full_payment <= deadline,
    request allows partial, user accepts it, 0 < safe < requested
  * affordable_now  => earliest_date_for_full_payment == request_date
  * spending_changes: <= 3, correct syntax, only flexible non-protected
    events in permitted categories, and never stop+reduce the same event
  * not_recommended  => payment_plan == "none"

`validate_row` returns a list of human-readable violations. main.py treats a
non-empty list as a hard failure for that row and falls back to a
conservative, provably-valid row rather than emitting something malformed.
"""

from __future__ import annotations
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import List, Dict, Optional
import re

from .models import OutputRow, Request, UserProfile, PaymentOption, FinancialEvent
from .config import AFFORDABILITY_STATUSES, PAYMENT_METHODS, money

PLAN_ENTRY = re.compile(r"^(\d{4}-\d{2}-\d{2}):(\d+(?:\.\d+)?)$")
CHANGE_STOP = re.compile(r"^stop:([A-Za-z0-9_]+)$")
CHANGE_REDUCE = re.compile(r"^reduce_to:([A-Za-z0-9_]+):(\d+(?:\.\d+)?)$")


def _parse_plan(plan: str):
    if plan == "none":
        return []
    entries = []
    for part in plan.split("|"):
        m = PLAN_ENTRY.match(part)
        if not m:
            raise ValueError(f"malformed payment_plan entry: {part!r}")
        y, mo, d = m.group(1).split("-")
        entries.append((date(int(y), int(mo), int(d)), Decimal(m.group(2))))
    return entries


def validate_row(row: OutputRow, req: Request, profile: UserProfile,
                  options: List[PaymentOption],
                  events_by_id: Dict[str, FinancialEvent]) -> List[str]:
    errs: List[str] = []

    if row.request_id != req.request_id:
        errs.append(f"request_id mismatch: {row.request_id} != {req.request_id}")

    if row.affordability_status not in AFFORDABILITY_STATUSES:
        errs.append(f"invalid affordability_status {row.affordability_status!r}")
    if row.recommended_payment_method not in PAYMENT_METHODS:
        errs.append(f"invalid recommended_payment_method {row.recommended_payment_method!r}")

    # bounds
    requested = money(req.requested_amount)
    safe = money(row.amount_safe_to_pay)
    if safe < 0 or safe > requested:
        errs.append(f"amount_safe_to_pay {safe} outside [0, {requested}]")

    # plan syntax + chronology
    try:
        entries = _parse_plan(row.payment_plan)
    except ValueError as e:
        errs.append(str(e))
        entries = []
    if entries:
        dates = [d for d, _ in entries]
        if dates != sorted(dates):
            errs.append("payment_plan is not in chronological order")
        if any(a <= 0 for _, a in entries):
            errs.append("payment_plan contains a non-positive amount")

    method = row.recommended_payment_method

    if method == "not_recommended" and row.payment_plan != "none":
        errs.append("not_recommended must have payment_plan 'none'")
    if method != "not_recommended" and row.payment_plan == "none":
        errs.append(f"{method} must have a non-empty payment_plan")

    # installments must match a supplied option exactly
    if method == "installments":
        if not _matches_any_option(entries, options):
            errs.append("installments plan does not exactly match any supplied payment option")
        if "installments" not in profile.payment_methods_user_will_consider:
            errs.append("installments recommended but user does not accept installments")

    # partial payment rules
    if method == "partial_payment":
        if row.affordability_status != "affordable_with_plan":
            errs.append("partial_payment requires affordability_status affordable_with_plan")
        if not req.allows_partial_payment:
            errs.append("partial_payment recommended but request does not allow partial payment")
        if "partial_payment" not in profile.payment_methods_user_will_consider:
            errs.append("partial_payment recommended but user does not accept it")
        if len(entries) != 2:
            errs.append(f"partial_payment must have exactly 2 payments, got {len(entries)}")
        else:
            total = entries[0][1] + entries[1][1]
            if money(total) != requested:
                errs.append(f"partial_payment total {money(total)} != requested {requested}")
            if entries[0][0] != req.request_date:
                errs.append("partial_payment first payment must be on request_date")
            if money(entries[0][1]) != safe:
                errs.append("partial_payment first payment must equal amount_safe_to_pay")
            if row.earliest_date_for_full_payment:
                if entries[1][0].isoformat() != row.earliest_date_for_full_payment:
                    errs.append("partial_payment second payment must be on "
                                 "earliest_date_for_full_payment")
            if entries[1][0] > req.desired_completion_date:
                errs.append("partial_payment second payment is after desired_completion_date")
        if not (Decimal("0") < safe < requested):
            errs.append("partial_payment requires 0 < amount_safe_to_pay < requested_amount")

    # affordable_now invariant
    if row.affordability_status == "affordable_now":
        if row.earliest_date_for_full_payment != req.request_date.isoformat():
            errs.append("affordable_now requires earliest_date_for_full_payment == request_date")

    if row.earliest_date_for_full_payment:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", row.earliest_date_for_full_payment):
            errs.append("earliest_date_for_full_payment is not YYYY-MM-DD")

    # spending changes
    errs.extend(_validate_changes(row.spending_changes_needed, profile, events_by_id))

    if not row.decision_explanation.strip():
        errs.append("decision_explanation is empty")

    return errs


def _matches_any_option(entries, options: List[PaymentOption]) -> bool:
    from .planner import build_installment_schedule
    for opt in options:
        if opt.payment_method != "installments":
            continue
        sched = build_installment_schedule(opt)
        if len(sched) != len(entries):
            continue
        ok = all(sd == ed and money(sa) == money(ea)
                 for (sd, sa), (ed, ea) in zip(sched, entries))
        if ok:
            return True
    return False


def _validate_changes(spec: str, profile: UserProfile,
                       events_by_id: Dict[str, FinancialEvent]) -> List[str]:
    errs: List[str] = []
    if spec == "none" or not spec:
        return errs
    parts = spec.split("|")
    if len(parts) > 3:
        errs.append(f"spending_changes_needed has {len(parts)} changes (max 3)")

    seen_events = set()
    protect = set(profile.expense_categories_to_protect)
    can_stop = set(profile.expense_categories_user_is_willing_to_stop)
    can_reduce = set(profile.expense_categories_user_is_willing_to_reduce)

    for part in parts:
        m_stop = CHANGE_STOP.match(part)
        m_red = CHANGE_REDUCE.match(part)
        if not (m_stop or m_red):
            errs.append(f"malformed spending change {part!r}")
            continue
        event_id = (m_stop or m_red).group(1)
        if event_id in seen_events:
            errs.append(f"event {event_id} appears in more than one spending change "
                         f"(stop and reduce are mutually exclusive)")
        seen_events.add(event_id)

        ev = events_by_id.get(event_id)
        if ev is None:
            errs.append(f"spending change references unknown event {event_id}")
            continue
        if ev.flexibility == "fixed":
            errs.append(f"spending change targets non-flexible event {event_id}")
        if ev.category in protect:
            errs.append(f"spending change targets protected category {ev.category!r}")
        if m_stop:
            if ev.flexibility not in ("stoppable", "reducible_or_stoppable"):
                errs.append(f"event {event_id} is not stoppable")
            if ev.category not in can_stop:
                errs.append(f"user is not willing to stop category {ev.category!r}")
        else:
            new_amt = Decimal(m_red.group(2))
            if ev.flexibility not in ("reducible", "reducible_or_stoppable"):
                errs.append(f"event {event_id} is not reducible")
            if ev.category not in can_reduce:
                errs.append(f"user is not willing to reduce category {ev.category!r}")
            if ev.minimum_allowed_amount is not None and new_amt < ev.minimum_allowed_amount:
                errs.append(f"reduce_to {new_amt} below minimum_allowed_amount "
                             f"{ev.minimum_allowed_amount} for {event_id}")
            if ev.amount is not None and new_amt >= ev.amount:
                errs.append(f"reduce_to {new_amt} is not a reduction "
                             f"(current {ev.amount}) for {event_id}")
    return errs


def validate_file(path: str, requests: List[Request]) -> List[str]:
    """Whole-file checks: header, row count, one row per request, no duplicates."""
    import csv
    expected_header = ["request_id", "amount_safe_to_pay", "affordability_status",
                       "recommended_payment_method", "payment_plan",
                       "earliest_date_for_full_payment", "spending_changes_needed",
                       "decision_explanation"]
    errs = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return ["output.csv is empty"]
    if rows[0] != expected_header:
        errs.append(f"header mismatch.\n  got:      {rows[0]}\n  expected: {expected_header}")
    body = rows[1:]
    if len(body) != len(requests):
        errs.append(f"row count {len(body)} != request count {len(requests)}")
    ids = [r[0] for r in body if r]
    if len(set(ids)) != len(ids):
        dupes = {i for i in ids if ids.count(i) > 1}
        errs.append(f"duplicate request_ids: {sorted(dupes)}")
    missing = {r.request_id for r in requests} - set(ids)
    if missing:
        errs.append(f"missing request_ids: {sorted(missing)}")
    extra = set(ids) - {r.request_id for r in requests}
    if extra:
        errs.append(f"unexpected request_ids: {sorted(extra)}")
    return errs
