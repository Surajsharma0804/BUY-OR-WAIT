"""
The decision engine: one request in, one fully-populated output row out.

PIPELINE FOR A SINGLE REQUEST
----------------------------
  1. reconcile     - resolve blank amounts from images, extract facts from
                      messages, apply cash-state rules            (upstream)
  2. build_timeline- 90-day dated cash flows in home currency     (forecast)
  3. capacity      - amount_safe_to_pay and earliest_date_for_full_payment,
                      both computed WITHOUT optional spending changes as the
                      spec requires
  4. candidates    - every eligible, SAFE plan (full / partial / installments
                      / wait), plus spending-change-enabled variants
  5. rank          - the spec's 6-rule ordering
  6. explain       - grounded, human-readable justification
  7. validate      - hard invariants before the row is allowed out (validators)

The model is never in this loop. Every number written to output.csv is
produced by deterministic arithmetic over the timeline, which is what makes
the output reproducible and auditable.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import List, Optional, Tuple, Dict

from .models import Request, UserProfile, PaymentOption, OutputRow, FinancialEvent, Fact
from .forecast import Timeline, build_timeline
from .planner import (
    Plan, SpendingChange, rank_key, apply_changes, find_minimal_changes,
    build_installment_schedule, option_allowed_for_user, installment_reading_note, _fmt,
)
from .config import money, FORECAST_DAYS
from .currency import ExchangeRates
from .explain import build_explanation


@dataclass
class Decision:
    row: OutputRow
    timeline: Timeline
    audit: List[str]


def _safe_with(tl: Timeline, schedule: List[Tuple[date, Decimal]]) -> bool:
    return tl.is_safe(schedule)


def decide(req: Request,
           profile: UserProfile,
           events: List[FinancialEvent],
           facts: List[Fact],
           options: List[PaymentOption],
           fx: ExchangeRates,
           unresolved_event_ids: Optional[List[str]] = None) -> Decision:

    audit: List[str] = []
    tl = build_timeline(profile, events, facts, req.request_date, fx,
                         unresolved_event_ids=unresolved_event_ids,
                         horizon_days=FORECAST_DAYS)
    audit.extend(tl.notes)

    requested = money(req.requested_amount)
    accepts = set(profile.payment_methods_user_will_consider)

    # ---- 3. capacity (no spending changes) ----
    headroom = tl.headroom_today()
    amount_safe = max(Decimal("0"), min(requested, headroom))
    amount_safe = money(amount_safe)

    earliest = tl.earliest_full_payment_date(requested)
    audit.append(f"headroom_today={_fmt(headroom)} amount_safe={_fmt(amount_safe)} "
                 f"earliest_full={earliest}")

    full_safe_today = headroom >= requested
    deadline = req.desired_completion_date

    # ---- 4. candidate plans ----
    candidates: List[Plan] = []

    # (a) full payment today
    if "full_payment" in accepts and full_safe_today:
        candidates.append(Plan(
            method="full_payment",
            schedule=[(req.request_date, requested)],
            total_paid=requested,
            completes_request=True,
            completes_by_deadline=req.request_date <= deadline,
            rationale="full amount is safe on the request date",
        ))

    # (b) wait: full payment becomes safe later
    if "full_payment" in accepts and not full_safe_today and earliest is not None:
        candidates.append(Plan(
            method="wait",
            schedule=[(earliest, requested)],
            total_paid=requested,
            completes_request=True,
            completes_by_deadline=earliest <= deadline,
            rationale=f"full amount becomes safe on {earliest.isoformat()}",
        ))

    # (c) installments from supplied options only
    for opt in options:
        if opt.payment_method != "installments":
            continue
        note = installment_reading_note(opt, profile)
        if note:
            audit.append(f"max_installment_months ambiguity: {note}")
        if not option_allowed_for_user(opt, profile):
            continue
        sched = build_installment_schedule(opt)
        if not _safe_with(tl, sched):
            continue
        last_date = sched[-1][0]
        candidates.append(Plan(
            method="installments",
            schedule=sched,
            total_paid=money(opt.total_payable_amount),
            completes_request=True,
            completes_by_deadline=last_date <= deadline,
            option_id=opt.payment_option_id,
            option_sort=_opt_num(opt.payment_option_id),
            rationale=f"{opt.number_of_payments} installments per {opt.payment_option_id}",
        ))

    # (d) partial payment: pay what's safe today, remainder on the earliest safe date
    if ("partial_payment" in accepts and req.allows_partial_payment
            and Decimal("0") < amount_safe < requested and earliest is not None
            and earliest <= deadline):
        remainder = money(requested - amount_safe)
        sched = [(req.request_date, amount_safe), (earliest, remainder)]
        if _safe_with(tl, sched):
            candidates.append(Plan(
                method="partial_payment",
                schedule=sched,
                total_paid=requested,
                completes_request=True,
                completes_by_deadline=earliest <= deadline,
                rationale=f"pay {_fmt(amount_safe)} today, remainder on {earliest.isoformat()}",
            ))

    # ---- 4b. spending-change-enabled variants ----
    # Only explored when no change-free plan already completes by the deadline,
    # because ranking rule 2 prefers plans with no spending changes.
    has_clean_ontime = any(p.completes_by_deadline for p in candidates)
    if not has_clean_ontime:
        candidates.extend(_changed_plans(req, profile, tl, options, accepts,
                                          requested, deadline, audit,
                                          amount_safe=amount_safe, earliest=earliest))

    # ---- 5. rank ----
    safe_candidates = [p for p in candidates if p.method != "not_recommended"]
    if safe_candidates:
        safe_candidates.sort(key=rank_key)
        best = safe_candidates[0]
    else:
        best = Plan(method="not_recommended", schedule=[], total_paid=Decimal("0"),
                    completes_request=False, completes_by_deadline=False,
                    rationale="no eligible plan keeps the minimum balance protected")

    status = _status_for(best, req, amount_safe, requested, earliest, full_safe_today, accepts)

    # `earliest_date_for_full_payment` measures capacity, independent of the
    # user's method preferences, and must equal request_date for affordable_now.
    earliest_out = ""
    if status == "affordable_now":
        earliest_out = req.request_date.isoformat()
    elif earliest is not None:
        earliest_out = earliest.isoformat()

    explanation = build_explanation(
        req=req, profile=profile, plan=best, status=status,
        amount_safe=amount_safe, requested=requested, earliest=earliest, tl=tl,
    )

    row = OutputRow(
        request_id=req.request_id,
        amount_safe_to_pay=amount_safe,
        affordability_status=status,
        recommended_payment_method=best.method,
        payment_plan=best.render_plan(),
        earliest_date_for_full_payment=earliest_out,
        spending_changes_needed=best.render_changes(),
        decision_explanation=explanation,
    )
    audit.append(f"selected={best.method} status={status} "
                 f"changes={best.render_changes()} plan={best.render_plan()[:80]}")
    return Decision(row=row, timeline=tl, audit=audit)


def _opt_num(option_id: Optional[str]) -> int:
    if not option_id:
        return 10**9
    digits = "".join(ch for ch in option_id if ch.isdigit())
    return int(digits) if digits else 10**9


def _changed_plans(req, profile, tl, options, accepts, requested, deadline, audit,
                    amount_safe: Decimal, earliest: Optional[date]) -> List[Plan]:
    """
    Build plans that become safe only after permitted spending changes.

    `amount_safe` and `earliest` are the CAPACITY figures already computed on
    the unmodified timeline (spec: both are computed BEFORE optional spending
    changes), so every changed variant below reuses them as fixed inputs
    rather than recomputing capacity under the changed timeline -- otherwise
    a changed partial-payment schedule could silently disagree with the
    amount_safe_to_pay / earliest_date_for_full_payment values written to the
    same output row.
    """
    out: List[Plan] = []

    # (i) full payment today, enabled by changes
    if "full_payment" in accepts:
        sched = [(req.request_date, requested)]
        changes = find_minimal_changes(tl, profile, lambda t: t.is_safe(sched))
        if changes:
            out.append(Plan(
                method="full_payment", schedule=sched, total_paid=requested,
                completes_request=True, completes_by_deadline=req.request_date <= deadline,
                spending_changes=changes,
                rationale="full payment today becomes safe after permitted spending changes",
            ))
            audit.append(f"spending changes enable full payment today: "
                         f"{[c.render() for c in changes]}")

    # (ii) installments, enabled by changes
    for opt in options:
        if opt.payment_method != "installments" or not option_allowed_for_user(opt, profile):
            continue
        sched = build_installment_schedule(opt)
        if tl.is_safe(sched):
            continue  # already covered as a change-free candidate
        changes = find_minimal_changes(tl, profile, lambda t: t.is_safe(sched))
        if changes:
            out.append(Plan(
                method="installments", schedule=sched,
                total_paid=money(opt.total_payable_amount),
                completes_request=True, completes_by_deadline=sched[-1][0] <= deadline,
                option_id=opt.payment_option_id, option_sort=_opt_num(opt.payment_option_id),
                spending_changes=changes,
                rationale=f"installments per {opt.payment_option_id} after spending changes",
            ))

    # (iii) partial payment "enabled by changes" is intentionally NOT built here.
    #
    # amount_safe and earliest are fixed, pre-change quantities (spec: both
    # are computed BEFORE optional spending changes), and change-free
    # candidate (d) already builds the two-payment schedule
    # [(request_date, amount_safe), (earliest, requested-amount_safe)]
    # whenever 0 < amount_safe < requested and earliest <= deadline. That
    # schedule is PROVABLY safe under those conditions regardless of what
    # else is in the timeline, with no role for spending changes to play:
    #   - amount_safe == headroom_today() in this branch (by construction of
    #     amount_safe = clamp(headroom, 0, requested)), so paying it today
    #     shifts the whole curve down by exactly headroom, and the curve's
    #     global minimum -- by definition of headroom -- cannot then drop
    #     below minimum_balance anywhere in [request_date, earliest).
    #   - earliest is defined as the first date whose SUFFIX minimum already
    #     covers minimum_balance + requested, so paying the remainder there
    #     keeps every date from earliest onward safe too.
    #   Together every point in the window is covered, so is_safe(sched) is
    #   always True here -- confirmed both by this proof and empirically
    #   with synthetic scenarios (see PR notes / audit trail).
    # A prior version of this function called find_minimal_changes() against
    # this same fixed schedule and discarded the result unconditionally
    # ("pass"), which was at best confusing dead code. It is not, however,
    # a correctness gap: validators.py itself requires partial_payment's
    # first leg to equal the WRITTEN (pre-change) amount_safe_to_pay, so a
    # "partial payment that only becomes valid after a spending change"
    # could never be spec-valid in the first place -- there is no amount to
    # substitute in its place. (i)'s full-payment-after-changes search is
    # the correct place for spending changes to unlock a previously-unsafe
    # request; a changed PARTIAL schedule was never a distinct case.

    return out


def _status_for(plan: Plan, req: Request, amount_safe: Decimal, requested: Decimal,
                 earliest: Optional[date], full_safe_today: bool, accepts: set) -> str:
    """
    Map the chosen plan onto the allowed affordability_status values.

    affordable_now        - full amount safe today AND user accepts full_payment
    affordable_with_plan  - completed via partial schedule, installments, or
                            permitted spending changes
    affordable_later      - full amount expected to become safe later
    not_affordable        - cannot be completed safely within the forecast
    """
    if plan.method == "not_recommended":
        return "not_affordable"
    if plan.method == "full_payment" and not plan.spending_changes and full_safe_today:
        return "affordable_now"
    if plan.method == "wait":
        return "affordable_later"
    # full_payment with changes, installments, partial_payment
    return "affordable_with_plan"
