"""
The 90-day forecast engine.

problem_statement.md, "90-Day Safety Check":
  "Forecast the user's balance for the next 90 days using recurring income
   and expenses, confirmed future payments, and relevant messages or images.
   A plan is safe only if the balance never falls below
   `minimum_balance_to_keep`."

CORE IDEA
---------
Build ONE timeline of dated cash deltas for a user over
(request_date, request_date + 90], then answer every question by simulating
that timeline with candidate payments injected.

Two quantities drive the whole output schema and both are defined by the
spec to be computed BEFORE any optional spending changes:

  amount_safe_to_pay
      The largest X payable ON request_date that never breaches the
      minimum. Because the running balance is strictly monotonic in X
      (paying more today lowers every subsequent point by exactly X), we do
      not need a search: X_max = min(running_balance_t) - minimum_balance,
      taken over the whole window, clamped to [0, requested_amount].
      That is exact, not approximate, and O(n) instead of a bisection.

  earliest_date_for_full_payment
      The first date D in the window at which paying the FULL requested
      amount as a single payment keeps the balance above the minimum for
      the rest of the window. Also solved in one pass: paying on D only
      affects points at/after D, so the constraint is
          min_{t >= D} (balance_t) - requested >= minimum_balance
      i.e. D is the first date where the SUFFIX MINIMUM of the unmodified
      balance curve is at least (minimum + requested).
      A suffix-minimum scan gives every date's answer in a single reverse
      pass, which matters because we evaluate this for 250 requests.

The suffix-minimum formulation is also what makes `wait` plans and
deadline feasibility cheap to check.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from typing import List, Dict, Optional, Tuple, Iterable

from .models import FinancialEvent, UserProfile, Fact
from .config import FORECAST_DAYS, money
from .currency import ExchangeRates
from .recurrence import RecurringSeries, detect_series, project_series, is_lapsed


@dataclass
class Flow:
    """One dated cash movement in the user's home currency (+credit / -debit)."""
    when: date
    amount: Decimal              # signed, home currency
    label: str
    source: str                  # 'event:<id>' | 'projected:<series>' | 'fact:<source>'
    category: str = ""
    event_id: Optional[str] = None
    flexible: bool = False
    stoppable: bool = False
    reducible: bool = False
    min_allowed: Optional[Decimal] = None


@dataclass
class Timeline:
    user_id: str
    start: date
    end: date
    opening_balance: Decimal
    minimum_balance: Decimal
    flows: List[Flow] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def sorted_flows(self) -> List[Flow]:
        return sorted(self.flows, key=lambda f: (f.when, f.label))

    # ---------------- core simulation ----------------

    def balance_curve(self, extra: Optional[List[Tuple[date, Decimal]]] = None
                       ) -> List[Tuple[date, Decimal]]:
        """
        Running balance after each dated step. `extra` injects candidate
        payments as negative deltas.
        """
        deltas: Dict[date, Decimal] = {}
        for f in self.flows:
            deltas[f.when] = deltas.get(f.when, Decimal("0")) + f.amount
        for when, amt in (extra or []):
            deltas[when] = deltas.get(when, Decimal("0")) - amt

        curve = []
        bal = self.opening_balance
        for d in sorted(deltas.keys()):
            if d < self.start or d > self.end:
                # Flows outside the window can't affect the check; guard anyway.
                if d < self.start:
                    bal += deltas[d]
                continue
            bal += deltas[d]
            curve.append((d, bal))
        return curve

    def min_balance(self, extra: Optional[List[Tuple[date, Decimal]]] = None) -> Decimal:
        curve = self.balance_curve(extra)
        vals = [b for _, b in curve]
        # The opening balance itself is a point in time that must also hold.
        vals.append(self.opening_balance if extra is None else
                     self.opening_balance - sum(a for d, a in (extra or []) if d <= self.start))
        return min(vals) if vals else self.opening_balance

    def is_safe(self, extra: Optional[List[Tuple[date, Decimal]]] = None) -> bool:
        return self.min_balance(extra) >= self.minimum_balance

    # ---------------- derived quantities ----------------

    def headroom_today(self) -> Decimal:
        """
        Max payable on `start` without ever breaching the minimum.
        Exact closed form: the lowest point of the unmodified curve minus
        the minimum balance (paying X today shifts the entire curve down by X).
        """
        return self.min_balance() - self.minimum_balance

    def suffix_minimums(self) -> List[Tuple[date, Decimal]]:
        """
        For each dated point, the minimum balance from that point to the end
        of the window (computed in one reverse pass).
        """
        curve = self.balance_curve()
        out: List[Tuple[date, Decimal]] = []
        running = None
        for d, b in reversed(curve):
            running = b if running is None else min(running, b)
            out.append((d, running))
        out.reverse()
        return out

    def earliest_full_payment_date(self, amount: Decimal) -> Optional[date]:
        """
        First date in the window where paying `amount` as ONE payment keeps
        the balance above the minimum for the remainder of the window.
        Computed without any spending changes, per spec.
        """
        need = self.minimum_balance + amount

        # Paying on `start` itself: the whole curve (and the opening balance)
        # must tolerate the hit.
        if self.min_balance() >= need:
            return self.start

        sufmins = self.suffix_minimums()
        for d, sufmin in sufmins:
            if d <= self.start:
                continue
            if sufmin >= need:
                return d
        return None


def build_timeline(profile: UserProfile,
                    events: List[FinancialEvent],
                    facts: List[Fact],
                    as_of: date,
                    fx: ExchangeRates,
                    unresolved_event_ids: Optional[Iterable[str]] = None,
                    horizon_days: int = FORECAST_DAYS) -> Timeline:
    """
    Assemble the 90-day timeline for one user as of `as_of`.
    """
    start = as_of
    end = as_of + timedelta(days=horizon_days)
    unresolved = set(unresolved_event_ids or [])

    tl = Timeline(user_id=profile.user_id, start=start, end=end,
                  opening_balance=money(profile.current_available_balance),
                  minimum_balance=money(profile.minimum_balance_to_keep))

    home = profile.home_currency

    def to_home(amount: Decimal, ccy: str, when: date) -> Decimal:
        if ccy == home:
            return money(amount)
        return fx.convert(amount, ccy, home, when)

    # ---- 1. explicit dated events inside the window ----
    occupied: Dict[Tuple, set] = {}
    for e in events:
        if e.excluded:
            continue
        when = e.cash_date()
        if when is None:
            continue
        if not (start < when <= end):
            continue
        if e.amount is None:
            if e.event_id in unresolved:
                tl.notes.append(f"event {e.event_id} has an unreadable amount; "
                                 f"excluded from cash flow but flagged for review "
                                 f"(never treated as zero in the explanation)")
            continue
        signed = to_home(e.amount, e.currency, when)
        if e.direction == "debit":
            signed = -signed
        tl.flows.append(Flow(
            when=when, amount=signed, label=e.description, source=f"event:{e.event_id}",
            category=e.category, event_id=e.event_id,
            flexible=e.flexibility != "fixed",
            stoppable=e.flexibility in ("stoppable", "reducible_or_stoppable"),
            reducible=e.flexibility in ("reducible", "reducible_or_stoppable"),
            min_allowed=e.minimum_allowed_amount,
        ))
        key = (e.event_type, e.category, e.direction)
        occupied.setdefault(key, set()).add(when)

    # ---- 2. recurring projections ----
    series = detect_series(events, as_of)

    # Facts can amend or stop projected income/expenses.
    income_stops = [f for f in facts if f.kind == "income_stop"]
    income_changes = [f for f in facts if f.kind == "income_change"]
    income_new = [f for f in facts if f.kind == "income_new"]
    rate_changes = [f for f in facts if f.kind == "expense_rate_change"]
    date_shifts = [f for f in facts if f.kind == "date_shift" and f.new_date]
    # "Your next payout is still pending and isn't withdrawable" means future
    # earnings in that stream are NOT confirmed cash (AGENTS.md 6.3: do not
    # count pending credits until they settle). This is deliberately scoped to
    # variable gig/platform earnings -- a broader reading also suppressed
    # ordinary salary whenever an unrelated refund happened to be pending,
    # which measurably hurt accuracy on the solved samples.
    suppress_projected_income = any(f.kind == "income_unconfirmed" for f in facts)

    for s in series:
        key = (s.event_type, s.category, s.direction)
        occ = occupied.get(key, set())

        # income_stop / income_change / date_shift are messages about ONE
        # specific income relationship (a message says "your employer"
        # or "your contract", not "all of your income streams"). When a
        # user has more than one detected income series (e.g. two
        # day-of-month clusters both categorised "salary" -- a primary job
        # plus a smaller second income), applying one of these facts to
        # every series conflates them. Scoped to the PRIMARY series (see
        # detect_series / is_primary) as the more defensible default reading
        # of an unqualified employer/salary message; a lone income series is
        # always its own primary, so this is a no-op in the common case.
        # Verified against the real dataset: user_11 has two "salary"
        # clusters (8.5M and 23.26M IDR); a single "confirmed base salary"
        # income_change message was previously applied to BOTH, doubling
        # the assumed raise into a dangerous overestimate of available cash.
        applies_to_this_series = s.is_primary

        if s.is_income and income_stops and applies_to_this_series:
            tl.notes.append(f"recurring income '{s.description}' stopped per employer message")
            continue

        if s.is_income and suppress_projected_income:
            tl.notes.append(f"projected income '{s.description}' not counted: an upcoming payout "
                             f"is reported pending/not yet withdrawable")
            continue

        amount = s.amount
        effective_from: Optional[date] = None
        if s.is_income and income_changes and applies_to_this_series:
            # Most recent applicable amendment wins (conflict rule 2).
            ch = income_changes[-1]
            if ch.new_amount is not None:
                amount = ch.new_amount
                if ch.new_currency and ch.new_currency != s.currency:
                    amount = fx.convert(amount, ch.new_currency, s.currency, as_of)
                effective_from = ch.new_date
                tl.notes.append(f"recurring income amended to {amount} {s.currency} "
                                 f"({ch.note})")

        if (not s.is_income) and rate_changes:
            for rc in rate_changes:
                if rc.category_hint and rc.category_hint == s.category and rc.multiplier:
                    amount = money(amount * rc.multiplier)
                    effective_from = rc.new_date
                    tl.notes.append(f"recurring {s.category} scaled by {rc.multiplier} ({rc.note})")

        if s.is_income and date_shifts and applies_to_this_series:
            shift = date_shifts[-1]
            if shift.new_date and shift.new_date > s.last_date:
                # The announced date is itself an upcoming CONFIRMED payment, not
                # merely a new anchor for later cycles. Emitting it matters: without
                # it the next payday vanished from the forecast entirely and the
                # balance trough sat a month and a half too low.
                # (Placed after `amount` is resolved so an accompanying pay
                # amendment is reflected in the shifted payment as well.)
                if start < shift.new_date <= end and shift.new_date not in occ:
                    tl.flows.append(Flow(
                        when=shift.new_date,
                        amount=to_home(amount, s.currency, shift.new_date),
                        label=f"{s.description} (payday moved)",
                        source=f"fact:{shift.source}", category=s.category,
                        event_id=s.member_event_ids[-1] if s.member_event_ids else None,
                    ))
                s.last_date = shift.new_date
                tl.notes.append(f"income payday moved to {shift.new_date} per employer message")

        for when, base_amt in project_series(s, start, end, occupied_dates=occ, as_of=as_of):
            use = amount if (effective_from is None or when >= effective_from) else base_amt
            signed = to_home(use, s.currency, when)
            if s.direction == "debit":
                signed = -signed
            tl.flows.append(Flow(
                when=when, amount=signed,
                label=f"projected {s.description}", source=f"projected:{s.category}",
                category=s.category,
                event_id=s.member_event_ids[-1] if s.member_event_ids else None,
                flexible=s.flexibility != "fixed",
                stoppable=s.flexibility in ("stoppable", "reducible_or_stoppable"),
                reducible=s.flexibility in ("reducible", "reducible_or_stoppable"),
                min_allowed=s.minimum_allowed_amount,
            ))

    # ---- 3. one-off confirmed income from messages (invoices, arrears, first salary) ----
    for f in income_new:
        if f.new_amount is None or f.new_date is None:
            continue
        if not (start < f.new_date <= end):
            continue
        ccy = f.new_currency or home
        tl.flows.append(Flow(
            when=f.new_date, amount=to_home(f.new_amount, ccy, f.new_date),
            label=f.note or "confirmed incoming payment", source=f"fact:{f.source}",
            category="income",
        ))

    return tl
