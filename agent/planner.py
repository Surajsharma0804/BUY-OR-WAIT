"""
Plan generation and selection.

problem_statement.md, "Choosing Between Safe Plans":
  An immediate method (full_payment / partial_payment / installments) is
  eligible only when it appears in the user's payment_methods_user_will_consider.
  `wait` is eligible when full payment becomes safe later AND the user
  accepts full_payment. `not_recommended` is the fallback.

  When more than one eligible plan is safe, rank by:
    1. Complete the full request by desired_completion_date
    2. Require no spending changes
    3. Minimize the total amount paid
    4. Start payment earlier
    5. Use fewer payments
    6. Lowest payment_option_id as final tie-breaker

That ranking is implemented literally as a sort key in `rank_key`, so the
precedence order is auditable against the spec line by line rather than
buried in branching logic.

SPENDING CHANGES
----------------
Only non-protected, flexible events in a category the user permits may be
changed, at most three, and stopping vs reducing the SAME event are mutually
exclusive. Candidate changes are searched greedily by how much headroom they
free per change, because the spec prefers plans with FEWER changes (rule 2
prefers none at all), and a minimal set is easier for a user to act on.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from typing import List, Optional, Tuple, Dict

from .models import UserProfile, PaymentOption, Request
from .forecast import Timeline, Flow
from .config import money

MAX_SPENDING_CHANGES = 3


@dataclass
class SpendingChange:
    action: str            # 'stop' | 'reduce_to'
    event_id: str
    new_amount: Optional[Decimal] = None
    freed: Decimal = Decimal("0")
    label: str = ""
    category: str = ""

    def render(self) -> str:
        if self.action == "stop":
            return f"stop:{self.event_id}"
        return f"reduce_to:{self.event_id}:{_fmt(self.new_amount)}"


@dataclass
class Plan:
    method: str                       # full_payment|partial_payment|installments|wait|not_recommended
    schedule: List[Tuple[date, Decimal]] = field(default_factory=list)
    total_paid: Decimal = Decimal("0")
    completes_request: bool = False
    completes_by_deadline: bool = False
    spending_changes: List[SpendingChange] = field(default_factory=list)
    option_id: Optional[str] = None
    option_sort: int = 10**9
    rationale: str = ""

    @property
    def num_payments(self) -> int:
        return len(self.schedule)

    @property
    def start_date(self) -> Optional[date]:
        return self.schedule[0][0] if self.schedule else None

    def render_plan(self) -> str:
        if not self.schedule:
            return "none"
        return "|".join(f"{d.isoformat()}:{_fmt(a)}" for d, a in self.schedule)

    def render_changes(self) -> str:
        if not self.spending_changes:
            return "none"
        return "|".join(c.render() for c in self.spending_changes)


def _fmt(x: Optional[Decimal]) -> str:
    """Format money without scientific notation, trimming pointless trailing zeros."""
    if x is None:
        return ""
    q = money(x)
    s = format(q, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def _opt_sort(option_id: Optional[str]) -> int:
    if not option_id:
        return 10**9
    digits = "".join(ch for ch in option_id if ch.isdigit())
    return int(digits) if digits else 10**9


def rank_key(p: Plan):
    """
    The spec's ranking, expressed directly. Lower tuple sorts better.
      1. completes by deadline   (False sorts worse)
      2. no spending changes
      3. minimize total paid
      4. start earlier
      5. fewer payments
      6. lowest payment_option_id
    """
    return (
        0 if p.completes_by_deadline else 1,
        len(p.spending_changes),
        p.total_paid,
        p.start_date or date.max,
        p.num_payments,
        p.option_sort,
    )


# ---------------------------------------------------------------------------
# Spending-change search
# ---------------------------------------------------------------------------

def candidate_changes(tl: Timeline, profile: UserProfile) -> List[SpendingChange]:
    """
    Every permitted change to a flexible, non-protected recurring expense.

    A change is permitted only when ALL hold:
      * the flow is a debit and marked flexible in the event data
      * its category is NOT in expense_categories_to_protect
      * the action matches the user's stated willingness:
          stop      -> category in expense_categories_user_is_willing_to_stop
          reduce_to -> category in expense_categories_user_is_willing_to_reduce
                       and the event declares a minimum_allowed_amount
    """
    protect = set(profile.expense_categories_to_protect)
    can_stop = set(profile.expense_categories_user_is_willing_to_stop)
    can_reduce = set(profile.expense_categories_user_is_willing_to_reduce)

    # Aggregate by event_id: stopping a recurring series frees every projected
    # occurrence of it inside the window, not just one.
    by_event: Dict[str, List[Flow]] = {}
    for f in tl.flows:
        if f.amount >= 0 or not f.flexible or not f.event_id:
            continue
        if f.category in protect:
            continue
        by_event.setdefault(f.event_id, []).append(f)

    out: List[SpendingChange] = []
    for event_id, flows in by_event.items():
        cat = flows[0].category
        total = sum(-f.amount for f in flows)

        if cat in can_stop and flows[0].stoppable:
            out.append(SpendingChange("stop", event_id, None, money(total),
                                       flows[0].label, cat))

        if cat in can_reduce and flows[0].reducible and flows[0].min_allowed is not None:
            floor = money(flows[0].min_allowed)
            per_occurrence = -flows[0].amount
            if floor < per_occurrence:
                freed = money((per_occurrence - floor) * len(flows))
                out.append(SpendingChange("reduce_to", event_id, floor, freed,
                                           flows[0].label, cat))

    # Largest headroom gain first: the spec prefers FEWER changes, so the
    # greedy search should reach feasibility in as few changes as possible.
    out.sort(key=lambda c: (-c.freed, c.event_id))
    return out


def apply_changes(tl: Timeline, changes: List[SpendingChange]) -> Timeline:
    """Return a copy of the timeline with the given changes applied."""
    stopped = {c.event_id for c in changes if c.action == "stop"}
    reduced = {c.event_id: c.new_amount for c in changes if c.action == "reduce_to"}

    new_flows: List[Flow] = []
    for f in tl.flows:
        if f.event_id and f.event_id in stopped and f.amount < 0:
            continue
        if f.event_id and f.event_id in reduced and f.amount < 0:
            new_flows.append(Flow(
                when=f.when, amount=-money(reduced[f.event_id]), label=f.label + " (reduced)",
                source=f.source, category=f.category, event_id=f.event_id,
                flexible=f.flexible, stoppable=f.stoppable, reducible=f.reducible,
                min_allowed=f.min_allowed,
            ))
            continue
        new_flows.append(f)

    return Timeline(user_id=tl.user_id, start=tl.start, end=tl.end,
                     opening_balance=tl.opening_balance, minimum_balance=tl.minimum_balance,
                     flows=new_flows, notes=list(tl.notes))


def find_minimal_changes(tl: Timeline, profile: UserProfile,
                          test) -> Optional[List[SpendingChange]]:
    """
    Smallest set of permitted changes (<= 3) that makes `test(timeline)` true.
    `test` is a predicate over a Timeline.

    Tries 1 change, then 2, then 3, greedily ordered by freed headroom.
    Stopping and reducing the same event are mutually exclusive (spec), which
    is enforced by skipping combinations that repeat an event_id.
    """
    cands = candidate_changes(tl, profile)
    if not cands:
        return None

    from itertools import combinations
    for size in range(1, MAX_SPENDING_CHANGES + 1):
        for combo in combinations(cands, size):
            ids = [c.event_id for c in combo]
            if len(set(ids)) != len(ids):
                continue  # same event stopped AND reduced -> not allowed
            if test(apply_changes(tl, list(combo))):
                return list(combo)
    return None


# ---------------------------------------------------------------------------
# Candidate plan construction
# ---------------------------------------------------------------------------

def build_installment_schedule(opt: PaymentOption) -> List[Tuple[date, Decimal]]:
    out = []
    freq = opt.payment_frequency_days or 0
    for i in range(opt.number_of_payments):
        d = opt.first_payment_date + timedelta(days=freq * i)
        out.append((d, money(opt.payment_amount)))
    return out


def installment_months(opt: PaymentOption) -> Decimal:
    """Span of the payment schedule in months (last payment minus first, /30)."""
    freq = opt.payment_frequency_days or 30
    span_days = freq * max(opt.number_of_payments - 1, 0)
    return Decimal(span_days) / Decimal(30)


def installment_reading_note(opt: PaymentOption, profile: UserProfile) -> Optional[str]:
    """
    `max_installment_months` is genuinely ambiguous against this dataset: every
    sample's CHOSEN option has 3 payments and every REJECTED one has >= 15
    (README, "Known limitations"), so a payment-COUNT reading and a
    span-IN-MONTHS reading (installment_months, above) draw the same line on
    every solved sample -- there is no evidence in the 25 samples to
    disambiguate them. `option_allowed_for_user` below deliberately takes the
    stricter, payment-count reading. This helper's only job is to flag, for
    the audit trail, any option on the FULL 250-request set where the two
    readings would actually have disagreed, so the ambiguity stays visible
    instead of being silently resolved one way with no record of the other.
    Returns None when the readings agree (i.e. on every sample seen so far).
    """
    if profile.max_installment_months is None:
        return None
    by_count = opt.number_of_payments <= profile.max_installment_months
    by_span = installment_months(opt) <= profile.max_installment_months
    if by_count == by_span:
        return None
    return (f"{opt.payment_option_id}: ambiguous under max_installment_months="
            f"{profile.max_installment_months} -- payment-count reading "
            f"({opt.number_of_payments} payments) {'allows' if by_count else 'rejects'}, "
            f"span-in-months reading ({installment_months(opt):.1f} months) "
            f"{'allows' if by_span else 'rejects'}; used the payment-count reading")


def option_allowed_for_user(opt: PaymentOption, profile: UserProfile) -> bool:
    if opt.payment_method == "installments":
        if "installments" not in profile.payment_methods_user_will_consider:
            return False
        if profile.max_installment_months is None:
            return False
        # Deliberate, tested choice of the STRICTER reading -- compare the
        # payment COUNT, not installment_months(opt)'s span in months. See
        # installment_reading_note(), which audits any option where the two
        # readings would actually disagree, and the README for the evidence
        # behind this choice.
        if opt.number_of_payments > profile.max_installment_months:
            return False
        return True
    if opt.payment_method == "full_payment":
        return "full_payment" in profile.payment_methods_user_will_consider
    return False
