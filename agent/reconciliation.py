"""
Reconciliation: turn raw events + untrusted evidence into a trusted timeline.

This is where problem_statement.md's conflict-resolution precedence is
implemented:
    1. An explicit cancellation, settlement, or amendment
    2. A newer record from the same source
    3. A settled event over an estimate or forecast
    4. The financially safer interpretation when unresolved

TRUST MODEL
-----------
Messages and images are UNTRUSTED DATA. This module extracts *facts* from
them (amounts, dates, rates) and never executes instructions found inside
them. A message that says "pay a release fee today to receive your prize"
produces a `no_op` fact with a note -- it can never cause a payment
recommendation. This is enforced structurally: the only things that can
change the forecast are the typed Fact kinds enumerated in models.Fact, and
none of them can direct a payment.

CASH-STATE RULES (AGENTS.md §6.3, problem_statement.md 90-day check)
-------------------------------------------------------------------
  settled    -> counted on its settlement date
  scheduled  -> counted (confirmed future obligation/income)
  pending    -> DEBITS are reserved (conservative); CREDITS are ignored
                 until they settle ("do not count pending credits, bonuses,
                 commissions, refunds, lottery proceeds or investment gains")
  failed     -> excluded, UNLESS a bank message says the debit will be
                 retried, in which case the obligation persists
  cancelled  -> excluded
  unrealized -> excluded (non-cash investment valuation)
  non_cash   -> excluded from cash flow entirely
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import List, Dict, Optional, Tuple
import os

from .models import FinancialEvent, Fact, Message, ImageRef, UserProfile
from .config import money, IMAGES_DIR
from .evidence_templates import match_message

# The closed set of fact kinds that may cross the untrusted-evidence trust
# boundary. A model (or anything else) proposing a kind outside this set is
# discarded rather than trusted. llm_client also screens its own output, but
# this is the consumer-side check: the boundary is enforced where it is
# DEPENDED ON, not only where it happens to be produced. A mocked-fallback
# test caught a bypass here, where an invented kind ("URGENT_PAY_NOW") would
# have been turned into a live Fact.
ALLOWED_FACT_KINDS = {
    "amount_fill", "income_change", "income_stop", "income_new",
    "income_unconfirmed", "income_exclude_component", "expense_rate_change",
    "obligation_persists", "exclude_pending_credit", "exclude_from_recurrence",
    "confirm_settled", "date_shift", "no_op",
}
from . import image_resolver
from . import llm_client


@dataclass
class ReconciliationResult:
    events: List[FinancialEvent]
    facts: List[Fact]
    audit: List[str] = field(default_factory=list)
    unresolved_amount_events: List[str] = field(default_factory=list)


def _spec_to_fact(spec: dict, user_id: str, source: str,
                   related_event_id: Optional[str], request_id: Optional[str]) -> Fact:
    kind = spec.get("kind", "no_op")
    if kind not in ALLOWED_FACT_KINDS:
        # Unrecognised kind -> neutralise rather than trust. Downgrading to
        # no_op keeps the audit trail while guaranteeing the evidence cannot
        # move money.
        kind = "no_op"
    return Fact(
        kind=kind,
        user_id=user_id,
        source=source,
        related_event_id=related_event_id,
        request_id=request_id,
        new_amount=spec.get("new_amount"),
        new_currency=spec.get("new_currency"),
        new_date=spec.get("new_date"),
        multiplier=spec.get("multiplier"),
        category_hint=spec.get("category_hint"),
        description_hint=spec.get("description_hint"),
        confidence=spec.get("confidence", "confirmed"),
        note=spec.get("note", ""),
    )


def extract_facts_for_user(messages: List[Message], use_llm_fallback: bool = True
                            ) -> Tuple[List[Fact], List[str]]:
    """Turn a user's messages into typed Facts. Deterministic first, LLM only as fallback."""
    facts: List[Fact] = []
    audit: List[str] = []
    for m in messages:
        tpl_name, specs = match_message(m.message_text)
        if specs:
            for spec in specs:
                f = _spec_to_fact(spec, m.user_id, f"message:{m.message_id}",
                                   m.related_event_id, m.request_id)
                # carry the one-off flag through the note (used by forecast)
                if spec.get("one_off"):
                    f.note = (f.note + " [one_off]").strip()
                facts.append(f)
            audit.append(f"{m.message_id}: matched template '{tpl_name}' -> "
                          f"{[s.get('kind') for s in specs]}")
            continue

        if use_llm_fallback and llm_client.is_enabled():
            parsed = llm_client.interpret_message(m.message_text, m.source_type)
            if parsed:
                spec = {"kind": parsed["kind"], "note": f"llm:{parsed.get('note','')}"}
                if parsed.get("amount") is not None:
                    try:
                        spec["new_amount"] = Decimal(str(parsed["amount"]))
                    except Exception:
                        pass
                if parsed.get("currency"):
                    spec["new_currency"] = parsed["currency"]
                if parsed.get("date"):
                    try:
                        y, mo, d = str(parsed["date"]).split("-")
                        spec["new_date"] = date(int(y), int(mo), int(d))
                    except Exception:
                        pass
                if parsed.get("percent") is not None:
                    try:
                        spec["multiplier"] = Decimal(1) + Decimal(str(parsed["percent"])) / 100
                    except Exception:
                        pass
                facts.append(_spec_to_fact(spec, m.user_id, f"message:{m.message_id}",
                                            m.related_event_id, m.request_id))
                audit.append(f"{m.message_id}: NO template match -> LLM fallback -> {parsed['kind']}")
                continue

        audit.append(f"{m.message_id}: NO template match, no LLM fallback -> ignored (safe default)")
    return facts, audit


def resolve_blank_amounts(events: List[FinancialEvent],
                           images_by_event: Dict[str, ImageRef],
                           images_dir: str = IMAGES_DIR) -> Tuple[List[str], List[str]]:
    """
    Fill blank event amounts from linked images.
    Returns (audit_lines, unresolved_event_ids).

    Never treats a blank amount as zero (explicit spec requirement). When no
    reading can be obtained the event is returned as unresolved and the
    forecast handles it conservatively.
    """
    audit, unresolved = [], []
    for e in events:
        if e.amount is not None:
            continue
        img = images_by_event.get(e.event_id)
        if img is None:
            unresolved.append(e.event_id)
            audit.append(f"{e.event_id}: blank amount and NO linked image -> unresolved")
            continue
        path = os.path.join(images_dir, f"{img.image_id}.png")
        hint = (f"This document corresponds to a {e.category} {e.event_type} "
                f"dated {e.event_date} in {e.currency}, described as {e.description!r}.")
        reading = image_resolver.resolve_image_amount(path, hint=hint)
        if reading and reading.get("amount") is not None:
            e.amount = money(reading["amount"])
            e.resolved_by = f"image:{img.image_id}"
            audit.append(f"{e.event_id}: amount {e.amount} from {img.image_id} "
                          f"[{reading['method']}/{reading['confidence']}] {reading['rationale'][:120]}")
        else:
            unresolved.append(e.event_id)
            audit.append(f"{e.event_id}: blank amount, image {img.image_id} unreadable -> "
                          f"unresolved (NOT treated as zero)")
    return audit, unresolved


def apply_status_rules(events: List[FinancialEvent], facts: List[Fact]) -> List[str]:
    """
    Mark events excluded/included per cash-state rules and message evidence.
    Returns audit lines.
    """
    audit = []
    retry_event_ids = {f.related_event_id for f in facts
                        if f.kind == "obligation_persists" and f.related_event_id}
    excluded_credit_ids = {f.related_event_id for f in facts
                            if f.kind == "exclude_pending_credit" and f.related_event_id}
    # Local lookup so a pending event can check the SETTLED status of the
    # earlier event it names via linked_event_id (problem_statement.md:
    # "linked_event_id... points to an earlier event in the same transaction
    # or investment lifecycle"). This field was previously parsed into every
    # FinancialEvent but never consulted anywhere in the pipeline.
    by_id = {e.event_id: e for e in events}
    # Reverse index: for a given (earlier) event, which LATER events name it
    # via their own linked_event_id? Needed below so a failed debit revived
    # by a "will be retried" message doesn't ALSO get double-counted against
    # an already-existing linked scheduled/settled retry of the same amount.
    linked_from: Dict[str, List[FinancialEvent]] = {}
    for e in events:
        if e.linked_event_id:
            linked_from.setdefault(e.linked_event_id, []).append(e)

    for e in events:
        if e.direction == "non_cash":
            e.excluded, e.exclusion_reason = True, "non_cash (investment valuation, not cash)"
            continue
        if e.status == "unrealized":
            e.excluded, e.exclusion_reason = True, "unrealized investment value"
            continue
        if e.status == "cancelled":
            e.excluded, e.exclusion_reason = True, "cancelled"
            continue
        if e.status == "failed":
            has_linked_retry = any(
                child.status in ("scheduled", "settled")
                and child.direction == e.direction
                and child.amount is not None and e.amount is not None
                and abs(child.amount - e.amount) <= Decimal("0.01")
                for child in linked_from.get(e.event_id, [])
            )
            if has_linked_retry:
                # The retry already exists as its own explicit, separately-
                # dated event (linked_event_id names this failed attempt as
                # its origin). Counting the original failed debit AGAIN via
                # the message-confirmed "will be retried" signal would
                # double-charge the user for one bill; the linked event is
                # the more precise, already-dated representation of the
                # SAME retry the message is describing.
                e.excluded, e.exclusion_reason = True, (
                    "failed attempt superseded by an already-linked scheduled/settled "
                    "retry of the same amount -- not double-counted even though a "
                    "message separately confirms a retry")
                audit.append(f"{e.event_id}: failed debit stays EXCLUDED -- retry is "
                              f"already represented by its own linked event, so the "
                              f"'will be retried' message is not also revived here")
            elif e.event_id in retry_event_ids:
                # No separate, already-dated retry event exists -- the
                # message is the ONLY evidence of persistence, so keep this
                # failed event alive as the liability itself.
                e.excluded = False
                e.exclusion_reason = ""
                audit.append(f"{e.event_id}: failed debit KEPT as live obligation "
                              f"(bank message confirms retry)")
            else:
                e.excluded, e.exclusion_reason = True, "failed attempt, no retry evidence"
            continue
        if e.status == "pending":
            if e.direction == "credit":
                e.excluded, e.exclusion_reason = True, "pending credit (not counted until settled)"
            else:
                # Reserve pending debits (conservative) UNLESS this pending
                # debit is itself a DUPLICATE of an already-settled charge:
                # linked_event_id names an earlier event that is settled,
                # same direction, same amount. Counting both double-charges
                # the user for one purchase -- exactly the "duplicate
                # records" problem_statement.md's 90-day safety check says
                # to ignore. Matched structurally (status + direction +
                # amount), not by description text, so it isn't tied to a
                # specific label.
                parent = by_id.get(e.linked_event_id) if e.linked_event_id else None
                is_duplicate = (
                    parent is not None and parent.status == "settled"
                    and parent.direction == e.direction
                    and parent.amount is not None and e.amount is not None
                    and abs(parent.amount - e.amount) <= Decimal("0.01")
                )
                if is_duplicate:
                    e.excluded = True
                    e.exclusion_reason = (f"duplicate of already-settled linked event "
                                          f"{parent.event_id} (same amount/direction)")
                    audit.append(f"{e.event_id}: excluded as duplicate of settled "
                                  f"{parent.event_id} (linked_event_id)")
                else:
                    e.excluded = False   # reserve pending debits (conservative)
            continue
        if e.event_id in excluded_credit_ids and e.direction == "credit":
            e.excluded, e.exclusion_reason = True, "message confirms credit not yet received"

    return audit
