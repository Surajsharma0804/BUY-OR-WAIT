"""
End-to-end orchestration: dataset in, validated rows out.

One function, `run`, so both `code/main.py` (produces output.csv) and
`code/evaluation/evaluate.py` (scores against the solved samples) drive the
exact same code path. Anything that only runs in one of those two places is
a source of "works in eval, breaks in submission" bugs, so there is
deliberately no separate scoring pipeline.

Per-request isolation: a single bad row can never abort a 250-request run.
Every request is processed in its own try/except; on failure the request
falls back to a conservative, provably-valid row and the failure is logged
rather than silently dropped.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import List, Dict, Optional
import os
import traceback

from .data_loader import load_dataset, Dataset
from .currency import ExchangeRates
from .models import Request, OutputRow
from .config import DATASET_DIR, money
from .reconciliation import (
    extract_facts_for_user, resolve_blank_amounts, apply_status_rules,
)
from .decide import decide
from .validators import validate_row


@dataclass
class RunResult:
    rows: List[OutputRow] = field(default_factory=list)
    audit: Dict[str, List[str]] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)
    validation_problems: Dict[str, List[str]] = field(default_factory=dict)
    unresolved_amounts: List[str] = field(default_factory=list)


def _fallback_row(req: Request, reason: str) -> OutputRow:
    """
    Conservative, always-schema-valid row used when a request cannot be
    processed. Recommending nothing is the safe default for a financial
    decision we could not verify.
    """
    return OutputRow(
        request_id=req.request_id,
        amount_safe_to_pay=Decimal("0"),
        affordability_status="not_affordable",
        recommended_payment_method="not_recommended",
        payment_plan="none",
        earliest_date_for_full_payment="",
        spending_changes_needed="none",
        decision_explanation=("This request could not be verified against the 90-day "
                              "safety check, so no payment is recommended."),
    )


def run(dataset_dir: str = DATASET_DIR,
        which: str = "requests",
        use_llm: bool = True,
        limit: Optional[int] = None,
        verbose: bool = False) -> RunResult:
    ds = load_dataset(dataset_dir)
    fx = ExchangeRates.load(os.path.join(dataset_dir, "exchange_rates.csv"))
    result = RunResult()

    targets = ds.requests if which == "requests" else ds.sample_requests
    if limit:
        targets = targets[:limit]

    # Resolve blank amounts once for the whole dataset (image reads are cached).
    img_audit, unresolved = resolve_blank_amounts(ds.events, ds.images_by_related_event)
    result.unresolved_amounts = unresolved
    result.audit["_image_resolution"] = img_audit

    for req in targets:
        try:
            profile = ds.profiles.get(req.user_id)
            if profile is None:
                raise KeyError(f"no financial profile for {req.user_id}")

            user_events = ds.events_by_user.get(req.user_id, [])
            user_messages = ds.messages_by_user.get(req.user_id, [])

            facts, fact_audit = extract_facts_for_user(user_messages, use_llm_fallback=use_llm)
            status_audit = apply_status_rules(user_events, facts)

            options = ds.options.get(req.request_id, [])
            decision = decide(req, profile, user_events, facts, options, fx,
                              unresolved_event_ids=unresolved)

            problems = validate_row(decision.row, req, profile, options, ds.events_by_id)
            if problems:
                result.validation_problems[req.request_id] = problems
                row = _fallback_row(req, "; ".join(problems))
                result.failures.append(f"{req.request_id}: validation failed -> fallback: {problems}")
            else:
                row = decision.row

            result.rows.append(row)
            result.audit[req.request_id] = fact_audit + status_audit + decision.audit

        except Exception as e:
            result.rows.append(_fallback_row(req, str(e)))
            result.failures.append(f"{req.request_id}: EXCEPTION {e}\n{traceback.format_exc()}")
            if verbose:
                print(f"  !! {req.request_id}: {e}")

    return result
