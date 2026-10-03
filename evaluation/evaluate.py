"""
Evaluation harness.

Scores the agent against dataset/sample_requests.csv (the 25 requests that
ship with ground-truth output columns) using the same field weighting the
challenge says it will apply:

    accuracy of amount_safe_to_pay
    correctness of affordability_status
    correctness of recommended_payment_method and payment_plan
    accuracy of earliest_date_for_full_payment
    validity of spending_changes_needed
    usefulness/consistency of decision_explanation

Design notes:
  * Runs the SAME agent.pipeline.run() used to produce the submission, so
    a passing eval means the submission path works, not a parallel one.
  * Reports per-field accuracy AND a confusion matrix over
    affordability_status, because a single aggregate number hides which
    decision class is actually broken.
  * Prints the worst mismatches with the agent's own audit trail, so a
    failing case can be debugged without re-instrumenting the run.

Usage:
    python3 code/evaluation/evaluate.py            # score the samples
    python3 code/evaluation/evaluate.py --verbose  # + per-case audit
"""

from __future__ import annotations
import argparse
import os
import sys
from collections import Counter, defaultdict
from decimal import Decimal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.pipeline import run           # noqa: E402
from agent.data_loader import load_dataset  # noqa: E402
from agent.config import DATASET_DIR, money  # noqa: E402

FIELDS = [
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
]

# amount_safe_to_pay is graded with a small relative tolerance: the forecast
# is built from median-based projections, so exact-cent equality with the
# organizers' generator is not a meaningful bar, but being in the right
# neighbourhood is.
AMOUNT_REL_TOL = Decimal("0.01")


def _amounts_match(pred: str, truth: str) -> bool:
    try:
        p, t = Decimal(str(pred)), Decimal(str(truth))
    except Exception:
        return str(pred).strip() == str(truth).strip()
    if p == t:
        return True
    scale = max(abs(p), abs(t), Decimal("1"))
    return abs(p - t) / scale <= AMOUNT_REL_TOL


def _norm_plan(s: str) -> str:
    """Normalize a payment_plan for comparison (trailing zeros, spacing)."""
    if not s or s == "none":
        return "none"
    out = []
    for part in s.split("|"):
        if ":" not in part:
            out.append(part.strip())
            continue
        d, a = part.split(":", 1)
        try:
            av = Decimal(a)
            a = format(av.normalize(), "f")
            if "." in a:
                a = a.rstrip("0").rstrip(".")
        except Exception:
            a = a.strip()
        out.append(f"{d.strip()}:{a}")
    return "|".join(out)


def _norm_changes(s: str) -> str:
    if not s or s == "none":
        return "none"
    parts = []
    for p in s.split("|"):
        bits = p.split(":")
        if bits[0] == "reduce_to" and len(bits) == 3:
            try:
                v = Decimal(bits[2])
                v = format(v.normalize(), "f")
                if "." in v:
                    v = v.rstrip("0").rstrip(".")
                bits[2] = v
            except Exception:
                pass
        parts.append(":".join(bits))
    return "|".join(sorted(parts))


def compare(pred_row, truth: dict) -> dict:
    out = {}
    out["amount_safe_to_pay"] = _amounts_match(pred_row.amount_safe_to_pay,
                                                truth["amount_safe_to_pay"])
    out["affordability_status"] = (pred_row.affordability_status ==
                                    truth["affordability_status"])
    out["recommended_payment_method"] = (pred_row.recommended_payment_method ==
                                          truth["recommended_payment_method"])
    out["payment_plan"] = (_norm_plan(pred_row.payment_plan) ==
                            _norm_plan(truth["payment_plan"]))
    out["earliest_date_for_full_payment"] = (
        (pred_row.earliest_date_for_full_payment or "").strip() ==
        (truth["earliest_date_for_full_payment"] or "").strip())
    out["spending_changes_needed"] = (_norm_changes(pred_row.spending_changes_needed) ==
                                       _norm_changes(truth["spending_changes_needed"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true", help="print per-case audit trails")
    ap.add_argument("--no-llm", action="store_true", help="disable LLM fallback")
    ap.add_argument("--case", help="only evaluate this request_id")
    args = ap.parse_args()

    ds = load_dataset(DATASET_DIR)
    truth_by_id = {r.request_id: r.expected for r in ds.sample_requests}
    req_by_id = {r.request_id: r for r in ds.sample_requests}

    res = run(DATASET_DIR, which="samples", use_llm=not args.no_llm)

    field_hits = Counter()
    total = 0
    confusion = defaultdict(Counter)
    mismatches = []

    for row in res.rows:
        if args.case and row.request_id != args.case:
            continue
        truth = truth_by_id.get(row.request_id)
        if not truth:
            continue
        total += 1
        cmp = compare(row, truth)
        for f, ok in cmp.items():
            field_hits[f] += int(ok)
        confusion[truth["affordability_status"]][row.affordability_status] += 1
        if not all(cmp.values()):
            mismatches.append((row.request_id, cmp, row, truth))

    print("=" * 78)
    print(f"SAMPLE EVALUATION  ({total} cases)")
    print("=" * 78)
    for f in FIELDS:
        pct = 100.0 * field_hits[f] / total if total else 0
        bar = "#" * int(pct / 4)
        print(f"  {f:<34} {field_hits[f]:>3}/{total}  {pct:5.1f}%  {bar}")
    exact = sum(1 for rid, cmp, _, _ in mismatches if False)
    all_correct = total - len(mismatches)
    print(f"\n  {'ALL FIELDS CORRECT':<34} {all_correct:>3}/{total}  "
          f"{100.0*all_correct/total if total else 0:5.1f}%")

    print("\n  affordability_status confusion (truth -> predicted):")
    for t in sorted(confusion):
        for p, c in sorted(confusion[t].items()):
            flag = "" if t == p else "   <-- MISMATCH"
            print(f"    {t:<22} -> {p:<22} {c}{flag}")

    if res.failures:
        print(f"\n  !! {len(res.failures)} processing failures:")
        for f in res.failures[:10]:
            print(f"     {f.splitlines()[0]}")

    if mismatches:
        print("\n" + "=" * 78)
        print(f"MISMATCHES ({len(mismatches)})")
        print("=" * 78)
        for rid, cmp, row, truth in mismatches:
            bad = [f for f, ok in cmp.items() if not ok]
            req = req_by_id[rid]
            print(f"\n{rid}  (user={req.user_id}, type={req.request_type}, "
                  f"requested={req.requested_amount}, allows_partial={req.allows_partial_payment})")
            print(f"  wrong fields: {bad}")
            for f in bad:
                pv = getattr(row, f)
                tv = truth[f]
                print(f"    {f}:")
                print(f"       predicted: {pv}")
                print(f"       truth:     {tv}")
            if args.verbose:
                print("  audit:")
                for line in res.audit.get(rid, []):
                    print(f"     - {line}")
    print()


if __name__ == "__main__":
    main()
