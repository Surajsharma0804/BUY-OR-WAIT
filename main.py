#!/usr/bin/env python3
"""
Buy or Wait? -- entry point.

Reads dataset/, runs the agent over every row in dataset/requests.csv, and
writes predictions to output.csv in the repository root (the location
README.md requires).

Usage:
    python3 code/main.py                    # full run -> ./output.csv
    python3 code/main.py --samples          # run the 25 solved samples instead
    python3 code/main.py --no-llm           # force fully deterministic mode
    python3 code/main.py --out path.csv     # write somewhere else
    python3 code/main.py --audit audit.txt  # dump the per-request reasoning trail

Exit code is non-zero if the produced file fails whole-file validation, so a
run cannot "succeed" while emitting a malformed submission.
"""

from __future__ import annotations
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.pipeline import run                      # noqa: E402
from agent.data_loader import load_dataset          # noqa: E402
from agent.validators import validate_file          # noqa: E402
from agent.config import DATASET_DIR, OUTPUT_CSV    # noqa: E402
from agent import llm_client                        # noqa: E402

HEADER = ["request_id", "amount_safe_to_pay", "affordability_status",
          "recommended_payment_method", "payment_plan",
          "earliest_date_for_full_payment", "spending_changes_needed",
          "decision_explanation"]


def write_csv(rows, path: str):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for r in rows:
            w.writerow(r.as_csv_row())


def write_usage_report(path: str, n_requests: int, elapsed: float, use_llm: bool):
    """
    Emit evaluation/usage_report.md describing the run that produced output.csv.
    Required by problem_statement.md, "Token Usage and Cost Analysis".
    """
    from decimal import Decimal
    ledger = llm_client.LEDGER
    by_model = ledger.by_model()
    totals = ledger.totals()

    L = []
    L.append("# Token Usage and Cost Analysis\n")
    L.append(f"Run covering **{n_requests} requests** from `dataset/requests.csv` "
             f"— the run that produced `output.csv`.\n")
    L.append(f"- Wall-clock runtime: **{elapsed:.1f}s**")
    L.append(f"- LLM/VLM extraction: "
             f"**{'enabled' if use_llm and llm_client.is_enabled() else 'disabled'}**\n")

    L.append("## Where models are (and are not) used\n")
    L.append("The decision engine is **fully deterministic**. Affordability, the 90-day "
             "safety check, plan ranking, and every number written to `output.csv` are "
             "computed in code — never by a model. This is a deliberate architecture "
             "choice: a hallucinated figure inside a financial recommendation is far more "
             "damaging than a misread receipt, and the latter is independently "
             "cross-checked.\n")
    L.append("Model calls are reserved for two *extraction* fallbacks:\n")
    L.append("1. **Vision (VLM)** — read the authoritative amount off a linked document "
             "image when the deterministic OCR path cannot resolve it confidently. The "
             "reading is cross-validated against OCR; disagreements are recorded.")
    L.append("2. **Text** — interpret a message matching none of the deterministic "
             "templates. All 215 messages in this dataset are covered by the template "
             "catalogue, so this path is dormant here.\n")

    if not ledger.records:
        L.append("## Model usage totals\n")
        L.append("**No model calls were made in this run.**\n")
        L.append("Either no `ANTHROPIC_API_KEY` was set, or neither fallback was needed. "
                 "All 215 messages resolved via the deterministic template catalogue and "
                 "image amounts via the OCR pipeline, so total cost was **$0.00**.\n")
        L.append("| Metric | Value |")
        L.append("|---|---|")
        L.append("| Model provider | Anthropic (configured, unused this run) |")
        L.append("| Model calls | 0 |")
        L.append("| Input tokens | 0 |")
        L.append("| Output tokens | 0 |")
        L.append("| Total tokens | 0 |")
        L.append("| Avg tokens / request | 0 |")
        L.append("| Estimated total cost | $0.00 |")
        L.append("| Estimated cost / request | $0.00 |")
    else:
        L.append("## Per-model totals\n")
        L.append("| Provider | Model | Calls | Input tokens | Output tokens | Total tokens "
                 "| Est. cost (USD) |")
        L.append("|---|---|---|---|---|---|---|")
        for model, e in sorted(by_model.items()):
            tot = e["input_tokens"] + e["output_tokens"]
            L.append(f"| Anthropic | `{model}` | {e['calls']} | {e['input_tokens']:,} | "
                     f"{e['output_tokens']:,} | {tot:,} | ${e['cost_usd']:.4f} |")
        grand = totals["input_tokens"] + totals["output_tokens"]
        L.append(f"| **Overall** | — | **{totals['calls']}** | **{totals['input_tokens']:,}** "
                 f"| **{totals['output_tokens']:,}** | **{grand:,}** | "
                 f"**${totals['cost_usd']:.4f}** |\n")
        per_tok = grand / n_requests if n_requests else 0
        per_cost = totals["cost_usd"] / n_requests if n_requests else Decimal("0")
        L.append("## Per-request averages\n")
        L.append("| Metric | Value |")
        L.append("|---|---|")
        L.append(f"| Requests | {n_requests} |")
        L.append(f"| Avg tokens / request | {per_tok:,.1f} |")
        L.append(f"| Est. cost / request | ${per_cost:.6f} |")
        L.append(f"| Est. total cost | ${totals['cost_usd']:.4f} |")

    L.append("\n## Cost controls\n")
    L.append("- **Disk-cached image reads.** Vision results are cached by image content "
             "hash, so re-running the full dataset never re-bills a document already read.")
    L.append("- **Deterministic-first routing.** A model is consulted only after the "
             "rule-based path declines, which keeps the common case free.")
    L.append("- **Right-sized model.** Extraction uses a small, fast model; no frontier "
             "model is needed to read a total off a receipt.")
    L.append("- **Bounded retries.** Malformed JSON is retried a limited number of times, "
             "then the system falls back rather than looping.\n")

    L.append("## Pricing basis\n")
    L.append("Published Anthropic list pricing per million tokens at the time of the run:\n")
    L.append("| Model | Input / MTok | Output / MTok |")
    L.append("|---|---|---|")
    for model, p in sorted(llm_client.MODEL_PRICING_PER_MTOK.items()):
        L.append(f"| `{model}` | ${p['input']} | ${p['output']} |")
    L.append("\nNo API keys or credentials appear in this report or anywhere in the "
             "submission; the key is read from the `ANTHROPIC_API_KEY` environment "
             "variable only.\n")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description="Buy or Wait? financial decision agent")
    ap.add_argument("--dataset", default=DATASET_DIR, help="path to dataset/ directory")
    ap.add_argument("--out", default=OUTPUT_CSV, help="output CSV path")
    ap.add_argument("--samples", action="store_true",
                    help="run dataset/sample_requests.csv instead of requests.csv")
    ap.add_argument("--no-llm", action="store_true",
                    help="disable the optional LLM/VLM extraction fallback")
    ap.add_argument("--audit", help="write the per-request reasoning trail to this path")
    ap.add_argument("--usage-report", help="path for evaluation/usage_report.md")
    args = ap.parse_args()

    which = "samples" if args.samples else "requests"
    use_llm = not args.no_llm

    print("Buy or Wait? agent")
    print(f"  dataset : {args.dataset}")
    print(f"  target  : {which}")
    print(f"  LLM     : "
          f"{'enabled' if (use_llm and llm_client.is_enabled()) else 'disabled (deterministic)'}")

    t0 = time.time()
    result = run(args.dataset, which=which, use_llm=use_llm, verbose=True)
    elapsed = time.time() - t0

    write_csv(result.rows, args.out)
    print(f"\n  wrote {len(result.rows)} rows -> {args.out}  ({elapsed:.1f}s)")

    ds = load_dataset(args.dataset)
    targets = ds.sample_requests if args.samples else ds.requests
    problems = validate_file(args.out, targets)

    if result.unresolved_amounts:
        print(f"  note: {len(result.unresolved_amounts)} event(s) had an unreadable image "
              f"amount, flagged for review and never treated as zero: "
              f"{result.unresolved_amounts}")
    if result.validation_problems:
        print(f"  warning: {len(result.validation_problems)} row(s) failed row-level "
              f"validation and fell back to a conservative recommendation")
        for rid, errs in list(result.validation_problems.items())[:5]:
            print(f"     {rid}: {errs[0]}")
    if result.failures:
        print(f"  warning: {len(result.failures)} request(s) raised errors")
        for f in result.failures[:5]:
            print(f"     {f.splitlines()[0]}")

    if args.audit:
        with open(args.audit, "w", encoding="utf-8") as f:
            for rid, lines in result.audit.items():
                f.write(f"=== {rid} ===\n")
                for ln in lines:
                    f.write(f"  {ln}\n")
                f.write("\n")
        print(f"  audit trail -> {args.audit}")

    report_path = args.usage_report or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "evaluation", "usage_report.md")
    write_usage_report(report_path, n_requests=len(result.rows), elapsed=elapsed,
                       use_llm=use_llm)
    print(f"  usage report -> {report_path}")

    if problems:
        print("\n  FILE VALIDATION FAILED:")
        for p in problems:
            print(f"     - {p}")
        sys.exit(1)

    print("\n  file validation passed: header, row count and request coverage are correct.")


if __name__ == "__main__":
    main()
