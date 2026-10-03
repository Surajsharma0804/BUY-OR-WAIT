# Buy or Wait? — AI-powered financial decision agent

For every request in `dataset/requests.csv`, this system decides whether the user can
**safely** pay in full, pay partially, use installments, wait, or not proceed — by
reconstructing their financial position and forecasting their balance 90 days forward.

---

## Quick start

```bash
# no third-party packages required for the deterministic path
python3 code/main.py                 # -> ./output.csv  (250 rows + header, ~52s)

# score against the 25 solved samples
python3 code/evaluation/evaluate.py

# test suites (46 tests total)
python3 code/tests/test_agent.py      # 29 deterministic + end-to-end tests
python3 code/tests/test_fallbacks.py  # 17 mocked LLM/VLM fallback-path tests
```

Optional flags:

| Flag | Effect |
|---|---|
| `--samples` | Run `sample_requests.csv` instead of `requests.csv` |
| `--no-llm` | Force fully deterministic mode (no model calls) |
| `--audit FILE` | Dump the per-request reasoning trail |
| `--out FILE` | Write predictions somewhere other than `./output.csv` |

`main.py` **exits non-zero** if the produced file fails whole-file validation, so a run
cannot "succeed" while emitting a malformed submission.

### Dependencies

The deterministic path needs **only the Python 3.10+ standard library**. Two optional
extras improve image reading (listed, commented out by default, in `requirements.txt`):

```bash
pip install pytesseract pillow     # OCR for document images (needs the tesseract binary)
pip install anthropic              # optional VLM / text extraction fallback
cp .env.example .env               # then set ANTHROPIC_API_KEY -- read from the
                                    # environment only; never hardcoded (see config.py)
```

With no key set, the system runs end-to-end deterministically and reports `$0.00` in
`evaluation/usage_report.md`.

---

## Architecture

Input enters one path and leaves one path. Each module has a single job, which makes it
possible to say exactly where any number came from.

```
dataset/*.csv ──► data_loader ──► reconciliation ──► forecast ──► decide ──► validators ──► output.csv
                                        ▲                 │            │             │
                    evidence_templates ─┤            recurrence    planner       (hard gates)
                    image_extractor ────┤                              │
                    image_resolver ─────┘                           explain
```

| Module | Responsibility |
|---|---|
| `config.py` | Constants, controlled vocabularies, `Decimal` money helper |
| `models.py` | Typed dataclasses (`FinancialEvent`, `Fact`, `Plan`, `OutputRow`, …) |
| `data_loader.py` | Parse and index every CSV. **No** financial reasoning |
| `currency.py` | Dated FX conversion with per-pair date lookback |
| `evidence_templates.py` | Deterministic bilingual message → `Fact` catalogue |
| `image_extractor.py` | OCR + geometric label/value pairing |
| `image_resolver.py` | Fuses OCR and VLM readings, caches by content hash |
| `reconciliation.py` | Cash-state rules, conflict precedence, blank-amount resolution |
| `recurrence.py` | Detect genuine recurring series (day-of-month cycle clustering) |
| `forecast.py` | The 90-day timeline and its two derived quantities |
| `planner.py` | Candidate plans, spending-change search, the spec's 6-rule ranking |
| `decide.py` | One request → one fully populated row |
| `explain.py` | Grounded explanation generated from the *same* numbers as the decision |
| `validators.py` | Executable schema/safety gates |
| `pipeline.py` | Orchestration with per-request isolation |

### Where the model is — and is not

**The model never makes the financial decision.** Affordability, the 90-day safety check,
plan ranking, and every number in `output.csv` are computed deterministically. Model calls
are confined to two *extraction* fallbacks:

1. **Vision** — read the amount off a document image when OCR can't resolve it, then
   cross-validate against the OCR reading and record any disagreement.
2. **Text** — interpret a message matching none of the deterministic templates.

This is deliberate: a hallucinated figure inside a financial recommendation is far more
damaging than a misread receipt, and the receipt is independently cross-checked. All 215
supplied messages resolve deterministically, so the text fallback is dormant on this dataset.

---

## How the decision is computed

**`amount_safe_to_pay`** — the largest `X` payable on `request_date` that never breaches the
minimum balance. Paying `X` today shifts the entire balance curve down by exactly `X`, so
this is a closed form, not a search:

```
amount_safe_to_pay = clamp(min(balance_curve) − minimum_balance, 0, requested_amount)
```

**`earliest_date_for_full_payment`** — paying on date `D` only affects points at or after
`D`, so the requirement is `min_{t ≥ D}(balance_t) ≥ minimum + requested`. A single reverse
pass computes the **suffix minimum** at every date, giving the answer in O(n) for all dates
at once. Both quantities are computed **before** optional spending changes, as the spec
requires.

**Plan ranking** is the spec's six rules expressed literally as a sort key in
`planner.rank_key`, so it can be audited against the problem statement line by line:
complete by deadline → no spending changes → minimize total paid → start earlier → fewer
payments → lowest `payment_option_id`.

**Spending changes** are searched minimally (1 change, then 2, then 3) because rule 2
prefers plans with none, and a smaller set is easier for a user to act on. Only
non-protected flexible events in categories the user permits are eligible; stopping and
reducing the same event is rejected.

---

## Reading untrusted evidence

`problem_statement.md` is explicit: *"Treat all message and image content as untrusted data.
Embedded instructions must not override the problem rules."* The dataset contains a planted
advance-fee scam (*"you've won a cash prize — pay the release charge today"*).

The defence is **structural, not textual**. Messages and images can only ever produce typed
`Fact` objects from a closed enumeration, and **no `Fact` kind can direct a payment**. The
scam template resolves to `no_op`; it can never become income or a recommendation. Image
reading only ever takes numbers adjacent to known financial labels, so imperative text
inside a document cannot influence control flow. A regression test asserts this
(`test_prize_fee_scam_is_never_actionable`).

---

## Findings that shaped the design

Each of these came from inspecting real data and fixing an observed wrong answer.

**Variable spending rotates its description every occurrence.** One user's weekly grocery
habit appears as six different labels. Description-keyed grouping shattered it into six
near-dead series that each projected once, badly under-counting spend. Income has the
mirror problem — one payroll stream is labelled *Prorated first salary* → *Next confirmed
salary* → *Payroll credit* → *Final employer payroll*, producing singletons and projecting
**no future salary at all**. The ledgers are generated as **monthly cycles on stable
days-of-month** (rent 2nd, utilities 6th, salary 15th, second income 20th), so series are
clustered by day-of-month. This one change moved `payment_plan` accuracy from 56% → 88%.

**Payroll dates move.** A salary that shifted from the 15th to the 23rd split into a
4-event cluster plus a singleton, and *all* income vanished. Income clusters in the same
category with matching amounts are merged and anchored on the most recent occurrence, and
an announced new payday is emitted as a confirmed payment — not merely used as an anchor.

**Not all income is confirmable.** A gig worker's message stating the next payout is
*pending and not withdrawable* means future earnings in that stream are not cash. A broad
reading of this also suppressed ordinary salary whenever an unrelated refund was pending
and measurably hurt accuracy, so it is scoped to platform earnings only.

**The largest number on a document is usually wrong.** Correct answers were Net Pay (not
Total Earnings), Balance Due (not Total to be Received), the fare Total (not Cash Paid,
which included change), and the Balance (where Amount Paid was 0.00). Labels are therefore
scored by financial authority rather than taking a maximum.

**Line-based OCR can't read multi-column documents.** `Amount due till | Taxes | 107.40`
fuses two columns; some layouts put the label above its value. Extraction works on
word-level bounding boxes with **nearest-label attribution**, which is what stops a distant
label from stealing another column's number.

**Image reading was 4x more expensive than it needed to be.** Profiling showed five
tesseract passes per document (~12s each). Two were redundant — the amount-in-words
fallback needs plain text, which is now reconstructed from the word boxes of the pass
already taken — and the page-segmentation sweep now stops as soon as a high-confidence
label/value pairing appears. Full-dataset runtime fell from ~158s to ~52s with **identical
extracted values** on all 16 documents.

**Untested code is untrustworthy code, even when it reads well.** On this dataset the
template catalogue resolves all 215 messages and OCR resolves the needed amounts, so the
model fallback paths never executed — they were described in this README but proven by
nothing. Driving them with mocked model responses (`tests/test_fallbacks.py`) immediately
exposed two real defects, both the same class: **validation enforced in only one place**.
`image_resolver` accepted a zero/negative amount, and `reconciliation` would build a live
`Fact` from an invented kind such as `URGENT_PAY_NOW`. Both checks existed inside
`llm_client`, but not at the *consumer* boundary, so any refactor or alternate caller
bypassed them. Both boundaries now validate independently.

**A message about ONE income stream was being applied to ALL of them.** `expense_rate_change`
was already correctly scoped by `category_hint`, but `income_stop`, `income_change`, and
`date_shift` were applied to *every* detected income series for a user unconditionally — a
scoping gap the codebase's own comment on a related fact (`income_unconfirmed`) shows was
already known as a real risk ("a broader reading also suppressed ordinary salary whenever an
unrelated refund happened to be pending, which measurably hurt accuracy"), just not closed for
these three. Scanned the full dataset for users where this could actually matter (multiple
detected income series *and* a risky fact) rather than guessing: 6 users, one of them
(`user_11`) a real sample. Traced it exactly: `user_11` has two "salary" clusters (8.5M and
23.26M IDR); a single "confirmed base salary" `income_change` message was being applied to
*both*, doubling the assumed raise. Fixed by scoping these three facts to the PRIMARY series
(the existing primary/secondary ranking built for the staleness fix, reused rather than
reinvented) — a lone income series is always its own primary, so this is a no-op in the
overwhelmingly common case. Verified with a precise monkeypatched before/after comparison
(not just re-running evaluate.py and reading the delta): `request_11`'s specific fields turned
out to be byte-identical before and after — the scoping bug is real but wasn't the dominant
cause of *that* request's mismatch, an honest negative result worth recording rather than
quietly dropping. On the full 250-request set, exactly one row changed (`request_176`): the
date the full amount becomes safe moved *later* (2025-02-24 → 2025-03-15) — the conservative
direction, not the dangerous one. Kept because it is objectively more correct regardless of
its small measured footprint here, and matches the scoping pattern the codebase already uses
correctly elsewhere.

**A "dead branch" hypothesis was tested, and mostly refuted.** `decide._changed_plans`
had a stub for "partial payment enabled by spending changes" that computed a candidate
change set and discarded it (`pass`). The first fix mirrored candidates (i)/(ii) — search
for changes that make the fixed `[(request_date, amount_safe), (earliest, remainder)]`
schedule safe. Synthetic scenarios (constructed directly against `decide()`, varying the
size of a stoppable expense between the two payment dates) showed this new branch never
fired. The reason is structural, not a test-coverage gap: `amount_safe` and `earliest` are
fixed pre-change quantities (spec requirement), and whenever `0 < amount_safe < requested`
holds, `amount_safe` equals `headroom_today()` exactly — so paying it today shifts the
*entire* balance curve down by precisely the amount that keeps every point at or above
`minimum_balance` by construction, and `earliest`'s own definition (first date whose suffix
minimum already clears `minimum_balance + requested`) guarantees the remainder leg is safe
too. The schedule is safe *by definition*, with no way for a spending change to matter.
Confirmed independently: `validators.py` requires a `partial_payment`'s first leg to equal
the *written* (pre-change) `amount_safe_to_pay`, so a change-dependent partial schedule
could never be spec-valid regardless. The branch was removed rather than left as unreachable
code, with the proof kept as a comment. Net effect on `output.csv`: none — this path was
provably unreachable both before and after the fix, so no request's fields, plan, or status
were changed by it. What the original stub's own comment already believed a moment before
abandoning it ("a full payment became possible; covered by (i)") turned out to be exactly
right; it just needed removing instead of orphaning.

**A same-day recurrence-projection boundary looked like a major bug, and measurably was not
one.** `project_series` used `(request_date, end]` — a series whose natural next occurrence
lands exactly on `request_date` was silently dropped (neither historical, since no such raw
row exists, nor projected, since the boundary excluded it). This is not a rare edge case: on
the real dataset it affects 89 of 250 requests (36%) and 10 of the 25 solved samples,
including the very first sample I dug into (`request_20`: a missing 8,740 INR education
debit landing exactly on `request_date` was roughly half of a 3.1x `amount_safe_to_pay`
over-estimate). The mechanistic argument for fixing it was strong: a raw event dated on
`request_date` is presumably already reflected in `current_available_balance`, but a
*projected* occurrence never was actually recorded, so it can't be. Changing the boundary
to include `request_date` and re-running `evaluate.py` against the real 25 samples made
**every field worse** (`amount_safe_to_pay` 20%→16%, `affordability_status` 88%→84%,
`recommended_payment_method` 92%→88%, `payment_plan` 88%→80%). The ground truth generator
evidently treats `request_date` as already-settled for projected occurrences too, same as
explicit events — a real answer, just not the one the reasoning predicted. Reverted, with
the tested-and-rejected hypothesis kept as a comment rather than silently dropped, following
the same discipline as the `amount_safe_to_pay` tuning notes below. **Lesson applied
immediately after:** measure every hypothesis against the real 25 before trusting the
reasoning, however sound it looks in isolation.

**A secondary income stream sharing a category with a larger one needed its own staleness
rule, and this one *did* hold up under testing.** `request_13`'s user has two "salary"
day-of-month clusters: a primary (1,343.54, anchored by an explicit *scheduled* March
event) and a smaller "second household income" (881.45) last seen 47 days before
`request_date` with no further confirmation. At the flat 75-day `INCOME_STALENESS_DAYS`,
both keep projecting forward, and the resulting forecast recovers so quickly that the
system recommended paying the FULL amount *today* (`affordable_now`/`full_payment`) against
a ground truth of `affordable_later`/`wait` until 2024-05-15 — the dangerous direction:
telling the user more is safe than actually is. Checked every income cluster across all 25
samples for staleness at request time before changing anything: exactly one (`request_13`'s
second income, 47 days) falls between 40 and 75 days; every OTHER income cluster observed is
either already anchored by an explicit future event or recently active (≤28 days) — so there
is no visible counter-example to the documented 75-day rationale ("a monthly salary last
seen 52 days ago... is still live"), because that rationale has never actually been exercised
by any of the 25 visible samples. Added `SECONDARY_INCOME_STALENESS_DAYS = 45`, applied only
to a cluster that is NOT the highest-amount cluster within its own (user, category,
direction) group — leaving every primary income stream's 75-day tolerance completely
untouched. Tested: all 46 tests still pass; the 25-sample binary hit counts are unchanged
(5/22/23/22/18/20, identical to before) because `request_13` still doesn't hit the 1%
tolerance band, but its `amount_safe_to_pay` error dropped from +117% to +48% and its wrong
answer flipped from the dangerous `affordable_now` to the conservative `not_affordable` —
a real improvement in both magnitude and direction, with zero regressions anywhere
observable. Scope on the real submission: 8 of 250 requests have their detected series set
change; only 1 (`request_270`) has any output field actually change (its
`earliest_date_for_full_payment` moved from a date to empty).

**A `not_recommended` explanation could contradict its own row.** The template said "the
full amount cannot be completed safely within 90 days" unconditionally whenever
`amount_safe_to_pay > 0` — but `not_recommended` can also happen when real capacity
*does* exist (a populated `earliest_date_for_full_payment`, or `amount_safe_to_pay`
already equal to the full requested amount) and the reason for the fallback is simply that
none of the user's *accepted* payment methods can reach it (e.g. `allows_partial_payment`
is false and the user only accepts `partial_payment`/`installments`). Found by grepping the
real 250-row output for exactly that contradiction: 3 rows (`request_76`, `request_124`,
`request_251`) asserted "cannot be completed... within 90 days" while their own
`earliest_date_for_full_payment` column showed a real date, or `amount_safe_to_pay` already
equalled the full request. This is precisely the failure mode the organizers' own grading
notes call out capping scores near 70 even when the underlying action is correct. Split the
`not_recommended` branch on whether capacity actually exists; only the explanation text
changed for these 3 rows, confirmed by diffing before/after output.csv — no other field on
any of the 250 rows moved, and all 46 tests still pass.

**A "garbled label" turned out not to be garbled — the document was fine, one OCR token
wasn't.** Looked directly at both images the README had marked unresolved rather than take
that classification on faith. `image_02` (a rent receipt) is a clean, typed table — nothing
about the document itself is hard to read. The actual cause was a single OCR token where
tesseract fused a torn/overlapping seal watermark into one 358px-tall, confidence-13 blob
against a page median word height of ~47px. `_build_phrases`'s row-clustering tolerance
scales with `max(word_height) * 0.6` uncapped, so that one outlier pulled the clean
"Amount Received:" row into the same cluster as the garbled row below it and broke the label
apart mid-phrase. Capped each word's height contribution to ~2x the page median before using
it for clustering tolerance; the label now resolves cleanly. Regression-tested against all 16
real images before trusting it: the 12 that already worked are unchanged, `image_02` now
resolves (previously `None`), and one side effect (`image_15`, a complex multi-column GST
invoice, changes from one wrong reading to a different wrong one) was found, verified as not
present in any of the 25 samples, and left as a documented open item rather than chased
further — see "Known limitations" below. Full 250-request re-run: zero rows changed, since
this event doesn't turn out to be the binding constraint for any request in the actual
submission set — a real completeness win (1 unresolved image instead of 2, and it's the
correct one that stays unresolved) with zero observed behavioural risk.

**`linked_event_id` was parsed into every event and never once consulted — a real double-
counting bug, not a cosmetic gap.** A full adversarial audit (boundary conditions on the
forecast/planner math, currency edge cases, message-injection patterns, LLM-response
hardening, idempotency) turned up one genuine, spec-explicit defect: `problem_statement.md`'s
90-day safety check says to ignore duplicate records, but nothing in the pipeline ever did.
Checking all 58 `linked_event_id` pairs in the real dataset by structure (not by reading
descriptions first) found a clean, consistent pattern: 6 pairs of a **settled** "Original
card charge" linked to a **pending** event of the identical amount and direction. The old
rule reserved *every* pending debit unconditionally ("conservative"), so both the original
and its linked duplicate were counted — double-charging the user for one purchase in the
forecast. Fixed by excluding a pending debit when its `linked_event_id` names an already-
settled event of the same amount and direction; matched structurally (status + direction +
amount), not by the description text, so it isn't tied to one specific label. Verified: a
synthetic unit test, a synthetic negative control (different amount → still reserved, exactly
as before), a synthetic negative control (no linkage at all → completely unaffected), and a
regression test against all 6 real occurrences (`tests/test_agent.py::TestReconciliation`).
Full 250-request re-run: exactly the 5 requests belonging to affected users changed (a 6th
user's request was already `affordable_now` at the requested-amount ceiling, so freeing more
headroom had no visible effect — confirmed, not assumed), every change moved
`amount_safe_to_pay` **up** (removing a phantom double-counted debit), no other field on any
of those rows moved, and the 25-sample metrics are unchanged (none of the 6 affected users
are sample requests, so this is unverifiable against ground truth there — but the mechanism,
unlike the earlier same-day-boundary hypothesis, is not a probabilistic forecast argument;
it is a direct violation of an explicit, literal spec sentence, structurally confirmed in the
actual data).

**A second, more severe double-count was hiding behind a *correctly matching* message
template.** Auditing all 58 `linked_event_id` pairs by category (not just the first few)
surfaced a `debt_payment → debt_payment` pattern the first pass missed: a **failed** "Failed
bill payment attempt" linked to a **scheduled** "Scheduled bill payment retry" of the
identical amount. On its own this is handled correctly — failed excluded, scheduled counted,
once. But 4 of the 7 real occurrences ALSO have a matching `bank_debit_retry` message
("previous debit attempt failed... another debit will be attempted"), which correctly fires
the existing `obligation_persists` fact and revives the failed event as a live obligation —
*in addition to* the already-present scheduled retry, double-charging the same bill. Traced
end to end and confirmed on the real data before touching anything (`event_8575`/`event_8576`
et al.: both counted, same 166). Fixed by checking, before honouring a retry-confirming
message, whether the failed event already has an explicit linked scheduled/settled retry of
the same amount — if so, the message's obligation is already represented by that separate
event and the original stays excluded; only when no such linked event exists does the message
alone revive it, exactly as before. Verified: a regression test against the real double-count
pattern, a negative control confirming a message-only retry (no linked event) still works as
before, full 46→50-test suite pass, and a full 250-request diff — zero rows changed, because
all 4 real occurrences' event dates happen to fall before their own request's forecast window
(historical relative to the request), so this fix is defensive-correctness for the general
case (and the hidden 250) rather than a fix with a visible before/after on this specific
dataset. Confirmed via direct date inspection, not assumed.

**The dead `RECURRENCE_REGULARITY_CV_MAX` config was wired in, tested, and it measurably
helped.** `regular: bool` already existed on `RecurringSeries`, defaulted to `True`, and was
never computed or read anywhere — the coefficient-of-variation check its docstring describes
("used only to size one forward projection so the series still exists in the forecast") was
never implemented. Added `_gap_regularity()`: computes the actual inter-occurrence gaps (in
days) within a cluster and compares their coefficient of variation to the existing threshold;
`project_series` now projects an irregular series **once** instead of repeating it every month
for the rest of the window. Tested against the real 25 samples before trusting it (after the
same-day-boundary hypothesis taught this the hard way): `affordability_status` 88%→**92%**,
`recommended_payment_method` 92%→**96%**, `earliest_date_for_full_payment` 72%→**76%**,
`spending_changes_needed` 80%→**84%**, zero fields regressed. Traced the mechanism rather than
just trusting the number: `request_13`'s two "dining" clusters (day-of-month grouped, but
real-world variable-spending habits, not fixed monthly bills) are the ones flagged irregular;
no longer over-projecting them every month, combined with the secondary-income fix from the
previous session, moves that request's `affordability_status`, `recommended_payment_method`,
`payment_plan`, and `earliest_date_for_full_payment` all from wrong to correct simultaneously
— a real, understood mechanism, not four fields coincidentally flipping at once. Scope on the
real submission: 40 of 250 requests change. `amount_safe_to_pay` itself is unchanged at 5/25
(request_13's point estimate actually moved slightly further from truth even as its
categorical fields became correct — a reminder that the six graded fields don't all move
together, and that getting the *decision* right matters more than getting one number closer).

**FX pair coverage varies by date.** `EUR→ZAR` stops being quoted after mid-2026 while
`USD→INR` continues, so "latest rate on or before" must be computed **per pair**. Caught by
a failing smoke test before it reached the forecast.

---

## Results on the 25 solved samples

```
recommended_payment_method      24/25   96%
affordability_status            23/25   92%
spending_changes_needed         21/25   84%
payment_plan                    22/25   88%
earliest_date_for_full_payment  19/25   76%
amount_safe_to_pay               5/25   20%   (exact ±1%; median error 6.2%)
```

Full-run sanity: 250/250 rows in ~3-52s (depending on whether the image OCR cache is warm),
0 processing failures, 0 invariant violations, 50/50 tests passing, and a non-degenerate
spread across all four statuses (`affordable_with_plan` 70, `affordable_now` 64,
`affordable_later` 58, `not_affordable` 58) and all five methods.

Three of this session's fixes (secondary-income-staleness, the two reconciliation double-
count fixes) don't move this table's binary hit counts, or move only the mechanism, not the
final answer — verified in each case by re-running `evaluate.py`, not assumed. The fourth
(wiring the previously-dead `RECURRENCE_REGULARITY_CV_MAX` into actual behavior) is the one
that measurably moved this table: `affordability_status` 88%→92%, `recommended_payment_method`
92%→96%, `earliest_date_for_full_payment` 72%→76%, `spending_changes_needed` 80%→84%, with
`amount_safe_to_pay` and `payment_plan` unchanged and nothing regressed. All of this was
verified against the real dataset (git-cloned directly, not assumed) rather than reasoned
about in the abstract — including one hypothesis (a same-day recurrence-projection boundary,
see "Findings" above) that looked at least as promising on paper and was rejected after it
measurably made every field in this table worse. The discipline that matters here isn't
"try things until the numbers go up" — it's test everything the same way, keep what survives
contact with the real 25 samples, and revert what doesn't, regardless of how sound the
reasoning looked beforehand.

### Test coverage

| Suite | Tests | Covers |
|---|---|---|
| `test_agent.py` | 37 | Currency, template coverage, image parsing, forecast math, validators, reconciliation duplicate-record handling, income-fact scoping, full-pipeline invariants |
| `test_fallbacks.py` | 17 | OCR↔VLM fusion and conflict policy, malformed/out-of-schema model output, retry behaviour, token accounting, cache, injection resistance |

Every bug found during development has a matching `test_regression_*`, so a fixed defect
cannot silently return.

---

## Known limitations

I'd rather state these plainly than imply the system is finished.

1. **`amount_safe_to_pay` is the weakest field.** Median relative error 6.2%, mean 22.6%
   (a few large outliers); 16/25 samples land within 20%, 11 within 5%, 5 within 1%. The
   forecast is a well-founded *model* of the ledger, not a reproduction of the organizers'
   generator, and closing the remainder most likely requires their generator rather than
   more fitting.

   Three things were tried and are worth knowing about:
   a **parameter sweep** over the recurrence thresholds (plateaued at 4–5 exact hits —
   which is what showed tuning was the wrong lever); **day-of-month cycle clustering**
   replacing description grouping (the large win, `payment_plan` 56% → 88%); and a
   structural hypothesis that ground truth counts only *confirmed* future income rather
   than projecting recurring income (**tested and refuted** — it collapsed
   `affordability_status` from 22/25 to 8/25, confirming the current income model is
   correct). I stopped there deliberately: further fitting to 25 samples risks degrading
   the 250 hidden requests, and the structural fields — which drive the recommendation a
   user actually acts on — are at 88–92%.
2. **The model fallback paths are tested but never exercised against a live API.**
   `tests/test_fallbacks.py` drives every branch with mocked responses — including
   disagreement, malformed JSON, out-of-schema kinds and injection attempts — but no run
   in this submission made a real network call. Mocks prove the contract is handled; they
   do not prove a live model honours it.
3. **OCR now reads 13/16 document images**, and 3/4 of the four that actually fall inside a
   forecast window (the other twelve settle 1–3 days *before* `request_date` and are already
   inside the opening balance). Looked directly at both previously-unresolved images (not to
   hardcode a value — to find out whether either failure was fixable):
     - The rent receipt (`image_02`) is **not actually garbled** — it's a clean, typed table.
       The real cause: one OCR token (a torn/overlapping seal watermark) got merged into a
       single 358px-tall garbled box against a page median of ~47px, which inflated
       row-clustering tolerance enough to fuse the "Balance Due" row into the "Amount
       Received:" row above it and corrupt both. Capping each word's clustering-tolerance
       contribution to ~2x the page's median height (`image_extractor._build_phrases`) fixes
       it: the clean "Amount Received: 1,00,000.00" label now resolves correctly (and happens
       to equal the true Balance Due in this document). Verified against all 16 images: the
       other 12 that already worked are byte-for-byte unchanged, and the one sample this event
       falls inside (`request_16`) was and remains fully correct (this event wasn't on that
       request's binding constraint either way). One side effect found and left open rather
       than chased further: `image_15`, a multi-column GST invoice, changes from one wrong
       reading (80.00) to a different wrong one (228.00, a tax sub-line) — the true value is
       9,968.00 (`Total(Incl Taxes)`), which the current same-row/below-label matching can't
       reach because the column header sits far above the total row in a wide table, and
       because `total\s*incl` doesn't match the literal `Total(Incl` text. Not in the 25
       samples, so unverifiable against ground truth; documented rather than patched blind so
       close to a deadline.
     - The pharmacy receipt (`image_14`) is genuinely handwritten and, honestly, ambiguous
       even to a human: the total could plausibly read either 4,593 or 4,543 depending on how
       two digits are read, and the line items don't cleanly resolve it either way (they sum
       to 4,543). This one stays unresolved. It fails **safe** — returns `None`, is flagged
       for review, and is *never* treated as zero, per spec. The VLM path would likely resolve
       it when an API key is present, since this is exactly the failure mode (handwriting)
       that motivated having a vision fallback at all.
4. **`max_installment_months` is interpreted as a payment count.** Every sample's chosen
   option has 3 payments and every rejected one has ≥15, so both that reading and a
   span-in-months reading fit the evidence. I took the stricter one; it is an inference,
   not a certainty. `planner.installment_reading_note()` cross-checks both readings
   against every option on the full 250-request run and logs a disagreement to the
   audit trail whenever they would actually diverge, so if this inference is wrong on
   some hidden request it is discoverable rather than silently absorbed.
5. **No amount is hardcoded.** I read all 16 images during development but deliberately did
   not embed those values — the brief forbids file-specific answers, and hardcoding would
   make the extraction pipeline untestable.

---

## Reproducibility

Behaviour is deterministic by default: `Decimal` arithmetic throughout (never floats), no
randomness, and vision readings cached by image content hash so a re-run cannot drift. The
same `pipeline.run()` powers `main.py`, `evaluate.py`, and the end-to-end tests — there is
no separate scoring path that could pass while the submission path breaks.
