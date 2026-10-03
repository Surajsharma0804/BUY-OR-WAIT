# Token Usage and Cost Analysis

Run covering **250 requests** from `dataset/requests.csv` — the run that produced `output.csv`.

- Wall-clock runtime: **3.5s**
- LLM/VLM extraction: **disabled**

## Where models are (and are not) used

The decision engine is **fully deterministic**. Affordability, the 90-day safety check, plan ranking, and every number written to `output.csv` are computed in code — never by a model. This is a deliberate architecture choice: a hallucinated figure inside a financial recommendation is far more damaging than a misread receipt, and the latter is independently cross-checked.

Model calls are reserved for two *extraction* fallbacks:

1. **Vision (VLM)** — read the authoritative amount off a linked document image when the deterministic OCR path cannot resolve it confidently. The reading is cross-validated against OCR; disagreements are recorded.
2. **Text** — interpret a message matching none of the deterministic templates. All 215 messages in this dataset are covered by the template catalogue, so this path is dormant here.

## Model usage totals

**No model calls were made in this run.**

Either no `ANTHROPIC_API_KEY` was set, or neither fallback was needed. All 215 messages resolved via the deterministic template catalogue and image amounts via the OCR pipeline, so total cost was **$0.00**.

| Metric | Value |
|---|---|
| Model provider | Anthropic (configured, unused this run) |
| Model calls | 0 |
| Input tokens | 0 |
| Output tokens | 0 |
| Total tokens | 0 |
| Avg tokens / request | 0 |
| Estimated total cost | $0.00 |
| Estimated cost / request | $0.00 |

## Cost controls

- **Disk-cached image reads.** Vision results are cached by image content hash, so re-running the full dataset never re-bills a document already read.
- **Deterministic-first routing.** A model is consulted only after the rule-based path declines, which keeps the common case free.
- **Right-sized model.** Extraction uses a small, fast model; no frontier model is needed to read a total off a receipt.
- **Bounded retries.** Malformed JSON is retried a limited number of times, then the system falls back rather than looping.

## Pricing basis

Published Anthropic list pricing per million tokens at the time of the run:

| Model | Input / MTok | Output / MTok |
|---|---|---|
| `claude-haiku-4-5-20251001` | $1.00 | $5.00 |
| `claude-opus-5` | $5.00 | $25.00 |
| `claude-sonnet-5` | $2.00 | $10.00 |

No API keys or credentials appear in this report or anywhere in the submission; the key is read from the `ANTHROPIC_API_KEY` environment variable only.
