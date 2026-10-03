"""
Resolves the amount for a blank-amount financial event from its linked image.

FUSION POLICY
-------------
Two independent readers are used and their agreement determines confidence:

  * OCR (tesseract + geometric label/value pairing) -- deterministic, free,
    offline, but brittle on handwriting and on glyphs like the rupee sign
    that bleed into the leading digit.
  * VLM (Claude vision) -- far more robust on real-world layouts and
    handwriting, but costs tokens and is non-deterministic.

Decision table:
  both agree (within tolerance)  -> accept, confidence "high"
  only one produced a value       -> accept it, confidence from that reader
  both produced, they DISAGREE     -> prefer the VLM (empirically stronger on
                                       these documents) but mark "conflict" so
                                       the audit trail records the discrepancy
  neither produced a value          -> return None; caller must NOT treat the
                                       amount as zero (spec is explicit), and
                                       instead flags the event for review and
                                       handles it conservatively

Results are cached to disk keyed by image content hash, so re-running the
full dataset does not re-bill vision calls, and so a graded run is
reproducible.
"""

from __future__ import annotations
from decimal import Decimal
from typing import Optional, Dict, Any, Tuple
import hashlib
import json
import os

from . import image_extractor
from . import llm_client

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".image_cache.json")
# Relative tolerance for declaring two independent readings "in agreement".
AGREEMENT_TOLERANCE = Decimal("0.005")


def _load_cache() -> Dict[str, Any]:
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(cache: Dict[str, Any]):
    try:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, sort_keys=True)
    except Exception:
        pass


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _valid_reading(amount) -> bool:
    """
    Defence in depth: a document total is never <= 0, so such a value is a
    parse failure rather than data. llm_client already screens this, but the
    resolver re-checks because a trust boundary enforced in only ONE place is
    one refactor away from being bypassed. (A mocked-fallback test caught the
    resolver accepting a zero amount that the producer would have rejected.)
    """
    if amount is None:
        return False
    try:
        return Decimal(str(amount)) > 0
    except Exception:
        return False


def _agree(a: Decimal, b: Decimal) -> bool:
    if a == b:
        return True
    bigger = max(abs(a), abs(b))
    if bigger == 0:
        return True
    return abs(a - b) / bigger <= AGREEMENT_TOLERANCE


def resolve_image_amount(image_path: str, hint: str = "",
                          use_cache: bool = True) -> Optional[Dict[str, Any]]:
    """
    Returns dict:
      {amount: Decimal, confidence: str, method: str, rationale: str}
    or None when no reading could be obtained.
    """
    if not os.path.exists(image_path):
        return None

    cache = _load_cache() if use_cache else {}
    key = _file_hash(image_path)
    if use_cache and key in cache:
        c = cache[key]
        if c.get("amount") is not None:
            return {"amount": Decimal(str(c["amount"])), "confidence": c.get("confidence", "medium"),
                    "method": c.get("method", "cache"), "rationale": c.get("rationale", "cached")}

    ocr_amt, ocr_conf, ocr_why = image_extractor.extract_amount(image_path)
    if not _valid_reading(ocr_amt):
        ocr_amt = None

    vlm = llm_client.read_amount_from_image(image_path, hint=hint) if llm_client.is_enabled() else None
    vlm_amt = vlm["amount"] if vlm else None
    if not _valid_reading(vlm_amt):
        vlm_amt = None
    vlm_conf = vlm.get("confidence", "low") if vlm else "none"
    vlm_label = vlm.get("label", "") if vlm else ""

    result: Optional[Dict[str, Any]] = None
    if ocr_amt is not None and vlm_amt is not None:
        if _agree(ocr_amt, vlm_amt):
            result = {"amount": vlm_amt, "confidence": "high", "method": "ocr+vlm-agree",
                      "rationale": f"OCR {ocr_amt} == VLM {vlm_amt} (label {vlm_label!r})"}
        else:
            result = {"amount": vlm_amt, "confidence": "conflict", "method": "vlm-over-ocr",
                      "rationale": f"DISAGREEMENT OCR={ocr_amt} ({ocr_why}) vs VLM={vlm_amt} "
                                     f"(label {vlm_label!r}); took VLM"}
    elif vlm_amt is not None:
        result = {"amount": vlm_amt, "confidence": vlm_conf, "method": "vlm-only",
                  "rationale": f"VLM read {vlm_amt} from label {vlm_label!r}"}
    elif ocr_amt is not None:
        result = {"amount": ocr_amt, "confidence": ocr_conf, "method": "ocr-only",
                  "rationale": ocr_why}

    if use_cache and result is not None:
        cache[key] = {"amount": str(result["amount"]), "confidence": result["confidence"],
                      "method": result["method"], "rationale": result["rationale"],
                      "image": os.path.basename(image_path)}
        _save_cache(cache)

    return result
