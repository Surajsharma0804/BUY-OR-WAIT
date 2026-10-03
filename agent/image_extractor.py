"""
Image evidence extraction.

problem_statement.md: "When a financial event has a blank `amount`, use its
`event_id` to find the matching `related_event_id` in `images.csv`, then
extract the amount from that image. Do not treat a blank amount as zero."

There are exactly 16 blank-amount events and exactly 16 images, in a clean
1:1 mapping via `related_event_id`.

WHY THIS IS GEOMETRY-BASED, NOT LINE-BASED
------------------------------------------
The first implementation matched labels and numbers on the same OCR *text
line* and scored 6/16. Dumping the raw OCR revealed the failure was not OCR
quality -- the right number was almost always present -- but document
LAYOUT. These are real multi-column documents, and line-based OCR fuses
unrelated columns together:

    "Amount due till   Taxes   107.40"   <- label from col 1, value from col 2
    "Item Total (2 items)"                <- label alone; its value is 2 lines below
    "Total (2)  704.05"                   <- correct pair, but competing with the above

So extraction works on tesseract's WORD-LEVEL BOUNDING BOXES and pairs a
label to the value that is geometrically nearest on the same visual row
(or, failing that, directly beneath it). That handles multi-column receipts,
right-aligned amount columns, and label-above-value layouts uniformly.

SELECTION IS PRIORITY-BASED, NOT max()
--------------------------------------
On real documents the largest number on the page is frequently the WRONG
answer. Real examples from this dataset:
  * payslip      -> Net Pay 4,365,000, NOT Total Earnings 4,780,800
  * rent receipt -> Balance Due 100,000, NOT "Total to be Received" 200,000
  * taxi receipt -> Total 33.50, NOT Cash Paid 40.00 (customer got change)
  * hospital bill-> Balance 3,650.00, while "Amount Paid" is 0.00
So labels are scored by financial authority and the best-scoring label wins.

TRUST BOUNDARY
--------------
Everything read here is DATA, never instruction. We only ever read numbers
adjacent to known financial labels; imperative text inside an image
("pay now", "call this number", "click here") can never influence control
flow, by construction.
"""

from __future__ import annotations
from decimal import Decimal, InvalidOperation
from typing import Optional, List, Tuple, Dict
import os
import re
import statistics

try:
    from PIL import Image
    import pytesseract
    _OCR_AVAILABLE = True
except Exception:  # pragma: no cover
    _OCR_AVAILABLE = False


# Label keywords ranked by financial authority. Higher wins.
LABEL_PRIORITY: List[Tuple[str, int]] = [
    (r"net\s*pay", 100),
    (r"balance\s*due", 98),
    (r"grand\s*total", 96),
    (r"total\s*amount\s*received", 94),
    (r"total\s*paid", 93),
    (r"amount\s*due\s*till", 92),
    (r"net\s*amount", 90),
    (r"total\s*incl", 88),
    (r"^\s*balance\s*:?\s*$", 86),
    (r"\bbalance\b", 84),
    (r"amount\s*payable", 82),
    (r"total\s*bill\s*amount", 80),
    (r"^\s*total\b", 78),
    (r"\btotal\b", 72),
    (r"item\s*total", 66),
    (r"item\s*bill", 64),
    (r"net\s*total", 62),
    # On a receipt, bare "Amount Received" is the cash actually collected --
    # distinct from the gross "Total Amount to be Received" (a negative label).
    (r"amount\s*received", 70),
    (r"sub\s*total", 45),
    (r"total\s*earnings", 40),   # deliberately below net_pay
    (r"cash\s*paid", 28),         # can exceed the bill when change is returned
    (r"amount\s*paid", 22),
]

# Labels whose adjacent value must never be selected.
NEGATIVE_LABELS = [
    r"\bchange\b",
    r"previous\s*balance",
    r"convenience\s*fee",
    r"tax\s*ref",
    r"\bdiscount\b",
    r"amount\s*due\s*after",
    r"total\s*deductions",
    r"subtotal\s*deductions",
    r"\bdeduction\b",
    r"\ballowance\b",
    r"usage\s*charges",
    r"\brentals\b",
    r"\btaxes?\b",
    r"\bcgst\b", r"\bsgst\b", r"\bigst\b",
    r"paid\s*amount\s*in\s*words",
    r"amount\s*in\s*words",
    # Gross "to be received" figures are the invoice face value, not the cash
    # obligation; a Balance Due / Amount Received line is the real number.
    r"total\s*amount\s*to\s*be\s*receiv",
    r"amount\s*to\s*be\s*receiv",
    r"previous\s*bal",
    r"sub\s*total",
]

# Currency/grouping-tolerant number token.
NUM_TOKEN = re.compile(r"^[^\d]{0,3}(\d{1,3}(?:[,\s]\d{2,3})*(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?)[^\d]{0,3}$")


def _parse_amount(tok: str) -> Optional[Decimal]:
    """
    Parse a numeric token. Handles:
      * western grouping   1,234.56
      * Indian grouping    1,00,000.00
      * decimal-comma OCR  $33,50 -> 33.50 and 41272,00 -> 41272.00

    Key generic rule: a trailing comma-group of EXACTLY 2 digits is never
    valid thousands grouping in any convention (western groups are 3, Indian
    groups are 2 only in the *middle*, never last). So a 2-digit tail after a
    comma is always a decimal separator. This single rule fixed several
    real misreads (41272,00 was being read as 4,127,200).
    """
    t = tok.strip()
    # A token that begins with a separator is an OCR fragment of a larger
    # number (e.g. ",50" split off from "$33 ,50"), never a value in itself.
    if re.match(r"^[^\d]*[.,]", t) and not re.match(r"^[^\d.,]*[.,]?\d", t):
        return None
    if re.match(r"^\s*[.,]", t):
        return None
    t = re.sub(r"^[^\d]+", "", t)          # strip leading currency glyphs
    t = re.sub(r"[^\d.,]+$", "", t)        # strip trailing junk
    if not t or not t[0].isdigit():
        return None
    t = t.replace(" ", "")

    if "." in t and "," in t:
        if t.rindex(".") > t.rindex(","):
            t = t.replace(",", "")           # 1,234.56 / 1,00,000.00
        else:
            t = t.replace(".", "").replace(",", ".")   # 1.234,56 (euro style)
    elif "," in t:
        tail = t.split(",")[-1]
        if len(tail) == 2:
            t = t[:t.rindex(",")].replace(",", "") + "." + tail
        else:
            t = t.replace(",", "")
    if t.count(".") > 1:
        return None
    try:
        v = Decimal(t)
    except InvalidOperation:
        return None
    if v <= 0:
        return None
    return v


# --- amount-in-words fallback (many Indian invoices print the authoritative
# --- total in words, which survives OCR better than a noisy numeric cell) ---
_WORD_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}
_WORD_SCALE = {"hundred": 100, "thousand": 1000, "lakh": 100000, "lakhs": 100000,
                "million": 1000000, "crore": 10000000}


def parse_amount_in_words(text: str) -> Optional[Decimal]:
    """Parse e.g. 'One Thousand And Nine Hundred And Ninety-Five Rupees'."""
    m = re.search(r"(?:amount\s*in\s*words|in\s*words|total\s*in\s*words)\s*:?\s*(.{0,160})",
                  text, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    phrase = m.group(1).lower()
    # The currency noun can appear BEFORE the number ("Rupees Fifteen Thousand")
    # or AFTER it ("Fifteen Thousand Rupees And Twenty Paise"). Strip any
    # leading currency words first, then split on a trailing occurrence so the
    # sub-unit tail (paise) is never summed into the main value.
    phrase = re.sub(r"^\s*(?:indian\s+)?(?:rupees?|rupiah|dollars?|euros?)\s+", " ", phrase)
    phrase = re.split(r"\b(?:only|paise|paisa|rupees|rupee|rupiah|dollars?|euros?)\b", phrase)[0]
    phrase = phrase.replace("-", " ").replace("&", " and ")
    tokens = [t for t in re.split(r"[^a-z]+", phrase) if t]
    if not tokens:
        return None
    total, current, seen = 0, 0, False
    for tok in tokens:
        if tok in _WORD_NUM:
            current += _WORD_NUM[tok]
            seen = True
        elif tok in _WORD_SCALE:
            scale = _WORD_SCALE[tok]
            if scale == 100:
                current = (current or 1) * 100
            else:
                total += (current or 1) * scale
                current = 0
            seen = True
        elif tok in ("and", "indian", "only"):
            continue
        else:
            if seen:
                break
    value = total + current
    return Decimal(value) if value > 0 else None


def _prep(image_path: str):
    img = Image.open(image_path)
    w, h = img.size
    # 3000px on the long edge with LANCZOS measurably improved digit accuracy
    # on thermal/scanned receipts (e.g. a hospital total misread as 3450 became
    # a correct 3650). Verified empirically, not assumed.
    if max(w, h) < 3000:
        scale = 3000 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img.convert("L")


def _ocr_words(image_path: str, psm: int = 6) -> List[Dict]:
    """Return word boxes: [{text, left, top, width, height, cx, cy, conf}]."""
    if not _OCR_AVAILABLE or not os.path.exists(image_path):
        return []
    try:
        img = _prep(image_path)
        data = pytesseract.image_to_data(img, config=f"--psm {psm}",
                                          output_type=pytesseract.Output.DICT)
    except Exception:
        return []

    words = []
    for i, txt in enumerate(data.get("text", [])):
        if not txt or not txt.strip():
            continue
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1.0
        left, top = data["left"][i], data["top"][i]
        width, height = data["width"][i], data["height"][i]
        words.append({
            "text": txt.strip(), "left": left, "top": top,
            "width": width, "height": height,
            "cx": left + width / 2.0, "cy": top + height / 2.0,
            "right": left + width,
            "conf": conf,
            "block": data.get("block_num", [0] * len(data["text"]))[i],
            "line": data.get("line_num", [0] * len(data["text"]))[i],
            "par": data.get("par_num", [0] * len(data["text"]))[i],
        })
    return words


def words_to_text(words: List[Dict]) -> str:
    """
    Rebuild the page text from word boxes produced by the SAME OCR pass.

    This exists purely for speed: the amount-in-words fallback needs plain
    text, and calling tesseract a second time to get it doubled the cost per
    image for no extra information. Reconstructing from the existing word
    boxes removed two full OCR passes per document.
    """
    lines: Dict[tuple, List[Dict]] = {}
    for w in words:
        lines.setdefault((w.get("block", 0), w.get("par", 0), w.get("line", 0)), []).append(w)
    out = []
    for key in sorted(lines.keys()):
        row = sorted(lines[key], key=lambda x: x["left"])
        out.append(" ".join(x["text"] for x in row))
    return "\n".join(out)


def ocr_text(image_path: str, psm: int = 6) -> str:
    """Plain text OCR, used for the amount-in-words fallback and diagnostics."""
    if not _OCR_AVAILABLE or not os.path.exists(image_path):
        return ""
    try:
        return pytesseract.image_to_string(_prep(image_path), config=f"--psm {psm}")
    except Exception:
        return ""


def _label_score(text: str) -> int:
    low = text.lower()
    if any(re.search(neg, low) for neg in NEGATIVE_LABELS):
        return -1
    score = 0
    for pat, pts in LABEL_PRIORITY:
        if re.search(pat, low):
            score = max(score, pts)
    return score


def _build_phrases(words: List[Dict]) -> List[Dict]:
    """
    Group words into short left-to-right phrases on the same visual row so
    multi-word labels ("Total Amount Received", "Net Pay", "Balance Due")
    can be scored as a unit. Rows are clustered by vertical overlap.

    Clustering tolerance is capped per word (see height_cap below): a single
    OCR "word" with an abnormally large bounding box -- tesseract
    occasionally fuses a watermark, seal graphic, or several visual lines
    into one spurious low-confidence token -- must not get to inflate the
    tolerance used to place ITS OWN row. Uncapped, that one garbled token
    can swallow neighbouring rows into itself and corrupt an otherwise
    clean, high-confidence label two rows away. Found on a real document in
    this dataset: a torn/overlapping seal watermark merged into one 358px-
    tall garbled token (page median word height ~47px) sitting on the
    "Balance Due" row, which pulled the clean "Amount Received:" row into
    the same cluster and broke that label's phrase apart mid-word.
    """
    if not words:
        return []
    heights = sorted(w["height"] for w in words if w["height"] > 0)
    median_h = statistics.median(heights) if heights else 1
    height_cap = max(median_h * 2, 1)

    def _tol_height(w):
        return min(w["height"], height_cap)

    rows: List[List[Dict]] = []
    for w in sorted(words, key=lambda x: (x["cy"], x["left"])):
        placed = False
        for row in rows:
            ref = row[0]
            tol = max(_tol_height(ref), _tol_height(w)) * 0.6
            if abs(w["cy"] - ref["cy"]) <= tol:
                row.append(w)
                placed = True
                break
        if not placed:
            rows.append([w])

    phrases = []
    for row in rows:
        row.sort(key=lambda x: x["left"])
        for i in range(len(row)):
            for span in (1, 2, 3, 4):
                if i + span > len(row):
                    continue
                chunk = row[i:i + span]
                # only join words that are close together horizontally
                ok = True
                for a, b in zip(chunk, chunk[1:]):
                    gap = b["left"] - (a["left"] + a["width"])
                    if gap > a["height"] * 2.2:
                        ok = False
                        break
                if not ok:
                    continue
                text = " ".join(c["text"] for c in chunk)
                phrases.append({
                    "text": text,
                    "left": chunk[0]["left"],
                    "right": chunk[-1]["left"] + chunk[-1]["width"],
                    "cy": sum(c["cy"] for c in chunk) / len(chunk),
                    "height": max(c["height"] for c in chunk),
                    "row": row,
                })
    return phrases


def _merge_numeric_fragments(words: List[Dict]) -> List[Dict]:
    """
    Re-join numbers that OCR split across word boxes, e.g. "$33" + ",50".
    Thermal receipts frequently insert a space before the decimal separator,
    which otherwise yields a bogus integer (33) and an orphan fragment (,50).
    """
    if not words:
        return words
    out, i = [], 0
    ordered = sorted(words, key=lambda w: (round(w["cy"] / 8), w["left"]))
    while i < len(ordered):
        cur = dict(ordered[i])
        if i + 1 < len(ordered):
            nxt = ordered[i + 1]
            same_row = abs(nxt["cy"] - cur["cy"]) <= max(cur["height"], nxt["height"]) * 0.6
            gap = nxt["left"] - cur.get("right", cur["left"] + cur["width"])
            if (same_row and gap <= cur["height"] * 0.9
                    and re.match(r"^[^\d]*\d", cur["text"])
                    and re.match(r"^[.,]\d{1,2}$", nxt["text"])):
                cur["text"] = cur["text"] + nxt["text"]
                cur["width"] = (nxt["left"] + nxt["width"]) - cur["left"]
                cur["right"] = nxt["left"] + nxt["width"]
                out.append(cur)
                i += 2
                continue
        out.append(cur)
        i += 1
    return out


def extract_amount_from_words(words: List[Dict]) -> Tuple[Optional[Decimal], str, str]:
    """
    Geometry-aware label->value pairing with NEAREST-LABEL attribution.

    The critical rule: a numeric cell belongs to the label CLOSEST to its
    left on the same visual row, not to any label anywhere on that row.
    Without this, a multi-column bill like

        Amount due till   |   Taxes   107.40
        06-Feb-2026  =  704.05

    lets the far-left "Amount due till" steal the Taxes column's value.
    Nearest-label attribution makes "Taxes" (a negative label) own 107.40,
    so it is correctly suppressed and the real total wins.
    """
    if not words:
        return None, "none", "no OCR words"

    words = _merge_numeric_fragments(words)
    numerics = []
    for w in words:
        if not NUM_TOKEN.match(w["text"]):
            continue
        val = _parse_amount(w["text"])
        if val is None:
            continue
        numerics.append({**w, "value": val})
    if not numerics:
        return None, "none", "no numeric tokens found"

    phrases = _build_phrases(words)
    candidates = []  # (score, value, rationale, cy)

    for ph in phrases:
        score = _label_score(ph["text"])
        if score <= 0:
            continue
        # CONTEXT CHECK: a bare "Balance" is authoritative, but "Previous
        # balance" is not. Sub-phrases are generated per word-window, so the
        # qualifier can sit just outside the matched phrase. Re-score the
        # phrase together with the word immediately to its left and suppress
        # it if that fuller reading is a negative label.
        left_neighbours = [w for w in ph["row"] if w["left"] < ph["left"]]
        if left_neighbours:
            prev = max(left_neighbours, key=lambda w: w["left"])
            if _label_score(f"{prev['text']} {ph['text']}") < 0:
                continue
        right_neighbours = [w for w in ph["row"] if w["left"] >= ph["right"]]
        if right_neighbours:
            nxt = sorted(right_neighbours, key=lambda w: w["left"])[:4]
            extended = ph["text"] + " " + " ".join(w["text"] for w in nxt)
            if _label_score(extended) < 0:
                continue
        tol = ph["height"] * 0.75

        # The label phrase may itself contain the value when OCR merged the
        # label and its number into one box ("Amount Payable: 3650.00").
        # That in-phrase number is the most reliable pairing available.
        inline = re.search(r"([\d][\d,\s]*(?:\.\d{1,2})?)\s*$", ph["text"])
        if inline:
            inline_val = _parse_amount(inline.group(1))
            if inline_val is not None:
                candidates.append((score, inline_val,
                                   f"label={ph['text']!r} value inline", ph["cy"]))
                continue

        same_row = [n for n in numerics
                    if abs(n["cy"] - ph["cy"]) <= tol and n["left"] >= ph["right"] - ph["height"] * 0.5]
        chosen, how = None, ""
        if same_row:
            chosen = min(same_row, key=lambda n: n["left"])   # first value right of the label
            # NEAREST-LABEL CHECK: is there a different, competing label sitting
            # between this label and the value? If so, the value is not ours.
            blockers = [p for p in phrases
                        if abs(p["cy"] - ph["cy"]) <= tol
                        and p["left"] > ph["right"]
                        and p["right"] <= chosen["left"] + 1
                        and _label_score(p["text"]) != 0]
            if blockers:
                continue
            # prefer the rightmost value when the row's amount column is far right
            far_right = [n for n in same_row if n["left"] > chosen["left"]]
            if far_right and not any(
                p for p in phrases
                if abs(p["cy"] - ph["cy"]) <= tol and p["left"] > chosen["right"]
                and _label_score(p["text"]) != 0
            ):
                chosen = max(same_row, key=lambda n: n["left"])
            how = "same-row"
        else:
            below = [n for n in numerics
                     if n["cy"] > ph["cy"] and n["cy"] - ph["cy"] <= ph["height"] * 4.0]
            if below:
                chosen = min(below, key=lambda n: (n["cy"] - ph["cy"], abs(n["cx"] - ph["left"])))
                how = "below-label"
        if chosen is None:
            continue
        candidates.append((score, chosen["value"],
                           f"label={ph['text']!r} value={chosen['text']!r} via {how}",
                           chosen["cy"]))

    if not candidates:
        return None, "none", "no label/value pair resolved"

    candidates.sort(key=lambda c: (c[0], c[3]), reverse=True)
    score, value, rationale, _cy = candidates[0]

    # ARITHMETIC SANITY: an invoice total can never plausibly exceed ~2x its
    # own sub-total (that would imply >100% tax). When it does, and dropping
    # the leading digit lands the value back in a plausible band above the
    # sub-total, the leading digit was a currency glyph misread (the Rs sign
    # is routinely OCR'd as a "2"). Verified on a real invoice in this
    # dataset where Rs79,679.26 was read as 279,679.26 against a 72,045.00
    # sub-total.
    subtotals = []
    for ph in phrases:
        if re.search(r"sub\s*total", ph["text"], re.IGNORECASE):
            tol_s = ph["height"] * 0.75
            row_nums = [n for n in numerics
                        if abs(n["cy"] - ph["cy"]) <= tol_s and n["left"] >= ph["right"] - ph["height"]]
            if row_nums:
                subtotals.append(max(row_nums, key=lambda n: n["left"])["value"])
    if subtotals:
        sub = max(subtotals)
        s_int = str(int(value))
        if value > sub * 2 and len(s_int) > 1:
            stripped = _parse_amount(s_int[1:] + (format(value, "f").split(".")[1:] or [""])[0].join(["." , ""]) if False else s_int[1:])
            try:
                frac = value - Decimal(int(value))
                stripped_val = Decimal(s_int[1:]) + frac
            except Exception:
                stripped_val = None
            if stripped_val is not None and sub <= stripped_val <= sub * Decimal("1.6"):
                rationale = (f"{rationale}; value exceeded 2x sub-total {sub}, dropped leading "
                              f"currency-glyph digit -> {stripped_val}")
                value = stripped_val

    # GLYPH-CONTAMINATION CONSENSUS: a leading currency glyph (Rs/INR sign)
    # frequently OCRs as an extra leading digit, turning 79,679.26 into
    # 279,679.26. If another independent labelled candidate equals our value
    # minus exactly one leading digit, trust the corroborated shorter value.
    vs = format(value, "f")
    for c_score, c_val, c_rat, _ in candidates[1:]:
        cs = format(c_val, "f")
        if len(vs) - len(cs) == 1 and vs.endswith(cs):
            rationale = (f"{rationale}; corroborated shorter value {cs} from {c_rat} "
                          f"-> treated leading digit as currency-glyph OCR artifact")
            value = c_val
            break

    conf = "high" if score >= 78 else ("medium" if score >= 55 else "low")
    return value, conf, f"{rationale} (label score {score})"


def extract_amount(image_path: str) -> Tuple[Optional[Decimal], str, str]:
    """
    Read the authoritative amount off a financial document.

    Page-segmentation modes are tried in order of how often they win, and the
    sweep STOPS as soon as a high-confidence reading appears: different PSMs
    win on different layouts, but running all of them unconditionally cost
    ~12s per document for a result the first pass usually already had. The
    text needed by the amount-in-words fallback is reconstructed from the same
    word boxes rather than re-invoking OCR. Together these cut full-dataset
    image resolution by roughly 4x with no change in extracted values.
    """
    rank = {"high": 3, "medium": 2, "low": 1, "none": 0}
    best = (None, "none", "no reading")
    readings = []
    words_text = ""

    for psm in (6, 4, 11):
        words = _ocr_words(image_path, psm=psm)
        if not words:
            continue
        if not words_text:
            words_text = words_to_text(words)
        amt, conf, why = extract_amount_from_words(words)
        if amt is not None:
            readings.append((amt, conf, f"[psm{psm}] {why}"))
            if rank[conf] > rank[best[1]]:
                best = (amt, conf, f"[psm{psm}] {why}")
        # Early exit: a high-confidence label/value pairing is not improved on
        # by further segmentations.
        if best[1] == "high":
            break

    if best[0] is not None and best[1] != "high":
        agree = sum(1 for a, _, _ in readings if a == best[0])
        if agree >= 2:
            best = (best[0], "high", best[2] + f"; agreed across {agree} PSM passes")

    words_amt = parse_amount_in_words(words_text) if words_text else None

    if best[0] is None:
        if words_amt is not None:
            return words_amt, "high", "amount-in-words line (no numeric cell resolved)"
        return best
    if words_amt is None:
        return best

    num = best[0]
    int_part = int(num)
    # (a) integer parts agree -> the numeric reading wins (it carries decimals)
    if int_part == int(words_amt):
        return best
    # (b) the numeric's integer part ENDS WITH the words value and is 1-2 digits
    #     longer -> a currency glyph was OCR'd as a leading digit. Strip it but
    #     keep the decimal precision from the numeric reading.
    s_num, s_words = str(int_part), str(int(words_amt))
    if len(s_num) - len(s_words) in (1, 2) and s_num.endswith(s_words):
        try:
            frac = num - Decimal(int_part)
            corrected = Decimal(s_num[len(s_num) - len(s_words):]) + frac
        except Exception:
            corrected = None
        if corrected and corrected > 0:
            return corrected, "high", (f"{best[2]}; leading digit(s) were a currency-glyph "
                                        f"OCR artifact -- corroborated by amount-in-words "
                                        f"{words_amt}")
    # (c) materially different -> issuer-printed words are more reliable
    return words_amt, "high", (f"amount-in-words={words_amt} preferred over numeric "
                                 f"reading {num} ({best[2]})")
