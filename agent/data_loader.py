"""
Loads every file in dataset/ into the typed models from models.py and
provides simple indexed lookups (by user, by request, by related_event_id).

This module does *no* financial reasoning. Its only job is: read the CSVs
exactly as given, parse types, and organize them for lookup. Blank amounts
stay `None` here — filling them from evidence happens in reconciliation.py,
which is a deliberate separation so it's always possible to tell "what did
the raw dataset say" from "what did we infer".
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Dict, List, Optional
import csv
import os

from .config import DATASET_DIR
from .models import FinancialEvent, UserProfile, PaymentOption, Message, ImageRef, Request


def _pdate(s: str) -> Optional[date]:
    if not s:
        return None
    y, m, d = s.split("-")
    return date(int(y), int(m), int(d))


def _pdec(s: str) -> Optional[Decimal]:
    if s is None or s == "":
        return None
    return Decimal(s)


def _plist(s: str) -> List[str]:
    return [x for x in s.split("|") if x] if s else []


def _pbool(s: str) -> bool:
    return s.strip().lower() == "true"


def _pint(s: str) -> Optional[int]:
    if not s:
        return None
    return int(Decimal(s))  # tolerate "7" or "7.0"-style values defensively


@dataclass
class Dataset:
    events: List[FinancialEvent]
    profiles: Dict[str, UserProfile]
    options: Dict[str, List[PaymentOption]]           # request_id -> options
    messages: List[Message]
    images: List[ImageRef]
    requests: List[Request]                            # dataset/requests.csv (to predict)
    sample_requests: List[Request]                      # dataset/sample_requests.csv (has ground truth)

    # ---- derived indices, built once in __post_init__ ----
    events_by_id: Dict[str, FinancialEvent] = field(default_factory=dict)
    events_by_user: Dict[str, List[FinancialEvent]] = field(default_factory=dict)
    messages_by_user: Dict[str, List[Message]] = field(default_factory=dict)
    messages_by_request: Dict[str, List[Message]] = field(default_factory=dict)
    messages_by_related_event: Dict[str, List[Message]] = field(default_factory=dict)
    images_by_related_event: Dict[str, ImageRef] = field(default_factory=dict)
    images_by_request: Dict[str, List[ImageRef]] = field(default_factory=dict)

    def build_indices(self):
        self.events_by_id = {e.event_id: e for e in self.events}
        by_user: Dict[str, List[FinancialEvent]] = {}
        for e in self.events:
            by_user.setdefault(e.user_id, []).append(e)
        for u in by_user:
            by_user[u].sort(key=lambda e: (e.event_date, e.event_id))
        self.events_by_user = by_user

        mbu: Dict[str, List[Message]] = {}
        mbr: Dict[str, List[Message]] = {}
        mbe: Dict[str, List[Message]] = {}
        for m in self.messages:
            mbu.setdefault(m.user_id, []).append(m)
            if m.request_id:
                mbr.setdefault(m.request_id, []).append(m)
            if m.related_event_id:
                mbe.setdefault(m.related_event_id, []).append(m)
        self.messages_by_user, self.messages_by_request, self.messages_by_related_event = mbu, mbr, mbe

        ibe: Dict[str, ImageRef] = {}
        ibr: Dict[str, List[ImageRef]] = {}
        for img in self.images:
            if img.related_event_id:
                ibe[img.related_event_id] = img
            if img.request_id:
                ibr.setdefault(img.request_id, []).append(img)
        self.images_by_related_event, self.images_by_request = ibe, ibr

    def all_requests(self) -> List[Request]:
        return self.sample_requests + self.requests


def load_dataset(dataset_dir: str = DATASET_DIR) -> Dataset:
    events = _load_events(os.path.join(dataset_dir, "financial_events.csv"))
    profiles = _load_profiles(os.path.join(dataset_dir, "financial_profiles.csv"))
    options = _load_options(os.path.join(dataset_dir, "request_payment_options.csv"))
    messages = _load_messages(os.path.join(dataset_dir, "messages.csv"))
    images = _load_images(os.path.join(dataset_dir, "images.csv"))
    requests = _load_requests(os.path.join(dataset_dir, "requests.csv"), with_expected=False)
    sample_requests = _load_requests(os.path.join(dataset_dir, "sample_requests.csv"), with_expected=True)

    ds = Dataset(events=events, profiles=profiles, options=options, messages=messages,
                 images=images, requests=requests, sample_requests=sample_requests)
    ds.build_indices()
    return ds


def _load_events(path: str) -> List[FinancialEvent]:
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.append(FinancialEvent(
                event_id=r["event_id"], user_id=r["user_id"], event_type=r["event_type"],
                description=r["description"], category=r["category"], direction=r["direction"],
                amount=_pdec(r["amount"]), currency=r["currency"],
                event_date=_pdate(r["event_date"]), settlement_date=_pdate(r["settlement_date"]),
                status=r["status"], linked_event_id=r["linked_event_id"] or None,
                flexibility=r["flexibility"], minimum_allowed_amount=_pdec(r["minimum_allowed_amount"]),
            ))
    return out


def _load_profiles(path: str) -> Dict[str, UserProfile]:
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["user_id"]] = UserProfile(
                user_id=r["user_id"], home_currency=r["home_currency"],
                current_available_balance=_pdec(r["current_available_balance"]),
                minimum_balance_to_keep=_pdec(r["minimum_balance_to_keep"]),
                financial_priorities=_plist(r["financial_priorities"]),
                expense_categories_to_protect=_plist(r["expense_categories_to_protect"]),
                expense_categories_user_is_willing_to_reduce=_plist(r["expense_categories_user_is_willing_to_reduce"]),
                expense_categories_user_is_willing_to_stop=_plist(r["expense_categories_user_is_willing_to_stop"]),
                payment_methods_user_will_consider=_plist(r["payment_methods_user_will_consider"]),
                max_installment_months=_pint(r["max_installment_months"]),
            )
    return out


def _load_options(path: str) -> Dict[str, List[PaymentOption]]:
    out: Dict[str, List[PaymentOption]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            opt = PaymentOption(
                payment_option_id=r["payment_option_id"], request_id=r["request_id"],
                payment_method=r["payment_method"], payment_amount=_pdec(r["payment_amount"]),
                number_of_payments=int(r["number_of_payments"]),
                first_payment_date=_pdate(r["first_payment_date"]),
                payment_frequency_days=_pint(r["payment_frequency_days"]),
                financing_fee=_pdec(r["financing_fee"]) or Decimal("0"),
                total_payable_amount=_pdec(r["total_payable_amount"]),
            )
            out.setdefault(r["request_id"], []).append(opt)
    # Keep a stable, deterministic order (ascending payment_option_id) so tie
    # breaks on "lowest payment_option_id" are trivial downstream.
    for k in out:
        out[k].sort(key=lambda o: _option_sort_key(o.payment_option_id))
    return out


def _option_sort_key(option_id: str):
    # "payment_option_12" -> 12, for correct numeric (not lexicographic) ordering
    digits = "".join(ch for ch in option_id if ch.isdigit())
    return int(digits) if digits else option_id


def _load_messages(path: str) -> List[Message]:
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.append(Message(
                message_id=r["message_id"], user_id=r["user_id"],
                request_id=r["request_id"] or None, related_event_id=r["related_event_id"] or None,
                sent_at=r["sent_at"], source_type=r["source_type"], message_text=r["message_text"],
            ))
    return out


def _load_images(path: str) -> List[ImageRef]:
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.append(ImageRef(
                image_id=r["image_id"], user_id=r["user_id"],
                request_id=r["request_id"] or None, related_event_id=r["related_event_id"] or None,
            ))
    return out


def _load_requests(path: str, with_expected: bool) -> List[Request]:
    out = []
    expected_cols = ["amount_safe_to_pay", "affordability_status", "recommended_payment_method",
                      "payment_plan", "earliest_date_for_full_payment", "spending_changes_needed",
                      "decision_explanation"]
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            expected = {c: r[c] for c in expected_cols} if with_expected else None
            out.append(Request(
                request_id=r["request_id"], user_id=r["user_id"], request_date=_pdate(r["request_date"]),
                request_type=r["request_type"], requested_amount=_pdec(r["requested_amount"]),
                desired_completion_date=_pdate(r["desired_completion_date"]),
                allows_partial_payment=_pbool(r["allows_partial_payment"]),
                request_text=r["request_text"], expected=expected,
            ))
    return out
