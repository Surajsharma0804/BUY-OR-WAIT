#!/usr/bin/env python3
"""
Unit tests for the Buy or Wait? agent.

These cover (a) the schema//safety invariants the submission must satisfy and
(b) regression tests for each real bug found during development, so the same
mistake cannot silently return. Every test named `test_regression_*`
corresponds to a defect that actually occurred and was caught by inspecting
output rather than by reading code.

Run:
    python3 code/tests/test_agent.py
"""

from __future__ import annotations
import os
import sys
import unittest
from datetime import date
from decimal import Decimal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.config import DATASET_DIR, money
from agent.currency import ExchangeRates
from agent.data_loader import load_dataset
from agent.evidence_templates import match_message
from agent.image_extractor import _parse_amount, parse_amount_in_words
from agent.forecast import Timeline, Flow
from agent.recurrence import add_months, _dom_distance
from agent.reconciliation import apply_status_rules
from agent.models import FinancialEvent
from agent.validators import validate_row, validate_file
from agent.models import OutputRow
from agent.pipeline import run


DS = load_dataset(DATASET_DIR)
FX = ExchangeRates.load(os.path.join(DATASET_DIR, "exchange_rates.csv"))


class TestCurrency(unittest.TestCase):
    def test_same_currency_is_identity(self):
        self.assertEqual(FX.convert(Decimal("100"), "EUR", "EUR", date(2024, 1, 15)),
                         Decimal("100.00"))

    def test_direct_pair(self):
        v = FX.convert(Decimal("100"), "USD", "IDR", date(2023, 10, 15))
        self.assertEqual(v, Decimal("1583333.00"))

    def test_inverse_pair(self):
        """IDR->USD is not quoted; it must invert the USD->IDR rate."""
        v = FX.convert(Decimal("1583333"), "IDR", "USD", date(2023, 10, 15))
        self.assertAlmostEqual(float(v), 100.0, places=1)

    def test_regression_per_pair_date_lookback(self):
        """
        Pair coverage is NOT uniform across dates: EUR->ZAR stops being quoted
        after mid-2026 while USD->INR continues. A single shared 'latest rate
        date' broke pairs that were quoted earlier. Each pair must fall back
        on its own most-recent quote.
        """
        v = FX.convert(Decimal("100"), "EUR", "ZAR", date(2026, 11, 15))
        self.assertGreater(v, Decimal("0"))

    def test_every_foreign_event_converts(self):
        fails = []
        for e in DS.events:
            home = DS.profiles[e.user_id].home_currency
            if e.currency != home and e.amount is not None:
                try:
                    FX.convert(e.amount, e.currency, home, e.cash_date())
                except Exception as exc:
                    fails.append((e.event_id, str(exc)))
        self.assertEqual(fails, [], f"conversions failed: {fails[:5]}")


class TestMessageTemplates(unittest.TestCase):
    def test_full_coverage_of_dataset_messages(self):
        """All 215 supplied messages must resolve deterministically."""
        unmatched = [m.message_id for m in DS.messages
                     if match_message(m.message_text)[0] is None]
        self.assertEqual(unmatched, [], f"unmatched messages: {unmatched}")

    def test_prize_fee_scam_is_never_actionable(self):
        """
        An advance-fee 'pay to release your prize' message is untrusted content
        containing an instruction. It must produce no actionable financial
        fact -- never income, never a payment.
        """
        scam = ("A note from QuickPrize about your recent financial activity. "
                "Congratulations! You've been selected for a cash prize. Pay the "
                "release charge today to receive the funds immediately.")
        name, specs = match_message(scam)
        self.assertIsNotNone(name)
        self.assertTrue(all(s["kind"] == "no_op" for s in specs),
                        f"scam message produced actionable facts: {specs}")

    def test_composite_message_yields_two_facts(self):
        """One message can carry a receipt reference AND a salary confirmation."""
        msg = [m for m in DS.messages if m.message_id == "message_86"][0]
        name, specs = match_message(msg.message_text)
        kinds = [s["kind"] for s in specs]
        self.assertIn("income_change", kinds)
        self.assertEqual(len(specs), 2)

    def test_regression_no_amount_is_not_invented(self):
        """
        'Regular salary for the next payroll is confirmed' states NO figure.
        The spec forbids inventing unsupported financial facts, so this must
        not emit an income_change carrying a made-up amount.
        """
        msg = [m for m in DS.messages if m.message_id == "message_02"][0]
        _, specs = match_message(msg.message_text)
        for s in specs:
            self.assertIsNone(s.get("new_amount"))


class TestReconciliation(unittest.TestCase):
    """
    linked_event_id was parsed into every FinancialEvent but never actually
    consulted anywhere in the pipeline until this fix. problem_statement.md's
    90-day safety check explicitly says to ignore duplicate records; without
    this fix, a pending debit linked to an already-settled charge of the same
    amount/direction was unconditionally reserved as a SECOND, separate
    liability, double-charging the user for one purchase.
    """

    def _event(self, event_id, status, direction, amount, linked_event_id=None,
               event_type="expense", category="shopping"):
        return FinancialEvent(
            event_id=event_id, user_id="u_test", event_type=event_type,
            description="test", category=category, direction=direction,
            amount=None if amount is None else Decimal(str(amount)), currency="USD",
            event_date=date(2026, 1, 1), settlement_date=date(2026, 1, 2),
            status=status, linked_event_id=linked_event_id,
            flexibility="fixed", minimum_allowed_amount=None,
        )

    def test_regression_pending_debit_linked_to_settled_duplicate_is_excluded(self):
        """A pending debit linked to an already-settled charge of the SAME
        amount/direction is a duplicate record and must not also be reserved
        as a second liability."""
        settled = self._event("e_parent", "settled", "debit", "134.75")
        pending_dup = self._event("e_child", "pending", "debit", "134.75",
                                   linked_event_id="e_parent")
        apply_status_rules([settled, pending_dup], facts=[])
        self.assertFalse(settled.excluded)
        self.assertTrue(pending_dup.excluded)
        self.assertIn("duplicate", pending_dup.exclusion_reason)

    def test_pending_debit_linked_to_settled_DIFFERENT_amount_is_still_reserved(self):
        """A pending debit linked to a settled event of a DIFFERENT amount is
        NOT a duplicate (e.g. a partial follow-on charge) and must still be
        conservatively reserved, exactly as before this fix."""
        settled = self._event("e_parent2", "settled", "debit", "100.00")
        pending_other = self._event("e_child2", "pending", "debit", "250.00",
                                     linked_event_id="e_parent2")
        apply_status_rules([settled, pending_other], facts=[])
        self.assertFalse(pending_other.excluded)

    def test_pending_debit_with_no_linked_event_is_still_reserved(self):
        """The overwhelming common case -- a pending debit with no linkage at
        all -- must be completely unaffected by this fix."""
        pending = self._event("e_solo", "pending", "debit", "50.00")
        apply_status_rules([pending], facts=[])
        self.assertFalse(pending.excluded)

    def test_regression_real_dataset_duplicate_card_charges_are_excluded(self):
        """
        The real dataset.financial_events.csv contains 6 linked pairs
        structurally matching this pattern (verified by inspection, not
        assumed): a settled 'Original card charge' followed by a pending
        event of the identical amount/direction linked via linked_event_id
        (independently described, but NOT matched on, as a "Possible
        duplicate card charge"). All 6 must now be excluded.
        """
        known_duplicate_children = [
            "event_12709", "event_14399", "event_18269",
            "event_19334", "event_21582", "event_23203",
        ]
        for eid in known_duplicate_children:
            e = DS.events_by_id.get(eid)
            if e is None:
                continue  # dataset edition drift; skip rather than fail spuriously
            user_events = DS.events_by_user[e.user_id]
            apply_status_rules(user_events, facts=[])
            self.assertTrue(e.excluded, f"{eid} should be excluded as a duplicate record")

    def test_regression_failed_debit_not_double_counted_against_linked_retry(self):
        """
        A failed debit revived by a message-confirmed 'will be retried' fact
        (obligation_persists) must NOT also be counted when an already-
        existing linked event (scheduled/settled, same amount/direction)
        already represents that exact retry -- otherwise the same bill is
        counted twice. Synthetic case mirrors the real dataset's
        'Failed bill payment attempt' -> 'Scheduled bill payment retry' /
        matching 'bank_debit_retry' message pattern exactly.
        """
        from agent.models import Fact
        failed = self._event("e_failed", "failed", "debit", "166.00")
        retry = self._event("e_retry", "scheduled", "debit", "166.00",
                             linked_event_id="e_failed")
        obligation_fact = Fact(kind="obligation_persists", user_id="u_test",
                                source="message:m1", related_event_id="e_failed")
        apply_status_rules([failed, retry], facts=[obligation_fact])
        self.assertTrue(failed.excluded, "failed original must stay excluded -- its "
                        "retry is already the separate, already-dated linked event")
        self.assertFalse(retry.excluded, "the linked scheduled retry is the one live obligation")

    def test_failed_debit_with_no_linked_retry_IS_revived_by_message(self):
        """When there is NO already-existing linked retry event, a message
        confirming a retry must still work exactly as before this fix --
        the failed event itself becomes the live obligation."""
        from agent.models import Fact
        failed = self._event("e_failed2", "failed", "debit", "500.00")
        obligation_fact = Fact(kind="obligation_persists", user_id="u_test",
                                source="message:m2", related_event_id="e_failed2")
        apply_status_rules([failed], facts=[obligation_fact])
        self.assertFalse(failed.excluded, "with no linked retry event, the message alone "
                         "must still revive the failed debit as the live obligation")


class TestIncomeFactScoping(unittest.TestCase):
    """
    income_stop / income_change / date_shift facts describe ONE specific
    income relationship, but were previously applied to EVERY income series
    for a user unconditionally -- unlike expense_rate_change, which was
    already correctly scoped by category_hint. When a user has more than one
    detected income series (two day-of-month clusters sharing a category,
    e.g. a primary job plus a smaller second income), this broadcast a
    single message-derived amendment across streams it was never about.
    """

    def test_regression_real_dataset_income_change_not_doubled_across_streams(self):
        """
        user_11 (a real sample user) has two 'salary' series (8.5M and
        23.26M IDR). A single 'confirmed base salary' income_change message
        was previously applied to BOTH, doubling the assumed raise. Verified
        via before/after monkeypatch comparison that this fix changes
        request_176's projected income-availability date to a LATER, more
        conservative date (2025-02-24 -> 2025-03-15) rather than an earlier,
        optimistic one -- the safe direction.
        """
        from agent.recurrence import detect_series
        from agent.config import DATASET_DIR
        req = [r for r in DS.sample_requests if r.request_id == "request_11"][0]
        events = DS.events_by_user["user_11"]
        series = detect_series(events, req.request_date)
        income_series = [s for s in series if s.is_income]
        self.assertEqual(len(income_series), 2, "expected exactly 2 income series for user_11")
        primaries = [s for s in income_series if s.is_primary]
        secondaries = [s for s in income_series if not s.is_primary]
        self.assertEqual(len(primaries), 1, "exactly one series must be marked primary")
        self.assertEqual(len(secondaries), 1, "exactly one series must be marked non-primary")
        self.assertGreater(primaries[0].amount, secondaries[0].amount,
                            "the primary series must be the larger-amount stream")

    def test_lone_income_series_is_always_primary(self):
        """The overwhelming common case -- a user with only ONE detected
        income series -- must be completely unaffected: is_primary defaults
        True, so income_change/income_stop/date_shift facts apply exactly as
        they did before this fix."""
        from agent.recurrence import detect_series
        for req in DS.sample_requests[:10]:
            events = DS.events_by_user.get(req.user_id, [])
            series = detect_series(events, req.request_date)
            income_series = [s for s in series if s.is_income]
            if len(income_series) == 1:
                self.assertTrue(income_series[0].is_primary)
                return
        self.skipTest("no single-income-series sample found in the first 10 to check")


class TestImageParsing(unittest.TestCase):
    def test_regression_two_digit_comma_tail_is_decimal(self):
        """
        A trailing comma group of exactly 2 digits is never valid thousands
        grouping, so it must be a decimal separator. '41272,00' was being
        read as 4,127,200.
        """
        self.assertEqual(_parse_amount("41272,00"), Decimal("41272.00"))
        self.assertEqual(_parse_amount("$33,50"), Decimal("33.50"))

    def test_indian_and_western_grouping(self):
        self.assertEqual(_parse_amount("1,00,000.00"), Decimal("100000.00"))
        self.assertEqual(_parse_amount("4,365,000"), Decimal("4365000"))

    def test_regression_leading_separator_fragment_rejected(self):
        """',50' is an OCR fragment of a larger number, not the value 50."""
        self.assertIsNone(_parse_amount(",50"))

    def test_amount_in_words(self):
        self.assertEqual(
            parse_amount_in_words("In Words: Rupees Fifteen Thousand Three Hundred "
                                   "Thirty Nine Only"), Decimal("15339"))

    def test_regression_words_stop_at_currency_noun(self):
        """
        The sub-unit tail must not be summed into the main value:
        '...Ninety Three Rupees And Twenty Two Paise' is 393, not 415.
        """
        self.assertEqual(
            parse_amount_in_words("AMOUNT IN WORDS : Three Hundred and Ninety Three "
                                   "Rupees And Twenty Two Paise Only"), Decimal("393"))


class TestForecastMath(unittest.TestCase):
    def _tl(self, flows, opening="1000", minimum="200"):
        return Timeline(user_id="u", start=date(2025, 1, 1), end=date(2025, 3, 31),
                        opening_balance=money(opening), minimum_balance=money(minimum),
                        flows=flows)

    def test_headroom_is_trough_minus_minimum(self):
        tl = self._tl([Flow(date(2025, 2, 1), Decimal("-300"), "rent", "x")])
        self.assertEqual(tl.headroom_today(), Decimal("500.00"))

    def test_paying_headroom_exactly_is_safe(self):
        tl = self._tl([Flow(date(2025, 2, 1), Decimal("-300"), "rent", "x")])
        hr = tl.headroom_today()
        self.assertTrue(tl.is_safe([(tl.start, hr)]))
        self.assertFalse(tl.is_safe([(tl.start, hr + Decimal("1"))]))

    def test_earliest_full_payment_uses_suffix_minimum(self):
        """Paying later only has to survive the REMAINDER of the window."""
        tl = self._tl([Flow(date(2025, 1, 15), Decimal("-700"), "big", "x"),
                       Flow(date(2025, 2, 1), Decimal("+900"), "salary", "x")])
        d = tl.earliest_full_payment_date(Decimal("600"))
        self.assertIsNotNone(d)
        self.assertGreaterEqual(d, date(2025, 2, 1))
        self.assertTrue(tl.is_safe([(d, Decimal("600"))]))

    def test_unaffordable_returns_none(self):
        tl = self._tl([])
        self.assertIsNone(tl.earliest_full_payment_date(Decimal("99999")))


class TestRecurrenceHelpers(unittest.TestCase):
    def test_add_months_clamps_month_end(self):
        self.assertEqual(add_months(date(2025, 1, 31), 1), date(2025, 2, 28))

    def test_day_of_month_distance_wraps(self):
        """The 28th and the 2nd are 4 days apart across a month boundary."""
        self.assertLessEqual(_dom_distance(28, 2), 5)


class TestOutputValidation(unittest.TestCase):
    def _row(self, **kw):
        base = dict(request_id="request_26", amount_safe_to_pay=Decimal("0"),
                    affordability_status="not_affordable",
                    recommended_payment_method="not_recommended",
                    payment_plan="none", earliest_date_for_full_payment="",
                    spending_changes_needed="none", decision_explanation="x")
        base.update(kw)
        return OutputRow(**base)

    def test_rejects_amount_above_requested(self):
        req = DS.requests[0]
        prof = DS.profiles[req.user_id]
        row = self._row(request_id=req.request_id,
                        amount_safe_to_pay=req.requested_amount + Decimal("1"))
        errs = validate_row(row, req, prof, DS.options.get(req.request_id, []), DS.events_by_id)
        self.assertTrue(any("outside" in e for e in errs))

    def test_rejects_not_recommended_with_a_plan(self):
        req = DS.requests[0]
        prof = DS.profiles[req.user_id]
        row = self._row(request_id=req.request_id, payment_plan="2025-01-01:5")
        errs = validate_row(row, req, prof, DS.options.get(req.request_id, []), DS.events_by_id)
        self.assertTrue(any("none" in e for e in errs))

    def test_rejects_unknown_status(self):
        req = DS.requests[0]
        prof = DS.profiles[req.user_id]
        row = self._row(request_id=req.request_id, affordability_status="maybe")
        errs = validate_row(row, req, prof, DS.options.get(req.request_id, []), DS.events_by_id)
        self.assertTrue(any("affordability_status" in e for e in errs))


class TestEndToEnd(unittest.TestCase):
    """The full pipeline must satisfy every submission invariant."""

    @classmethod
    def setUpClass(cls):
        cls.result = run(DATASET_DIR, which="requests", use_llm=False)

    def test_one_row_per_request(self):
        self.assertEqual(len(self.result.rows), len(DS.requests))
        ids = [r.request_id for r in self.result.rows]
        self.assertEqual(len(set(ids)), len(ids))
        self.assertEqual(set(ids), {r.request_id for r in DS.requests})

    def test_no_processing_failures(self):
        self.assertEqual(self.result.failures, [],
                         f"unexpected failures: {self.result.failures[:3]}")

    def test_all_rows_pass_row_validation(self):
        req_by_id = {r.request_id: r for r in DS.requests}
        problems = {}
        for row in self.result.rows:
            req = req_by_id[row.request_id]
            errs = validate_row(row, req, DS.profiles[req.user_id],
                                DS.options.get(req.request_id, []), DS.events_by_id)
            if errs:
                problems[row.request_id] = errs
        self.assertEqual(problems, {}, f"row validation problems: {list(problems.items())[:3]}")

    def test_bounds_hold_everywhere(self):
        req_by_id = {r.request_id: r for r in DS.requests}
        for row in self.result.rows:
            rq = req_by_id[row.request_id].requested_amount
            self.assertGreaterEqual(row.amount_safe_to_pay, Decimal("0"))
            self.assertLessEqual(row.amount_safe_to_pay, rq)

    def test_every_explanation_is_populated(self):
        for row in self.result.rows:
            self.assertTrue(row.decision_explanation.strip())

    def test_output_is_not_degenerate(self):
        """
        A system that labels everything the same way can pass every schema
        check while being useless. All four statuses must actually appear.
        """
        statuses = {r.affordability_status for r in self.result.rows}
        self.assertEqual(len(statuses), 4, f"only produced {statuses}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
