"""
Currency conversion against dataset/exchange_rates.csv.

Design notes (informed by actually inspecting the rate sheet, not assumed):
  * Every conversion this dataset needs turns out to be a single direct pair
    (USD->INR, USD->IDR, USD->EUR, EUR->ZAR, EUR->USD) -- confirmed by
    scanning every event whose currency differs from its user's
    home_currency. A 2-hop path-search is still implemented for
    robustness/generalization, but is not required to make this dataset's
    numbers correct.
  * Pair coverage is *not* uniform across dates: EUR->ZAR and USD->IDR stop
    being quoted after mid-2026, while USD->INR continues. So "most recent
    quoted date on/before the target" must be computed **per pair**, not
    once globally -- otherwise a date with partial coverage silently breaks
    pairs that were actually quoted a month earlier. This was caught by a
    failing smoke test before it could reach the forecast engine.
"""

from __future__ import annotations
from decimal import Decimal
from datetime import date
from typing import Dict, Tuple, Optional, List
import csv


class ExchangeRates:
    def __init__(self, rows):
        # by_pair[(from_ccy, to_ccy)] = sorted list of (date, rate)
        self.by_pair: Dict[Tuple[str, str], List[Tuple[date, Decimal]]] = {}
        all_currencies = set()
        for r in rows:
            d = _parse_date(r["rate_date"])
            pair = (r["from_currency"], r["to_currency"])
            self.by_pair.setdefault(pair, []).append((d, Decimal(r["rate"])))
            all_currencies.add(r["from_currency"])
            all_currencies.add(r["to_currency"])
        for pair in self.by_pair:
            self.by_pair[pair].sort(key=lambda t: t[0])
        self.all_currencies = all_currencies

    @classmethod
    def load(cls, path: str) -> "ExchangeRates":
        with open(path, newline="", encoding="utf-8") as f:
            return cls(list(csv.DictReader(f)))

    def _rate_on_or_before(self, pair: Tuple[str, str], on_date: date) -> Optional[Decimal]:
        series = self.by_pair.get(pair)
        if not series:
            return None
        best = None
        for d, rate in series:
            if d <= on_date:
                best = rate  # series is sorted ascending, so this keeps advancing
            else:
                break
        if best is not None:
            return best
        # Nothing on/before -> fall back to the earliest quote we do have,
        # rather than failing outright on an out-of-range date.
        return series[0][1]

    def _leg_rate(self, a: str, b: str, on_date: date) -> Optional[Decimal]:
        """Rate to convert 1 unit of `a` into `b`, direct or inverted, using
        that pair's own most-recent-available date independently."""
        direct = self._rate_on_or_before((a, b), on_date)
        if direct is not None:
            return direct
        inverse = self._rate_on_or_before((b, a), on_date)
        if inverse is not None and inverse != 0:
            return Decimal(1) / inverse
        return None

    def convert(self, amount: Decimal, from_ccy: str, to_ccy: str, on_date: date) -> Decimal:
        from .config import money
        if from_ccy == to_ccy:
            return money(amount)

        direct = self._leg_rate(from_ccy, to_ccy, on_date)
        if direct is not None:
            return money(amount * direct)

        for hub in self.all_currencies:
            if hub in (from_ccy, to_ccy):
                continue
            leg1 = self._leg_rate(from_ccy, hub, on_date)
            if leg1 is None:
                continue
            leg2 = self._leg_rate(hub, to_ccy, on_date)
            if leg2 is None:
                continue
            return money(amount * leg1 * leg2)

        raise ValueError(f"No conversion path {from_ccy}->{to_ccy} found near {on_date}")


def _parse_date(s: str) -> date:
    y, m, d = s.split("-")
    return date(int(y), int(m), int(d))
