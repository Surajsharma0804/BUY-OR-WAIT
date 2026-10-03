"""
Rule-based message template catalog.

Why regex instead of an LLM call for this: dataset/messages.csv turned out
(after reading every single row) to be generated from ~20 fixed templates,
each appearing in an English and an Indonesian phrasing, with company names,
amounts, dates, and currency codes substituted in. For money-moving facts,
a small, auditable, 100%-deterministic template matcher is strictly more
trustworthy than a free-text LLM call: it cannot hallucinate a figure, and
every match is traceable to an exact regex. evidence_extractor.py still
carries an optional LLM fallback (agent/llm_client.py) for any message that
matches *no* template here, so the system degrades gracefully on inputs
outside this catalog rather than silently dropping them.

Each TEMPLATE has:
  name        - short id, used for logging/audit
  patterns    - list of compiled regexes (EN + ID variants); first match wins
  build       - callable(match, message) -> list[FactSpec] (kind + kwargs,
                turned into agent.models.Fact by evidence_extractor.py)

FactSpec is a plain dict; evidence_extractor.py fills in user_id/source.
"""

from __future__ import annotations
import re
from decimal import Decimal
from datetime import date


def _d(y, m, d):
    return date(int(y), int(m), int(d))


AMT = r"(?P<amount>[\d,]+(?:\.\d+)?)"
AMT2 = r"(?P<amount2>[\d,]+(?:\.\d+)?)"
CUR = r"(?P<currency>INR|ZAR|IDR|USD|EUR)"
CUR2 = r"(?:INR|ZAR|IDR|USD|EUR)"
DATE = r"(?P<date>\d{4}-\d{2}-\d{2})"


def _num(s: str) -> Decimal:
    return Decimal(s.replace(",", ""))


def _mkdate(s: str) -> date:
    y, m, d = s.split("-")
    return _d(y, m, d)


_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}


def _month_name_date(year: str, monthname: str, day: str) -> date:
    return date(int(year), _MONTHS[monthname.strip().lower()], int(day))


class Template:
    def __init__(self, name, patterns, build):
        self.name = name
        self.patterns = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in patterns]
        self.build = build

    def try_match(self, text):
        for p in self.patterns:
            m = p.search(text)
            if m:
                return self.build(m)
        return None


# ---------------------------------------------------------------------------
# EMPLOYER / payroll templates
# ---------------------------------------------------------------------------

TPL_RAISE = Template(
    "employer_raise",
    [
        rf"(?:monthly )?salary has increased to {CUR}\s*{AMT}.*?applies from {DATE}",
        rf"[Gg]aji bulanan Anda naik menjadi {CUR}\s*{AMT}.*?berlaku mulai {DATE}",
    ],
    lambda m: [dict(kind="income_change", new_amount=_num(m["amount"]),
                     new_currency=m["currency"], new_date=_mkdate(m["date"]),
                     note="employer: monthly salary raised")],
)

TPL_DATE_SHIFT = Template(
    "employer_date_shift",
    [
        rf"confirmed salary is now expected on {DATE}\. This replaces the payroll date",
        rf"[Gg]aji yang sudah dikonfirmasi kini diperkirakan masuk pada {DATE}\. Tanggal ini menggantikan",
    ],
    lambda m: [dict(kind="date_shift", new_date=_mkdate(m["date"]),
                     note="employer: next payroll date moved")],
)

TPL_REDUCED_NEXT = Template(
    "employer_reduced_next",
    [
        rf"next salary is reduced to {CUR}\s*{AMT}\. The adjustment is due to approved unpaid leave",
        rf"[Gg]aji (?:Anda )?(?:berikutnya|selanjutnya)? ?(?:yang )?(?:akan datang )?(?:turun|berkurang|dikurangi)",  # ID variants vary; amount not always ID here
    ],
    lambda m: [dict(kind="income_change", new_amount=_num(m["amount"]) if "amount" in m.groupdict() and m["amount"] else None,
                     new_currency=m.groupdict().get("currency"),
                     note="employer: next salary reduced (unpaid leave)")],
)

TPL_COMMISSION_PENDING = Template(
    "employer_commission_pending",
    [
        rf"confirmed base salary is {CUR}\s*{AMT}\. The commission",
        rf"[Gg]aji pokok yang dikonfirmasi adalah {CUR}\s*{AMT}\. ?[Kk]omisi",
        rf"confirmed base salary is {AMT}\. The commission",
    ],
    lambda m: [dict(kind="income_change", new_amount=_num(m["amount"]),
                     new_currency=m.groupdict().get("currency"),
                     note="employer: confirmed base salary (commission excluded, not confirmed)")],
)

TPL_FIRST_SALARY = Template(
    "employer_first_salary",
    [
        rf"first salary will be {CUR}\s*{AMT}\. The confirmed credit date is {DATE}",
        rf"[Gg]aji pertama (?:dari perusahaan baru )?(?:Anda )?(?:sebesar |adalah )?{CUR}\s*{AMT}.{{0,40}}?(?:dikonfirmasi untuk|dijadwalkan pada) {DATE}",
        rf"first salary from the new employer is {CUR}\s*{AMT}\. It is confirmed for {DATE}",
        rf"first salary of {CUR}\s*{AMT} (?:is )?scheduled for {DATE}",
        # ID variant where amount and confirmed credit date are in separate sentences:
        # "Gaji pertama Anda sebesar IDR 30400000. Tanggal kredit yang dikonfirmasi adalah 2026-04-15."
        rf"[Gg]aji pertama Anda sebesar {CUR}\s*{AMT}\.\s*Tanggal kredit yang dikonfirmasi adalah {DATE}",
    ],
    lambda m: [dict(kind="income_new", new_amount=_num(m["amount"]), new_currency=m["currency"],
                     new_date=_mkdate(m["date"]), note="employer: first salary at new/updated job")],
)

TPL_ARREARS = Template(
    "employer_arrears_onetime",
    [
        rf"regular salary for the next payroll is {CUR}\s*{AMT}\. The same payroll includes a one-time arrears adjustment of {CUR2}\s*{AMT2}",
        rf"[Gg]aji rutin (?:Anda )?untuk penggajian berikutnya (?:adalah|sudah dikonfirmasi) {CUR}\s*{AMT}\..{{0,20}}?[Pp]enggajian yang sama mencakup penyesuaian tunggakan satu kali sebesar {CUR2}\s*{AMT2}",
    ],
    lambda m: [
        dict(kind="income_change", new_amount=_num(m["amount"]), new_currency=m["currency"],
             note="employer: regular salary confirmed"),
        dict(kind="income_new", new_amount=_num(m.groupdict().get("amount2") or "0"), new_currency=m["currency"],
             one_off=True, note="employer: one-time arrears adjustment"),
    ],
)

TPL_INCOME_SOURCE_ENDED = Template(
    "employer_income_source_ended",
    [
        rf"household employment record has ended\. The remaining confirmed monthly salary is {CUR}\s*{AMT}",
        rf"[Ss]alah satu sumber pendapatan kerja rumah tangga telah berakhir\. Sisa gaji bulanan yang dikonfirmasi adalah {CUR}\s*{AMT}",
    ],
    lambda m: [dict(kind="income_change", new_amount=_num(m["amount"]), new_currency=m["currency"],
                     note="employer: one income source ended, remaining confirmed salary given")],
)

TPL_TEMP_REDUCED = Template(
    "employer_temp_reduced",
    [
        rf"temporary monthly pay is {CUR}\s*{AMT}\. The reduced amount continues",
        rf"[Gg]aji bulanan sementara Anda adalah {CUR}\s*{AMT}\. Jumlah yang lebih rendah masih berlaku",
    ],
    lambda m: [dict(kind="income_change", new_amount=_num(m["amount"]), new_currency=m["currency"],
                     note="employer: temporary reduced pay continues")],
)

TPL_EMPLOYMENT_ENDED = Template(
    "employer_employment_ended",
    [
        r"[Yy]our employment has ended\. There are no regular salary payments scheduled",
        r"[Hh]ubungan kerja Anda telah berakhir\. Tidak ada pembayaran gaji rutin",
    ],
    lambda m: [dict(kind="income_stop", note="employer: employment ended")],
)

TPL_SEASONAL_ENDED = Template(
    "employer_seasonal_ended",
    [
        r"current seasonal contract has ended\. No off-season income",
        r"[Kk]ontrak musiman saat ini telah berakhir\. Belum ada pendapatan di luar musim",
    ],
    lambda m: [dict(kind="income_stop", note="employer: seasonal contract ended, no renewal confirmed")],
)

TPL_RESUME_CHILDCARE = Template(
    "employer_resume_with_childcare",
    [
        rf"[Rr]egular salary of {CUR}\s*{AMT} resumes on {DATE}",
        rf"[Gg]aji (?:pokok |rutin )?(?:sebesar )?{CUR}\s*{AMT}.{{0,10}}(?:dimulai kembali|resumes) (?:pada |on )?{DATE}",
    ],
    lambda m: [dict(kind="income_change", new_amount=_num(m["amount"]), new_currency=m["currency"],
                     new_date=_mkdate(m["date"]), note="employer: salary resumes (childcare deduction mentioned, no amount given)")],
)

TPL_FOREIGN_SALARY = Template(
    "employer_foreign_currency_confirmed",
    [
        rf"salary of {CUR}\s*{AMT} is confirmed for {DATE}\. The receiving bank will convert",
        rf"[Gg]aji sebesar {CUR}\s*{AMT} dikonfirmasi untuk {DATE}\. Bank penerima akan mengonversinya",
    ],
    lambda m: [dict(kind="income_change", new_amount=_num(m["amount"]), new_currency=m["currency"],
                     new_date=_mkdate(m["date"]), note="employer: foreign-currency salary confirmed, convert at settlement rate")],
)

TPL_REIMBURSEMENT_NOT_SALARY = Template(
    "employer_reimbursement_not_salary",
    [
        r"latest employer credit is the reimbursement for your earlier work expense",
        r"[Dd]ana terbaru dari perusahaan adalah penggantian atas biaya kerja",
    ],
    lambda m: [dict(kind="exclude_from_recurrence",
                     note="employer: this credit is a one-off reimbursement, not recurring salary")],
)

TPL_BONUS_PENDING = Template(
    "employer_bonus_pending",
    [
        r"quarterly bonus is still (?:subject to|pending) the? final performance review",
        r"[Bb]onus kuartalan Anda masih menunggu hasil akhir penilaian kinerja",
    ],
    lambda m: [dict(kind="no_op", note="employer: bonus not yet approved -- must not be counted as income")],
)

# ---------------------------------------------------------------------------
# BANK templates
# ---------------------------------------------------------------------------

TPL_REGULAR_CONFIRMED_NO_AMOUNT = Template(
    "employer_regular_confirmed_no_amount",
    [
        r"[Rr]egular (?:salary|pay) for the next payroll (?:is|has been) confirmed",
        r"[Gg]aji rutin untuk penggajian berikutnya sudah dikonfirmasi",
    ],
    # Deliberately emits NO income_change: the message confirms that the usual
    # salary continues but states no figure, and problem_statement.md forbids
    # inventing unsupported financial facts. The existing recurring-salary
    # projection already covers it; the one-off adjustment is unquantified and
    # therefore must not be added.
    lambda m: [dict(kind="no_op",
                     note="employer: regular salary confirmed but no amount stated; keep existing "
                          "recurring salary, do not invent the separate one-off adjustment")],
)

TPL_COMPOSITE_WALLET_AND_SALARY = Template(
    "financial_wallet_charge_plus_salary_confirmation",
    [
        rf"wallet was charged for the session.*?employer has confirmed a {CUR}\s*{AMT} salary credit for "
        rf"(?P<day>\d{{1,2}}) (?P<monthname>January|February|March|April|May|June|July|August|September|October|November|December) (?P<year>\d{{4}})",
    ],
    lambda m: [
        # The wallet charge amount itself lives on the linked receipt image, not in this text.
        dict(kind="confirm_settled",
             note="financial_service: wallet session charge confirmed; final amount on linked receipt image"),
        dict(kind="income_change", new_amount=_num(m["amount"]), new_currency=m["currency"],
             new_date=_month_name_date(m["year"], m["monthname"], m["day"]),
             note="financial_service: employer-confirmed foreign-currency salary credit, "
                  "converted at settlement-date rate"),
    ],
)

TPL_WASH_TRANSFER = Template(
    "bank_wash_transfer",
    [
        r"matching debit and credit came from a transfer between your two accounts",
        r"[Dd]ebit dan kredit dengan jumlah yang sama berasal dari transfer antara dua rekening",
    ],
    lambda m: [dict(kind="no_op", note="bank: self-transfer wash, already settled, nets to zero")],
)

TPL_DISPUTE_PENDING = Template(
    "bank_dispute_pending",
    [
        r"extra card charge is still being investigated\. A reversal has not been posted",
        r"[Tt]agihan kartu tambahan masih dalam penyelidikan\. Dana pembalikannya belum tercatat",
    ],
    lambda m: [dict(kind="exclude_pending_credit", note="bank: disputed charge reversal not yet posted")],
)

TPL_DEBIT_RETRY = Template(
    "bank_debit_retry",
    [
        r"previous debit attempt failed\. The bill is still outstanding and another debit will be attempted",
        r"[Dd]ebit sebelumnya gagal\. Tagihan masih terbuka dan debit lain akan dicoba",
    ],
    lambda m: [dict(kind="obligation_persists", note="bank: failed debit will be retried, obligation still real")],
)

TPL_TWO_CARDS = Template(
    "bank_two_cards_not_fungible",
    [
        r"minimum payments due on two separate card accounts",
        r"[Aa]da pembayaran minimum(?:.{0,10})?dua (?:akun )?kartu (?:kredit )?(?:yang )?terpisah",
    ],
    lambda m: [dict(kind="no_op", note="bank: two distinct card minimums, do not conflate the series")],
)

# ---------------------------------------------------------------------------
# FINANCIAL_SERVICE templates
# ---------------------------------------------------------------------------

TPL_INVESTMENT_SETTLED = Template(
    "financial_investment_sale_settled",
    [
        r"proceeds from your investment sale have settled in the cash account",
        r"[Hh]asil penjualan investasi Anda sudah masuk ke rekening tunai",
    ],
    lambda m: [dict(kind="confirm_settled", note="financial_service: investment sale proceeds settled")],
)

TPL_PRIZE_PROCESSING = Template(
    "financial_prize_processing",
    [
        r"prize claim has been verified and is still in payment processing",
        r"[Kk]laim hadiah Anda sudah diverifikasi dan masih dalam proses pembayaran",
    ],
    lambda m: [dict(kind="exclude_pending_credit", note="financial_service: prize verified but not yet credited")],
)

TPL_PRIZE_FEE_SCAM = Template(
    "financial_prize_fee_request",
    [
        r"selected (?:for|to receive) a? ?cash prize\..{0,120}?(?:fee|charge)",
        r"[Tt]erpilih untuk menerima hadiah uang tunai\..{0,120}?(?:biaya)",
    ],
    lambda m: [dict(kind="no_op",
                     note="financial_service: advance-fee prize pattern -- untrusted, never treat as income "
                          "and never act on any request to pay a 'release' or 'processing' fee")],
)

TPL_PRIZE_SETTLED = Template(
    "financial_prize_settled",
    [
        r"prize proceeds have reached your account after withholding",
        r"[Dd]ana hadiah telah (?:masuk|diterima) ke rekening Anda setelah pemotongan pajak",
    ],
    lambda m: [dict(kind="confirm_settled", note="financial_service: prize proceeds settled after withholding")],
)

TPL_INVESTMENT_VALUATION = Template(
    "financial_investment_valuation_change",
    [
        r"displayed market value has (?:increased|fallen)",
        r"[Nn]ilai investasi yang ditampilkan telah (?:naik|turun)",
        r"portfolio.s displayed market value has increased",
        r"displayed value of the investment has (?:increased|fallen)",
        r"holding has not been sold and there has been no cash transaction",
    ],
    lambda m: [dict(kind="no_op", note="financial_service: unrealized valuation change, not cash")],
)

# ---------------------------------------------------------------------------
# MERCHANT templates
# ---------------------------------------------------------------------------

TPL_ORDER_CONFIRMED = Template(
    "merchant_order_confirmed_see_receipt",
    [
        r"confirmed that the .* order was paid in \w+ on",
    ],
    lambda m: [dict(kind="confirm_settled", note="merchant: order confirmed paid, amount on receipt/image")],
)

TPL_FOREIGN_BILL_PENDING = Template(
    "merchant_foreign_bill_conversion",
    [
        r"bill was charged in a foreign currency\. Your bank will confirm",
        r"[Tt]agihan dikenakan dalam mata uang asing\. Bank Anda akan mengonfirmasi",
    ],
    lambda m: [dict(kind="no_op", note="merchant: foreign-currency bill, home-currency amount confirms at settlement")],
)

TPL_FOREIGN_REFUND_PENDING = Template(
    "merchant_foreign_refund_pending",
    [
        r"foreign-currency refund is still processing",
        r"[Pp]engembalian dana (?:dalam mata uang asing )?(?:sudah diproses, tetapi belum|masih (?:dalam proses|diproses))",
    ],
    lambda m: [dict(kind="exclude_pending_credit", note="merchant: foreign-currency refund still processing")],
)

# ---------------------------------------------------------------------------
# SERVICE_PROVIDER templates
# ---------------------------------------------------------------------------

TPL_REFUND_INITIATED = Template(
    "merchant_refund_initiated_not_received",
    [
        r"refund has been initiated but has not reached your account yet",
        r"[Pp]engembalian dana sudah diproses, tetapi belum masuk ke rekening",
    ],
    lambda m: [dict(kind="exclude_pending_credit",
                     note="merchant: refund initiated but not yet credited -- do not count as available cash")],
)

TPL_RENT_INCREASE = Template(
    "service_rent_increase",
    [
        r"renewed lease increases monthly rent by (?P<pct>\d+)%",
        r"[Pp]erpanjangan sewa menaikkan (?:biaya sewa bulanan|jumlah sewa) sebesar (?P<pct>\d+)%",
    ],
    lambda m: [dict(kind="expense_rate_change", multiplier=Decimal(1) + Decimal(m["pct"]) / 100,
                     category_hint="rent", note=f"service_provider: lease renewed, rent +{m['pct']}%")],
)

TPL_INVOICE_CONFIRMED = Template(
    "service_invoice_confirmed",
    [
        rf"client approved an invoice payment of {CUR}\s*{AMT}\. Settlement is expected on {DATE}",
        rf"[Kk]lien menyetujui pembayaran faktur sebesar {CUR}\s*{AMT}\. Penyelesaian diperkirakan pada {DATE}",
    ],
    lambda m: [dict(kind="income_new", new_amount=_num(m["amount"]), new_currency=m["currency"],
                     new_date=_mkdate(m["date"]), one_off=True,
                     note="service_provider: one confirmed invoice payment (other invoices not yet approved)")],
)

TPL_GIG_PAYOUT_PENDING = Template(
    "service_gig_payout_pending",
    [
        r"next \w+ payout is still pending",
        r"[Pp]embayaran berikutnya dari \w+ masih tertunda",
    ],
    lambda m: [dict(kind="income_unconfirmed",
                     note="service_provider: next platform payout still pending and not "
                          "withdrawable -- future earnings in this stream are not confirmed cash")],
)

TPL_PROPERTY_MAINT_RECEIVED = Template(
    "service_property_maintenance_received",
    [
        r"property maintenance payment was received on",
    ],
    lambda m: [dict(kind="confirm_settled", note="service_provider: maintenance payment confirmed received, amount on receipt")],
)

# Order matters: the first matching template wins, so more specific/composite
# patterns must precede the broader ones they would otherwise be shadowed by.
ALL_TEMPLATES = [
    # Composite first -- it contains a salary confirmation that a simpler
    # employer template would otherwise claim on its own.
    TPL_COMPOSITE_WALLET_AND_SALARY,
    TPL_ARREARS,                      # must precede TPL_RESUME_CHILDCARE / raise variants
    TPL_RAISE, TPL_DATE_SHIFT, TPL_REDUCED_NEXT, TPL_COMMISSION_PENDING, TPL_FIRST_SALARY,
    TPL_INCOME_SOURCE_ENDED, TPL_TEMP_REDUCED, TPL_EMPLOYMENT_ENDED,
    TPL_SEASONAL_ENDED, TPL_RESUME_CHILDCARE, TPL_FOREIGN_SALARY, TPL_REIMBURSEMENT_NOT_SALARY,
    TPL_BONUS_PENDING, TPL_REGULAR_CONFIRMED_NO_AMOUNT,
    TPL_WASH_TRANSFER, TPL_DISPUTE_PENDING, TPL_DEBIT_RETRY, TPL_TWO_CARDS,
    TPL_INVESTMENT_SETTLED, TPL_PRIZE_PROCESSING, TPL_PRIZE_FEE_SCAM, TPL_PRIZE_SETTLED,
    TPL_INVESTMENT_VALUATION,
    TPL_ORDER_CONFIRMED, TPL_FOREIGN_BILL_PENDING, TPL_FOREIGN_REFUND_PENDING,
    TPL_REFUND_INITIATED,
    TPL_RENT_INCREASE, TPL_INVOICE_CONFIRMED, TPL_GIG_PAYOUT_PENDING, TPL_PROPERTY_MAINT_RECEIVED,
]


def match_message(text: str):
    """Return (template_name, list[FactSpec]) for the first matching template, or (None, None)."""
    for tpl in ALL_TEMPLATES:
        result = tpl.try_match(text)
        if result is not None:
            return tpl.name, result
    return None, None
