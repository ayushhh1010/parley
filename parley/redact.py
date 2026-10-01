"""Mask personal data in transcripts before any text leaves the machine (PRD F3).

ponytail: text only. Muting the matching audio needs word timestamps and an audio
pipeline; add it with the MVP's normalize stage.
ponytail: Hindi number words (ek, do / एक, दो) and spelled-out PAN letters are not
detected; add them when partner transcripts show them.
"""
import re
import unicodedata

from .transcribe import Call

WORD_DIGITS = {"zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "three": "3", "four": "4",
               "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}
REPEAT = {"double": 2, "triple": 3}
TOKEN = re.compile(r"\+?\d+|[A-Za-z]+")
SEP = re.compile(r"[\s,.\-()/]*")
CARD_CONTEXT = re.compile(r"\b(card|credit|debit|visa|master ?card|rupay|amex|expiry)\b", re.I)
CVV_CONTEXT = re.compile(r"\b(cvv|cvc|security code|three digits?|back of (the|your) card)\b", re.I)
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b|\b\w+ at \w+ dot (com|in|net|org|co)\b", re.I)
PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", re.I)

_D = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
      [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
      [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
      [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
_P = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
      [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
      [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]]


def luhn(d: str) -> bool:
    total = 0
    for i, c in enumerate(reversed(d)):
        n = int(c) * (2 if i % 2 else 1)
        total += n - 9 if n > 9 else n
    return total % 10 == 0


def verhoeff(d: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(d)):
        c = _D[c][_P[i % 8][int(ch)]]
    return c == 0


def digit_runs(text):
    """(start, end, digits) for each run of written or spoken digits ("four double one" -> "411")."""
    runs, cur, mult, mult_start = [], None, 1, None
    for m in TOKEN.finditer(text):
        word = m.group().lower().lstrip("+")
        if word in REPEAT:
            mult, mult_start = REPEAT[word], m.start()
            continue
        if word.isdigit():
            d = "".join(str(unicodedata.digit(c)) for c in word)  # also Devanagari digits
        else:
            d = WORD_DIGITS.get(word)
        if d is None:
            if cur:
                runs.append(cur)
            cur, mult, mult_start = None, 1, None
            continue
        start = m.start() if mult_start is None else mult_start
        d = d * mult if len(d) == 1 else d
        mult, mult_start = 1, None
        if cur and SEP.fullmatch(text[cur[1]:start]):
            cur[1], cur[2] = m.end(), cur[2] + d
        else:
            if cur:
                runs.append(cur)
            cur = [start, m.end(), d]
    if cur:
        runs.append(cur)
    return [tuple(r) for r in runs]


def _kind(d: str, card_ctx: bool, cvv_ctx: bool) -> str | None:
    n = len(d)
    if n == 12 and d[0] in "23456789" and verhoeff(d):
        return "AADHAAR"
    if 13 <= n <= 19 and luhn(d):
        return "CARD"
    if (n == 10 and d[0] in "6789") or (n == 12 and d.startswith("91") and d[2] in "6789"):
        return "PHONE"
    if n >= 9:  # account numbers, IDs, mis-heard cards: safer masked than leaked
        return "NUMBER"
    if cvv_ctx and n in (3, 4):
        return "CVV"
    if card_ctx and n in (4, 8):  # card numbers read out in groups, split across turns
        return "CARD"
    return None


def redact_text(text: str, card_ctx=False, cvv_ctx=False) -> tuple[str, list[str]]:
    card_ctx = card_ctx or bool(CARD_CONTEXT.search(text))
    cvv_ctx = cvv_ctx or bool(CVV_CONTEXT.search(text))
    spans = [(s, e, k) for s, e, d in digit_runs(text) if (k := _kind(d, card_ctx, cvv_ctx))]
    spans += [(m.start(), m.end(), "EMAIL") for m in EMAIL.finditer(text)]
    spans += [(m.start(), m.end(), "PAN") for m in PAN.finditer(text)]
    out, last, kinds = [], 0, []
    for s, e, k in sorted(spans, key=lambda x: (x[0], -x[1])):
        if s < last:
            continue
        out += [text[last:s], f"[{k}]"]
        last = e
        kinds.append(k)
    return "".join(out) + text[last:], kinds


def redact_call(call: Call) -> tuple[Call, dict[str, int]]:
    """Redacted copy of the call plus counts per kind. Payment context carries over from the previous turns."""
    turns, counts = [], {}
    for i, t in enumerate(call.turns):
        before = " ".join(p.text for p in call.turns[max(0, i - 2):i])
        text, kinds = redact_text(t.text, bool(CARD_CONTEXT.search(before)),
                                  bool(CVV_CONTEXT.search(call.turns[i - 1].text)) if i else False)
        turns.append(t.model_copy(update={"text": text}))
        for k in kinds:
            counts[k] = counts.get(k, 0) + 1
    return call.model_copy(update={"turns": turns}), counts
