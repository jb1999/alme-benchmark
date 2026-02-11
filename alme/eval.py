#!/usr/bin/env python3
"""
Evaluate audio-LLM on ALME stimuli.

Runs the model through four conditions:
  - audio_only: audio input only
  - text_only: transcript text only
  - audio_text_aligned: audio + correct transcript (multimodal ceiling)
  - audio_text_conflict: audio + modified transcript (conflict scenario)

Supports checkpointing for long runs (per-condition JSONL checkpoints).
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any


@dataclass
class TrialResult:
    """Result of a single trial."""

    stimulus_id: str
    condition: str  # "audio_only", "text_only", "audio_text_aligned", "audio_text_conflict"
    expected_answer: str
    model_answer: Optional[str]
    raw_response: str
    latency_sec: float
    correct: bool
    model: str = ""
    language: str = ""
    followed_text: Optional[bool] = None  # For conflict condition
    flip_type: str = ""
    metadata: Dict = field(default_factory=dict)


class EvalMonitor:
    """Live monitoring for eval runs.

    Tracks per-language and per-condition error rates, parse failures,
    refusals, and empty responses.  Issues warnings when error rates
    exceed a configurable threshold over a rolling window.
    """

    _REFUSAL_KEYWORDS = [
        "i cannot", "i can't", "i'm unable", "i am unable",
        "content policy", "i apologize", "i'm sorry, i",
        "not able to process", "cannot process audio",
        "unable to analyze", "not supported",
        "as an ai", "as a language model",
    ]

    def __init__(
        self,
        error_rate_threshold: float = 0.3,
        window_size: int = 100,
    ):
        self._threshold = error_rate_threshold
        self._window = window_size
        # Per-language counters
        self._by_lang: Dict[str, Dict[str, int]] = {}
        # Per-condition counters
        self._by_cond: Dict[str, Dict[str, int]] = {}
        # Totals
        self._total = 0
        self._errors = 0
        self._parse_failures = 0
        self._refusals = 0
        self._empty_responses = 0
        self._api_errors = 0
        # Rolling window for circuit breaker (True = ok, False = error)
        self._recent: List[bool] = []
        self._warnings_issued: set = set()

    # -- internal helpers ---------------------------------------------------

    def _counters(self, bucket: dict, key: str) -> Dict[str, int]:
        if key not in bucket:
            bucket[key] = {
                "total": 0, "errors": 0, "parse_failures": 0,
                "refusals": 0, "empty": 0, "api_errors": 0,
            }
        return bucket[key]

    def _is_refusal(self, text: str) -> bool:
        low = text.lower()
        return any(kw in low for kw in self._REFUSAL_KEYWORDS)

    # -- public API ---------------------------------------------------------

    def record(self, result: "TrialResult", condition: str):
        """Record a single trial result."""
        self._total += 1
        lang = result.language or "UNKNOWN"
        lc = self._counters(self._by_lang, lang)
        cc = self._counters(self._by_cond, condition)
        lc["total"] += 1
        cc["total"] += 1

        raw = result.raw_response
        is_error = False

        # Check for adapter-level API error
        if result.metadata.get("error"):
            self._api_errors += 1
            lc["api_errors"] += 1
            cc["api_errors"] += 1
            is_error = True
        # Empty response (OOM, timeout, crash)
        elif not raw.strip():
            self._empty_responses += 1
            lc["empty"] += 1
            cc["empty"] += 1
            is_error = True
        # Refusal (content policy, "I cannot", etc.)
        elif self._is_refusal(raw):
            self._refusals += 1
            lc["refusals"] += 1
            cc["refusals"] += 1
            is_error = True
        # Parse failure (non-empty response but no answer extracted)
        elif result.model_answer is None:
            self._parse_failures += 1
            lc["parse_failures"] += 1
            cc["parse_failures"] += 1
            is_error = True

        if is_error:
            self._errors += 1
            lc["errors"] += 1
            cc["errors"] += 1

        # Rolling window
        self._recent.append(not is_error)
        if len(self._recent) > self._window:
            self._recent.pop(0)

        self._check_alerts(lang)

    def record_batch(self, results: List["TrialResult"], condition: str):
        """Record a batch of trial results."""
        for r in results:
            self.record(r, condition)

    def _check_alerts(self, lang: str):
        """Emit warnings when error rates cross thresholds."""
        # Rolling-window alert
        if len(self._recent) >= self._window:
            recent_errors = sum(1 for ok in self._recent if not ok)
            rate = recent_errors / len(self._recent)
            if rate >= self._threshold:
                key = f"rolling_{self._total // self._window}"
                if key not in self._warnings_issued:
                    self._warnings_issued.add(key)
                    print(
                        f"\n  WARNING [monitor]: High error rate in last "
                        f"{self._window} trials: {rate:.0%} "
                        f"({recent_errors}/{len(self._recent)})"
                    )

        # Per-language alert (check every 50 trials per lang)
        lc = self._by_lang.get(lang, {})
        n = lc.get("total", 0)
        if n >= 20:
            lang_rate = lc["errors"] / n
            if lang_rate >= self._threshold:
                key = f"lang_{lang}_{n // 50}"
                if key not in self._warnings_issued:
                    self._warnings_issued.add(key)
                    parts = []
                    if lc["empty"]:
                        parts.append(f"{lc['empty']} empty")
                    if lc["refusals"]:
                        parts.append(f"{lc['refusals']} refusals")
                    if lc["parse_failures"]:
                        parts.append(f"{lc['parse_failures']} parse-fail")
                    if lc["api_errors"]:
                        parts.append(f"{lc['api_errors']} api-err")
                    print(
                        f"\n  WARNING [monitor]: {lang} error rate "
                        f"{lang_rate:.0%} ({lc['errors']}/{n})"
                        f" — {', '.join(parts)}"
                    )

    def summary(self) -> Dict[str, Any]:
        """Return monitoring summary as a dict (serialisable to JSON)."""
        return {
            "total_trials": self._total,
            "total_errors": self._errors,
            "error_rate": round(self._errors / self._total, 4) if self._total else 0,
            "empty_responses": self._empty_responses,
            "parse_failures": self._parse_failures,
            "refusals": self._refusals,
            "api_errors": self._api_errors,
            "by_language": {k: dict(v) for k, v in self._by_lang.items()},
            "by_condition": {k: dict(v) for k, v in self._by_cond.items()},
        }

    def print_summary(self):
        """Print a human-readable monitoring summary."""
        if self._total == 0:
            return
        rate = self._errors / self._total
        print(f"\n{'=' * 60}")
        print(f"MONITORING SUMMARY ({self._total} trials)")
        print("=" * 60)
        print(f"  Overall error rate: {rate:.1%} ({self._errors}/{self._total})")
        print(f"    Empty responses:  {self._empty_responses}")
        print(f"    Parse failures:   {self._parse_failures}")
        print(f"    Refusals:         {self._refusals}")
        print(f"    API errors:       {self._api_errors}")

        problem_langs = {
            lang: c for lang, c in sorted(self._by_lang.items())
            if c["errors"] > 0
        }
        if problem_langs:
            print(f"\n  Per-language errors:")
            for lang, c in problem_langs.items():
                parts = []
                if c["empty"]:
                    parts.append(f"empty={c['empty']}")
                if c["refusals"]:
                    parts.append(f"refusal={c['refusals']}")
                if c["parse_failures"]:
                    parts.append(f"parse={c['parse_failures']}")
                if c["api_errors"]:
                    parts.append(f"api_err={c['api_errors']}")
                print(
                    f"    {lang}: {c['errors']}/{c['total']} "
                    f"({c['errors'] / c['total']:.0%})"
                    f" — {', '.join(parts)}"
                )


def _is_cjk(s: str) -> bool:
    """Return True if s consists entirely of CJK Unified Ideograph characters."""
    return all("\u4e00" <= ch <= "\u9fff" or "\u3040" <= ch <= "\u30ff" for ch in s)


# ---------------------------------------------------------------------------
# Multilingual native→English translation map for choice matching.
# Maps native-language choice words to their English equivalents so that
# when a model responds in English, we can match against native choices.
# ---------------------------------------------------------------------------

_NATIVE_TO_ENGLISH: Dict[str, str] = {}

def _build_native_map() -> Dict[str, str]:
    """Build comprehensive native→English mapping for all 8 languages."""
    m: Dict[str, str] = {}

    # Numbers (DE, FR, IT, PT, AR, JA, ZH)
    _numbers = {
        "en": ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"],
        "de": ["eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun", "zehn"],
        "fr": ["un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix"],
        "it": ["uno", "due", "tre", "quattro", "cinque", "sei", "sette", "otto", "nove", "dieci"],
        "pt": ["um", "dois", "três", "quatro", "cinco", "seis", "sete", "oito", "nove", "dez"],
        "ar": ["واحد", "اثنان", "ثلاثة", "أربعة", "خمسة", "ستة", "سبعة", "ثمانية", "تسعة", "عشرة"],
        "ja": ["一つ", "二つ", "三つ", "四つ", "五つ", "六つ", "七つ", "八つ", "九つ", "十"],
        "zh": ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"],
    }
    en_nums = _numbers["en"]
    for lang, words in _numbers.items():
        if lang == "en":
            continue
        for native, english in zip(words, en_nums):
            m[native.lower()] = english

    # Ordinals
    _ordinals = {
        "de": {"erste": "first", "zweite": "second", "dritte": "third", "ersten": "first",
               "zweiten": "second", "dritten": "third"},
        "fr": {"premier": "first", "deuxième": "second", "troisième": "third",
               "première": "first"},
        "it": {"primo": "first", "secondo": "second", "terzo": "third",
               "prima": "first", "seconda": "second", "terza": "third"},
        "pt": {"primeiro": "first", "segundo": "second", "terceiro": "third",
               "primeira": "first", "segunda": "second", "terceira": "third"},
        "zh": {"第一": "first", "第二": "second", "第三": "third"},
        "ja": {"第一": "first", "第二": "second", "第三": "third",
               "一番目": "first", "二番目": "second"},
    }
    for lang_map in _ordinals.values():
        for native, english in lang_map.items():
            m[native.lower()] = english

    # Adjective antonym pairs
    _adj_pairs = {
        "de": {"groß": "big", "klein": "small", "alt": "old", "neu": "new", "jung": "young",
               "heiß": "hot", "kalt": "cold", "schnell": "fast", "langsam": "slow",
               "gut": "good", "schlecht": "bad", "hoch": "high", "niedrig": "low",
               "lang": "long", "kurz": "short", "leicht": "easy", "schwer": "hard",
               "hell": "light", "dunkel": "dark", "stark": "strong", "schwach": "weak",
               "reich": "rich", "arm": "poor", "sicher": "safe", "gefährlich": "dangerous",
               "sauber": "clean", "schmutzig": "dirty", "offen": "open", "geschlossen": "closed",
               "glücklich": "happy", "traurig": "sad"},
        "fr": {"grand": "big", "petit": "small", "vieux": "old", "nouveau": "new",
               "jeune": "young", "chaud": "hot", "froid": "cold", "rapide": "fast",
               "lent": "slow", "bon": "good", "mauvais": "bad", "haut": "high",
               "bas": "low", "long": "long", "court": "short", "facile": "easy",
               "difficile": "hard", "clair": "light", "sombre": "dark",
               "fort": "strong", "faible": "weak", "riche": "rich", "pauvre": "poor",
               "propre": "clean", "sale": "dirty", "ouvert": "open", "fermé": "closed",
               "heureux": "happy", "triste": "sad", "grande": "big", "petite": "small",
               "vieille": "old", "nouvelle": "new"},
        "it": {"grande": "big", "piccolo": "small", "vecchio": "old", "nuovo": "new",
               "giovane": "young", "caldo": "hot", "freddo": "cold", "veloce": "fast",
               "lento": "slow", "buono": "good", "cattivo": "bad", "alto": "high",
               "basso": "low", "lungo": "long", "corto": "short", "facile": "easy",
               "difficile": "hard", "chiaro": "light", "scuro": "dark",
               "forte": "strong", "debole": "weak", "ricco": "rich", "povero": "poor",
               "pulito": "clean", "sporco": "dirty", "aperto": "open", "chiuso": "closed",
               "felice": "happy", "triste": "sad"},
        "pt": {"grande": "big", "pequeno": "small", "velho": "old", "novo": "new",
               "jovem": "young", "quente": "hot", "frio": "cold", "rápido": "fast",
               "lento": "slow", "bom": "good", "mau": "bad", "alto": "high",
               "baixo": "low", "longo": "long", "curto": "short", "fácil": "easy",
               "difícil": "hard", "claro": "light", "escuro": "dark",
               "forte": "strong", "fraco": "weak", "rico": "rich", "pobre": "poor",
               "limpo": "clean", "sujo": "dirty", "aberto": "open", "fechado": "closed",
               "feliz": "happy", "triste": "sad"},
        "ar": {"كبير": "big", "صغير": "small", "قديم": "old", "جديد": "new",
               "شاب": "young", "حار": "hot", "بارد": "cold", "سريع": "fast",
               "بطيء": "slow", "جيد": "good", "سيء": "bad", "عالي": "high",
               "منخفض": "low", "طويل": "long", "قصير": "short",
               "سهل": "easy", "صعب": "hard", "قوي": "strong", "ضعيف": "weak",
               "غني": "rich", "فقير": "poor", "سعيد": "happy", "حزين": "sad"},
        "zh": {"大": "big", "小": "small", "旧": "old", "新": "new",
               "年轻": "young", "热": "hot", "冷": "cold", "快": "fast",
               "慢": "slow", "好": "good", "坏": "bad", "高": "high",
               "低": "low", "长": "long", "短": "short", "容易": "easy",
               "困难": "difficult", "对": "right", "错": "wrong",
               "开": "open", "关": "closed", "满": "full", "空": "empty",
               "高兴": "happy", "难过": "sad", "强": "strong", "弱": "weak"},
        "ja": {"大きい": "big", "小さい": "small", "古い": "old", "新しい": "new",
               "若い": "young", "暑い": "hot", "寒い": "cold", "速い": "fast",
               "遅い": "slow", "良い": "good", "悪い": "bad", "高い": "high",
               "低い": "low", "長い": "long", "短い": "short",
               "簡単": "easy", "難しい": "hard", "強い": "strong", "弱い": "weak"},
    }
    for lang_map in _adj_pairs.values():
        for native, english in lang_map.items():
            m[native.lower()] = english

    # Time expressions
    _time = {
        "de": {"morgen": "morning", "abend": "evening", "gestern": "yesterday",
               "heute": "today", "morgens": "morning", "abends": "evening",
               "früh": "early", "spät": "late", "vorher": "before", "nachher": "after",
               "immer": "always", "nie": "never", "oft": "often", "selten": "rarely"},
        "fr": {"matin": "morning", "soir": "evening", "hier": "yesterday",
               "aujourd'hui": "today", "demain": "tomorrow",
               "tôt": "early", "tard": "late", "avant": "before", "après": "after",
               "toujours": "always", "jamais": "never", "souvent": "often",
               "rarement": "rarely"},
        "it": {"mattina": "morning", "sera": "evening", "ieri": "yesterday",
               "oggi": "today", "domani": "tomorrow",
               "presto": "early", "tardi": "late", "prima": "before", "dopo": "after",
               "sempre": "always", "mai": "never", "spesso": "often",
               "raramente": "rarely"},
        "pt": {"manhã": "morning", "noite": "evening", "ontem": "yesterday",
               "hoje": "today", "amanhã": "tomorrow",
               "cedo": "early", "tarde": "late", "antes": "before", "depois": "after",
               "sempre": "always", "nunca": "never", "frequentemente": "often",
               "raramente": "rarely"},
        "ar": {"صباح": "morning", "مساء": "evening", "أمس": "yesterday",
               "اليوم": "today", "غداً": "tomorrow", "غدا": "tomorrow",
               "قبل": "before", "بعد": "after",
               "دائماً": "always", "دائما": "always", "أبداً": "never", "أبدا": "never"},
        "zh": {"早上": "morning", "晚上": "evening", "昨天": "yesterday",
               "今天": "today", "明天": "tomorrow",
               "之前": "before", "之后": "after",
               "总是": "always", "从不": "never", "经常": "often", "很少": "rarely"},
        "ja": {"朝": "morning", "夜": "evening", "昨日": "yesterday",
               "今日": "today", "明日": "tomorrow",
               "前": "before", "後": "after",
               "いつも": "always", "決して": "never"},
    }
    for lang_map in _time.values():
        for native, english in lang_map.items():
            m[native.lower()] = english

    # Negation / boolean
    _negation = {
        "de": {"ja": "yes", "nein": "no", "positiv": "positive", "negativ": "negative",
               "wahr": "true", "falsch": "false", "kann": "can", "kann nicht": "cannot",
               "nicht": "not", "wieder": "again", "nicht mehr": "no longer"},
        "fr": {"oui": "yes", "non": "no", "positif": "positive", "négatif": "negative",
               "vrai": "true", "faux": "false", "peut": "can", "ne peut pas": "cannot",
               "présence": "presence", "absence": "absence"},
        "it": {"sì": "yes", "no": "no", "positivo": "positive", "negativo": "negative",
               "vero": "true", "falso": "false", "può": "can", "non può": "cannot"},
        "pt": {"sim": "yes", "não": "no", "positivo": "positive", "negativo": "negative",
               "verdadeiro": "true", "falso": "false", "pode": "can", "não pode": "cannot"},
        "ar": {"نعم": "yes", "لا": "no", "إيجابي": "positive", "سلبي": "negative",
               "صحيح": "true", "خاطئ": "false"},
        "zh": {"是": "yes", "不是": "no", "否": "no", "会": "can", "不会": "cannot",
               "有": "have", "没有": "not have", "能": "can", "不能": "cannot",
               "可以": "can", "不可以": "cannot",
               "正面": "positive", "负面": "negative"},
        "ja": {"はい": "yes", "いいえ": "no", "できる": "can", "できない": "cannot",
               "ある": "have", "ない": "not have"},
    }
    for lang_map in _negation.values():
        for native, english in lang_map.items():
            m[native.lower()] = english

    return m

_NATIVE_TO_ENGLISH = _build_native_map()


def _build_english_to_choice(choices: List[str]) -> Dict[str, str]:
    """Build English→choice mapping for the given native-language choices."""
    en_to_choice: Dict[str, str] = {}
    for choice in choices:
        cl = choice.lower().strip()
        if cl in _NATIVE_TO_ENGLISH:
            en = _NATIVE_TO_ENGLISH[cl]
            en_to_choice[en] = choice
    return en_to_choice


def parse_answer(response: str, choices: List[str]) -> Optional[str]:
    """Extract answer from response.

    Handles multilingual choices: if the model responds in English but
    choices are in a native language, uses a translation map to match.
    """
    import re as _re
    import unicodedata

    # Guard against catastrophic regex backtracking on degenerate model
    # output (e.g. Ultravox repeating Arabic fragments to fill the token
    # budget).  Valid JSON answers are always <500 chars; 2000 is generous.
    if len(response) > 2000:
        response = response[:2000]

    # Normalize a string for comparison: strip, lower, normalize quotes,
    # remove leading punctuation
    def _clean(s: str) -> str:
        s = s.strip().lower()
        # Normalize curly/typographic apostrophes and quotes to ASCII
        s = s.replace("\u2019", "'").replace("\u2018", "'")
        s = s.replace("\u00b4", "'")  # acute accent used as apostrophe
        s = s.replace("\u201c", '"').replace("\u201d", '"')
        # Collapse repeated apostrophes
        while "''" in s:
            s = s.replace("''", "'")
        # Strip leading CJK/Latin punctuation
        while s and unicodedata.category(s[0]).startswith("P"):
            s = s[1:]
        return s.strip()

    def _strip_diacritics(s: str) -> str:
        """Strip diacritics and normalize ß→ss for accent-insensitive matching."""
        s = s.replace("ß", "ss")
        decomposed = unicodedata.normalize("NFKD", s)
        return "".join(ch for ch in decomposed if not unicodedata.category(ch).startswith("M"))

    # Build English→choice translation map for current choices
    en_to_choice = _build_english_to_choice(choices)

    # Try JSON parse, then ast.literal_eval, then regex extraction
    json_answer = None
    data = None
    try:
        data = json.loads(response.strip())
    except json.JSONDecodeError:
        import ast
        try:
            data = ast.literal_eval(response.strip())
        except (ValueError, SyntaxError):
            m = _re.search(
                r"""['"]answer['"]\s*:\s*['"]([^'"]{0,500})['"]""",
                response,
            )
            if m:
                raw_val = m.group(1)
                try:
                    raw_val = raw_val.encode('utf-8').decode('unicode_escape')
                except (UnicodeDecodeError, UnicodeError):
                    pass
                data = {"answer": raw_val}
    if isinstance(data, dict):
        raw = data.get("answer")
        if raw:
            raw_str = str(raw)
            if "\\u" in raw_str:
                raw_str = _re.sub(
                    r"\\u([0-9a-fA-F]{4})",
                    lambda m: chr(int(m.group(1), 16)),
                    raw_str,
                )
            cleaned = _clean(raw_str)
            # Exact match
            for choice in choices:
                if _clean(choice) == cleaned:
                    return choice
            # Strip common prefixes like "choice" (model artifact)
            for prefix in ("choice", "选项"):
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
                    for choice in choices:
                        if _clean(choice) == cleaned:
                            return choice
            # English→native translation
            if cleaned in en_to_choice:
                return en_to_choice[cleaned]
            # Synonym resolution
            if cleaned in _NATIVE_TO_ENGLISH:
                en_word = _NATIVE_TO_ENGLISH[cleaned]
                if en_word in en_to_choice:
                    return en_to_choice[en_word]
            # Check if the answer starts with or contains a choice.
            for choice in sorted(choices, key=len, reverse=True):
                if cleaned.startswith(_clean(choice)):
                    return choice
            for choice in sorted(choices, key=len, reverse=True):
                if _clean(choice) in cleaned:
                    return choice
            json_answer = raw_str

    # Exact match (stripped response equals a choice)
    response_stripped = response.strip()
    for choice in choices:
        if choice == response_stripped or _clean(choice) == _clean(response_stripped):
            return choice

    # Substring match — sort choices longest-first to avoid partial CJK hits
    search_texts = [_clean(response)]
    if json_answer:
        search_texts.append(_clean(json_answer))
    for search_text in search_texts:
        for choice in sorted(choices, key=len, reverse=True):
            cl = _clean(choice)
            if len(cl) <= 2 and _is_cjk(cl):
                import re as _re2
                pattern = (
                    r"(?<![\u4e00-\u9fff\u3040-\u30ff\u31f0-\u31ff])"
                    + _re2.escape(cl)
                    + r"(?![\u4e00-\u9fff\u3040-\u30ff\u31f0-\u31ff])"
                )
                if _re2.search(pattern, search_text):
                    return choice
            else:
                if cl in search_text:
                    return choice

    # Reverse substring
    if json_answer:
        ja_clean = _clean(json_answer)
        if len(ja_clean) >= 4:
            for choice in sorted(choices, key=len, reverse=True):
                cl = _clean(choice)
                if ja_clean in cl and len(ja_clean) >= len(cl) * 0.6:
                    return choice

    # English translation substring match
    for search_text in search_texts:
        for en_word, choice in sorted(en_to_choice.items(), key=lambda x: len(x[0]), reverse=True):
            import re as _re3
            if _re3.search(r'\b' + _re3.escape(en_word) + r'\b', search_text):
                return choice

    # Diacritic-insensitive fallback
    for search_text in search_texts:
        stripped_text = _strip_diacritics(search_text)
        for choice in sorted(choices, key=len, reverse=True):
            if _strip_diacritics(_clean(choice)) == stripped_text:
                return choice
            if _strip_diacritics(_clean(choice)) in stripped_text:
                return choice

    return None


def build_system_prompt() -> str:
    """Build system prompt for evaluation."""
    return """You are an expert speech and language analyst. Answer the question based on the provided inputs.

RESPONSE FORMAT:
- Output ONLY valid JSON: {"answer": "<choice>", "confidence": 0.0-1.0, "rationale": "brief"}
- Your "answer" MUST be EXACTLY one of the provided choices, copied verbatim.
- Use the SAME language and script as the choices (e.g. if choices are in Chinese, answer in Chinese).
- Do NOT translate, paraphrase, or reformat the choice. Copy it character-for-character.
- Do not include any extra text or markdown formatting."""


def run_trial(
    adapter,
    stimulus: Dict,
    condition: str,
) -> TrialResult:
    """Run a single trial using a model adapter."""
    from .models.base import ModelResponse

    question = stimulus["question"]
    choices = question["choices"]
    correct_answer = question["correct_answer"]
    system_prompt = build_system_prompt()
    model_id = adapter.model_id
    language = stimulus.get("language", stimulus.get("accent_bucket", "EN"))
    flip_type = question.get("flip_type", "")

    start = time.time()

    if condition == "audio_only":
        response = adapter.run(
            audio_path=stimulus["audio_path"],
            text=None,
            question=question["question_text"],
            choices=choices,
            system_prompt=system_prompt,
            condition=condition,
        )
        latency = time.time() - start
        model_answer = parse_answer(response.raw_output, choices)
        correct = model_answer == correct_answer

        return TrialResult(
            stimulus_id=stimulus["stimulus_id"],
            condition=condition,
            expected_answer=correct_answer,
            model_answer=model_answer,
            raw_response=response.raw_output,
            latency_sec=latency,
            correct=correct,
            model=model_id,
            language=language,
            flip_type=flip_type,
        )

    elif condition == "text_only":
        response = adapter.run(
            audio_path=None,
            text=stimulus["ref_text"],
            question=question["question_text"],
            choices=choices,
            system_prompt=system_prompt,
            condition=condition,
        )
        latency = time.time() - start
        model_answer = parse_answer(response.raw_output, choices)
        correct = model_answer == correct_answer

        return TrialResult(
            stimulus_id=stimulus["stimulus_id"],
            condition=condition,
            expected_answer=correct_answer,
            model_answer=model_answer,
            raw_response=response.raw_output,
            latency_sec=latency,
            correct=correct,
            model=model_id,
            language=language,
            flip_type=flip_type,
        )

    elif condition == "audio_text_aligned":
        response = adapter.run(
            audio_path=stimulus["audio_path"],
            text=stimulus["ref_text"],
            question=question["question_text"],
            choices=choices,
            system_prompt=system_prompt,
            condition=condition,
        )
        latency = time.time() - start
        model_answer = parse_answer(response.raw_output, choices)
        correct = model_answer == correct_answer

        return TrialResult(
            stimulus_id=stimulus["stimulus_id"],
            condition=condition,
            expected_answer=correct_answer,
            model_answer=model_answer,
            raw_response=response.raw_output,
            latency_sec=latency,
            correct=correct,
            model=model_id,
            language=language,
            flip_type=flip_type,
        )

    elif condition == "audio_text_conflict":
        response = adapter.run(
            audio_path=stimulus["audio_path"],
            text=stimulus["conflict_text"],
            question=question["question_text"],
            choices=choices,
            system_prompt=system_prompt,
            condition=condition,
        )
        latency = time.time() - start
        model_answer = parse_answer(response.raw_output, choices)

        followed_audio = model_answer == correct_answer
        conflict_answer = next(
            (c for c in choices if c != correct_answer), None
        )
        followed_text = model_answer == conflict_answer

        return TrialResult(
            stimulus_id=stimulus["stimulus_id"],
            condition=condition,
            expected_answer=correct_answer,
            model_answer=model_answer,
            raw_response=response.raw_output,
            latency_sec=latency,
            correct=followed_audio,
            model=model_id,
            language=language,
            followed_text=followed_text,
            flip_type=flip_type,
            metadata={"conflict_answer": conflict_answer},
        )

    else:
        raise ValueError(f"Unknown condition: {condition}")


def _build_batch_items(stimuli_batch: List[Dict], condition: str) -> List[tuple]:
    """Build (audio_path, text, question, choices) tuples for a batch."""
    items = []
    for stim in stimuli_batch:
        q = stim["question"]
        if condition == "text_only":
            items.append((None, stim["ref_text"], q["question_text"], q["choices"]))
        elif condition == "audio_only":
            items.append((stim["audio_path"], None, q["question_text"], q["choices"]))
        elif condition == "audio_text_aligned":
            items.append((stim["audio_path"], stim["ref_text"],
                          q["question_text"], q["choices"]))
        elif condition == "audio_text_conflict":
            items.append((stim["audio_path"], stim["conflict_text"],
                          q["question_text"], q["choices"]))
        else:
            raise ValueError(f"Unknown condition: {condition}")
    return items


def _score_batch(
    stimuli_batch: List[Dict],
    responses: list,
    condition: str,
    model_id: str,
    batch_wall_sec: float,
) -> List[TrialResult]:
    """Convert a batch of model responses into TrialResult objects."""
    per_item_sec = batch_wall_sec / len(stimuli_batch) if stimuli_batch else 0
    results = []
    for stim, resp in zip(stimuli_batch, responses):
        q = stim["question"]
        choices = q["choices"]
        correct_answer = q["correct_answer"]
        language = stim.get("language", stim.get("accent_bucket", "EN"))
        flip_type = q.get("flip_type", "")
        model_answer = parse_answer(resp.raw_output, choices)

        if condition == "audio_text_conflict":
            followed_audio = model_answer == correct_answer
            conflict_answer = next(
                (c for c in choices if c != correct_answer), None
            )
            followed_text = model_answer == conflict_answer

            results.append(TrialResult(
                stimulus_id=stim["stimulus_id"],
                condition=condition,
                expected_answer=correct_answer,
                model_answer=model_answer,
                raw_response=resp.raw_output,
                latency_sec=per_item_sec,
                correct=followed_audio,
                model=model_id,
                language=language,
                followed_text=followed_text,
                flip_type=flip_type,
                metadata={"conflict_answer": conflict_answer},
            ))
        else:
            correct = model_answer == correct_answer
            results.append(TrialResult(
                stimulus_id=stim["stimulus_id"],
                condition=condition,
                expected_answer=correct_answer,
                model_answer=model_answer,
                raw_response=resp.raw_output,
                latency_sec=per_item_sec,
                correct=correct,
                model=model_id,
                language=language,
                flip_type=flip_type,
            ))
    return results


# ---------------------------------------------------------------------------
# Checkpointing helpers
# ---------------------------------------------------------------------------

def _checkpoint_path(checkpoint_dir: Path, model: str, condition: str) -> Path:
    """Return path for a condition checkpoint JSONL file."""
    safe_model = model.replace("/", "_")
    return checkpoint_dir / f"{safe_model}_{condition}.jsonl"


def _load_checkpoint(path: Path) -> List[TrialResult]:
    """Load TrialResults from a checkpoint JSONL file."""
    results = []
    if not path.exists():
        return results
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            results.append(TrialResult(
                stimulus_id=d["stimulus_id"],
                condition=d["condition"],
                expected_answer=d["expected_answer"],
                model_answer=d.get("model_answer"),
                raw_response=d["raw_response"],
                latency_sec=d["latency_sec"],
                correct=d["correct"],
                model=d.get("model", ""),
                language=d.get("language", ""),
                followed_text=d.get("followed_text"),
                flip_type=d.get("flip_type", ""),
                metadata=d.get("metadata", {}),
            ))
    return results


def _append_checkpoint(path: Path, results: List[TrialResult]):
    """Append TrialResults to a checkpoint JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def evaluate_stimuli(
    adapter,
    stimuli: List[Dict],
    conditions: List[str],
    output_path: str,
    max_stimuli: Optional[int] = None,
    batch_size: int = 1,
    checkpoint_dir: Optional[str] = None,
):
    """Evaluate model on stimuli across conditions.

    Runs ALL conditions on ALL eligible stimuli (no text-only filtering).
    Text-only results are collected as diagnostics; text-validated metrics
    are computed as a secondary sensitivity analysis in compute_metrics().

    Supports checkpointing: completed conditions are skipped on resume,
    and mid-condition progress is flushed every batch.
    """
    results = []
    system_prompt = build_system_prompt()
    monitor = EvalMonitor()

    # Set up checkpoint directory
    output_dir = Path(output_path).parent
    if checkpoint_dir:
        ckpt_dir = Path(checkpoint_dir)
    else:
        ckpt_dir = output_dir / ".checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Collect stimuli with valid audio
    eligible = []
    for stim in stimuli:
        if max_stimuli and len(eligible) >= max_stimuli:
            break
        if not Path(stim["audio_path"]).exists():
            continue
        eligible.append(stim)

    total = len(eligible)
    print(f"\n{total} stimuli with valid audio (max_stimuli={max_stimuli})")

    # ------------------------------------------------------------------
    # Run ALL conditions on ALL eligible stimuli
    # ------------------------------------------------------------------
    for condition in conditions:
        ckpt_path = _checkpoint_path(ckpt_dir, adapter.model_id, condition)

        # Check for completed checkpoint
        existing = _load_checkpoint(ckpt_path)
        if existing:
            good = [r for r in existing
                    if r.raw_response and r.model_answer is not None]
            error_ids = {r.stimulus_id for r in existing} - {r.stimulus_id for r in good}
            if error_ids:
                print(f"\n  Dropping {len(error_ids)} error results for retry")
                _path_tmp = ckpt_path.with_suffix(".tmp")
                _append_checkpoint(_path_tmp, good)
                _path_tmp.replace(ckpt_path)

            existing_ids = {r.stimulus_id for r in good}
            if existing_ids >= {s["stimulus_id"] for s in eligible}:
                print(f"\nSkipping {condition}: checkpoint has all {len(good)} results")
                results.extend(good)
                continue
            print(f"\nResuming {condition}: {len(good)} already done, "
                  f"continuing from there")
            results.extend(good)
            remaining = [s for s in eligible if s["stimulus_id"] not in existing_ids]
        else:
            remaining = eligible

        print(f"\nPhase: {condition} on {len(remaining)} stimuli "
              f"(batch_size={batch_size})...")

        processed_cond = total - len(remaining)

        for i in range(0, len(remaining), batch_size):
            batch = remaining[i : i + batch_size]
            items = _build_batch_items(batch, condition)

            start = time.time()
            try:
                responses = adapter.run_batch(items, system_prompt, condition)
            except Exception as e:
                print(f"  Batch {i//batch_size + 1} ERROR: {e}")
                processed_cond += len(batch)
                continue
            wall = time.time() - start

            batch_results = _score_batch(batch, responses, condition,
                                         adapter.model_id, wall)
            results.extend(batch_results)
            monitor.record_batch(batch_results, condition)
            processed_cond += len(batch)

            # Append to checkpoint
            _append_checkpoint(ckpt_path, batch_results)

            # Summary line per batch
            n_correct = sum(1 for r in batch_results if r.correct)
            n_err = sum(
                1 for r in batch_results
                if not r.raw_response.strip() or r.model_answer is None
            )
            extra = ""
            if condition == "text_only":
                extra = f" [{n_correct}/{len(batch)} pass]"
            elif condition == "audio_text_conflict":
                n_text = sum(1 for r in batch_results if r.followed_text)
                n_audio = n_correct
                extra = f" [audio={n_audio}, text={n_text}]"
            if n_err:
                extra += f" err={n_err}"
            done_pct = processed_cond / total * 100 if total else 0
            print(f"  [{processed_cond}/{total}] ({done_pct:.0f}%) "
                  f"batch {i//batch_size + 1}: "
                  f"{n_correct}/{len(batch)} correct "
                  f"({wall:.1f}s){extra}")

    # Print and save monitoring summary
    monitor.print_summary()
    monitoring_path = Path(output_path).with_name("monitoring.json")
    with open(monitoring_path, "w") as f:
        json.dump(monitor.summary(), f, indent=2, ensure_ascii=False)
    print(f"Monitoring data saved to {monitoring_path}")

    # Save combined results
    _save_results(results, output_path)

    # Save per-condition results
    out_stem = Path(output_path).stem
    out_dir = Path(output_path).parent
    out_ext = Path(output_path).suffix
    for cond in conditions:
        cond_results = [r for r in results if r.condition == cond]
        if cond_results:
            cond_path = out_dir / f"{out_stem}_{cond}{out_ext}"
            _save_results(cond_results, str(cond_path))

    return results


def _save_results(results: List[TrialResult], output_path: str):
    """Save trial results to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            [asdict(r) for r in results],
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nResults saved to {output_path}")


def compute_metrics(
    results: List[TrialResult],
    stimuli_lookup: Dict[str, Dict] = None,
) -> Dict[str, Any]:
    """Compute evaluation metrics with stratification by language and flip type.

    Primary TDR is computed on ALL stimuli (unfiltered).
    Secondary tdr_text_validated is computed only on stimuli that passed
    text-only validation (for sensitivity analysis / appendix).
    """

    metrics = {
        "total_trials": len(results),
        "by_condition": {},
        "by_language": {},
        "by_flip_type": {},
        "tdr": None,
        # Diagnostic fields
        "text_only_accuracy": None,
        "audio_only_accuracy": None,
        "aligned_accuracy": None,
        # Secondary metric (sensitivity analysis)
        "tdr_text_validated": None,
    }

    stim_info = stimuli_lookup or {}

    # Group by condition
    by_condition = {}
    for r in results:
        if r.condition not in by_condition:
            by_condition[r.condition] = []
        by_condition[r.condition].append(r)

    for condition, cond_results in by_condition.items():
        n = len(cond_results)
        correct = sum(1 for r in cond_results if r.correct)
        accuracy = correct / n if n > 0 else 0

        metrics["by_condition"][condition] = {
            "n": n,
            "correct": correct,
            "accuracy": accuracy,
        }

        if condition == "audio_text_conflict":
            followed_text = sum(1 for r in cond_results if r.followed_text)
            followed_audio = sum(1 for r in cond_results if r.correct)
            metrics["by_condition"][condition]["followed_text"] = followed_text
            metrics["by_condition"][condition]["followed_audio"] = followed_audio

            total_clear = followed_text + followed_audio
            if total_clear > 0:
                metrics["tdr"] = followed_text / total_clear

    # Stratify conflict results by language
    conflict_results = by_condition.get("audio_text_conflict", [])
    if conflict_results and stim_info:
        by_language = {}
        by_flip = {}

        for r in conflict_results:
            stim = stim_info.get(r.stimulus_id, {})
            language = stim.get("language", stim.get("accent_bucket", "UNKNOWN"))
            flip_type = stim.get("question", {}).get("flip_type", "UNKNOWN")

            # By language
            if language not in by_language:
                by_language[language] = {"followed_audio": 0, "followed_text": 0, "other": 0}
            if r.correct:
                by_language[language]["followed_audio"] += 1
            elif r.followed_text:
                by_language[language]["followed_text"] += 1
            else:
                by_language[language]["other"] += 1

            # By flip type
            if flip_type not in by_flip:
                by_flip[flip_type] = {"followed_audio": 0, "followed_text": 0, "other": 0}
            if r.correct:
                by_flip[flip_type]["followed_audio"] += 1
            elif r.followed_text:
                by_flip[flip_type]["followed_text"] += 1
            else:
                by_flip[flip_type]["other"] += 1

        # Compute TDR per language
        for lang, counts in by_language.items():
            total = counts["followed_audio"] + counts["followed_text"]
            tdr = counts["followed_text"] / total if total > 0 else None
            metrics["by_language"][lang] = {
                "n": total,
                "followed_audio": counts["followed_audio"],
                "followed_text": counts["followed_text"],
                "other": counts["other"],
                "tdr": tdr,
            }

        # Compute TDR per flip type
        for flip_type, counts in by_flip.items():
            total = counts["followed_audio"] + counts["followed_text"]
            tdr = counts["followed_text"] / total if total > 0 else None
            metrics["by_flip_type"][flip_type] = {
                "n": total,
                "followed_audio": counts["followed_audio"],
                "followed_text": counts["followed_text"],
                "other": counts["other"],
                "tdr": tdr,
            }

    # ------------------------------------------------------------------
    # Diagnostic accuracies
    # ------------------------------------------------------------------
    text_only_results = by_condition.get("text_only", [])
    if text_only_results:
        n = len(text_only_results)
        correct = sum(1 for r in text_only_results if r.correct)
        metrics["text_only_accuracy"] = correct / n if n > 0 else None

    audio_only_results = by_condition.get("audio_only", [])
    if audio_only_results:
        n = len(audio_only_results)
        correct = sum(1 for r in audio_only_results if r.correct)
        metrics["audio_only_accuracy"] = correct / n if n > 0 else None

    aligned_results = by_condition.get("audio_text_aligned", [])
    if aligned_results:
        n = len(aligned_results)
        correct = sum(1 for r in aligned_results if r.correct)
        metrics["aligned_accuracy"] = correct / n if n > 0 else None

    # ------------------------------------------------------------------
    # Secondary metric: TDR on text-validated subset (sensitivity analysis)
    # ------------------------------------------------------------------
    if text_only_results and conflict_results:
        text_pass_ids = {
            r.stimulus_id for r in text_only_results if r.correct
        }
        validated_conflict = [
            r for r in conflict_results if r.stimulus_id in text_pass_ids
        ]
        if validated_conflict:
            followed_text_v = sum(1 for r in validated_conflict if r.followed_text)
            followed_audio_v = sum(1 for r in validated_conflict if r.correct)
            total_clear_v = followed_text_v + followed_audio_v
            if total_clear_v > 0:
                metrics["tdr_text_validated"] = followed_text_v / total_clear_v

    return metrics


def main():
    from .models import ALL_MODELS

    ALL_CONDITIONS = [
        "audio_only", "text_only", "audio_text_aligned", "audio_text_conflict",
    ]

    parser = argparse.ArgumentParser(
        description="Evaluate audio-LLM on ALME stimuli.",
        epilog=(
            "Checkpointing: results are saved per-condition after every batch. "
            "If interrupted, re-run the same command to resume automatically — "
            "completed conditions are skipped, partial conditions continue "
            "from the last checkpoint, and failed stimuli are retried."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stimuli",
        type=str,
        default=None,
        help="Path to stimuli JSONL file. "
             "(default: data/stimuli.jsonl relative to package root)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/eval_results.json",
        help="Output path for results JSON. "
             "(default: results/eval_results.json)",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        nargs="+",
        choices=ALL_CONDITIONS,
        default=ALL_CONDITIONS,
        metavar="COND",
        help="Conditions to evaluate. Choices: %(choices)s. "
             "(default: all four conditions)",
    )
    parser.add_argument(
        "--max-stimuli",
        type=int,
        default=None,
        help="Maximum total number of stimuli to evaluate. "
             "(default: all 57,602)",
    )
    parser.add_argument(
        "--max-per-language",
        type=int,
        default=None,
        help="Maximum stimuli per language for balanced sampling. "
             "(default: no limit)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ultravox",
        choices=ALL_MODELS,
        help="Model to evaluate. (default: ultravox)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference. "
             "(default: 8, suitable for 8-bit quantization on 24GB GPU)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stimulus shuffling. (default: 42)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory for checkpoint files. "
             "(default: {output_dir}/.checkpoints/)",
    )
    parser.add_argument(
        "--cv-root",
        type=str,
        default=None,
        help="Root directory for Common Voice 22.0 corpus. "
             "Stimuli audio paths are relative; this prefix is prepended. "
             "Also reads from ALME_CV_ROOT env var. (required)",
    )
    args = parser.parse_args()

    # Resolve CV root
    cv_root = args.cv_root or os.environ.get("ALME_CV_ROOT")
    if not cv_root:
        parser.error(
            "--cv-root is required (or set ALME_CV_ROOT environment variable). "
            "This should point to the Common Voice 22.0 corpus directory, e.g. "
            "/path/to/cv-corpus-22.0-2025-06-20"
        )
    cv_root = Path(cv_root)
    if not cv_root.is_dir():
        parser.error(f"--cv-root directory does not exist: {cv_root}")

    # Resolve stimuli path
    stimuli_path = args.stimuli
    if stimuli_path is None:
        # Default: data/stimuli.jsonl relative to package root
        pkg_dir = Path(__file__).resolve().parent.parent
        stimuli_path = str(pkg_dir / "data" / "stimuli.jsonl")

    # Load stimuli and prepend cv_root to relative audio paths
    stimuli = []
    with open(stimuli_path, "r") as f:
        for line in f:
            stim = json.loads(line)
            stim["audio_path"] = str(cv_root / stim["audio_path"])
            stimuli.append(stim)

    import random
    random.seed(args.seed)
    random.shuffle(stimuli)

    print(f"Loaded {len(stimuli)} stimuli from {stimuli_path}")

    # Per-language balanced sampling
    if args.max_per_language:
        from collections import defaultdict
        by_lang = defaultdict(list)
        for s in stimuli:
            lang = s.get("language", s.get("accent_bucket", "UNKNOWN"))
            by_lang[lang].append(s)
        stimuli = []
        for lang in sorted(by_lang):
            stimuli.extend(by_lang[lang][:args.max_per_language])
        random.seed(args.seed)
        random.shuffle(stimuli)
        print(f"Balanced to {args.max_per_language}/lang: {len(stimuli)} total")

    # Show language/bucket distribution
    lang_counts = {}
    for s in stimuli[:args.max_stimuli] if args.max_stimuli else stimuli:
        lang = s.get("language", s.get("accent_bucket", "UNKNOWN"))
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    print(f"Language distribution: {dict(sorted(lang_counts.items()))}")

    # Load model adapter
    from .models import get_adapter

    adapter = get_adapter(args.model)
    adapter.load()

    # Run evaluation
    results = evaluate_stimuli(
        adapter,
        stimuli,
        args.conditions,
        args.output,
        args.max_stimuli,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Build stimuli lookup for stratification
    stimuli_lookup = {s["stimulus_id"]: s for s in stimuli}

    # Compute and display metrics
    metrics = compute_metrics(results, stimuli_lookup)

    print("\n" + "=" * 60)
    print(f"EVALUATION METRICS ({args.model})")
    print("=" * 60)

    for condition, cond_metrics in metrics["by_condition"].items():
        print(f"\n{condition}:")
        print(f"  Accuracy: {cond_metrics['accuracy']:.1%} "
              f"({cond_metrics['correct']}/{cond_metrics['n']})")
        if condition == "audio_text_conflict":
            print(f"  Followed audio: {cond_metrics['followed_audio']}")
            print(f"  Followed text: {cond_metrics['followed_text']}")

    # Diagnostic accuracies
    if metrics["text_only_accuracy"] is not None:
        print(f"\n  Diagnostic — text_only accuracy:  "
              f"{metrics['text_only_accuracy']:.1%}")
    if metrics["audio_only_accuracy"] is not None:
        print(f"  Diagnostic — audio_only accuracy: "
              f"{metrics['audio_only_accuracy']:.1%}")
    if metrics["aligned_accuracy"] is not None:
        print(f"  Diagnostic — aligned accuracy:    "
              f"{metrics['aligned_accuracy']:.1%}")

    if metrics["tdr"] is not None:
        print(f"\n{'='*60}")
        print(f"PRIMARY TDR (all stimuli): {metrics['tdr']:.1%}")
        if metrics["tdr"] > 0.5:
            print("  -> Model is TEXT-DOMINANT in conflict scenarios")
        else:
            print("  -> Model is AUDIO-DOMINANT in conflict scenarios")

    if metrics["tdr_text_validated"] is not None:
        print(f"SECONDARY TDR (text-validated subset): "
              f"{metrics['tdr_text_validated']:.1%}")

    if metrics["by_language"]:
        print(f"\n{'='*60}")
        print("TDR BY LANGUAGE:")
        print("-" * 60)
        for lang in sorted(metrics["by_language"].keys()):
            data = metrics["by_language"][lang]
            tdr_str = f"{data['tdr']:.1%}" if data['tdr'] is not None else "N/A"
            print(f"  {lang:15s}: TDR={tdr_str:>6s}  "
                  f"(n={data['n']:3d}, audio={data['followed_audio']}, "
                  f"text={data['followed_text']})")

    if metrics["by_flip_type"]:
        print(f"\n{'='*60}")
        print("TDR BY FLIP TYPE:")
        print("-" * 60)
        for flip_type in sorted(metrics["by_flip_type"].keys()):
            data = metrics["by_flip_type"][flip_type]
            tdr_str = f"{data['tdr']:.1%}" if data['tdr'] is not None else "N/A"
            print(f"  {flip_type:18s}: TDR={tdr_str:>6s}  "
                  f"(n={data['n']:3d}, audio={data['followed_audio']}, "
                  f"text={data['followed_text']})")

    # Save metrics
    metrics_path = args.output.replace(".json", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # Unload model
    adapter.unload()

    return 0


if __name__ == "__main__":
    sys.exit(main())
