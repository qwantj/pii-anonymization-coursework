"""
Утилиты морфологического анализа для обезличивания ФИО, организаций и локаций.

Интегрирует:
- Единый синглтон MorphologyEngine и LRU-кэширование (KR-PERF-2, LR-PERF-3).
- Снятие омонимии двуродовых имён Женя/Саша с контекстным окном (LR-REL-2).
- Изоляцию контекстного окна границами предложений (KR-REL-3).
- Склонение составных/дефисных сущностей и сохранение регистра (KR-REL-4).
"""

from __future__ import annotations

import re
from typing import Optional, Dict, Any, Tuple

from lab2.src.morphology_engine import (
    MorphologyEngine,
    get_morph_info_cached,
    inflect_word_cached,
    normalize_lemma_cached,
)

_morph = MorphologyEngine.get_analyzer()

_LATIN_CLEAN_PATTERN = re.compile(r"[^a-zA-Zа-яА-ЯёЁ]")
_SENTENCE_BOUNDARY_PATTERN = re.compile(r"[\.!\?\n;]")
_CYRILLIC_WORDS_PATTERN = re.compile(r"[а-яёА-ЯЁ]+")

AMBIGUOUS_NAMES = {
    "саша", "женя", "валя", "слава", "лена", "паша", "миша", "ярослава",
    "ваня", "тоня", "шура", "даня", "алекса"
}


def is_latin_text(text: str) -> bool:
    """Проверить, написан ли текст преимущественно на латинице (английский)."""
    clean = _LATIN_CLEAN_PATTERN.sub("", text)
    if not clean:
        return False
    latin_count = sum(1 for c in clean if 'a' <= c.lower() <= 'z')
    return (latin_count / len(clean)) >= 0.5


def strip_quotes_and_legal_forms(text: str) -> Tuple[str, str, str, str]:
    """
    Отделить организационно-правовую форму (ООО, ПАО, АО, ИП) и кавычки перед склонением.

    Returns:
        (prefix_form, left_quote, core_text, right_quote)
    """
    original = text.strip()
    prefix = ""
    left_quote = ""
    right_quote = ""

    forms = ("ООО", "ПАО", "АО", "ЗАО", "ИП", "ОАО", "НКО")
    for form in forms:
        if original.startswith(form + " "):
            prefix = form + " "
            original = original[len(form) + 1:].strip()
            break
        elif original.startswith(form):
            prefix = form
            original = original[len(form):].strip()
            break

    quote_pairs = [("«", "»"), ('"', '"'), ("'", "'")]
    for l_q, r_q in quote_pairs:
        if original.startswith(l_q) and original.endswith(r_q):
            left_quote = l_q
            right_quote = r_q
            original = original[len(l_q):-len(r_q)].strip()
            break

    return prefix, left_quote, original, right_quote


def preserve_text_casing(original: str, modified: str) -> str:
    """
    Сохранить авторский регистр (UPPERCASE, Title Case) оригинальной строки (KR-REL-4).
    """
    if not original or not modified:
        return modified

    clean = "".join(c for c in original if c.isalpha())
    if clean.isupper() and len(clean) > 1:
        return modified.upper()
    if clean.istitle() or original[0].isupper():
        return modified[0].upper() + modified[1:] if len(modified) > 1 else modified.upper()
    if clean.islower():
        return modified.lower()

    return modified


def handle_english_casing_and_possessive(original: str, replacement: str) -> str:
    """
    Сохранить регистр (UPPERCASE, Title Case) и притяжательное окончание ('s / ’s)
    для англоязычных сущностей.
    """
    has_possessive = original.endswith("'s") or original.endswith("’s")
    clean_orig = original[:-2] if has_possessive else original

    res = preserve_text_casing(clean_orig, replacement)

    if has_possessive:
        res = f"{res}'s"

    return res


def isolate_sentence_clause(left_part: str, right_part: str) -> Tuple[str, str]:
    """
    Изолировать фрагменты текста границами текущего предложения (KR-REL-3).
    Предотвращает ошибочное влияние сказуемых из соседних предложений на омонимию.
    """
    last_punct = -1
    for punct in ('.', '!', '?', '\n', ';'):
        idx = left_part.rfind(punct)
        if idx > last_punct:
            last_punct = idx
    bounded_left = left_part[last_punct + 1:].lstrip() if last_punct != -1 else left_part

    first_punct = len(right_part)
    for punct in ('.', '!', '?', '\n', ';'):
        idx = right_part.find(punct)
        if idx != -1 and idx < first_punct:
            first_punct = idx
    bounded_right = right_part[:first_punct].rstrip()

    return bounded_left, bounded_right


def _isolate_local_subclause(left_part: str, right_part: str) -> Tuple[str, str]:
    """Изолировать текущую синтаксическую подклаузу (до запятой/тире/двоеточия)."""
    last_sep = -1
    for sep in (',', '—', '–', ':'):
        idx = left_part.rfind(sep)
        if idx > last_sep:
            last_sep = idx
    sub_left = left_part[last_sep + 1:].lstrip() if last_sep != -1 else left_part

    first_sep = len(right_part)
    for sep in (',', '—', '–', ':'):
        idx = right_part.find(sep)
        if idx != -1 and idx < first_sep:
            first_sep = idx
    sub_right = right_part[:first_sep].rstrip()

    return sub_left, sub_right


def _interleave_by_proximity(left_tokens: list[str], right_tokens: list[str]) -> list[str]:
    """Упорядочить токены контекста по возрастанию расстояния от сущности (справа -> слева)."""
    ordered: list[str] = []
    max_len = max(len(left_tokens), len(right_tokens))
    for d in range(max_len):
        if d < len(right_tokens):
            ordered.append(right_tokens[d])
        if d < len(left_tokens):
            ordered.append(left_tokens[-(d + 1)])
    return ordered


def _is_ambiguous_surname_form(word: str) -> bool:
    """Проверить, является ли фамилия омонимичной (типа 'Иванова': ж.р. им.п. vs м.р. род.п.)."""
    low = word.lower()
    return low.endswith(("ова", "ева", "ёва", "ина", "ына", "ская", "цкая"))


def _infer_gender_from_tokens(tokens: list[str]) -> Optional[str]:
    """Определить явный грамматический род из упорядоченного списка контекстных слов."""
    morph = MorphologyEngine.get_analyzer()
    for tok in tokens:
        lower_t = tok.lower()
        if lower_t in AMBIGUOUS_NAMES:
            continue

        # 1. Маркерные местоимения и существительные
        if lower_t in ("он", "его", "ему", "им", "сам", "тот", "этот", "клиент", "сотрудник", "гражданин", "заявитель"):
            return "masc"
        if lower_t in ("она", "её", "ей", "ею", "сама", "та", "эта", "клиентка", "сотрудница", "гражданка", "заявительница"):
            return "femn"

        parses = morph.parse(tok)
        if not parses:
            continue
        p_tag = parses[0].tag

        # 2. Глаголы прошедшего времени и причастия в ед. ч.
        if ("VERB" in p_tag or "PRTF" in p_tag) and "past" in p_tag and "sing" in p_tag:
            if "femn" in p_tag:
                return "femn"
            if "masc" in p_tag:
                return "masc"

        # 3. Отчества (Patr)
        if "Patr" in p_tag:
            if "femn" in p_tag:
                return "femn"
            if "masc" in p_tag:
                return "masc"

        # 4. Однозначные имена (Name) или полные прилагательные муж/жен рода рядом
        if "Name" in p_tag and not _is_ambiguous_surname_form(tok):
            has_f = any("Name" in p.tag and "femn" in p.tag for p in parses)
            has_m = any("Name" in p.tag and "masc" in p.tag for p in parses)
            if has_f and not has_m:
                return "femn"
            if has_m and not has_f:
                return "masc"

    return None


def detect_explicit_context_gender(
    name: str,
    text: str = "",
    start_char: int = 0,
    end_char: int = 0,
) -> Optional[str]:
    """Определить род из синтаксического контекста (None, если явных маркеров нет)."""
    if not text:
        return None

    left_part = text[:start_char]
    right_part = text[end_char:]
    bounded_left, bounded_right = isolate_sentence_clause(left_part, right_part)

    # Сначала проверяем текущую клаузу без запятых/тире (защита от 'Саша пришла, а Женя ушёл')
    sub_left, sub_right = _isolate_local_subclause(bounded_left, bounded_right)
    sub_tokens = _interleave_by_proximity(
        _CYRILLIC_WORDS_PATTERN.findall(sub_left)[-4:],
        _CYRILLIC_WORDS_PATTERN.findall(sub_right)[:4],
    )
    res = _infer_gender_from_tokens(sub_tokens)
    if res is not None:
        return res

    # Затем расширенное окно в пределах того же предложения
    full_tokens = _interleave_by_proximity(
        _CYRILLIC_WORDS_PATTERN.findall(bounded_left)[-4:],
        _CYRILLIC_WORDS_PATTERN.findall(bounded_right)[:4],
    )
    return _infer_gender_from_tokens(full_tokens)


def resolve_ambivalent_gender(
    name: str,
    text: str = "",
    start_char: int = 0,
    end_char: int = 0,
) -> str:
    """
    Контекстное разрешение амбивалентного грамматического рода имен (Женя/Саша)
    и омонимичных фамилий типа «Иванова» (LR-REL-2, KR-REL-3).
    """
    lower_name = name.lower()
    is_ambiguous = (lower_name in AMBIGUOUS_NAMES) or _is_ambiguous_surname_form(name)

    info = get_morph_info_cached(name)
    if not is_ambiguous and info["gender"]:
        return info["gender"]

    ctx_gender = detect_explicit_context_gender(name, text, start_char, end_char)
    if ctx_gender is not None:
        return ctx_gender

    return info["gender"] or "masc"


def inflect_hyphenated_entity(
    entity: str,
    target_case: Optional[str],
    gender: Optional[str] = None,
) -> str:
    """
    Склонить составную дефисную сущность (Мамин-Сибиряк, Ростов-на-Дону) (KR-REL-4).
    Служебные частицы ('на', 'де', 'фон', 'эль') не склоняются.
    """
    if "-" not in entity:
        return inflect_word_cached(entity, target_case, gender)

    parts = entity.split("-")
    inflected_parts = []
    particles = {"на", "де", "фон", "эль", "ле", "ла", "ван", "дер"}

    for part in parts:
        if part.lower() in particles or not part:
            inflected_parts.append(part)
        else:
            inflected_parts.append(inflect_word_cached(part, target_case, gender))

    return "-".join(inflected_parts)


def get_morph_info(word: str) -> dict:
    """Получить морфологическую информацию о слове (с LRU-кэшированием, LR-PERF-3)."""
    return get_morph_info_cached(word)


def inflect_word(
    word: str,
    target_case: str | None,
    target_gender: str | None = None,
    target_number: str | None = None,
) -> str:
    """Склонить слово или составное словосочетание в заданный падеж (с LRU-кэшированием, LR-PERF-3)."""
    if not word or not target_case:
        return word

    if " " in word:
        parts = word.split()
        return " ".join(inflect_word(p, target_case, target_gender, target_number) for p in parts)

    if "-" in word:
        return inflect_hyphenated_entity(word, target_case, target_gender)

    return inflect_word_cached(word, target_case, target_gender, target_number)


def determine_gender_from_word(word: str) -> str | None:
    """Определить род слова."""
    info = get_morph_info_cached(word)
    return info["gender"]
