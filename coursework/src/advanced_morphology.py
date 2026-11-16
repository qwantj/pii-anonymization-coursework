"""
Модуль разрешения морфологической омонимии и контекстного определения рода.

Решает задачу на оценку «Отлично» из методических указаний к КР:
1. Определение рода для двуродовых имён (Женя, Саша, Валя, Слава) по контексту:
   - Анализ соседних глаголов прошедшего времени (например: «Саша оплатила» -> femn, «Саша оплатил» -> masc).
   - Анализ отчеств в контексте (на «-ич» -> masc, на «-на» -> femn).
   - Анализ согласованных прилагательных и причастий.
   - Изоляция контекстного окна границами предложений (KR-REL-3).
2. Определение рода и падежа для несклоняемых и амбивалентных фамилий.
3. Единый потокобезопасный синглтон MorphologyEngine (KR-PERF-2).
"""

from __future__ import annotations

import re
from typing import Optional
from lab2.src.morphology_engine import MorphologyEngine, get_morph_info_cached, get_morph_info_for_tag
from lab2.src.morph_utils import (
    isolate_sentence_clause,
    detect_explicit_context_gender,
    _interleave_by_proximity,
    _is_ambiguous_surname_form,
)

_morph = MorphologyEngine.get_analyzer()

# Список популярных имён с неоднозначным грамматическим родом (двуродовые / unisex)
AMBIGUOUS_NAMES = {
    "саша", "женя", "валя", "слава", "лена", "паша", "миша", "ярослава",
    "ваня", "тоня", "шура", "даня", "алекса"
}


def extract_context_window(text: str, start: int, end: int, window_words: int = 4) -> list[str]:
    """
    Извлечь слова из контекстного окна вокруг заданной сущности
    с изоляцией границами текущего предложения (KR-REL-3), упорядоченные по близости.
    """
    left_part = text[:start]
    right_part = text[end:]

    bounded_left, bounded_right = isolate_sentence_clause(left_part, right_part)

    left_words = re.findall(r'[а-яёА-ЯЁ]+', bounded_left)[-window_words:]
    right_words = re.findall(r'[а-яёА-ЯЁ]+', bounded_right)[:window_words]

    return _interleave_by_proximity(left_words, right_words)


def disambiguate_gender_by_context(
    word: str,
    text: str,
    start: int,
    end: int,
    entity_tag: Optional[str] = None,
) -> str:
    """
    Определить грамматический род имени или фамилии с учётом синтаксического контекста
    и изоляции предложений (KR-REL-3).

    Returns:
        'masc' или 'femn'.
    """
    lower_word = word.lower()
    is_ambiguous = (lower_word in AMBIGUOUS_NAMES) or _is_ambiguous_surname_form(word)

    # Для однозначных имён и фамилий используем словарь с учётом тега сущности
    if not is_ambiguous:
        info = get_morph_info_for_tag(word, entity_tag=entity_tag)
        if info["gender"]:
            return info["gender"]

    if text:
        ctx_gender = detect_explicit_context_gender(word, text, start, end)
        if ctx_gender is not None:
            return ctx_gender

    # Дефолт при отсутствии контекстных маркеров
    info = get_morph_info_for_tag(word, entity_tag=entity_tag)
    if info["gender"]:
        return info["gender"]

    parses = _morph.parse(word)
    if parses:
        for p in parses:
            if "femn" in p.tag:
                return "femn"
    return "masc"
