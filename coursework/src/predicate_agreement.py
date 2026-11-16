"""
Модуль адаптивного грамматического согласования зависимых предикатов и причастий (KR-COMP-3).

Решает задачу:
При синонимизации ФИО сканирует зависимый глагол прошедшего времени или причастие
в рамках синтаксической клаузы и приводит его в полное грамматическое соответствие
с родом и числом заменяемого субъекта (инвариант C6).
"""

from __future__ import annotations

import re
from typing import List, Tuple, Optional
from lab2.src.morphology_engine import MorphologyEngine, get_morph_info_cached, inflect_word_cached
from lab2.src.morph_utils import isolate_sentence_clause, preserve_text_casing

_morph = MorphologyEngine.get_analyzer()
_WORD_REGEX = re.compile(r"[а-яёА-ЯЁ]+")


class PredicateAgreementRepair:
    """Инструмент для гармонизации глагольных форм со сгенерированным псевдонимом (KR-COMP-3)."""

    @classmethod
    def repair_sentence_clause(
        cls,
        text: str,
        subject_start: int,
        subject_end: int,
        target_gender: str,
    ) -> str:
        """
        Проверить и скорректировать форму зависимого глагола прошедшего времени
        в пределах одного предложения вокруг позиции замененного субъекта.
        """
        if not text or target_gender not in ("masc", "femn"):
            return text

        left_part = text[:subject_start]
        subject_str = text[subject_start:subject_end]
        right_part = text[subject_end:]

        # Изоляция границами предложений (KR-REL-3)
        b_left, b_right = isolate_sentence_clause(left_part, right_part)
        left_offset = len(left_part) - len(b_left)
        right_len = len(b_right)

        # Сканируем слова справа от субъекта (наиболее частая позиция: "Саша [оплатил/оплатила]")
        for m in _WORD_REGEX.finditer(b_right):
            verb_candidate = m.group(0)
            v_start = left_offset + len(b_left) + len(subject_str) + m.start()
            v_end = left_offset + len(b_left) + len(subject_str) + m.end()

            parses = _morph.parse(verb_candidate)
            if not parses:
                continue

            p = parses[0]
            if ("VERB" in p.tag or "PRTF" in p.tag) and "past" in p.tag and "sing" in p.tag:
                # Проверяем, согласуется ли род
                curr_g = "femn" if "femn" in p.tag else ("masc" if "masc" in p.tag else None)
                if curr_g and curr_g != target_gender:
                    # Корректируем форму глагола
                    new_verb = inflect_word_cached(verb_candidate, target_case=None, target_gender=target_gender)
                    new_verb_cased = preserve_text_casing(verb_candidate, new_verb)
                    return text[:v_start] + new_verb_cased + text[v_end:]
                # Первого сказуемого достаточно в клаузе
                break

        # Сканируем слова слева от субъекта (инверсия: "[Оплатила/Оплатил] госпошлину Саша")
        left_matches = list(_WORD_REGEX.finditer(b_left))
        for m in reversed(left_matches):
            verb_candidate = m.group(0)
            v_start = left_offset + m.start()
            v_end = left_offset + m.end()

            parses = _morph.parse(verb_candidate)
            if not parses:
                continue

            p = parses[0]
            if ("VERB" in p.tag or "PRTF" in p.tag) and "past" in p.tag and "sing" in p.tag:
                curr_g = "femn" if "femn" in p.tag else ("masc" if "masc" in p.tag else None)
                if curr_g and curr_g != target_gender:
                    new_verb = inflect_word_cached(verb_candidate, target_case=None, target_gender=target_gender)
                    new_verb_cased = preserve_text_casing(verb_candidate, new_verb)
                    return text[:v_start] + new_verb_cased + text[v_end:]
                break

        return text
