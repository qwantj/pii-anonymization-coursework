"""
Централизованный синглтон морфологического анализатора (KR-PERF-2) с многоуровневым LRU-кэшированием.

Исключает многократную загрузку тяжелых DAWG-словарей OpenCorpora (~30 МБ на экземпляр),
предотвращает утечки памяти и ускоряет морфологический разбор на 80-88% (LR-PERF-3).
"""

from __future__ import annotations

import functools
import threading
from typing import Any, Dict, Optional, Tuple
import pymorphy3


class MorphologyEngine:
    """Потокобезопасный синглтон-контейнер для MorphAnalyzer."""

    _instance: Optional[pymorphy3.MorphAnalyzer] = None
    _lock: threading.RLock = threading.RLock()

    @classmethod
    def get_analyzer(cls) -> pymorphy3.MorphAnalyzer:
        """Возвращает глобальный экземпляр pymorphy3.MorphAnalyzer."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = pymorphy3.MorphAnalyzer()
        return cls._instance


@functools.lru_cache(maxsize=32768)
def get_morph_info_cached(word: str) -> Dict[str, Any]:
    """
    LRU-кэшированный разбор слова. Возвращает неизменяемые или плоские структуры.
    """
    morph = MorphologyEngine.get_analyzer()
    parsed = morph.parse(word)
    if not parsed:
        return {
            "normal_form": word,
            "gender": None,
            "case": None,
            "number": None,
        }

    best = parsed[0]
    if ("Fixd" in best.tag or "plur" in best.tag) and any("Name" in p.tag or "Surn" in p.tag for p in parsed):
        preferred = [
            p for p in parsed
            if ("Name" in p.tag or "Surn" in p.tag) and "Fixd" not in p.tag and "sing" in p.tag
        ]
        if not preferred:
            preferred = [
                p for p in parsed
                if ("Name" in p.tag or "Surn" in p.tag) and "Fixd" not in p.tag
            ]
        if preferred:
            best = preferred[0]

    if word.lower().endswith(("у", "ю")):
        for p in parsed:
            if "Surn" in p.tag and "datv" in p.tag and "masc" in p.tag:
                best = p
                break
        else:
            for p in parsed:
                if "datv" in p.tag:
                    best = p
                    break
    elif word.lower().endswith(("ым", "им", "ом", "ем")):
        for p in parsed:
            if "Surn" in p.tag and "ablt" in p.tag and "sing" in p.tag:
                best = p
                break
        else:
            for p in parsed:
                if "ablt" in p.tag and "sing" in p.tag:
                    best = p
                    break
    elif word.lower().endswith(("ой", "ей")):
        if "nomn" not in parsed[0].tag:
            for p in parsed:
                if "Surn" in p.tag and "gent" in p.tag and "femn" in p.tag:
                    best = p
                    break
            else:
                for p in parsed:
                    if "gent" in p.tag and "femn" in p.tag:
                        best = p
                        break

    # Именительный падеж для базовых форм при омонимии с винительным (Казань, Смоленск, Чебоксары, Тамбов)
    if "accs" in best.tag:
        for p in parsed:
            if "nomn" in p.tag and (p.normal_form == best.normal_form or "Geox" in p.tag or "NOUN" in p.tag):
                best = p
                break

    tag = best.tag

    gender = None
    for g in ("masc", "femn", "neut"):
        if g in tag:
            gender = g
            break

    case = None
    for c in ("nomn", "gent", "datv", "accs", "ablt", "loct", "voct"):
        if c in tag:
            case = c
            break

    number = None
    for n in ("sing", "plur"):
        if n in tag:
            number = n
            break

    return {
        "normal_form": best.normal_form,
        "gender": gender,
        "case": case,
        "number": number,
    }


@functools.lru_cache(maxsize=32768)
def get_morph_info_for_tag(
    word: str,
    entity_tag: Optional[str] = None,
    target_gender: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Контекстно-ориентированный морфологический разбор слова с учётом тега сущности
    (SURNAME/NAME) и определённого грамматического рода.
    Устраняет омонимию форм типа «Жени» (глагол «женить» vs имя «Женя») и
    «Иванова» (муж. род.п. «Иванова» vs жен. им.п. «Иванова»).
    """
    if not entity_tag and not target_gender:
        return get_morph_info_cached(word)

    morph = MorphologyEngine.get_analyzer()
    parsed = morph.parse(word)
    if not parsed:
        return get_morph_info_cached(word)

    # Приоритетная оценка каждого разбора по тегу сущности и роду
    def _score_parse(p) -> tuple[int, int, int, int, int]:
        t = p.tag
        tag_match = 0
        if entity_tag == "SURNAME" and "Surn" in t:
            tag_match = 3
        elif entity_tag == "NAME" and "Name" in t:
            tag_match = 3
        elif entity_tag in ("PER", "FIO") and ("Surn" in t or "Name" in t or "Patr" in t):
            tag_match = 2
        elif "NOUN" in t or "ADJF" in t:
            tag_match = 1

        not_fixd_bonus = 1 if "Fixd" not in t else 0
        gender_match = 1 if (target_gender and target_gender in t) else 0
        sing_bonus = 1 if "sing" in t else 0
        nomn_bonus = 1 if (target_gender == "femn" and "nomn" in t and word.lower().endswith(("а", "я"))) else 0
        return (tag_match, not_fixd_bonus, gender_match, sing_bonus, nomn_bonus)

    best = max(parsed, key=_score_parse)
    # Если ни один специфичный тег не совпал и род не задан, используем стандартный разбор
    if _score_parse(best)[:1] == (0,) and not target_gender:
        return get_morph_info_cached(word)

    # Если целевой род не задан, но выбрали по тегу (например, NAME для 'Жени'),
    # при омонимии жен/муж рода в одном падеже не фиксируем случайный род
    tag = best.tag
    gender = None
    for g in ("masc", "femn", "neut"):
        if g in tag:
            gender = g
            break

    case = None
    for c in ("nomn", "gent", "datv", "accs", "ablt", "loct", "voct"):
        if c in tag:
            case = c
            break

    number = None
    for n in ("sing", "plur"):
        if n in tag:
            number = n
            break

    return {
        "normal_form": best.normal_form,
        "gender": gender,
        "case": case,
        "number": number,
    }


@functools.lru_cache(maxsize=32768)
def inflect_word_cached(
    word: str,
    target_case: Optional[str],
    target_gender: Optional[str] = None,
    target_number: Optional[str] = None,
) -> str:
    """
    LRU-кэшированное склонение слова.
    """
    if not word:
        return word

    morph = MorphologyEngine.get_analyzer()
    parsed = morph.parse(word)
    if not parsed:
        return word

    best = parsed[0]
    if target_gender:
        for p in parsed:
            if "Surn" in p.tag and target_gender in p.tag:
                best = p
                break
        else:
            for p in parsed:
                if target_gender in p.tag and ("sing" in p.tag or target_number == "plur"):
                    best = p
                    break

    grammemes = set()
    if target_case:
        grammemes.add(target_case)
    if target_gender and target_number != "plur":
        grammemes.add(target_gender)
    if target_number:
        grammemes.add(target_number)

    if not grammemes:
        return word

    inflected = best.inflect(grammemes)
    if inflected is None:
        if target_gender and target_number == "plur":
            grammemes.remove(target_gender)
            inflected = best.inflect(grammemes)
        if inflected is None:
            return word

    result = inflected.word
    if word and word[0].isupper():
        result = result[0].upper() + result[1:]

    return result


@functools.lru_cache(maxsize=32768)
def normalize_lemma_cached(word: str) -> str:
    """Кэшированная нормализация одиночного слова."""
    morph = MorphologyEngine.get_analyzer()
    parses = morph.parse(word)
    return parses[0].normal_form if parses else word.lower()
