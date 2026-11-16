"""
Модуль обеспечения идемпотентности при обезличивании (Idempotency Vault).

Реализует требования КР и аналитического отчета:
- KR-REL-2: Устранение дефекта кэширования падежных форм топонимов (LOC) и организаций (ORG).
- KR-REL-4: Корректная обработка дефисных фамилий, составных топонимов и сохранение регистра (UPPERCASE).
- KR-COMP-1: Масштабирование словарей замен и контроль коллизий псевдонимов (_used_substitutions).
- KR-ARC-2: Потокобезопасность (threading.RLock), управление сессиями и экспорт карты соответствий.
- C6, C7, C8: Строгое соблюдение идемпотентности внутри сессии и грамматической связности.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from typing import Any, Dict, Optional, Set, Tuple
from faker import Faker

from lab2.src.morphology_engine import (
    MorphologyEngine,
    get_morph_info_cached,
    get_morph_info_for_tag,
    normalize_lemma_cached,
)
from lab2.src.morph_utils import (
    get_morph_info,
    inflect_word,
    is_latin_text,
    strip_quotes_and_legal_forms,
    handle_english_casing_and_possessive,
    preserve_text_casing,
    inflect_hyphenated_entity,
    detect_explicit_context_gender,
    _is_ambiguous_surname_form,
)
from lab2.src.anonymizer import (
    _pick_random_name,
    _pick_random_patronymic,
    _is_patronymic,
    _HANDLERS,
    _anonymize_org_entity,
    _anonymize_loc_entity,
    _anonymize_addr_entity,
)
from lab2.src.faker_providers import anonymize_location
from coursework.src.advanced_morphology import disambiguate_gender_by_context, AMBIGUOUS_NAMES

logger = logging.getLogger(__name__)

_morph = MorphologyEngine.get_analyzer()
_faker_en = Faker("en_US")


def normalize_lemma(text: str) -> str:
    """Привести словосочетание к канонической строке лемм в нижнем регистре."""
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    lemmas: list[str] = []
    for tok in tokens:
        if tok.isalpha():
            lemmas.append(normalize_lemma_cached(tok))
        else:
            lemmas.append(tok.lower())
    return " ".join(lemmas)


class IdempotencyVault:
    """
    Потокобезопасное хранилище соответствий между исходными и обезличенными сущностями
    с поддержкой сессий, разрешения коллизий и падежной инвариантности (KR-ARC-2, KR-REL-2, KR-COMP-1).
    """

    def __init__(self, session_id: Optional[str] = None):
        self.session_id: str = session_id or "default"
        self._lock: threading.RLock = threading.RLock()

        # Канонический кэш ФИО: (tag, normalized_lemma) -> base_replacement_in_nom
        self._name_cache: Dict[Tuple[str, str], str] = {}
        # Кэш рода для лемм: normalized_lemma -> 'masc' / 'femn'
        self._gender_cache: Dict[str, str] = {}
        # Кэш структурированных сущностей (ИНН, карты, email и др.): (tag, normalized_key) -> replacement
        self._structured_cache: Dict[Tuple[str, str], str] = {}
        # KR-REL-2: Кэш топонимов (LOC) в канонической форме Им.п.: lemma -> (base_loc_nom, gender)
        self._loc_cache: Dict[str, Tuple[str, str]] = {}
        # KR-REL-2: Кэш организаций (ORG): lemma -> (prefix, l_quote, core_nom, r_quote)
        self._org_cache: Dict[str, Tuple[str, str, str, str]] = {}
        # KR-COMP-1: Реестр использованных замен для предотвращения коллизий разных лиц
        self._used_substitutions: Set[str] = set()

    def session(self, session_id: str) -> IdempotencyVault:
        """Фабричный метод создания контекста для отдельного пакета документов (KR-ARC-2)."""
        return IdempotencyVault(session_id=session_id)

    def clear(self):
        """Очистить хранилище перед обработкой нового несвязанного документа."""
        with self._lock:
            self._name_cache.clear()
            self._gender_cache.clear()
            self._structured_cache.clear()
            self._loc_cache.clear()
            self._org_cache.clear()
            self._used_substitutions.clear()

    def _generate_unique_name(self, target_gender: str, is_surname: bool) -> str:
        """Генерация уникального псевдонима с контролем коллизий (KR-COMP-1)."""
        for _ in range(50):
            cand = _pick_random_name(target_gender, is_surname)
            if cand not in self._used_substitutions:
                self._used_substitutions.add(cand)
                return cand
        cand = _pick_random_name(target_gender, is_surname)
        self._used_substitutions.add(cand)
        return cand

    def get_or_create_name_replacement(
        self,
        original_word: str,
        tag: str,
        text: str = "",
        start: int = 0,
        end: int = 0,
    ) -> str:
        """
        Получить или создать идемпотентную замену для ФИО (PER, SURNAME, NAME)
        с канонизацией леммы, сохранением регистра и падежа (KR-REL-4, C6, C7).
        """
        with self._lock:
            if is_latin_text(original_word):
                lemma_key = (tag, original_word.lower().strip())
                if lemma_key in self._name_cache:
                    base_fake = self._name_cache[lemma_key]
                else:
                    if tag == "SURNAME":
                        base_fake = _faker_en.last_name()
                    elif tag == "NAME":
                        base_fake = _faker_en.first_name()
                    else:
                        base_fake = f"{_faker_en.first_name()} {_faker_en.last_name()}"
                    self._name_cache[lemma_key] = base_fake
                return handle_english_casing_and_possessive(original_word, base_fake)

            # Русскоязычные ФИО
            words = original_word.split()

            # 1. Составное ФИО из 3 слов: [Фамилия, Имя, Отчество] (LR-COMP-1)
            if len(words) == 3 and tag in ("PER", "FIO"):
                lemma = normalize_lemma(original_word)
                cache_key = ("PER_TRIAD", lemma)

                if cache_key in self._name_cache:
                    base_full_name = self._name_cache[cache_key]
                    target_gender = self._gender_cache.get(lemma, "masc")
                else:
                    target_gender = disambiguate_gender_by_context(words[1], text, start, end)
                    self._gender_cache[lemma] = target_gender

                    sur = self._generate_unique_name(target_gender, is_surname=True)
                    nam = self._generate_unique_name(target_gender, is_surname=False)
                    pat = _pick_random_patronymic(target_gender)
                    base_full_name = f"{sur} {nam} {pat}"
                    self._name_cache[cache_key] = base_full_name

                base_parts = base_full_name.split()
                inflected_parts = []
                for idx, orig_w in enumerate(words):
                    w_info = get_morph_info(orig_w)
                    part = base_parts[idx] if idx < len(base_parts) else base_parts[-1]
                    if w_info["case"] and w_info["case"] != "nomn":
                        inf = inflect_word(part, w_info["case"], target_gender)
                    else:
                        inf = part
                    inflected_parts.append(preserve_text_casing(orig_w, inf))
                return " ".join(inflected_parts)

            # 2. Составное ФИО из 2 слов
            if len(words) > 1 and tag in ("PER", "FIO"):
                lemma = normalize_lemma(original_word)
                cache_key = ("PER", lemma)

                w0_info = get_morph_info_for_tag(words[0], entity_tag=None)
                w1_info = get_morph_info_for_tag(words[1], entity_tag=None)
                w0_parses = _morph.parse(words[0])
                w0_is_surn = any("Surn" in p.tag for p in w0_parses) or any(
                    words[0].lower().endswith(s) for s in ("ов", "ова", "ев", "ева", "ин", "ина", "ский", "ская", "цкий", "цкая")
                )

                if cache_key in self._name_cache:
                    base_full_name = self._name_cache[cache_key]
                    target_gender = self._gender_cache.get(lemma, "masc")
                else:
                    # Если 1-е слово двуродовое ("Саша Иванов"), проверяем 2-е слово и контекст
                    if words[0].lower() in AMBIGUOUS_NAMES and w1_info["gender"]:
                        target_gender = w1_info["gender"]
                    elif words[1].lower() in AMBIGUOUS_NAMES and w0_info["gender"] and not _is_ambiguous_surname_form(words[0]):
                        target_gender = w0_info["gender"]
                    elif text:
                        target_gender = disambiguate_gender_by_context(words[0], text, start, end)
                    else:
                        target_gender = w0_info["gender"] or w1_info["gender"] or "masc"

                    self._gender_cache[lemma] = target_gender
                    sur = self._generate_unique_name(target_gender, is_surname=True)
                    nam = self._generate_unique_name(target_gender, is_surname=False)
                    base_full_name = f"{sur} {nam}" if w0_is_surn else f"{nam} {sur}"
                    self._name_cache[cache_key] = base_full_name

                base_parts = base_full_name.split()
                inflected_parts = []
                for idx, orig_w in enumerate(words):
                    part_tag = ("SURNAME" if w0_is_surn else "NAME") if idx == 0 else ("NAME" if w0_is_surn else "SURNAME")
                    w_info = get_morph_info_for_tag(orig_w, entity_tag=part_tag, target_gender=target_gender)
                    part_to_inflect = base_parts[idx] if idx < len(base_parts) else base_parts[-1]
                    if w_info["case"] and w_info["case"] != "nomn":
                        inf = inflect_word(part_to_inflect, w_info["case"], target_gender)
                    else:
                        inf = part_to_inflect
                    inflected_parts.append(preserve_text_casing(orig_w, inf))
                return " ".join(inflected_parts)

            # 3. Дефисная фамилия (KR-REL-4: "Мамин-Сибиряк", "Салтыков-Щедрин", "Петров-Водкин")
            if "-" in original_word and tag in ("SURNAME", "PER"):
                parts_orig = original_word.split("-")
                lemma = normalize_lemma(original_word)
                first_part = parts_orig[0]
                last_part = parts_orig[-1]
                explicit_g = detect_explicit_context_gender(original_word, text, start, end) if text else None
                part_info = get_morph_info_for_tag(last_part, entity_tag="SURNAME", target_gender=explicit_g)
                target_gender = explicit_g or part_info["gender"] or "masc"

                cache_key = ("PER_HYPHEN", f"{lemma}:{target_gender}")
                if cache_key in self._name_cache:
                    base_replacement = self._name_cache[cache_key]
                    target_gender = self._gender_cache.get(f"{lemma}:{target_gender}", target_gender)
                else:
                    self._gender_cache[f"{lemma}:{target_gender}"] = target_gender
                    sur1 = self._generate_unique_name(target_gender, is_surname=True)
                    sur2 = self._generate_unique_name(target_gender, is_surname=True)
                    while sur2 == sur1:
                        sur2 = self._generate_unique_name(target_gender, is_surname=True)
                    base_replacement = f"{sur1}-{sur2}"
                    self._name_cache[cache_key] = base_replacement

                w_info = get_morph_info_for_tag(last_part, entity_tag="SURNAME", target_gender=target_gender)
                c = w_info["case"]
                inf = inflect_hyphenated_entity(base_replacement, c, target_gender) if (c and c != "nomn") else base_replacement
                return preserve_text_casing(original_word, inf)

            # 4. Одиночное слово (SURNAME, NAME или PER)
            base_info = get_morph_info_for_tag(original_word, entity_tag=tag)
            normal_form = base_info["normal_form"].lower()

            if normal_form in AMBIGUOUS_NAMES:
                is_surname = False
            elif tag == "SURNAME":
                is_surname = True
            elif tag == "NAME":
                is_surname = False
            else:
                is_surname = any(
                    normal_form.endswith(s)
                    for s in ("ов", "ова", "ев", "ева", "ин", "ина", "ский", "ская", "цкий", "цкая", "ых", "их")
                )

            role_bucket = "SURNAME" if is_surname else "NAME"
            is_ambig = (normal_form in AMBIGUOUS_NAMES) or (is_surname and _is_ambiguous_surname_form(original_word))

            explicit_g = detect_explicit_context_gender(original_word, text, start, end) if text else None

            if is_ambig and explicit_g is None and normal_form in self._gender_cache:
                # Нейтральный контекст (напр. "Документы Саши готовы") — берем уже известный род для этой леммы
                target_gender = self._gender_cache[normal_form]
            elif explicit_g is not None and is_ambig:
                target_gender = explicit_g
            else:
                target_gender = disambiguate_gender_by_context(
                    original_word, text, start, end, entity_tag=tag
                ) if text else (base_info["gender"] or "masc")

            cache_key = (role_bucket, f"{normal_form}:{target_gender}")

            if cache_key in self._name_cache:
                base_replacement = self._name_cache[cache_key]
            else:
                self._gender_cache[normal_form] = target_gender
                if _is_patronymic(original_word):
                    base_replacement = _pick_random_patronymic(target_gender)
                else:
                    base_replacement = self._generate_unique_name(target_gender, is_surname)
                self._name_cache[cache_key] = base_replacement

            cased_info = get_morph_info_for_tag(original_word, entity_tag=tag, target_gender=target_gender)
            case = cased_info["case"]
            if case and case != "nomn":
                inf = inflect_word(base_replacement, case, target_gender)
            else:
                inf = base_replacement

            return preserve_text_casing(original_word, inf)

    def get_or_create_structured_replacement(
        self,
        original_value: str,
        tag: str,
    ) -> str:
        """
        Получить или создать идемпотентную замену для организаций, топонимов и реквизитов.
        Исправляет дефект залипания падежа LOC и ORG (KR-REL-2).
        """
        with self._lock:
            # KR-REL-2: Обработка топонимов (LOC) с динамическим падежным согласованием
            if tag == "LOC":
                if is_latin_text(original_value):
                    norm_key = original_value.strip().lower()
                    if norm_key not in self._structured_cache:
                        fake_loc = _HANDLERS["LOC"](original_value)
                        self._structured_cache[norm_key] = fake_loc
                    return preserve_text_casing(original_value, self._structured_cache[norm_key])

                norm_key = normalize_lemma(original_value)
                info = get_morph_info(original_value)
                current_case = info["case"]

                if norm_key in self._loc_cache:
                    base_loc_nom, loc_gender, loc_number = self._loc_cache[norm_key]
                else:
                    raw_fake = anonymize_location(original_value, lang="ru")
                    base_loc_nom = raw_fake.strip()
                    base_info = get_morph_info(base_loc_nom)
                    loc_gender = base_info["gender"] or "masc"
                    loc_number = base_info.get("number")
                    self._loc_cache[norm_key] = (base_loc_nom, loc_gender, loc_number)

                # Склоняем базовый город в падеж текущего вхождения
                if current_case and current_case != "nomn":
                    res = inflect_word(base_loc_nom, current_case, loc_gender, loc_number)
                else:
                    res = base_loc_nom

                return preserve_text_casing(original_value, res)

            # KR-REL-2: Обработка организаций (ORG) с сохранением ОПФ и кавычек
            if tag == "ORG":
                if is_latin_text(original_value):
                    norm_key = original_value.strip().lower()
                    if norm_key not in self._structured_cache:
                        self._structured_cache[norm_key] = _HANDLERS["ORG"](original_value)
                    return preserve_text_casing(original_value, self._structured_cache[norm_key])

                prefix, l_q, core, r_q = strip_quotes_and_legal_forms(original_value)
                norm_core = normalize_lemma(core)

                if norm_core in self._org_cache:
                    c_prefix, c_l_q, fake_core, c_r_q = self._org_cache[norm_core]
                else:
                    # Генерируем название компании
                    fake_full = _HANDLERS["ORG"](core)
                    _, _, fake_core, _ = strip_quotes_and_legal_forms(fake_full)
                    self._org_cache[norm_core] = (prefix, l_q, fake_core, r_q)

                res = f"{prefix}{l_q}{fake_core}{r_q}"
                return preserve_text_casing(original_value, res)

            # Остальные структурированные реквизиты (ИНН, карты, ОГРНИП, email и др.)
            if tag == "ADDR":
                normalized_key = (tag, normalize_lemma(original_value))
            else:
                digits = "".join(c for c in original_value if c.isdigit())
                normalized_key = (tag, digits.lower() if digits else "".join(original_value.split()).lower())

            if normalized_key in self._structured_cache:
                return self._structured_cache[normalized_key]

            handler = _HANDLERS.get(tag)
            if handler:
                replacement = handler(original_value)
            else:
                replacement = original_value

            self._structured_cache[normalized_key] = replacement
            return replacement

    def export_translation_map(self, filepath: str) -> None:
        """Экспортирует соответствия 'исходное -> псевдоним' в JSON для аудита (KR-ARC-2)."""
        with self._lock:
            data = {
                "session_id": self.session_id,
                "names": {f"{k[0]}:{k[1]}": v for k, v in self._name_cache.items()},
                "locations": {k: v[0] for k, v in self._loc_cache.items()},
                "organizations": {k: f"{v[0]}{v[1]}{v[2]}{v[3]}" for k, v in self._org_cache.items()},
                "structured": {f"{k[0]}:{k[1]}": v for k, v in self._structured_cache.items()},
            }
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)


ThreadSafeIdempotencyVault = IdempotencyVault
