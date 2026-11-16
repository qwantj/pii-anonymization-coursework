"""
Тестовый набор Спринта 2: Лингвистическая точность, устранение грамматических ошибок
и внутридокументная ссылочная целостность.

Покрывает:
1. LR-COMP-1: Поддержка отчеств в ФИО и устранение дефекта 'Смирнов Артём Кирилл'.
2. LR-REL-2: Контекстное разрешение амбивалентного рода Женя/Саша в ядре ЛР2.
3. KR-REL-2: Устранение дефекта кэширования падежных форм топонимов (LOC) и организаций (ORG) в IdempotencyVault.
4. KR-REL-3: Изоляция контекстного окна границами предложений при снятии омонимии.
5. KR-REL-4: Дефисные фамилии, составные топонимы и сохранение исходного регистра (UPPERCASE).
6. LR-ARC-2: Локальный контекст ссылочной целостности и внутридокументной идемпотентности AnonymizationContext.
"""

from __future__ import annotations

import pytest

from lab2.src.anonymizer import (
    anonymize_text,
    AnonymizationContext,
    _anonymize_person_name,
    _PATRONYMICS_M,
    _PATRONYMICS_F,
    _SURNAMES_M,
    _NAMES_M,
)
from lab2.src.morph_utils import (
    resolve_ambivalent_gender,
    isolate_sentence_clause,
    preserve_text_casing,
    inflect_hyphenated_entity,
    get_morph_info,
)
from coursework.src.advanced_morphology import disambiguate_gender_by_context
from coursework.src.idempotency_manager import IdempotencyVault


# ============================================================================
# 1. ТЕСТЫ LR-COMP-1: Поддержка отчеств и триад ФИО
# ============================================================================
class TestLRCOMP1PatronymicsAndLexicon:
    """Тестирование корректного синтеза ФИО с отчествами и согласованием рода."""

    def test_three_component_fio_has_surname_name_patronymic(self):
        orig_fio = "Иванов Иван Иванович"
        # Проверяем, что третье слово не является вторым именем
        for _ in range(10):
            anon_fio = _anonymize_person_name(orig_fio, "PER")
            parts = anon_fio.split()
            assert len(parts) == 3, f"ФИО должно состоять ровно из 3 частей: {anon_fio}"
            sur, nam, pat = parts[0], parts[1], parts[2]
            assert pat.lower().endswith(("ович", "евич", "ич")), f"Третья часть должна быть отчеством, а не именем: {pat}"

    def test_three_component_fio_female_gender_consistency(self):
        orig_fio = "Смирнова Анна Сергеевна"
        for _ in range(10):
            anon_fio = _anonymize_person_name(orig_fio, "PER")
            parts = anon_fio.split()
            assert len(parts) == 3
            pat = parts[2]
            assert pat.lower().endswith(("овна", "евна", "ична", "инична")), f"Женское отчество должно оканчиваться на -на: {pat}"

    def test_fio_case_inflection_preservation(self):
        # Дательный падеж
        orig_fio = "Иванову Ивану Ивановичу"
        anon_fio = _anonymize_person_name(orig_fio, "PER")
        parts = anon_fio.split()
        assert len(parts) == 3
        # Проверяем дательный падеж
        for p in parts:
            info = get_morph_info(p)
            assert info["case"] in ("datv", None), f"Компонент {p} должен быть в дательном падеже"

    def test_lexicon_size_minimum_requirements(self):
        assert len(_SURNAMES_M) >= 200, "Словарь мужских фамилий должен быть >= 200"
        assert len(_NAMES_M) >= 100, "Словарь мужских имен должен быть >= 100"
        assert len(_PATRONYMICS_M) >= 50, "Словарь отчеств должен быть >= 50"


# ============================================================================
# 2. ТЕСТЫ LR-REL-2: Контекстное снятие омонимии рода в ядре ЛР2
# ============================================================================
class TestLRREL2AmbivalentGenderResolution:
    """Тестирование контекстного разрешения Женя/Саша в lab2."""

    def test_sasha_female_context_past_verb(self):
        text = "Саша получила диплом с отличием."
        # start 0, end 4
        gender = resolve_ambivalent_gender("Саша", text, 0, 4)
        assert gender == "femn"

    def test_sasha_male_context_past_verb(self):
        text = "Саша получил диплом с отличием."
        gender = resolve_ambivalent_gender("Саша", text, 0, 4)
        assert gender == "masc"

    def test_zhenya_female_context_patronymic(self):
        text = "Женя Александровна подписала приказ."
        gender = resolve_ambivalent_gender("Женя", text, 0, 4)
        assert gender == "femn"

    def test_zhenya_male_context_patronymic(self):
        text = "Женя Александрович подписал приказ."
        gender = resolve_ambivalent_gender("Женя", text, 0, 4)
        assert gender == "masc"


# ============================================================================
# 3. ТЕСТЫ KR-REL-3: Изоляция контекстного окна границами предложений
# ============================================================================
class TestKRREL3SentenceBoundaryIsolation:
    """Тестирование исключения взаимного влияния соседних предложений."""

    def test_sentence_boundary_isolation_prevents_leak(self):
        # В первом предложении глагол 'сдала' (femn), но во втором 'Саша оплатил' (masc)
        text = "Елена Васильева успешно сдала проект. Саша оплатил госпошлину за регистрацию."
        sasha_start = text.index("Саша")
        sasha_end = sasha_start + len("Саша")

        gender = disambiguate_gender_by_context("Саша", text, sasha_start, sasha_end)
        assert gender == "masc", "Глагол 'сдала' из прошлого предложения не должен искажать род Саши!"

    def test_sentence_boundary_isolation_question_mark(self):
        text = "Мария пришла вовремя? Саша опоздал на поезд."
        sasha_start = text.index("Саша")
        sasha_end = sasha_start + len("Саша")

        gender = disambiguate_gender_by_context("Саша", text, sasha_start, sasha_end)
        assert gender == "masc"

    def test_isolate_sentence_clause_helper(self):
        left = "Первое предложение. Второе "
        right = " продолжается! Третье."
        b_left, b_right = isolate_sentence_clause(left, right)
        assert b_left == "Второе "
        assert b_right == " продолжается"


# ============================================================================
# 4. ТЕСТЫ KR-REL-2: Исправление бага кэширования падежей LOC и ORG
# ============================================================================
class TestKRREL2LocAndOrgCaseCaching:
    """Тестирование устранения залипания падежа при повторном упоминании топонимов/организаций."""

    def test_loc_prepositional_then_nominative(self):
        # Первое вхождение города — в предложном падеже ("в Москве")
        # Второе вхождение — в именительном ("Москва")
        vault = IdempotencyVault()

        # 1. Первое упоминание
        rep_prep = vault.get_or_create_structured_replacement("Москве", "LOC")
        # 2. Второе упоминание того же города в им. падеже
        rep_nom = vault.get_or_create_structured_replacement("Москва", "LOC")

        assert rep_prep != rep_nom, f"Форма в предложном падеже ({rep_prep}) не должна возвращаться для Им.п. ({rep_nom})"
        nom_info = get_morph_info(rep_nom)
        assert nom_info["case"] in ("nomn", None), f"Город {rep_nom} должен быть в именительном падеже"

    def test_loc_nominative_then_genitive(self):
        vault = IdempotencyVault()
        rep_nom = vault.get_or_create_structured_replacement("Казань", "LOC")
        rep_gent = vault.get_or_create_structured_replacement("Казани", "LOC")
        # При нормализации Казань -> Казани
        assert rep_nom != rep_gent or rep_nom.endswith("ь")

    def test_org_preserves_prefix_and_quotes(self):
        vault = IdempotencyVault()
        org_str = 'ПАО "Газпром"'
        rep = vault.get_or_create_structured_replacement(org_str, "ORG")
        assert rep.startswith("ПАО ")
        assert '"' in rep or '«' in rep


# ============================================================================
# 5. ТЕСТЫ KR-REL-4: Дефисные фамилии и сохранение регистра (UPPERCASE)
# ============================================================================
class TestKRREL4HyphenationAndCasing:
    """Тестирование дефисных фамилий, частиц и регистра."""

    def test_hyphenated_surname_inflection(self):
        orig = "Мамин-Сибиряк"
        inf = inflect_hyphenated_entity(orig, "datv", "masc")
        # Должно склоняться: Мамину-Сибиряку
        assert "-" in inf
        parts = inf.split("-")
        assert parts[0].endswith("у")
        assert parts[1].endswith("у")

    def test_hyphenated_particle_not_inflected(self):
        orig = "Ростов-на-Дону"
        inf = inflect_hyphenated_entity(orig, "datv", "masc")
        parts = inf.split("-")
        assert parts[1] == "на", "Частица 'на' не должна склоняться"

    def test_uppercase_preservation(self):
        orig = "ИВАНОВ"
        res = preserve_text_casing(orig, "смирнов")
        assert res == "СМИРНОВ"

    def test_titlecase_preservation(self):
        orig = "Иванов"
        res = preserve_text_casing(orig, "смирнов")
        assert res == "Смирнов"


# ============================================================================
# 6. ТЕСТЫ LR-ARC-2: Внутридокументная идемпотентность в ЛР2
# ============================================================================
class TestLRARC2ContextIdempotency:
    """Тестирование AnonymizationContext внутри одного документа."""

    def test_same_person_replaced_identically_across_document(self):
        text = "Иванов подписал договор. Позже Иванову перевели средства. Документы Иванова готовы."
        spans = [
            ("PER", 0, 6),
            ("PER", 31, 38),
            ("PER", 68, 75),
        ]
        anon = anonymize_text(text, spans)
        words = anon.split()
        # Первое слово — псевдоним в Им.п.
        fake_nom = words[0]
        # Проверяем, что в тексте повторяется корень того же псевдонима
        root = fake_nom[:-2] if len(fake_nom) > 4 else fake_nom
        assert anon.count(root) >= 3, f"Субъект должен сохраняться по всему документу: {anon}"

    def test_stochasticity_between_independent_calls_c8(self):
        text = "Иванов Иван подал заявку."
        spans = [("PER", 0, 11)]
        call1 = anonymize_text(text, spans)
        call2 = anonymize_text(text, spans)
        # При разных вызовах независимые случайные персонажи (C8)
        # Прогоняем несколько раз, чтобы исключить случайное совпадение
        results = {anonymize_text(text, spans) for _ in range(10)}
        assert len(results) > 1, "Разные вызовы функции должны генерировать новые случайные данные (C8)"
