"""
Интеграционный и модульный тестовый набор Спринта 1: Надежность и устранение утечек ПДн.

Покрывает:
1. LR-REL-1: Универсальный проектор масок форматирования project_digits_into_format
   с сохранением нецифровых символов (пробелы, дефисы, скобки, знаки №/N, плюс).
2. LR-REL-3: Обработка граничных форматов документов (паспорта 1997-1999 гг. и карты 13-19 цифр с Луном).
3. LR-COMP-2: Провайдер синонимизации дат DATE (сохранение разделителей, защита от утечек, C8).
4. KR-REL-1: Симметричный контекстный арбитраж бланков ЗАГС (рождение vs брак с окном +-50 символов).
5. LR-ARC-3: Слой санитарии, верификации и дедупликации спанов (Span Sanitizer, BIO-инвариант C3).
6. E2E: Сквозная детекция и обезличивание с сохранением форматов и контрольных сумм.
"""

from __future__ import annotations

import re
import pytest

from lab1.src.rules.inn_validator import validate_inn
from lab1.src.rules.luhn_validator import validate_luhn
from lab1.src.rules.ogrnip_validator import validate_ogrnip
from lab1.src.rules.regex_patterns import (
    PASSPORT_PATTERN,
    BANK_CARD_PATTERN,
)
from lab1.src.hybrid_pipeline import (
    detect_pii,
    _validate_passport,
    _validate_bank_card,
    _classify_zags_certificate,
)
from lab2.src.anonymizer import anonymize_text, _HANDLERS
from lab2.src.span_sanitizer import sanitize_spans, DEFAULT_TAG_PRIORITY
from lab2.src.faker_providers import (
    project_digits_into_format,
    anonymize_inn,
    anonymize_bank_card,
    anonymize_passport,
    anonymize_ogrnip,
    anonymize_snils,
    anonymize_phone,
    anonymize_date,
)


# ============================================================================
# 1. ТЕСТЫ LR-REL-1: Универсальный проектор масок форматирования
# ============================================================================
class TestLRREL1MaskProjection:
    """Тестирование сохранения нецифровых символов (пробелы, дефисы, скобки, №, N, плюс)."""

    def test_project_digits_preserves_spaces_and_hyphens(self):
        orig = "45-12 № 654321"
        new_digits = "4512999888"
        res = project_digits_into_format(orig, new_digits)
        assert res == "45-12 № 999888"

    def test_project_digits_preserves_phone_mask(self):
        orig = "+7 (999) 123-45-67"
        new_digits = "79998887766"
        res = project_digits_into_format(orig, new_digits)
        assert res == "+7 (999) 888-77-66"

    def test_project_digits_preserves_amex_mask(self):
        orig = "3782 822463 10005"
        new_digits = "378211111110005"
        res = project_digits_into_format(orig, new_digits)
        assert res == "3782 111111 10005"

    def test_project_digits_excess_new_digits_appended(self):
        orig = "№ 123"
        new_digits = "98765"
        res = project_digits_into_format(orig, new_digits)
        assert res == "№ 98765"

    def test_project_digits_deficit_padded_with_zero(self):
        orig = "(123) 456-78"
        new_digits = "99"
        res = project_digits_into_format(orig, new_digits)
        assert res == "(990) 000-00"

    def test_project_digits_empty_original(self):
        assert project_digits_into_format("", "12345") == "12345"

    def test_project_digits_no_digits_in_original(self):
        assert project_digits_into_format("NO_DIGITS - ", "123") == "NO_DIGITS - 123"

    def test_inn_12_preserves_hyphens_and_mod11(self):
        orig_inn = "7706-9895-9438"
        anon = anonymize_inn(orig_inn)
        assert anon != orig_inn
        assert anon.count("-") == 2
        parts = anon.split("-")
        assert len(parts[0]) == 4 and parts[0] == "7706"
        clean = anon.replace("-", "")
        assert len(clean) == 12
        assert validate_inn(clean)

    def test_inn_10_preserves_spaces_and_mod11(self):
        orig_inn = "7706 12345 6"
        anon = anonymize_inn(orig_inn)
        assert " " in anon
        clean = anon.replace(" ", "")
        assert len(clean) == 10
        assert clean.startswith("7706")
        assert validate_inn(clean)

    def test_passport_preserves_number_sign_and_series(self):
        orig_passport = "45-12 № 654321"
        anon = anonymize_passport(orig_passport)
        assert anon != orig_passport
        assert anon.startswith("45-12 № ")
        clean = re.sub(r"\D", "", anon)
        assert len(clean) == 10
        assert clean.startswith("4512")

    def test_passport_preserves_latin_n_sign(self):
        orig_passport = "45 12 N 654321"
        anon = anonymize_passport(orig_passport)
        assert anon != orig_passport
        assert " N " in anon
        clean = re.sub(r"\D", "", anon)
        assert len(clean) == 10
        assert clean.startswith("4512")

    def test_bank_card_amex_4_6_5_preservation(self):
        orig_card = "3782 822463 10005"
        anon = anonymize_bank_card(orig_card)
        assert anon != orig_card
        parts = anon.split()
        assert len(parts) == 3
        assert len(parts[0]) == 4 and len(parts[1]) == 6 and len(parts[2]) == 5
        clean = anon.replace(" ", "")
        assert len(clean) == 15
        assert clean.startswith("378282")
        assert clean.endswith("0005")
        assert validate_luhn(clean)

    def test_bank_card_hyphens_preservation(self):
        orig_card = "4276-3800-1234-5678"
        anon = anonymize_bank_card(orig_card)
        assert anon.count("-") == 3
        clean = anon.replace("-", "")
        assert len(clean) == 16
        assert clean.startswith("427638")
        assert clean.endswith("5678")
        assert validate_luhn(clean)

    def test_ogrnip_preserves_formatting(self):
        orig_ogrnip = "30450-01160-00157"
        anon = anonymize_ogrnip(orig_ogrnip)
        assert anon != orig_ogrnip
        assert anon.count("-") == 2
        clean = anon.replace("-", "")
        assert len(clean) == 15
        assert clean.startswith("30450")
        assert validate_ogrnip(clean)

    def test_snils_preserves_standard_dashes_and_checksum(self):
        orig_snils = "123-456-789 01"
        anon = anonymize_snils(orig_snils)
        assert anon != orig_snils
        assert re.match(r"^\d{3}-\d{3}-\d{3} \d{2}$", anon)
        clean = re.sub(r"\D", "", anon)
        total = sum(int(clean[i]) * (9 - i) for i in range(9))
        expected = total if total < 100 else (total % 101 if total % 101 < 100 else 0)
        assert int(clean[-2:]) == expected

    def test_phone_preserves_all_delimiters(self):
        orig_phone = "+7 (999) 123-45-67"
        anon = anonymize_phone(orig_phone)
        assert anon != orig_phone
        assert anon.startswith("+7 (999) ")
        assert anon.count("-") == 2


# ============================================================================
# 2. ТЕСТЫ LR-REL-3: Граничные форматы документов
# ============================================================================
class TestLRREL3DocumentBoundaryFormats:
    """Тестирование паспортов 1997-1999 годов и карт переменной длины (13-19 цифр)."""

    @pytest.mark.parametrize("passport_str", [
        "45 97 123456",  # 1997 год (начало выдачи по Указу № 696)
        "77 98 654321",  # 1998 год
        "50 99 987654",  # 1999 год
        "45 00 112233",  # 2000 год
        "45 24 445566",  # 2024 год
        "77 26 123456",  # 2026 год
    ])
    def test_passport_rf_valid_years_detection(self, passport_str):
        assert _validate_passport(passport_str)
        text = f"Паспорт гражданина РФ: {passport_str} выдан подразделением ОУФМС."
        spans = detect_pii(text)
        assert any(s[0] == "PASSPORT" for s in spans), f"Паспорт {passport_str} не обнаружен"

    def test_passport_with_hyphen_and_number_sign_detection(self):
        text = "Предъявлен паспорт 45-12 № 654321 гражданина."
        spans = detect_pii(text)
        passport_spans = [s for s in spans if s[0] == "PASSPORT"]
        assert len(passport_spans) == 1
        captured = text[passport_spans[0][1]:passport_spans[0][2]]
        assert "45-12" in captured
        assert "654321" in captured

    @pytest.mark.parametrize("invalid_passport", [
        "45 85 123456",  # 1985 год (советский бланк)
        "45 96 123456",  # 1996 год (до указа № 696)
        "45 35 123456",  # 2035 год (из будущего)
        "00 15 123456",  # невалидный регион 00
    ])
    def test_passport_rf_invalid_rejected(self, invalid_passport):
        assert not _validate_passport(invalid_passport)

    def test_bank_card_13_digits_visa_legacy(self):
        # 13 цифр: 4012888888881 (валидный Лун)
        card13 = "4012888888881"
        assert validate_luhn(card13)
        assert _validate_bank_card(card13)
        text = f"Номер счета привязан к карте {card13} Visa legacy."
        spans = detect_pii(text)
        assert any(s[0] == "BANK_CARD" for s in spans)

    def test_bank_card_15_digits_amex(self):
        # 15 цифр: 3782 822463 10005 (Amex 4-6-5, валидный Лун)
        amex = "3782 822463 10005"
        clean = amex.replace(" ", "")
        assert validate_luhn(clean)
        assert _validate_bank_card(amex)
        text = f"Оплата картой American Express: {amex} проведена успешно."
        spans = detect_pii(text)
        card_spans = [s for s in spans if s[0] == "BANK_CARD"]
        assert len(card_spans) == 1

    def test_bank_card_18_digits_maestro(self):
        # 18 цифр: 6762 1234 5678 9012 32 (валидный Лун)
        card18 = "6762 1234 5678 9012 32"
        clean = card18.replace(" ", "")
        assert len(clean) == 18
        assert validate_luhn(clean)
        assert _validate_bank_card(card18)
        text = f"Карта Maestro: {card18} активна."
        spans = detect_pii(text)
        card_spans = [s for s in spans if s[0] == "BANK_CARD"]
        assert len(card_spans) == 1

    def test_bank_card_19_digits_continuous(self):
        # 19 цифр сплошным номером: подберем валидный Лун
        base18 = "220012345678901234"
        total = 0
        for i, d in enumerate(reversed([int(c) for c in base18])):
            v = int(d)
            if i % 2 == 0:  # в 19-значном числе позиция справа от luhn check имеет индекс 1
                v *= 2
                if v > 9:
                    v -= 9
            total += v
        chk = (10 - (total % 10)) % 10
        card19 = base18 + str(chk)
        assert len(card19) == 19
        assert validate_luhn(card19)
        assert _validate_bank_card(card19)
        text = f"Номер социальной карты МИР: {card19}."
        spans = detect_pii(text)
        assert any(s[0] == "BANK_CARD" for s in spans)

    def test_bank_card_luhn_rejection(self):
        # 16-значный номер с невалидным Луном
        invalid_card = "4276 3800 1234 5678"
        clean = invalid_card.replace(" ", "")
        assert not validate_luhn(clean)
        assert not _validate_bank_card(invalid_card)
        text = f"Невалидная карта {invalid_card} не должна распознаваться."
        spans = detect_pii(text)
        assert not any(s[0] == "BANK_CARD" for s in spans)


# ============================================================================
# 3. ТЕСТЫ LR-COMP-2: Провайдер синонимизации дат (DATE)
# ============================================================================
class TestLRCOMP2DateAnonymization:
    """Тестирование провайдера дат DATE, сохранения формата и предотвращения утечек."""

    def test_anonymize_date_dots_format(self):
        orig = "25.10.1985"
        fakes = {anonymize_date(orig) for _ in range(15)}
        assert len(fakes) > 1, "Генератор должен быть стохастичным (C8)"
        for fake in fakes:
            assert fake != orig, "Дата должна быть изменена (Zero Leakage Rate)"
            assert "." in fake
            parts = fake.split(".")
            assert len(parts) == 3
            day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
            assert 1 <= month <= 12
            assert 1 <= day <= 31
            assert 1950 <= year <= 2026

    def test_anonymize_date_slashes_format(self):
        orig = "15/06/2018"
        fake = anonymize_date(orig)
        assert fake != orig
        assert "/" in fake
        assert "." not in fake
        parts = fake.split("/")
        assert len(parts) == 3

    def test_anonymize_date_iso_hyphens_format(self):
        orig = "1995-12-31"
        fake = anonymize_date(orig)
        assert fake != orig
        assert "-" in fake
        parts = fake.split("-")
        assert len(parts) == 3
        assert len(parts[0]) == 4  # Год в начале
        year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
        assert 1 <= month <= 12
        assert 1 <= day <= 31

    def test_anonymize_date_2digit_year(self):
        orig = "12.05.98"
        fake = anonymize_date(orig)
        assert fake != orig
        parts = fake.split(".")
        assert len(parts) == 3
        assert len(parts[2]) == 2

    def test_anonymize_date_text_russian_format(self):
        orig = "15 мая 2023 г."
        fakes = {anonymize_date(orig) for _ in range(10)}
        assert len(fakes) > 1
        for fake in fakes:
            assert fake != orig
            assert fake.endswith(" г.")
            tokens = fake.split()
            assert len(tokens) == 4
            assert tokens[1] in [
                "января", "февраля", "марта", "апреля", "мая", "июня",
                "июля", "августа", "сентября", "октября", "ноября", "декабря"
            ]

    def test_anonymize_text_date_integration(self):
        assert "DATE" in _HANDLERS, "Обработчик DATE должен быть в _HANDLERS"
        text = "Гражданин Иванов родился 15.06.1990 в городе Москве."
        spans = [["DATE", 25, 35]]
        anon_text = anonymize_text(text, spans)
        assert "15.06.1990" not in anon_text
        assert re.search(r"\d{2}\.\d{2}\.\d{4}", anon_text)


# ============================================================================
# 4. ТЕСТЫ KR-REL-1: Симметричный контекстный арбитраж ЗАГС
# ============================================================================
class TestKRREL1ZagsCertificateArbitration:
    """Тестирование симметричного контекстного арбитража бланков ЗАГС."""

    def test_marriage_long_left_context(self):
        # Дистанция контекста 38 символов от маркера брака до номера
        text = "Актовая запись о заключении брака между супругами произведена под номером II-АГ № 123456 в ЗАГС."
        spans = detect_pii(text)
        zags_spans = [s for s in spans if s[0] in ("MARRIAGE_CERTIFICATE", "BIRTH_CERTIFICATE")]
        assert len(zags_spans) == 1, "Номер должен порождать ровно один спан (C3)"
        assert zags_spans[0][0] == "MARRIAGE_CERTIFICATE"

    def test_marriage_right_context(self):
        text = "Был предъявлен документ II-АГ № 123456 о государственной регистрации заключения брака."
        spans = detect_pii(text)
        zags_spans = [s for s in spans if s[0] in ("MARRIAGE_CERTIFICATE", "BIRTH_CERTIFICATE")]
        assert len(zags_spans) == 1
        assert zags_spans[0][0] == "MARRIAGE_CERTIFICATE"

    def test_birth_long_left_context(self):
        text = "В книге записей актов о рождении зарегистрирован ребенок, бланк серии II-АГ № 123456 выдан матери."
        spans = detect_pii(text)
        zags_spans = [s for s in spans if s[0] in ("MARRIAGE_CERTIFICATE", "BIRTH_CERTIFICATE")]
        assert len(zags_spans) == 1
        assert zags_spans[0][0] == "BIRTH_CERTIFICATE"

    def test_birth_right_context(self):
        text = "Предъявлен бланк II-АГ № 123456, удостоверяющий факт рождения ребенка."
        spans = detect_pii(text)
        zags_spans = [s for s in spans if s[0] in ("MARRIAGE_CERTIFICATE", "BIRTH_CERTIFICATE")]
        assert len(zags_spans) == 1
        assert zags_spans[0][0] == "BIRTH_CERTIFICATE"

    def test_mixed_context_closer_marker_wins(self):
        # Брак упомянут в начале, но рождение значительно ближе к номеру
        text = "После заключения брака у супругов родился первенец, выдано свидетельство о рождении: II-АГ № 123456."
        spans = detect_pii(text)
        zags_spans = [s for s in spans if s[0] in ("MARRIAGE_CERTIFICATE", "BIRTH_CERTIFICATE")]
        assert len(zags_spans) == 1
        assert zags_spans[0][0] == "BIRTH_CERTIFICATE"

    def test_isolated_number_defaults_to_birth(self):
        text = "В архиве найден документ с номером II-АГ № 123456 без пометок."
        spans = detect_pii(text)
        zags_spans = [s for s in spans if s[0] in ("MARRIAGE_CERTIFICATE", "BIRTH_CERTIFICATE")]
        assert len(zags_spans) == 1
        assert zags_spans[0][0] == "BIRTH_CERTIFICATE"


# ============================================================================
# 5. ТЕСТЫ LR-ARC-3: Span Sanitizer и BIO-инвариант C3
# ============================================================================
class TestLRARC3SpanSanitizer:
    """Тестирование верификации, очистки и детерминированного NMS спанов."""

    def test_sanitizer_filters_out_of_bounds_and_inverted(self):
        text = "Пример текста для проверки санитарии."
        text_len = len(text)
        raw_spans = [
            ("PER", -5, 10),      # отрицательный start
            ("ORG", 5, -2),       # отрицательный end
            ("LOC", 100, 110),    # за пределами строки
            ("ADDR", 15, 10),     # инвертированные границы start > end
            ("DATE", 5, 5),       # нулевая длина start == end
            ("INN", 0, 6),        # валидный
        ]
        sanitized = sanitize_spans(raw_spans, text_len)
        assert len(sanitized) == 1
        assert sanitized[0] == ("INN", 0, 6)

    def test_sanitizer_clamps_end_to_text_len(self):
        text = "Короткий текст"
        sanitized = sanitize_spans([("PER", 0, 100)], len(text))
        assert len(sanitized) == 1
        assert sanitized[0] == ("PER", 0, len(text))

    def test_sanitizer_nested_longer_span_wins(self):
        # PER [0, 20] против SURNAME [0, 10] и NAME [11, 20]
        text = "Иванов Иван Иванович"
        raw_spans = [
            ("SURNAME", 0, 10),
            ("PER", 0, 20),
            ("NAME", 11, 20),
        ]
        sanitized = sanitize_spans(raw_spans, len(text))
        assert len(sanitized) == 1
        assert sanitized[0] == ("PER", 0, 20)

    def test_sanitizer_partial_overlap_longer_wins(self):
        # [0, 10] (len 10) vs [5, 25] (len 20)
        raw_spans = [
            ("ORG", 0, 10),
            ("PER", 5, 25),
        ]
        sanitized = sanitize_spans(raw_spans, 50)
        assert len(sanitized) == 1
        assert sanitized[0] == ("PER", 5, 25)

    def test_sanitizer_equal_length_priority_wins(self):
        # Равная длина 10: BANK_CARD (приоритет 100) vs LOC (приоритет 55)
        raw_spans = [
            ("LOC", 5, 15),
            ("BANK_CARD", 5, 15),
        ]
        sanitized = sanitize_spans(raw_spans, 50)
        assert len(sanitized) == 1
        assert sanitized[0][0] == "BANK_CARD"

    def test_sanitizer_equal_length_and_priority_earlier_start_wins(self):
        # Одинаковая длина (10), одинаковый тег PER: [0, 10] vs [5, 15]
        raw_spans = [
            ("PER", 5, 15),
            ("PER", 0, 10),
        ]
        sanitized = sanitize_spans(raw_spans, 50)
        assert len(sanitized) == 1
        assert sanitized[0] == ("PER", 0, 10)

    def test_sanitizer_dict_and_tuple_inputs(self):
        raw_spans = [
            {"tag": "PER", "start": 0, "end": 10},
            {"entity_type": "LOC", "start": 15, "end": 20},
            ["ORG", 25, 30],
        ]
        sanitized = sanitize_spans(raw_spans, 50)
        assert len(sanitized) == 3
        assert sanitized[0] == ("PER", 0, 10)
        assert sanitized[1] == ("LOC", 15, 20)
        assert sanitized[2] == ("ORG", 25, 30)

    def test_strict_bio_invariant_c3_randomized(self):
        """Инвариант C3: гарантированное отсутствие пересечений спанов."""
        import random
        rng = random.Random(1337)
        text_len = 250
        tags = list(DEFAULT_TAG_PRIORITY.keys())
        raw_spans = []
        for _ in range(60):
            st = rng.randint(-15, 270)
            en = rng.randint(st - 5, st + 50)
            tag = rng.choice(tags)
            raw_spans.append((tag, st, en))

        sanitized = sanitize_spans(raw_spans, text_len)
        for i in range(len(sanitized) - 1):
            curr_s = sanitized[i]
            next_s = sanitized[i + 1]
            assert curr_s[1] < curr_s[2], "Длина спана должна быть строго положительной"
            assert curr_s[2] <= next_s[1], f"Нарушен BIO-инвариант C3: {curr_s} перекрывается с {next_s}"

    def test_anonymize_text_resilience_to_malformed_spans(self):
        text = "Иван Иванов передал 100 рублей."
        malformed_spans = [
            ("PER", -10, 5),
            ("ORG", 100, 200),
            None,
            "invalid_span",
            {"tag": "DATE", "start": "bad", "end": 10},
            ("NAME", 0, 4),
            ("SURNAME", 0, 11),
        ]
        result = anonymize_text(text, malformed_spans)
        assert isinstance(result, str)
        assert len(result) > 0


# ============================================================================
# 6. СКВОЗНОЙ ИНТЕГРАЦИОННЫЙ ТЕСТ: ЛР1 (детекция) + ЛР2 (обезличивание)
# ============================================================================
class TestSprint1EndToEndPipeline:
    """Сквозная проверка конвейера защиты ПДн."""

    def test_full_pipeline_sprint1(self):
        text = (
            "Договор № 45-12/98. Паспорт гражданина РФ: 45 98 123456 выдан ОВД. "
            "Номер карты American Express: 3782 822463 10005. "
            "ИНН: 770698959438. "
            "Дата подписания: 15.06.2023. "
            "Актовая запись о заключении брака зарегистрирована бланком II-АГ № 123456."
        )

        # 1. Детекция
        spans = detect_pii(text)
        tags = {s[0] for s in spans}
        assert "PASSPORT" in tags, "Паспорт 1998 года должен быть обнаружен"
        assert "BANK_CARD" in tags, "Карта Amex должна быть обнаружена"
        assert "INN" in tags, "ИНН должен быть обнаружен"
        assert "DATE" in tags, "Дата должна быть обнаружена"
        assert "MARRIAGE_CERTIFICATE" in tags, "Бланк брака должен быть классифицирован как MARRIAGE_CERTIFICATE"

        # 2. Обезличивание
        anon_text = anonymize_text(text, spans)

        # 3. Верификация Zero Leakage
        assert "45 98 123456" not in anon_text
        assert "3782 822463 10005" not in anon_text
        assert "770698959438" not in anon_text
        assert "15.06.2023" not in anon_text
        assert "II-АГ № 123456" not in anon_text

        # 4. Проверка сохранения масок и контрольных сумм
        # ИНН валиден по mod 11
        inn_match = re.search(r"ИНН:\s*(\d{12})", anon_text)
        assert inn_match, f"ИНН должен присутствовать в тексте: {anon_text}"
        anon_inn = inn_match.group(1)
        assert validate_inn(anon_inn)

        # Карта сохранила маску Amex 4-6-5
        card_match = re.search(r"American Express:\s*(\d{4}\s\d{6}\s\d{5})", anon_text)
        assert card_match, f"Маска Amex должна сохраниться: {anon_text}"
        anon_card = card_match.group(1).replace(" ", "")
        assert validate_luhn(anon_card)
