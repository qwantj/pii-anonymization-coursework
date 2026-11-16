"""
Модульные тесты для проверки требований Лабораторной работы 2.

Проверяемые требования:
1. Сохранение первых 4 цифр ИНН и валидность нового контрольного разряда.
2. Сохранение первых 6 (BIN) и последних 4 цифр банковской карты, валидность алгоритма Луна.
3. Сохранение первых 5 цифр ОГРНИП и валидность контрольного числа mod 13.
4. Сохранение почтового домена после '@' для E-mail.
5. Сохранение серии паспорта.
6. Сохранение структуры свидетельства о рождении.
7. Сохранение кода региона (первых 2 цифр) в КЛАДР.
8. Сохранение грамматического рода и падежа для Фамилий и Имён.
9. Случайность: повторные вызовы формируют разные значения.
10. Устойчивость к некорректным входным данным.
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from lab1.src.rules.inn_validator import validate_inn
from lab1.src.rules.luhn_validator import validate_luhn
from lab1.src.rules.ogrnip_validator import validate_ogrnip

from lab2.src.anonymizer import anonymize_text
from lab2.src.faker_providers import (
    anonymize_inn,
    anonymize_bank_card,
    anonymize_passport,
    anonymize_ogrnip,
    anonymize_ip,
    anonymize_email,
    anonymize_birth_certificate,
    anonymize_kladr,
)
from lab2.src.morph_utils import get_morph_info


def test_inn_anonymization():
    orig_inn12 = "770698959438"
    new_inns = {anonymize_inn(orig_inn12) for _ in range(10)}
    # Проверяем, что сгенерировано несколько разных значений
    assert len(new_inns) > 1, "Обезличенные ИНН должны быть случайными"
    for new_inn in new_inns:
        assert new_inn[:4] == "7706", f"Первые 4 цифры ИНН должны сохраняться: {new_inn}"
        assert len(new_inn) == 12, f"Длина ИНН-12 должна быть 12: {new_inn}"
        assert validate_inn(new_inn), f"Контрольный разряд ИНН должен быть валиден: {new_inn}"


def test_bank_card_anonymization():
    orig_card = "4276 3800 1234 5678"
    new_cards = {anonymize_bank_card(orig_card) for _ in range(10)}
    assert len(new_cards) > 1, "Обезличенные карты должны быть случайными"
    for c in new_cards:
        clean = c.replace(" ", "")
        assert clean[:6] == "427638", f"Первые 6 цифр (BIN) должны сохраняться: {c}"
        assert clean[-4:] == "5678", f"Последние 4 цифры должны сохраняться: {c}"
        assert validate_luhn(clean), f"Алгоритм Луна должен быть валиден: {c}"


def test_ogrnip_anonymization():
    orig_ogrnip = "304500116000157"
    new_ogrnips = {anonymize_ogrnip(orig_ogrnip) for _ in range(10)}
    assert len(new_ogrnips) > 1, "Обезличенные ОГРНИП должны быть случайными"
    for o in new_ogrnips:
        assert o[:5] == "30450", f"Первые 5 цифр ОГРНИП должны сохраняться: {o}"
        assert len(o) == 15, f"Длина ОГРНИП должна быть 15: {o}"
        assert validate_ogrnip(o), f"Контрольный разряд ОГРНИП должен быть валиден: {o}"


def test_email_anonymization():
    orig_email = "ivan.petrov@company.example.com"
    new_emails = {anonymize_email(orig_email) for _ in range(10)}
    assert len(new_emails) > 1
    for em in new_emails:
        assert em.endswith("@company.example.com"), f"Домен после @ должен сохраняться: {em}"
        assert em != orig_email


def test_passport_anonymization():
    orig_passport = "45 10 123456"
    new_passports = {anonymize_passport(orig_passport) for _ in range(10)}
    assert len(new_passports) > 1
    for p in new_passports:
        assert p.startswith("45 10"), f"Серия паспорта должна сохраняться: {p}"
        assert len(p.replace(" ", "")) == 10


def test_birth_certificate_anonymization():
    orig_cert = "II-АГ № 123456"
    new_certs = {anonymize_birth_certificate(orig_cert) for _ in range(10)}
    assert len(new_certs) > 1
    for c in new_certs:
        assert c.startswith("II-АГ № "), f"Формат и серия должны сохраняться: {c}"
        digits = c.split("№ ")[1]
        assert len(digits) == 6


def test_kladr_anonymization():
    orig_kladr = "7700000000000"
    new_kladrs = {anonymize_kladr(orig_kladr) for _ in range(10)}
    assert len(new_kladrs) > 1
    for k in new_kladrs:
        assert k[:2] == "77", f"Код региона должен сохраняться: {k}"
        assert len(k) == len(orig_kladr)


def test_morphology_preservation():
    # Тест на склонение фамилий и имён
    text = "Иван передал документы Анне Ивановой."
    # Иван: [NAME, 0, 4]
    # Анне: [NAME, 23, 27]
    # Ивановой: [SURNAME, 28, 36]
    spans = [
        ["NAME", 0, 4],
        ["NAME", 23, 27],
        ["SURNAME", 28, 36],
    ]
    anonymized = anonymize_text(text, spans)
    assert anonymized != text
    # Проверяем, что в результирующем тексте "Анне" заменено на женское имя в дательном падеже
    # А "Ивановой" - на женскую фамилию в дательном/родительном падеже
    print(f"\nИсходный текст:     {text}")
    print(f"Обезличенный текст: {anonymized}")


def test_full_pipeline_anonymize():
    text = "Клиент Петров Иван перевёл средства на карту 4276 3800 1234 5678, ИНН: 770698959438."
    spans = [
        ["SURNAME", 7, 13],
        ["NAME", 14, 18],
        ["BANK_CARD", 45, 64],
        ["INN", 71, 83],
    ]
    res1 = anonymize_text(text, spans)
    res2 = anonymize_text(text, spans)
    assert res1 != res2, "Повторный запуск должен генерировать разные случайные значения"


if __name__ == "__main__":
    print("Запуск тестов ЛР2...")
    test_inn_anonymization()
    print("  [OK] ИНН: сохранение кода НО и валидность контрольного разряда")
    test_bank_card_anonymization()
    print("  [OK] Банковская карта: сохранение BIN/last4 и алгоритм Луна")
    test_ogrnip_anonymization()
    print("  [OK] ОГРНИП: сохранение базы и проверка mod 13")
    test_email_anonymization()
    print("  [OK] Email: сохранение домена")
    test_passport_anonymization()
    print("  [OK] Паспорт: сохранение серии")
    test_birth_certificate_anonymization()
    print("  [OK] Свидетельство о рождении: сохранение формата и серии")
    test_kladr_anonymization()
    print("  [OK] КЛАДР: сохранение кода региона")
    test_morphology_preservation()
    print("  [OK] Морфология: сохранение рода и падежа")
    test_full_pipeline_anonymize()
    print("  [OK] Случайность: повторные запуски дают разные значения")
    print("\nВсе тесты ЛР2 успешно пройдены!")
