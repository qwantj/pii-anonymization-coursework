"""
Тесты билингвальной маршрутизации (RU/EN) и сохранения морфологии:
1. Одновременное обнаружение и замена сущностей на русском и английском языках в одном тексте.
2. Сохранение грамматических падежей (родительный, дательный, творительный).
3. Корректное обезличивание почтовых адресов ADDR (ru/en).
4. Корректное выделение ОПФ и кавычек.
"""

import pytest
from lab2.src.anonymizer import anonymize_text
from lab2.src.morph_utils import (
    get_morph_info,
    inflect_word,
    is_latin_text,
    strip_quotes_and_legal_forms,
    handle_english_casing_and_possessive,
)
from lab2.src.faker_providers import anonymize_address, anonymize_org


def test_bilingual_routing_in_single_text():
    """Тест одновременного обезличивания русских и английских сущностей в одном документе."""
    text = "Контракт подписан между Петровым Алексеем из Москвы и John Doe из New York на 5th Avenue ."

    spans = [
        ("PER", text.find("Петровым Алексеем"), text.find("Петровым Алексеем") + len("Петровым Алексеем")),
        ("LOC", text.find("Москвы"), text.find("Москвы") + len("Москвы")),
        ("PER", text.find("John Doe"), text.find("John Doe") + len("John Doe")),
        ("LOC", text.find("New York"), text.find("New York") + len("New York")),
        ("ADDR", text.find("5th Avenue"), text.find("5th Avenue") + len("5th Avenue")),
    ]

    anon = anonymize_text(text, spans)

    # Проверяем, что исходные значения заменены
    assert "Петровым Алексеем" not in anon
    assert "John Doe" not in anon
    assert "New York" not in anon
    assert "5th Avenue" not in anon

    # Проверяем, что в результирующем тексте остались и кириллические, и латинские слова
    assert any("а" <= c.lower() <= "я" for c in anon)
    assert any("a" <= c.lower() <= "z" for c in anon)


def test_morphology_case_preservation():
    """Тест сохранения падежных форм при синонимизации."""
    # Родительный падеж
    text_gent = "Заявление от гражданки Смирновой Анны ."
    spans_gent = [("SURNAME", text_gent.find("Смирновой"), text_gent.find("Смирновой") + len("Смирновой"))]
    anon_gent = anonymize_text(text_gent, spans_gent)
    replaced_word = anon_gent.split()[3]
    info = get_morph_info(replaced_word)
    assert info["case"] == "gent", f"Ожидался родительный падеж, получено: {info['case']}"

    # Дательный падеж
    text_datv = "Справка выдана Петрову Ивану ."
    spans_datv = [("SURNAME", text_datv.find("Петрову"), text_datv.find("Петрову") + len("Петрову"))]
    anon_datv = anonymize_text(text_datv, spans_datv)
    replaced_word_datv = anon_datv.split()[2]
    info_datv = get_morph_info(replaced_word_datv)
    assert info_datv["case"] == "datv", f"Ожидался дательный падеж, получено: {info_datv['case']}"

    # Творительный падеж
    text_ablt = "Согласовано с генеральным директором Соколовым ."
    spans_ablt = [("SURNAME", text_ablt.find("Соколовым"), text_ablt.find("Соколовым") + len("Соколовым"))]
    anon_ablt = anonymize_text(text_ablt, spans_ablt)
    replaced_word_ablt = anon_ablt.split()[-2]
    info_ablt = get_morph_info(replaced_word_ablt)
    assert info_ablt["case"] == "ablt", f"Ожидался творительный падеж, получено: {info_ablt['case']}"


def test_english_casing_and_possessive():
    """Тест сохранения регистра и притяжательного 's для английских сущностей."""
    res_upper = handle_english_casing_and_possessive("MICROSOFT", "google")
    assert res_upper.isupper()

    res_possessive = handle_english_casing_and_possessive("Smith's", "Taylor")
    assert res_possessive.endswith("'s")


def test_address_anonymization_structure():
    """Тест обезличивания адресов (ADDR) для русского и английского языков."""
    addr_ru = "125009, г. Москва, ул. Тверская, д. 15, корп. 2, кв. 84"
    anon_ru = anonymize_address(addr_ru, lang="ru")
    assert "ул." in anon_ru or "проспект" in anon_ru
    assert "д." in anon_ru

    addr_en = "10001, 350 5th Avenue, Apt. 42"
    anon_en = anonymize_address(addr_en, lang="en")
    assert any(street in anon_en for street in ("Avenue", "Street", "Boulevard", "Lane", "Broadway"))


def test_strip_quotes_and_legal_forms():
    """Тест отделения кавычек и формы собственности (ООО, ПАО, АО)."""
    prefix, l_q, core, r_q = strip_quotes_and_legal_forms('ООО «Альфа-Инновации»')
    assert prefix == "ООО "
    assert l_q == "«"
    assert core == "Альфа-Инновации"
    assert r_q == "»"
