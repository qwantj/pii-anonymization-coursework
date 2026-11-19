"""
Регрессионные тесты исправлений КР (C1–C7):
1. C1: Женские имена заменяются женскими именами, а не фамилиями.
2. C2: Двойные дефисные фамилии распознаются и обезличиваются целиком в любом падеже.
3. C3: Двуполые имена (Саша/Женя) корректно определяют род с изоляцией клауз
   и сохраняют идемпотентность при нейтральном контексте.
4. C5/C7: Датасет coursework/data/dataset.json содержит ровно 10 атрибутов чётного варианта
   (>= 100 текстов и >= 50 экземпляров каждого атрибута).
5. C6: Разрешение коллизий OGRNIP vs BANK_CARD, KLADR/SNILS vs PHONE_NUMBER, URL без точки.
"""

from __future__ import annotations

import json
import os
from collections import Counter

from coursework.src.core_pipeline import CourseworkPipeline, EVEN_VARIANT_TAGS, anon
from coursework.src.idempotency_manager import IdempotencyVault
from lab1.src.hybrid_pipeline import detect_pii
from lab2.src.anonymizer import _NAMES_F, _NAMES_M, _SURNAMES_F, _SURNAMES_M


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_c1_female_first_names_not_replaced_by_surnames():
    """C1: Женские имена не должны заменяться на фамилии."""
    pipe = CourseworkPipeline(detector_type="logreg")
    names_f_low = {n.lower() for n in _NAMES_F}
    surnames_f_low = {s.lower() for s in _SURNAMES_F}

    for _ in range(10):
        out, spans = pipe.process("Анна пришла в банк.", mode="synonymic", vault=IdempotencyVault())
        first_word = out.split()[0].lower()
        assert first_word in names_f_low, f"Женское имя 'Анна' заменено не на имя: {out}"
        assert first_word not in surnames_f_low, f"Женское имя заменено на фамилию: {out}"

    out2, _ = pipe.process("Иванова Анна оплатила счёт.", mode="synonymic", vault=IdempotencyVault())
    w_sur, w_nam = out2.split()[0].lower(), out2.split()[1].lower()
    assert w_sur in surnames_f_low, f"Женская фамилия 'Иванова' не в пуле женских фамилий: {out2}"
    assert w_nam in names_f_low, f"Женское имя 'Анна' не в пуле женских имён: {out2}"


def test_c2_hyphenated_surname_full_span_and_anonymization():
    """C2: Двойные фамилии (Петров-Водкин) распознаются и обезличиваются целиком."""
    pipe = CourseworkPipeline(detector_type="logreg")

    text1 = "Петров-Водкин Иван подписал договор."
    out1, spans1 = pipe.process(text1, mode="synonymic", vault=IdempotencyVault())
    surname_texts1 = [text1[s[1]:s[2]] for s in spans1 if s[0] == "SURNAME"]
    assert "Петров-Водкин" in surname_texts1, f"Спан не покрыл всю дефисную фамилию: {spans1}"
    assert "Петров" not in out1 and "Водкин" not in out1, f"Часть фамилии осталась нетронутой: {out1}"
    assert "-" in out1.split()[0]

    text2 = "Справка выдана Петрову-Водкину Ивану."
    out2, spans2 = pipe.process(text2, mode="synonymic", vault=IdempotencyVault())
    surname_texts2 = [text2[s[1]:s[2]] for s in spans2 if s[0] == "SURNAME"]
    assert "Петрову-Водкину" in surname_texts2, f"Дательный падеж дефисной фамилии не покрыт: {spans2}"
    assert "Петрову" not in out2 and "Водкину" not in out2, f"Часть фамилии в дат.п. осталась: {out2}"


def test_c3_unisex_names_clause_isolation_and_idempotency():
    """C3: Двуполые имена в разных клаузах получают верный род и сохраняют идемпотентность."""
    pipe = CourseworkPipeline(detector_type="logreg")
    names_f_low = {n.lower() for n in _NAMES_F}
    names_m_low = {n.lower() for n in _NAMES_M}

    # Разные лица в одном сложном предложении
    text_complex = "Саша пришла в офис, а Женя ушёл домой."
    out_complex, _ = pipe.process(text_complex, mode="synonymic", vault=IdempotencyVault())
    tokens = out_complex.split()
    sasha_rep = tokens[0].lower()
    zhenya_rep = tokens[5].lower()
    assert sasha_rep in names_f_low, f"'Саша пришла' должна получить женское имя: {out_complex}"
    assert zhenya_rep in names_m_low, f"'Женя ушёл' должна получить мужское имя: {out_complex}"

    # Нейтральный повтор того же лица сохраняет идемпотентность
    text_idemp = "Саша оплатила госпошлину. Документы Саши готовы."
    out_idemp, _ = pipe.process(text_idemp, mode="synonymic", vault=IdempotencyVault())
    rep1 = out_idemp.split()[0]
    rep2 = out_idemp.split()[4]
    assert rep1[:3].lower() == rep2[:3].lower(), f"Нарушена идемпотентность унисекс-имени: {out_idemp}"


def test_c5_dataset_strictly_10_even_attributes():
    """C5: Бенчмарк coursework/data/dataset.json строго соответствует 10 атрибутам чётного варианта."""
    ds_path = os.path.join(_PROJECT_ROOT, "coursework", "data", "dataset.json")
    assert os.path.exists(ds_path)
    with open(ds_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) >= 100
    counts = Counter(e[0] for item in data for e in item["entities"])
    assert set(counts.keys()) == set(EVEN_VARIANT_TAGS)
    for tag, cnt in counts.items():
        assert cnt >= 50, f"Атрибут {tag} имеет {cnt} < 50 экземпляров"


def test_c6_collision_resolution_ogrnip_kladr_snils_url():
    """C6: OGRNIP не путается с картой, KLADR и SNILS с телефоном, URL не берет точку."""
    t_ogrnip = "Выписка по ИП, ОГРНИП 341084100459744 подтверждена."
    spans_o = detect_pii(t_ogrnip)
    assert [s[0] for s in spans_o if "341084100459744" in t_ogrnip[s[1]:s[2]]] == ["OGRNIP"]

    t_kladr = "Адрес по КЛАДР: 85460171573. Уведомление на https://portal-6.example.ru/docs/6."
    spans_k = detect_pii(t_kladr)
    extracted = {s[0]: t_kladr[s[1]:s[2]] for s in spans_k}
    assert extracted.get("KLADR") == "85460171573"
    assert "PHONE_NUMBER" not in extracted
    assert extracted.get("URL") == "https://portal-6.example.ru/docs/6"

    t_snils = "СНИЛС 843-373-186 13 принят."
    spans_s = detect_pii(t_snils)
    assert [s[0] for s in spans_s] == ["SNILS"]


def test_anon_api_signature_and_output_format():
    """Проверка высокоуровневой функции anon(text) -> (anonymized_text, spans)."""
    out, spans = anon("Иван получил кредит")
    assert isinstance(out, str)
    assert spans == [["NAME", 0, 4]]
    rep_name = out.split()[0]
    assert rep_name != "Иван"
    assert rep_name.lower() in {n.lower() for n in _NAMES_M}


def test_dative_name_anne_idempotency_and_inflection():
    """Дательный падеж имени 'Анне' в повторном упоминании сохраняет идемпотентность и склоняется."""
    text = "Петрова-Водкина Анна подписала договор. Справка выдана Петровой-Водкиной Анне."
    out, spans = anon(text)
    assert [s[0] for s in spans] == ["SURNAME", "NAME", "SURNAME", "NAME"]
    parts = out.replace(".", "").split()
    sur1, name1 = parts[0], parts[1]
    sur2, name2 = parts[6], parts[7]
    assert sur1[:4].lower() == sur2[:4].lower()
    assert name1[:3].lower() == name2[:3].lower()
    assert name2.lower() != name1.lower() or name1.lower().endswith(("и", "е"))
    assert name2.lower().endswith(("е", "и"))


def test_mixed_gender_pair_recognition():
    """Распознавание NAME в паре 'Иванов Пётр и Иванова Анна подписали договор.'."""
    text = "Иванов Пётр и Иванова Анна подписали договор."
    out, spans = anon(text)
    assert [s[0] for s in spans] == ["SURNAME", "NAME", "SURNAME", "NAME"]
    tokens = out.split()
    assert tokens[1].lower() in {n.lower() for n in _NAMES_M}
    assert tokens[4].lower() in {n.lower() for n in _NAMES_F}


def test_unisex_name_sentence_start_and_oblique_cases():
    """Начало предложения и косвенные падежи двуполых имён распознаются как NAME с единым родом."""
    text = "Женя пришёл вовремя. Документы Жени проверены, Жене выдали пропуск."
    out, spans = anon(text)
    assert [s[0] for s in spans] == ["NAME", "NAME", "NAME"]
    words = out.replace(",", "").replace(".", "").split()
    n1, n2, n3 = words[0], words[4], words[6]
    assert n1.lower() in {n.lower() for n in _NAMES_M}
    assert n1[:3].lower() == n2[:3].lower() == n3[:3].lower()
