"""
Генератор тестового бенчмарка персональных данных для Курсовой работы (Чётный вариант № 8).

Требования:
- Строго 10 целевых атрибутов чётного варианта:
  SURNAME, NAME, INN, PASSPORT, BANK_CARD, OGRNIP, IP_ADDRESS, EMAIL, BIRTH_CERTIFICATE, KLADR.
- Не менее 100 связных текстов и не менее 50 экземпляров каждого из 10 атрибутов.
- Основа реальных ФИО: предложения из открытого корпуса factRuEval-2016 (testset),
  дополненные связными текстами со всеми недостающими атрибутами ПДн чётного варианта.
- Валидные контрольные разряды (ИНН-10/12, Лун для карт, ОГРНИП mod 13).
- Тексты с повторными упоминаниями в разных падежах (проверка идемпотентности),
  амбивалентными именами (Женя/Саша) и двойными фамилиями (Петров-Водкин и др.).
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from lab1.generate_data import (
    _gen_inn_12, _gen_passport, _gen_card, _gen_ogrnip, _gen_ip,
    _gen_email, _gen_birth_cert, _gen_kladr,
    SURNAMES_M, SURNAMES_F, NAMES_M, NAMES_F, _find_in_text,
)
from lab1.src.rules.inn_validator import generate_inn_check_digits
from lab2.src.morph_utils import inflect_word


def _gen_inn_10() -> str:
    """Сгенерировать валидный 10-значный ИНН юридического лица."""
    codes = ["7707", "7706", "7801", "5001", "6601", "5200", "2300", "1600"]
    base = random.choice(codes) + "".join(str(random.randint(0, 9)) for _ in range(5))
    return generate_inn_check_digits(base)


def _assemble_text_with_spans(parts: list[tuple[str | None, str]]) -> tuple[str, list[list]]:
    """
    Собрать текст из фрагментов [(tag_or_None, chunk), ...] с точным расчётом границ спанов.
    Полностью исключает ошибки смещений и вложенные дубликаты при совпадении основ слов.
    """
    chunks: list[str] = []
    entities: list[list] = []
    cursor = 0
    for tag, chunk in parts:
        start = cursor
        end = cursor + len(chunk)
        chunks.append(chunk)
        if tag is not None:
            entities.append([tag, start, end])
        cursor = end
    return "".join(chunks), entities


def _load_factrueval_testset_samples(limit: int = 45) -> list[dict]:
    """Загрузить реальные предложения с ФИО из корпуса factRuEval-2016 (testset)."""
    test_dir = os.path.join(
        PROJECT_ROOT, "lab1", "data", "factRuEval", "factRuEval-2016-master", "testset"
    )
    if os.path.isdir(test_dir):
        from lab1.src.ner.train_ner_model import extract_clean_factrueval_sentences
        return extract_clean_factrueval_sentences(test_dir, limit=limit)

    fallback_json = os.path.join(os.path.dirname(__file__), "factrueval_testset_45.json")
    if os.path.exists(fallback_json):
        with open(fallback_json, "r", encoding="utf-8") as f:
            return json.load(f)[:limit]
    return []


def generate_coursework_dataset(output_path: str, min_per_attr: int = 60) -> list[dict]:
    """Сгенерировать и сохранить бенчмарк-датасет из 10 атрибутов чётного варианта для КР."""
    random.seed(12345)
    samples: list[dict] = []

    # 1. Реальные тексты из factRuEval-2016 (testset, невиданный при обучении)
    fre_samples = _load_factrueval_testset_samples(limit=45)
    samples.extend(fre_samples)

    # 2. Тексты с повторным упоминанием одного лица в разных падежах (идемпотентность)
    for idx in range(30):
        is_m = random.random() > 0.5
        sur_nom = random.choice(SURNAMES_M if is_m else SURNAMES_F)
        name_nom = random.choice(NAMES_M if is_m else NAMES_F)
        gender = "masc" if is_m else "femn"
        card = _gen_card()
        inn = _gen_inn_12() if idx % 2 == 0 else _gen_inn_10()
        email = _gen_email()

        sur_dat = inflect_word(sur_nom, "datv", gender)
        name_dat = inflect_word(name_nom, "datv", gender)
        sur_gen = inflect_word(sur_nom, "gent", gender)
        name_gen = inflect_word(name_nom, "gent", gender)

        text, entities = _assemble_text_with_spans([
            (None, "Заявитель "),
            ("SURNAME", sur_nom),
            (None, " "),
            ("NAME", name_nom),
            (None, " обратился в отделение. Кредит был одобрен гражданину "),
            ("SURNAME", sur_dat),
            (None, " "),
            ("NAME", name_dat),
            (None, ". В договоре указан ИНН "),
            ("INN", inn),
            (None, " и карта "),
            ("BANK_CARD", card),
            (None, ". Копия направлена на email "),
            ("EMAIL", email),
            (None, ". Личное дело "),
            ("SURNAME", sur_gen),
            (None, " "),
            ("NAME", name_gen),
            (None, " отправлено в архив."),
        ])
        samples.append({"text": text, "entities": entities})

    # 3. Тексты со снятием омонимии двуполых имён (Саша/Женя/Валя + глагол прошедшего времени)
    ambig_cases = [
        ("Саша", "оплатила", "Иванова"),
        ("Саша", "оплатил", "Иванов"),
        ("Женя", "составила", "Петрова"),
        ("Женя", "составил", "Петров"),
        ("Валя", "подписала", "Сидорова"),
        ("Валя", "подписал", "Сидоров"),
    ]
    for _ in range(20):
        name, verb, surname = random.choice(ambig_cases)
        passport = _gen_passport()
        ogrnip = _gen_ogrnip()
        text, entities = _assemble_text_with_spans([
            (None, "Заявитель "),
            ("SURNAME", surname),
            (None, " "),
            ("NAME", name),
            (None, f" {verb} государственную пошлину по паспорту "),
            ("PASSPORT", passport),
            (None, ", ОГРНИП: "),
            ("OGRNIP", ogrnip),
            (None, "."),
        ])
        samples.append({"text": text, "entities": entities})

    # 4. Тексты с двойными фамилиями через дефис
    hyphen_cases = [
        ("Петров-Водкин", "Иван", "подписал"),
        ("Мамин-Сибиряк", "Дмитрий", "передал"),
        ("Салтыков-Щедрин", "Михаил", "оформил"),
        ("Римский-Корсаков", "Николай", "подтвердил"),
        ("Новиков-Прибой", "Алексей", "направил"),
        ("Петрова-Водкина", "Анна", "подписала"),
        ("Мамина-Сибиряк", "Елена", "передала"),
        ("Лебедева-Кумач", "Ольга", "оформила"),
    ]
    for idx in range(16):
        h_sur, h_name, verb = hyphen_cases[idx % len(hyphen_cases)]
        inn = _gen_inn_12()
        email = _gen_email()
        text, entities = _assemble_text_with_spans([
            (None, "Клиент "),
            ("SURNAME", h_sur),
            (None, " "),
            ("NAME", h_name),
            (None, f" {verb} заявление (ИНН "),
            ("INN", inn),
            (None, ") и указал почту "),
            ("EMAIL", email),
            (None, "."),
        ])
        samples.append({"text": text, "entities": entities})

    # 5. Тексты со свидетельствами о рождении, IP и КЛАДР
    for _ in range(max(60, min_per_attr)):
        cert = _gen_birth_cert()
        ip = _gen_ip()
        kladr = _gen_kladr()
        email = _gen_email()
        text, entities = _assemble_text_with_spans([
            (None, "Зарегистрировано рождение ребёнка, свидетельство "),
            ("BIRTH_CERTIFICATE", cert),
            (None, ". Место регистрации по КЛАДР: "),
            ("KLADR", kladr),
            (None, ". Уведомление отправлено на "),
            ("EMAIL", email),
            (None, ", запрос поступил с IP "),
            ("IP_ADDRESS", ip),
            (None, "."),
        ])
        samples.append({"text": text, "entities": entities})

    # 6. Тексты с индивидуальными предпринимателями (ОГРНИП + ИНН + КЛАДР + Паспорт)
    for _ in range(40):
        is_m = random.random() > 0.5
        sur = random.choice(SURNAMES_M if is_m else SURNAMES_F)
        name = random.choice(NAMES_M if is_m else NAMES_F)
        ogrnip = _gen_ogrnip()
        inn = _gen_inn_12()
        passp = _gen_passport()
        kladr = _gen_kladr()
        text, entities = _assemble_text_with_spans([
            (None, "ИП "),
            ("SURNAME", sur),
            (None, " "),
            ("NAME", name),
            (None, " (ОГРНИП "),
            ("OGRNIP", ogrnip),
            (None, ", ИНН "),
            ("INN", inn),
            (None, "). Паспортные данные: "),
            ("PASSPORT", passp),
            (None, ". Адрес регистрации по классификатору КЛАДР: "),
            ("KLADR", kladr),
            (None, "."),
        ])
        samples.append({"text": text, "entities": entities})

    # 7. Тексты с банковскими картами, паспортами и IP
    for _ in range(35):
        card = _gen_card()
        ip = _gen_ip()
        email = _gen_email()
        passp = _gen_passport()
        text, entities = _assemble_text_with_spans([
            (None, "Транзакция по банковской карте "),
            ("BANK_CARD", card),
            (None, " (паспорт "),
            ("PASSPORT", passp),
            (None, ") инициирована с сетевого узла "),
            ("IP_ADDRESS", ip),
            (None, ". Чек выслан на "),
            ("EMAIL", email),
            (None, "."),
        ])
        samples.append({"text": text, "entities": entities})

    random.shuffle(samples)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    return samples


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "dataset.json")
    dataset = generate_coursework_dataset(out)
    c = Counter()
    for s in dataset:
        for e in s["entities"]:
            c[e[0]] += 1
    print(f"Сгенерировано {len(dataset)} текстов для КР (чётный вариант, 10 атрибутов).")
    for k, v in sorted(c.items()):
        print(f"  {k}: {v}")
