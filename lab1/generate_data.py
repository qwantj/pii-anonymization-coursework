"""
Генератор тестового набора данных для ЛР1.

Создаёт >= 100 текстов с >= 50 экземплярами каждого из 10 атрибутов ПДн
чётного варианта. Данные сохраняются в формате JSON.

Каждая запись:
{
    "text": "...",
    "entities": [["TAG", start, end], ...]
}
"""

import json
import random
import string
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from lab1.src.rules.inn_validator import generate_inn_check_digits
from lab1.src.rules.luhn_validator import generate_luhn_digit
from lab1.src.rules.ogrnip_validator import generate_ogrnip_check_digit

random.seed(42)

# === Словари ===
SURNAMES_M = [
    "Иванов", "Петров", "Сидоров", "Козлов", "Смирнов",
    "Кузнецов", "Попов", "Васильев", "Соколов", "Михайлов",
    "Новиков", "Фёдоров", "Морозов", "Волков", "Алексеев",
    "Лебедев", "Семёнов", "Егоров", "Павлов", "Николаев",
    "Марков", "Степанов", "Орлов", "Андреев", "Макаров",
    "Захаров", "Борисов", "Тарасов", "Белов", "Комаров",
]

SURNAMES_F = [
    "Иванова", "Петрова", "Сидорова", "Козлова", "Смирнова",
    "Кузнецова", "Попова", "Васильева", "Соколова", "Михайлова",
    "Новикова", "Фёдорова", "Морозова", "Волкова", "Алексеева",
    "Лебедева", "Семёнова", "Егорова", "Павлова", "Николаева",
    "Маркова", "Степанова", "Орлова", "Андреева", "Макарова",
    "Захарова", "Борисова", "Тарасова", "Белова", "Комарова",
]

NAMES_M = [
    "Иван", "Пётр", "Александр", "Дмитрий", "Андрей",
    "Сергей", "Михаил", "Алексей", "Николай", "Максим",
    "Евгений", "Владимир", "Артём", "Денис", "Роман",
    "Олег", "Виктор", "Кирилл", "Григорий", "Тимофей",
    "Павел", "Антон", "Игорь", "Юрий", "Фёдор",
    "Борис", "Геннадий", "Валерий", "Станислав", "Руслан",
]

NAMES_F = [
    "Анна", "Мария", "Елена", "Ольга", "Наталья",
    "Ирина", "Татьяна", "Светлана", "Екатерина", "Юлия",
    "Валентина", "Дарья", "Вера", "Надежда", "Людмила",
    "Галина", "Лариса", "Алина", "Полина", "Ксения",
    "Виктория", "Марина", "Зинаида", "Тамара", "Нина",
    "Раиса", "Любовь", "Жанна", "Диана", "Кристина",
]

TEMPLATES = [
    "Клиент {SURNAME} {NAME} обратился в банк с заявлением.",
    "Договор заключён с {NAME} {SURNAME}.",
    "По данным {SURNAME} {NAME}, оплата произведена.",
    "Заявка от {SURNAME} {NAME} принята к рассмотрению.",
    "Сотрудник {SURNAME} {NAME} подтвердил операцию.",
    "{NAME} {SURNAME} подал документы на регистрацию.",
    "В базе данных зарегистрирован {SURNAME} {NAME}.",
    "Обращение {SURNAME} {NAME} было рассмотрено.",
]

TEMPLATES_INN = [
    "ИНН налогоплательщика: {INN}.",
    "Присвоен ИНН {INN} клиенту.",
    "В декларации указан ИНН {INN}.",
    "Проверка ИНН {INN} завершена.",
]

TEMPLATES_PASSPORT = [
    "Паспорт серия {PASSPORT}, выдан УФМС.",
    "Документ, удостоверяющий личность: паспорт {PASSPORT}.",
    "Серия и номер паспорта: {PASSPORT}.",
]

TEMPLATES_CARD = [
    "Номер карты: {CARD}.",
    "Оплата по карте {CARD} успешна.",
    "На карту {CARD} зачислены средства.",
]

TEMPLATES_OGRNIP = [
    "ОГРНИП предпринимателя: {OGRNIP}.",
    "Зарегистрирован с ОГРНИП {OGRNIP}.",
    "Проверка ОГРНИП {OGRNIP} пройдена.",
]

TEMPLATES_IP = [
    "Подключение с IP-адреса {IP}.",
    "Зафиксирован вход с {IP}.",
    "Сервер доступен по адресу {IP}.",
]

TEMPLATES_EMAIL = [
    "Электронная почта: {EMAIL}.",
    "Свяжитесь по адресу {EMAIL}.",
    "Письмо направлено на {EMAIL}.",
]

TEMPLATES_BIRTH_CERT = [
    "Свидетельство о рождении {CERT}.",
    "Документ: свидетельство {CERT}.",
    "Предъявлено свидетельство {CERT}.",
]

TEMPLATES_KLADR = [
    "Код КЛАДР адреса: {KLADR}.",
    "Адрес по КЛАДР: {KLADR}.",
    "Объект зарегистрирован по КЛАДР {KLADR}.",
]

TEMPLATES_PHONE = [
    "Контактный номер: {PHONE}.",
    "Абонент доступен по номеру {PHONE}.",
    "Телефон для связи: {PHONE}.",
]


def _gen_inn_12() -> str:
    """Сгенерировать валидный ИНН-12 (физлицо)."""
    # Первые 4 цифры — код налогового органа (валидный)
    region_codes = [
        "7700", "7706", "7719", "7725", "5000", "5001", "5006",
        "7800", "7801", "7802", "6300", "6600", "1600", "0200",
        "2300", "3400", "3800", "5200", "5400", "6100",
    ]
    code = random.choice(region_codes)
    # Следующие 6 цифр — порядковый номер
    middle = ''.join(random.choices(string.digits, k=6))
    base = code + middle
    return generate_inn_check_digits(base)


def _gen_passport() -> str:
    """Сгенерировать серию и номер паспорта."""
    region = random.randint(1, 99)
    year = random.randint(0, 24)
    number = random.randint(100000, 999999)
    return f"{region:02d} {year:02d} {number}"


def _gen_card() -> str:
    """Сгенерировать валидный номер банковской карты."""
    # BIN: VISA (4xxx), MasterCard (5xxx), МИР (2200-2204)
    bins = ["4276", "4561", "5469", "5536", "2200", "2202"]
    bin_prefix = random.choice(bins)
    middle = ''.join(random.choices(string.digits, k=11))
    prefix = bin_prefix + middle
    full = generate_luhn_digit(prefix)
    return f"{full[:4]} {full[4:8]} {full[8:12]} {full[12:]}"


def _gen_ogrnip() -> str:
    """Сгенерировать валидный ОГРНИП."""
    first = random.choice(["3", "4"])
    rest = ''.join(random.choices(string.digits, k=13))
    base = first + rest
    return generate_ogrnip_check_digit(base)


def _gen_ip() -> str:
    """Сгенерировать случайный IPv4."""
    return ".".join(str(random.randint(1, 254)) for _ in range(4))


def _gen_email() -> str:
    """Сгенерировать случайный email."""
    names = ["ivan", "petr", "alex", "dmitry", "anna", "maria", "elena", "olga"]
    domains = ["mail.ru", "yandex.ru", "gmail.com", "rambler.ru", "inbox.ru"]
    name = random.choice(names)
    suffix = random.randint(1, 999)
    domain = random.choice(domains)
    return f"{name}{suffix}@{domain}"


def _gen_birth_cert() -> str:
    """Сгенерировать номер свидетельства о рождении."""
    roman_nums = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
    roman = random.choice(roman_nums)
    letters = ''.join(random.choices("АБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЭЮЯ", k=2))
    number = random.randint(100000, 999999)
    return f"{roman}-{letters} № {number}"


def _gen_kladr() -> str:
    """Сгенерировать код КЛАДР (11 цифр)."""
    region = random.randint(1, 89)
    rest = ''.join(random.choices(string.digits, k=9))
    return f"{region:02d}{rest}"


def _gen_phone() -> str:
    """Сгенерировать случайный номер телефона."""
    formats = [
        "+7 ({code}) {n1}-{n2}-{n3}",
        "8 {code} {n1} {n2} {n3}",
        "+7{code}{n1}{n2}{n3}"
    ]
    code = f"{random.randint(900, 999)}"
    n1 = f"{random.randint(100, 999):03d}"
    n2 = f"{random.randint(10, 99):02d}"
    n3 = f"{random.randint(10, 99):02d}"
    return random.choice(formats).format(code=code, n1=n1, n2=n2, n3=n3)

def _find_in_text(text: str, value: str) -> tuple[int, int]:
    """Найти позицию подстроки в тексте."""
    idx = text.find(value)
    return idx, idx + len(value)


def generate_dataset(
    output_path: str,
    min_per_class: int = 55,
    seed: int | None = None,
) -> None:
    """
    Сгенерировать тестовый набор данных и сохранить в JSON.

    Args:
        output_path: путь к выходному JSON-файлу.
        min_per_class: минимальное количество примеров на каждый атрибут.
        seed: начальное значение генератора случайных чисел.
    """
    if seed is not None:
        random.seed(seed)

    samples = []

    # === Фамилии и Имена ===
    for _ in range(min_per_class):
        is_male = random.random() > 0.5
        surname = random.choice(SURNAMES_M if is_male else SURNAMES_F)
        name = random.choice(NAMES_M if is_male else NAMES_F)
        template = random.choice(TEMPLATES)
        text = template.format(SURNAME=surname, NAME=name)

        entities = []
        s_start, s_end = _find_in_text(text, surname)
        n_start, n_end = _find_in_text(text, name)
        if s_start >= 0:
            entities.append(["SURNAME", s_start, s_end])
        if n_start >= 0:
            entities.append(["NAME", n_start, n_end])

        samples.append({"text": text, "entities": entities})

    # === ИНН ===
    for _ in range(min_per_class):
        inn = _gen_inn_12()
        template = random.choice(TEMPLATES_INN)
        text = template.format(INN=inn)
        start, end = _find_in_text(text, inn)
        samples.append({"text": text, "entities": [["INN", start, end]]})

    # === Паспорт ===
    for _ in range(min_per_class):
        passport = _gen_passport()
        template = random.choice(TEMPLATES_PASSPORT)
        text = template.format(PASSPORT=passport)
        start, end = _find_in_text(text, passport)
        samples.append({"text": text, "entities": [["PASSPORT", start, end]]})

    # === Банковская карта ===
    for _ in range(min_per_class):
        card = _gen_card()
        template = random.choice(TEMPLATES_CARD)
        text = template.format(CARD=card)
        start, end = _find_in_text(text, card)
        samples.append({"text": text, "entities": [["BANK_CARD", start, end]]})

    # === ОГРНИП ===
    for _ in range(min_per_class):
        ogrnip = _gen_ogrnip()
        template = random.choice(TEMPLATES_OGRNIP)
        text = template.format(OGRNIP=ogrnip)
        start, end = _find_in_text(text, ogrnip)
        samples.append({"text": text, "entities": [["OGRNIP", start, end]]})

    # === IP адрес ===
    for _ in range(min_per_class):
        ip = _gen_ip()
        template = random.choice(TEMPLATES_IP)
        text = template.format(IP=ip)
        start, end = _find_in_text(text, ip)
        samples.append({"text": text, "entities": [["IP_ADDRESS", start, end]]})

    # === Email ===
    for _ in range(min_per_class):
        email = _gen_email()
        template = random.choice(TEMPLATES_EMAIL)
        text = template.format(EMAIL=email)
        start, end = _find_in_text(text, email)
        samples.append({"text": text, "entities": [["EMAIL", start, end]]})

    # === Свидетельство о рождении ===
    for _ in range(min_per_class):
        cert = _gen_birth_cert()
        template = random.choice(TEMPLATES_BIRTH_CERT)
        text = template.format(CERT=cert)
        start, end = _find_in_text(text, cert)
        samples.append({"text": text, "entities": [["BIRTH_CERTIFICATE", start, end]]})

    # === КЛАДР ===
    for _ in range(min_per_class):
        kladr = _gen_kladr()
        template = random.choice(TEMPLATES_KLADR)
        text = template.format(KLADR=kladr)
        start, end = _find_in_text(text, kladr)
        samples.append({"text": text, "entities": [["KLADR", start, end]]})

    # === Телефон ===
    for _ in range(min_per_class):
        phone = _gen_phone()
        template = random.choice(TEMPLATES_PHONE)
        text = template.format(PHONE=phone)
        start, end = _find_in_text(text, phone)
        samples.append({"text": text, "entities": [["PHONE_NUMBER", start, end]]})

    # === Комплексные тексты (несколько атрибутов в одном) ===
    for _ in range(30):
        is_male = random.random() > 0.5
        surname = random.choice(SURNAMES_M if is_male else SURNAMES_F)
        name = random.choice(NAMES_M if is_male else NAMES_F)
        inn = _gen_inn_12()
        email = _gen_email()
        card = _gen_card()

        text = (
            f"Клиент {surname} {name} (ИНН: {inn}) "
            f"оплатил картой {card}. "
            f"Контакт: {email}."
        )

        entities = []
        for value, tag in [
            (surname, "SURNAME"), (name, "NAME"), (inn, "INN"),
            (card, "BANK_CARD"), (email, "EMAIL"),
        ]:
            start, end = _find_in_text(text, value)
            if start >= 0:
                entities.append([tag, start, end])

        samples.append({"text": text, "entities": entities})

    random.shuffle(samples)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    # Статистика
    from collections import Counter
    counter = Counter()
    for sample in samples:
        for ent in sample["entities"]:
            counter[ent[0]] += 1

    print(f"Сгенерировано {len(samples)} тестовых текстов:")
    for tag, count in sorted(counter.items()):
        print(f"  {tag}: {count}")
    print(f"Сохранено в {output_path}")


if __name__ == "__main__":
    output = os.path.join(os.path.dirname(__file__), "data", "test_samples.json")
    generate_dataset(output)
