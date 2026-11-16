"""
Генераторы синонимических обезличенных значений для структурированных атрибутов.

Все генераторы гарантируют:
  - Случайность (SystemRandom).
  - Сохранение косвенных признаков (BIN карты, код региона ИНН, серия паспорта).
  - Валидность контрольных разрядов по умолчанию (C5).
  - Поддержку настраиваемой политики обработки дефектов InvalidChecksumPolicy (LR-REL-4).
  - Статическую компиляцию всех регулярных выражений (LR-PERF-3).
  - Реалистичные структуры email и расширенные пулы организаций/топонимов (LR-COMP-3, KR-COMP-1).
"""

from __future__ import annotations

import calendar
from enum import Enum
import random
import re
import string
from typing import Optional

from lab1.src.rules.inn_validator import generate_inn_check_digits, validate_inn
from lab1.src.rules.luhn_validator import generate_luhn_digit, validate_luhn
from lab1.src.rules.ogrnip_validator import generate_ogrnip_check_digit, validate_ogrnip
from lab2.src.morphology_engine import normalize_lemma_cached

# Криптостойкий ГПСЧ
_rng = random.SystemRandom()

# Статическая компиляция регулярных выражений (LR-PERF-3)
_RE_DIGITS = re.compile(r"\D")
_RE_NON_DIGITS = re.compile(r"\d")
_RE_ZIP = re.compile(r"\b\d{5,6}\b")
_RE_URL = re.compile(r"(?i)(https?://)([^/]+)(.*)")
_RE_MARRIAGE = re.compile(r"\d{6}")
_RE_JOURNAL = re.compile(r"\d+")
_RE_DATE_TEXT = re.compile(
    r"^(?P<day>\d{1,2})\s+(?P<month>[а-яёА-ЯЁ]+)\s+(?P<year>\d{4})(?P<suffix>\s+(?:г\.|года|г))?$",
    re.IGNORECASE,
)
_RE_DATE_SEP = re.compile(r"[./-]")
_RE_BIRTH_CERT = re.compile(r"(.*?)(\d{6})(.*)")
_RE_EMAIL_SPLIT = re.compile(r"@")


class InvalidChecksumPolicy(Enum):
    """Политика обработки некорректных контрольных сумм во входных данных (LR-REL-4)."""
    FORCE_VALID = "force_valid"          # Всегда генерировать математически корректную сумму (дефолт)
    PRESERVE_DEFECT = "preserve_defect"  # Если вход был битым, выход генерируется намеренно с битой суммой
    PASS_THROUGH = "pass_through"        # Оставлять исходный ошибочный спан без изменений


_CURRENT_CHECKSUM_POLICY = InvalidChecksumPolicy.FORCE_VALID


def get_invalid_checksum_policy() -> InvalidChecksumPolicy:
    """Получить текущую глобальную политику контрольных сумм."""
    return _CURRENT_CHECKSUM_POLICY


def set_invalid_checksum_policy(policy: InvalidChecksumPolicy) -> None:
    """Установить текущую глобальную политику контрольных сумм."""
    global _CURRENT_CHECKSUM_POLICY
    _CURRENT_CHECKSUM_POLICY = policy


def project_digits_into_format(original_formatted: str, new_digits: str) -> str:
    """
    Проецирует сгенерированную последовательность цифр new_digits на маску
    исходной строки original_formatted, сохраняя любые нецифровые символы
    (пробелы, дефисы, тире, скобки, знаки №, N, +, слэши, точки и др.) (LR-REL-1).
    """
    if not original_formatted:
        return new_digits

    digit_iter = iter(new_digits)
    result: list[str] = []

    for char in original_formatted:
        if char.isdigit():
            val = next(digit_iter, "0")
            result.append(val)
        else:
            result.append(char)

    for rem in digit_iter:
        result.append(rem)

    return "".join(result)


def anonymize_inn(original: str, policy: Optional[InvalidChecksumPolicy] = None) -> str:
    """
    Обезличить ИНН с сохранением кода налогового органа (первые 4 цифры)
    и исходной маски форматирования (LR-REL-1, LR-REL-4).
    """
    pol = policy or _CURRENT_CHECKSUM_POLICY
    digits = "".join(c for c in original if c.isdigit())

    is_valid = validate_inn(digits) if len(digits) in (10, 12) else False

    if not is_valid:
        if pol == InvalidChecksumPolicy.PASS_THROUGH:
            return original

    if len(digits) == 12:
        prefix = digits[:4]
        middle = "".join(str(_rng.randint(0, 9)) for _ in range(6))
        base = prefix + middle
        new_digits = generate_inn_check_digits(base)
        if not is_valid and pol == InvalidChecksumPolicy.PRESERVE_DEFECT:
            # Намеренно искажаем контрольный разряд
            last_d = (int(new_digits[-1]) + 1) % 10
            new_digits = new_digits[:-1] + str(last_d)
        return project_digits_into_format(original, new_digits)
    elif len(digits) == 10:
        prefix = digits[:4]
        middle = "".join(str(_rng.randint(0, 9)) for _ in range(5))
        base = prefix + middle
        new_digits = generate_inn_check_digits(base)
        if not is_valid and pol == InvalidChecksumPolicy.PRESERVE_DEFECT:
            last_d = (int(new_digits[-1]) + 1) % 10
            new_digits = new_digits[:-1] + str(last_d)
        return project_digits_into_format(original, new_digits)
    else:
        return original


def anonymize_bank_card(original: str, policy: Optional[InvalidChecksumPolicy] = None) -> str:
    """
    Обезличить номер банковской карты с сохранением BIN (первые 6 цифр),
    последних 4 цифр и исходной маски форматирования (LR-REL-1, LR-REL-4).
    """
    pol = policy or _CURRENT_CHECKSUM_POLICY
    digits = "".join(c for c in original if c.isdigit())
    if len(digits) < 13:
        return original

    is_valid = validate_luhn(digits)
    if not is_valid:
        if pol == InvalidChecksumPolicy.PASS_THROUGH:
            return original

    bin_prefix = digits[:6]
    last4 = digits[-4:]

    mid_len = len(digits) - 6 - 4
    if mid_len <= 0:
        return original

    rand_mid = [_rng.randint(0, 9) for _ in range(mid_len - 1)]
    test_digits = [int(c) for c in bin_prefix] + rand_mid + [0] + [int(c) for c in last4]
    adj_index = len(bin_prefix) + len(rand_mid)

    total = 0
    for i, d in enumerate(reversed(test_digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d

    rem = total % 10
    adj_digit = (10 - rem) % 10
    if not is_valid and pol == InvalidChecksumPolicy.PRESERVE_DEFECT:
        adj_digit = (adj_digit + 1) % 10

    test_digits[adj_index] = adj_digit
    full_digits = "".join(str(d) for d in test_digits)

    return project_digits_into_format(original, full_digits)


def anonymize_passport(original: str) -> str:
    """Обезличить номер паспорта РФ с сохранением серии и маски форматирования (LR-REL-1)."""
    digits = "".join(c for c in original if c.isdigit())
    if len(digits) < 10:
        return original

    series = digits[:4]
    new_number = "".join(str(_rng.randint(0, 9)) for _ in range(6))
    full_digits = series + new_number
    return project_digits_into_format(original, full_digits)


def anonymize_ogrnip(original: str, policy: Optional[InvalidChecksumPolicy] = None) -> str:
    """Обезличить ОГРНИП с сохранением первых 5 цифр и валидным mod 13 (LR-REL-1, LR-REL-4)."""
    pol = policy or _CURRENT_CHECKSUM_POLICY
    digits = "".join(c for c in original if c.isdigit())
    if len(digits) != 15:
        return original

    is_valid = validate_ogrnip(digits)
    if not is_valid:
        if pol == InvalidChecksumPolicy.PASS_THROUGH:
            return original

    prefix = digits[:5]
    middle = "".join(str(_rng.randint(0, 9)) for _ in range(9))
    base14 = prefix + middle
    new_digits = generate_ogrnip_check_digit(base14)

    if not is_valid and pol == InvalidChecksumPolicy.PRESERVE_DEFECT:
        last_d = (int(new_digits[-1]) + 1) % 10
        new_digits = new_digits[:-1] + str(last_d)

    return project_digits_into_format(original, new_digits)


def anonymize_ip(original: str) -> str:
    """Обезличить IPv4-адрес."""
    return ".".join(str(_rng.randint(1, 254)) for _ in range(4))


_TRANSLIT_DICT = {
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'e',
    'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm',
    'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
    'ф': 'f', 'х': 'kh', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'shch',
    'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya',
}


def transliterate_text(text: str) -> str:
    """Транслитерация русского текста в латиницу для правдоподобных адресов email (LR-COMP-3)."""
    res = []
    for c in text.lower():
        res.append(_TRANSLIT_DICT.get(c, c))
    return "".join(res)


def anonymize_email_realistic(original_email: str, current_person_nom: Optional[str] = None) -> str:
    """
    Генерация правдоподобного email на основе имени субъекта или частотных лексем (LR-COMP-3).
    Сохраняет исходный домен после '@' (инвариант C5).
    """
    if "@" not in original_email:
        return original_email

    _, domain = original_email.rsplit("@", 1)

    if current_person_nom:
        words = current_person_nom.strip().split()
        if len(words) >= 2:
            w0 = transliterate_text(words[0])
            w1 = transliterate_text(words[1])
            user = f"{w0}.{w1}"
            user = re.sub(r"[^a-z0-9._-]", "", user)
            if user:
                return f"{user}@{domain}"
        elif len(words) == 1:
            tr = transliterate_text(words[0])
            tr = re.sub(r"[^a-z0-9._-]", "", tr)
            if tr:
                return f"{tr}.{_rng.randint(10, 99)}@{domain}"

    first_names = [
        "alex", "dmitry", "elena", "sergey", "anna", "ivan", "olga", "maxim",
        "kirill", "artem", "mikhail", "andrey", "pavel", "marina", "tatiana",
    ]
    last_names = [
        "smirnov", "kuznetsov", "popov", "ivanov", "vasiliev", "morozov",
        "novikov", "fedorov", "volkov", "alexeev", "lebedev", "semenov",
    ]
    first = _rng.choice(first_names)
    last = _rng.choice(last_names)
    num = _rng.randint(10, 99)
    return f"{first}.{last}{num}@{domain}"


def anonymize_email(original: str) -> str:
    """Обезличить email с сохранением домена после @ (LR-COMP-3)."""
    return anonymize_email_realistic(original)


def anonymize_birth_certificate(original: str) -> str:
    """Обезличить номер свидетельства о рождении с сохранением серии."""
    match = _RE_BIRTH_CERT.match(original)
    if not match:
        return original
    prefix = match.group(1)
    suffix = match.group(3)
    new_number = "".join(str(_rng.randint(0, 9)) for _ in range(6))
    return f"{prefix}{new_number}{suffix}"


def anonymize_kladr(original: str) -> str:
    """Обезличить код КЛАДР: сохраняем код субъекта РФ (первые 2 цифры)."""
    digits = "".join(c for c in original if c.isdigit())
    if len(digits) < 11:
        return original
    prefix = digits[:2]
    rest = "".join(str(_rng.randint(0, 9)) for _ in range(len(digits) - 2))
    return prefix + rest


def anonymize_phone(original: str) -> str:
    """Обезличить номер телефона с сохранением маски форматирования (LR-REL-1)."""
    digits = "".join(c for c in original if c.isdigit())
    if not digits:
        return original
    prefix = digits[:4]
    rest_len = max(0, len(digits) - len(prefix))
    new_rest = "".join(str(_rng.randint(0, 9)) for _ in range(rest_len))
    return project_digits_into_format(original, prefix + new_rest)


def anonymize_snils(original: str, policy: Optional[InvalidChecksumPolicy] = None) -> str:
    """Обезличить СНИЛС с валидной контрольной суммой (LR-REL-1, LR-REL-4)."""
    pol = policy or _CURRENT_CHECKSUM_POLICY
    digits = "".join(c for c in original if c.isdigit())
    if len(digits) != 11:
        return original

    # Проверка исходной валидности
    base_orig = digits[:9]
    chk_orig = int(digits[9:])
    tot_orig = sum(int(d) * (9 - i) for i, d in enumerate(base_orig))
    expected_chk = tot_orig if tot_orig < 100 else (tot_orig % 101 if tot_orig % 101 < 100 else 0)
    is_valid = (chk_orig == expected_chk)

    if not is_valid:
        if pol == InvalidChecksumPolicy.PASS_THROUGH:
            return original

    base = "".join(str(_rng.randint(0, 9)) for _ in range(9))
    total = sum(int(d) * (9 - i) for i, d in enumerate(base))
    check = total if total < 100 else (total % 101 if total % 101 < 100 else 0)

    if not is_valid and pol == InvalidChecksumPolicy.PRESERVE_DEFECT:
        check = (check + 1) % 100

    full_digits = base + f"{check:02d}"
    return project_digits_into_format(original, full_digits)


_MONTH_NAMES_RU = [
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря"
]
_MONTH_NAME_TO_NUM = {name: idx + 1 for idx, name in enumerate(_MONTH_NAMES_RU)}


def _days_in_month(month: int, year: int) -> int:
    try:
        return calendar.monthrange(year, month)[1]
    except Exception:
        if month in (1, 3, 5, 7, 8, 10, 12):
            return 31
        elif month in (4, 6, 9, 11):
            return 30
        else:
            is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
            return 29 if is_leap else 28


def anonymize_date(original_date_str: str, mode: str = "random") -> str:
    """Обезличить дату с сохранением разделителей (LR-COMP-2)."""
    if not original_date_str or not original_date_str.strip():
        return original_date_str

    raw = original_date_str.strip()

    # 1. Текстовый формат: "15 мая 2023 г."
    m_text = _RE_DATE_TEXT.match(raw)
    if m_text:
        orig_day_str = m_text.group("day")
        orig_month_name = m_text.group("month").lower()
        orig_year = int(m_text.group("year"))
        suffix = m_text.group("suffix") or ""

        if orig_month_name in _MONTH_NAME_TO_NUM:
            for _ in range(5):
                year_shift = _rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
                new_year = max(1950, min(2025, orig_year + year_shift))
                new_month_num = _rng.randint(1, 12)
                max_d = _days_in_month(new_month_num, new_year)
                new_day = _rng.randint(1, max_d)
                new_month_name = _MONTH_NAMES_RU[new_month_num - 1]

                day_fmt = f"{new_day:02d}" if len(orig_day_str) == 2 and orig_day_str.startswith("0") else str(new_day)
                cand = f"{day_fmt} {new_month_name} {new_year}{suffix}"
                if cand.lower() != raw.lower():
                    return cand

    # 2. Цифровые форматы с разделителями (., /, -)
    sep_match = _RE_DATE_SEP.search(raw)
    if sep_match:
        sep = sep_match.group(0)
        parts = raw.split(sep)
        if len(parts) == 3:
            p0, p1, p2 = parts[0].strip(), parts[1].strip(), parts[2].strip()

            if len(p0) == 4 and p0.isdigit() and p1.isdigit() and p2.isdigit():
                orig_year = int(p0)
                for _ in range(5):
                    year_shift = _rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
                    new_year = max(1950, min(2025, orig_year + year_shift))
                    new_month = _rng.randint(1, 12)
                    max_d = _days_in_month(new_month, new_year)
                    new_day = _rng.randint(1, max_d)
                    cand = f"{new_year:04d}{sep}{new_month:02d}{sep}{new_day:02d}"
                    if cand != raw:
                        return cand

            elif len(p2) == 4 and p2.isdigit() and p0.isdigit() and p1.isdigit():
                orig_year = int(p2)
                for _ in range(5):
                    year_shift = _rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
                    new_year = max(1950, min(2025, orig_year + year_shift))
                    new_month = _rng.randint(1, 12)
                    max_d = _days_in_month(new_month, new_year)
                    new_day = _rng.randint(1, max_d)
                    day_fmt = f"{new_day:02d}" if len(p0) == 2 else str(new_day)
                    month_fmt = f"{new_month:02d}" if len(p1) == 2 else str(new_month)
                    cand = f"{day_fmt}{sep}{month_fmt}{sep}{new_year:04d}"
                    if cand != raw:
                        return cand

            elif len(p2) == 2 and p2.isdigit() and p0.isdigit() and p1.isdigit():
                orig_year_2d = int(p2)
                for _ in range(5):
                    new_year_2d = (orig_year_2d + _rng.choice([-3, -2, -1, 1, 2, 3])) % 100
                    new_month = _rng.randint(1, 12)
                    full_year = 2000 + new_year_2d if new_year_2d <= 26 else 1900 + new_year_2d
                    max_d = _days_in_month(new_month, full_year)
                    new_day = _rng.randint(1, max_d)
                    cand = f"{new_day:02d}{sep}{new_month:02d}{sep}{new_year_2d:02d}"
                    if cand != raw:
                        return cand

    return "15.06.2018"


def anonymize_url(original: str) -> str:
    match = _RE_URL.match(original)
    if not match:
        return original
    host = match.group(2)
    return f"{match.group(1)}example-{_rng.randint(100, 999)}.{host.rsplit('.', 1)[-1]}{match.group(3)}"


def anonymize_marriage_certificate(original: str) -> str:
    return _RE_MARRIAGE.sub(lambda _: "".join(str(_rng.randint(0, 9)) for _ in range(6)), original, count=1)


def anonymize_cadastral_number(original: str) -> str:
    parts = original.split(":")
    if len(parts) != 4:
        return original
    return ":".join([parts[0], parts[1], parts[2], str(_rng.randint(1, 9999999))])


def anonymize_journal(original: str, odd: bool) -> str:
    digits = "".join(c for c in original if c.isdigit())
    if not digits:
        return original
    value = _rng.randint(1, 999999)
    if value % 2 != (1 if odd else 0):
        value += 1
    replacement = str(value)
    return _RE_JOURNAL.sub(replacement, original, count=1)


def anonymize_address(original: str, lang: str = "ru") -> str:
    """Обезличить почтовый адрес (ADDR) с сохранением структуры."""
    has_zip = bool(_RE_ZIP.search(original))
    has_apt = any(kw in original.lower() for kw in ("кв.", "квартира", "apt", "suite", "unit", "офис", "комн."))

    if lang == "en":
        candidates = [
            "Baker Street", "Broadway", "Sunset Boulevard", "Main Street",
            "Market Street", "Wall Street", "Oxford Street", "Park Lane",
            "Madison Avenue", "Michigan Avenue", "Pennsylvania Avenue"
        ]
        filtered = [s for s in candidates if s.lower() not in original.lower()]
        fake_street = _rng.choice(filtered or candidates)
        fake_house = _rng.randint(1, 999)
        apt_str = f", Apt. {_rng.randint(1, 200)}" if has_apt else ""
        zip_str = f"{_rng.randint(10000, 99999)}, " if has_zip else ""
        return f"{zip_str}{fake_house} {fake_street}{apt_str}"
    else:
        fake_streets = [
            "ул. Тверская", "Ленинский проспект", "ул. Арбат", "Невский проспект",
            "ул. Малая Бронная", "Кутузовский проспект", "ул. Баумана", "ул. Гагарина",
            "ул. Пушкина", "ул. Советская", "проспект Мира", "ул. Кирова", "ул. Ленина",
            "ул. Садовая", "ул. Большая Покровская", "ул. Красная", "проспект Ленина",
            "ул. Мира", "ул. Комсомольская", "ул. Победы", "ул. Чапаева", "ул. Чкалова",
        ]
        street = _rng.choice(fake_streets)
        house = _rng.randint(1, 150)
        corp = f", корп. {_rng.randint(1, 4)}" if "корп" in original.lower() else ""
        apt_str = f", кв. {_rng.randint(1, 350)}" if has_apt else ""
        zip_str = f"{_rng.randint(100000, 699999)}, " if has_zip else ""
        return f"{zip_str}{street}, д. {house}{corp}{apt_str}"


# 50+ реалистичных названий организаций (LR-COMP-3, KR-COMP-1)
_ORG_NAMES_RU = [
    "«Вектор»", "«Альфа»", "«Инновация»", "«Технологии будущего»", "«Омега»",
    "«Ресурс»", "«Стандарт»", "«Авангард»", "«Прогресс»", "«Кристалл»",
    "«Сириус»", "«Горизонт»", "«Энергия»", "«Спектр»", "«Импульс»",
    "«Квант»", "«Атлант»", "«Флагман»", "«Олимп»", "«Триумф»",
    "«Монолит»", "«Глобус»", "«Матрица»", "«Элемент»", "«Интеграл»",
    "«Сфера»", "«Альянс»", "«Гарант»", "«Орион»", "«Титан»",
    "«Созвездие»", "«Феникс»", "«Компас»", "«Форвард»", "«Вершина»",
    "«Рубин»", "«Алмаз»", "«Сапфир»", "«Магистраль»", "«Континент»",
    "«Мегаполис»", "«Интеллект»", "«Инвест-Групп»", "«Технопарк»", "«Капитал»",
    "«Северсталь»", "«Норильский Никель»", "«Яндекс»", "«Ростелеком»", "«Аэрофлот»",
    "«Сбербанк»", "«Лукойл»", "«Газпромнефть»", "«ФосАгро»", "«Магнит»",
]

_ORG_NAMES_EN = [
    "Apex Corporation", "Global Logistics LLC", "Summit Technologies Inc.",
    "Nexus Solutions Corp.", "Pinnacle Holdings LLC", "Vanguard Industries Inc.",
    "Horizon Media Group", "Starlight Systems Inc.", "Frontier Analytics Ltd.",
    "Beacon Capital LLC", "Omega Innovations Corp.", "Atlas Cloud Solutions",
]


def anonymize_org(original: str, lang: str = "ru") -> str:
    """Обезличить организацию с сохранением ОПФ и расширенным пулом (LR-COMP-3)."""
    if lang == "en":
        candidates = [o for o in _ORG_NAMES_EN if o.lower() not in original.lower() and original.lower() not in o.lower()]
        return _rng.choice(candidates) if candidates else _rng.choice(_ORG_NAMES_EN)

    forms = ("ООО", "ПАО", "АО", "ЗАО", "ИП", "НКО")
    detected_form = None
    for form in forms:
        if form in original:
            detected_form = form
            break

    candidates = [o for o in _ORG_NAMES_RU if o.replace("«", "").replace("»", "").lower() not in original.lower()]
    chosen_name = _rng.choice(candidates) if candidates else _rng.choice(_ORG_NAMES_RU)
    if detected_form:
        return f"{detected_form} {chosen_name}"
    return chosen_name


# 50+ реалистичных российских топонимов (города и регионы РФ) (LR-COMP-3, KR-COMP-1)
_LOCATIONS_RU = [
    "Москва", "Санкт-Петербург", "Новосибирск", "Екатеринбург", "Казань",
    "Нижний Новгород", "Самара", "Омск", "Ростов-на-Дону", "Красноярск",
    "Челябинск", "Уфа", "Волгоград", "Пермь", "Краснодар", "Воронеж",
    "Саратов", "Тюмень", "Смоленск", "Ижевск", "Барнаул", "Ульяновск",
    "Иркутск", "Хабаровск", "Ярославль", "Владивосток", "Махачкала", "Томск",
    "Оренбург", "Вологда", "Новокузнецк", "Рязань", "Астрахань", "Набережные Челны",
    "Пенза", "Киров", "Липецк", "Чебоксары", "Калининград", "Тула", "Курск",
    "Севастополь", "Кострома", "Тамбов", "Тверь", "Магнитогорск", "Иваново",
    "Брянск", "Белгород", "Сургут", "Владимир", "Чита", "Архангельск", "Калуга",
]

_LOCATIONS_EN = [
    "New York", "London", "Chicago", "San Francisco", "Boston",
    "Toronto", "Dublin", "Sydney", "Melbourne", "Seattle", "Austin",
]


def anonymize_location(original: str, lang: str = "ru") -> str:
    """Обезличить топоним с расширенным пулом городов (LR-COMP-3)."""
    pool = _LOCATIONS_EN if lang == "en" else _LOCATIONS_RU
    orig_clean = original.strip().lower()
    if lang == "en":
        candidates = [loc for loc in pool if loc.lower() != orig_clean]
    else:
        orig_norm = normalize_lemma_cached(orig_clean)
        candidates = [
            loc for loc in pool
            if loc.lower() != orig_clean and normalize_lemma_cached(loc.lower()) != orig_norm
        ]
    if candidates:
        return _rng.choice(candidates)
    return _rng.choice(pool)
