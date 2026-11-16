"""
Гибридный пайплайн распознавания ПДн (ЛР1).

Объединяет два подхода:
1. Rule-based: регулярные выражения + валидация контрольных разрядов
   (ИНН, Банковские карты, ОГРНИП, Паспорт, Email, IP, СНИЛС, КЛАДР и др.)
2. ML-based: нейросетевая модель ai-forever/ru-en-RoSBERTa
   (PER, SURNAME, NAME, ORG, LOC, ADDR).

Приоритеты разрешения коллизий (Overlap Resolution):
  Priority 1 — сущности с подтверждёнными контрольными суммами (INN, BANK_CARD, OGRNIP)
  Priority 2 — нейросетевые спаны ru-en-RoSBERTa (PER, SURNAME, NAME, ORG, LOC, ADDR)
  Priority 3 — структурные паттерны (EMAIL, IP, PASSPORT, SNILS, PHONE, KLADR и др.)
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Union

from lab1.src.rules.regex_patterns import (
    INN_PATTERN,
    PASSPORT_PATTERN,
    BANK_CARD_PATTERN,
    OGRNIP_PATTERN,
    IP_ADDRESS_PATTERN,
    EMAIL_PATTERN,
    BIRTH_CERTIFICATE_PATTERN,
    KLADR_PATTERN,
    JOURNAL_ODD_PATTERN,
    JOURNAL_EVEN_PATTERN,
    SNILS_PATTERN,
    URL_PATTERN,
    CADASTRAL_NUMBER_PATTERN,
    DATE_PATTERN,
    PHONE_PATTERN,
)
from lab1.src.rules.inn_validator import validate_inn
from lab1.src.rules.luhn_validator import validate_luhn
from lab1.src.rules.ogrnip_validator import validate_ogrnip

if TYPE_CHECKING:
    from lab1.src.ner.rosberta_detector import RoSBERTaNerDetector

# Тип спана: (метка, начало, конец)
Span = tuple[str, int, int]
_PrioritizedSpan = tuple[str, int, int, Union[int, float]]

# Единый шаблон номера бланка акта гражданского состояния ЗАГС (рождение / брак) (KR-REL-1)
_ZAGS_CERTIFICATE_PATTERN = re.compile(
    r'(?i)\b((?:[IVX]{1,4}|X{0,3}(?:IX|IV|V?I{0,3}))-[А-ЯЁ]{2}\s?[№N]\s?\d{6})\b'
)

# Симметричные контекстные маркеры заключения брака (окно +-50 символов) (KR-REL-1)
_MARRIAGE_CONTEXT_PATTERN = re.compile(
    r"(?iu)\b(?:брак[а-яё]*|бракосочетан[а-яё]*|супруг[а-яё]*|женитьб[а-яё]*|"
    r"муж[а-яё]*|жен[а-яё]*|брачн[а-яё]*|свадеб[а-яё]*|свадьб[а-яё]*|"
    r"marriage|wedding|spouse|husband|wife)\b"
)

# Симметричные контекстные маркеры рождения (окно +-50 символов) (KR-REL-1)
_BIRTH_CONTEXT_PATTERN = re.compile(
    r"(?iu)\b(?:рожден[а-яё]*|рожд[а-яё]*|родил[а-яё]*|новорожден[а-яё]*|новорождён[а-яё]*|"
    r"реб[её]н[а-яё]*|дет[ея][а-яё]*|матер[а-яё]*|мать|отц[а-яё]*|отец|сын[а-яё]*|"
    r"дочер[а-яё]*|дочь|роддом[а-яё]*|birth|born|child|infant|newborn)\b"
)
_PASSPORT_CONTEXT_PATTERN = re.compile(
    r"(?iu)(?:паспорт|серия\s+и\s+номер|passport)"
)
_KLADR_CONTEXT_PATTERN = re.compile(
    r"(?iu)(?:к[лЛ]адр|kladr|адресн(?:ый|ого)\s+код|код\s+адреса|"
    r"справочник[а-яё]*\s+указан\s+код)"
)
_OGRNIP_CONTEXT_PATTERN = re.compile(
    r"(?iu)(?:огрнип|ogrnip|\bип\b|предпринимател)"
)
_CARD_CONTEXT_PATTERN = re.compile(
    r"(?iu)(?:карт[а-яё]*|card|visa|mastercard|мир|amex|сч[её]т)"
)
_PHONE_CONTEXT_PATTERN = re.compile(
    r"(?iu)(?:телефон|тел\.?|phone|моб\.?|звонить|номер)"
)


def _classify_zags_certificate(text: str, start: int, end: int) -> tuple[str, float]:
    """
    Симметричный контекстный арбитраж коллизий бланков ЗАГС (KR-REL-1).

    Анализирует окно до 50 символов слева и справа от номера документа.
    Возвращает (тег, приоритет):
    - ('MARRIAGE_CERTIFICATE', 2.5) при подтверждении маркерами брака;
    - ('BIRTH_CERTIFICATE', 2.5) при подтверждении маркерами рождения;
    - при наличии обоих маркеров побеждает ближайший к номеру документ;
    - ('BIRTH_CERTIFICATE', 3.0) при отсутствии контекстных маркеров.
    """
    left_start = max(0, start - 50)
    right_end = min(len(text), end + 50)

    left_context = text[left_start:start]
    right_context = text[end:right_end]

    marriage_dists = []
    for m in _MARRIAGE_CONTEXT_PATTERN.finditer(left_context):
        marriage_dists.append(start - (left_start + m.end()))
    for m in _MARRIAGE_CONTEXT_PATTERN.finditer(right_context):
        marriage_dists.append(m.start())

    birth_dists = []
    for m in _BIRTH_CONTEXT_PATTERN.finditer(left_context):
        birth_dists.append(start - (left_start + m.end()))
    for m in _BIRTH_CONTEXT_PATTERN.finditer(right_context):
        birth_dists.append(m.start())

    min_m = min(marriage_dists) if marriage_dists else float("inf")
    min_b = min(birth_dists) if birth_dists else float("inf")

    if min_m < min_b:
        return "MARRIAGE_CERTIFICATE", 2.5
    elif min_b < min_m:
        return "BIRTH_CERTIFICATE", 2.5
    else:
        return "BIRTH_CERTIFICATE", 3.0


def _find_regex_spans(
    pattern: re.Pattern, text: str, tag: str, priority: int,
    validator=None,
) -> list[_PrioritizedSpan]:
    """Найти все совпадения regex-паттерна в тексте."""
    spans = []
    for match in pattern.finditer(text):
        matched_text = match.group(0)
        if validator and not validator(matched_text):
            continue
        spans.append((tag, match.start(), match.end(), priority))
    return spans


def _validate_bank_card(text: str) -> bool:
    digits = re.sub(r"\D", "", text)
    if len(digits) < 13 or len(digits) > 19:
        return False
    return validate_luhn(digits)


def _validate_inn_wrapper(text: str) -> bool:
    digits = re.sub(r"\D", "", text)
    return validate_inn(digits)


def _validate_ogrnip_wrapper(text: str) -> bool:
    digits = re.sub(r"\D", "", text)
    return validate_ogrnip(digits)


def _validate_passport(text: str) -> bool:
    digits = re.sub(r"\D", "", text)
    if len(digits) != 10:
        return False
    region_code = int(digits[:2])
    issue_year = int(digits[2:4])
    # Гражданский паспорт РФ выдается с 1997 г. (Указ № 696) по 2026 г. (LR-REL-3)
    is_valid_year = (issue_year <= 26) or (issue_year >= 97)
    return 1 <= region_code <= 99 and is_valid_year


def _validate_snils(text: str) -> bool:
    digits = re.sub(r"\D", "", text)
    if len(digits) != 11:
        return False
    total = sum(int(digits[i]) * (9 - i) for i in range(9))
    expected = total if total < 100 else (total % 101 if total % 101 < 100 else 0)
    return int(digits[-2:]) == expected


def _has_context(text: str, start: int, end: int, pattern: re.Pattern) -> bool:
    left = text[max(0, start - 40):start]
    right = text[end:end + 20]
    return bool(pattern.search(left + right))


def _resolve_overlaps(spans: list[_PrioritizedSpan]) -> list[_PrioritizedSpan]:
    """
    Разрешение коллизий:
    - Сортировка по возрастанию числового приоритета (1 - наивысший, 2 - нейросеть, 3 - паттерны).
    - При равном приоритете побеждает более длинный спан.
    - Частичные перекрытия со спанами более высокого приоритета разрешаются в пользу высокого приоритета.
    """
    sorted_spans = sorted(spans, key=lambda s: (s[3], -(s[2] - s[1])))

    accepted: list[_PrioritizedSpan] = []
    occupied_positions: set[int] = set()

    for tag, start, end, priority in sorted_spans:
        span_range = set(range(start, end))
        overlap = span_range & occupied_positions

        if not overlap:
            accepted.append((tag, start, end, priority))
            occupied_positions.update(span_range)
        else:
            # Если перекрытие небольшое по краям (<20% спана), пытаемся подрезать спан
            remaining_positions = sorted(list(span_range - occupied_positions))
            if len(remaining_positions) >= 4 and (len(remaining_positions) / len(span_range)) > 0.6:
                new_start = min(remaining_positions)
                new_end = max(remaining_positions) + 1
                if new_end - new_start == len(remaining_positions):
                    accepted.append((tag, new_start, new_end, priority))
                    occupied_positions.update(remaining_positions)

    # Финальная сортировка по позиции в тексте
    accepted.sort(key=lambda s: s[1])
    return accepted


class HybridDetector:
    """Гибридный детектор ПДн: контрольные суммы + ai-forever/ru-en-RoSBERTa + паттерны."""

    def __init__(self, ner_detector=None):
        if ner_detector is not None:
            self._ner = ner_detector
        else:
            try:
                from lab1.src.ner.trained_detector import TrainedTokenClassifierDetector
                self._ner = TrainedTokenClassifierDetector()
            except Exception:
                try:
                    from lab1.src.ner.rosberta_detector import RoSBERTaNerDetector
                    self._ner = RoSBERTaNerDetector()
                except Exception:
                    self._ner = None

    def detect(self, text: str, target_tags: set[str] | frozenset[str] | None = None) -> list[list]:
        """Обнаружить все сущности ПДн в тексте с соблюдением приоритетов."""
        if not text or not text.strip():
            return []

        all_spans: list[_PrioritizedSpan] = []

        # ===== Priority 1: Валидация контрольных разрядов =====
        all_spans.extend(_find_regex_spans(INN_PATTERN, text, "INN", 1, _validate_inn_wrapper))

        # ОГРНИП (15 цифр на 3/4, mod 13): при наличии контекста или сплошном 15-значном формате
        # без карточного контекста имеет приоритет над 15-значной картой Amex
        for m in OGRNIP_PATTERN.finditer(text):
            val = m.group(0)
            if _validate_ogrnip_wrapper(val):
                has_ogrnip_ctx = _has_context(text, m.start(), m.end(), _OGRNIP_CONTEXT_PATTERN)
                has_card_ctx = _has_context(text, m.start(), m.end(), _CARD_CONTEXT_PATTERN)
                prio = 0.9 if (has_ogrnip_ctx or not has_card_ctx) else 1.1
                all_spans.append(("OGRNIP", m.start(), m.end(), prio))

        for m in BANK_CARD_PATTERN.finditer(text):
            val = m.group(0)
            if _validate_bank_card(val):
                digits = re.sub(r"\D", "", val)
                # 15-значная сплошная строка может быть ОГРНИП; Amex всегда начинается на 34/37
                if len(digits) == 15 and digits == val and not digits.startswith(("34", "37")):
                    if not _has_context(text, m.start(), m.end(), _CARD_CONTEXT_PATTERN):
                        continue
                all_spans.append(("BANK_CARD", m.start(), m.end(), 1))

        # ===== Priority 2: Нейросеть / ML-детектор (PER, SURNAME, NAME, ORG, LOC, ADDR) =====
        if self._ner is not None:
            try:
                ner_results = self._ner.detect(text)
                for tag, start, end in ner_results:
                    all_spans.append((tag, start, end, 2))
            except Exception:
                pass
        else:
            # Фоллбэк морфологического анализа через pymorphy3
            try:
                import pymorphy3
                morph = pymorphy3.MorphAnalyzer()
                for m in re.finditer(r"\b[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)*\b", text):
                    w = m.group(0)
                    first_part = w.split("-")[0]
                    parses = morph.parse(first_part)
                    if parses:
                        tag_obj = parses[0].tag
                        if "Surn" in tag_obj or "-" in w:
                            all_spans.append(("SURNAME", m.start(), m.end(), 2))
                        elif "Name" in tag_obj:
                            all_spans.append(("NAME", m.start(), m.end(), 2))
            except Exception:
                pass

        # ===== Priority 1.5: Атомарные структурные паттерны (защита от вытеснения нейросетью) =====
        all_spans.extend(_find_regex_spans(EMAIL_PATTERN, text, "EMAIL", 1.5))
        all_spans.extend(_find_regex_spans(IP_ADDRESS_PATTERN, text, "IP_ADDRESS", 1.5))
        # Валидированный по контрольной сумме СНИЛС приоритетнее телефона
        all_spans.extend(_find_regex_spans(SNILS_PATTERN, text, "SNILS", 1.5, _validate_snils))

        # ===== Priority 2.5–3: Структурные regex-паттерны =====
        for m in PHONE_PATTERN.finditer(text):
            val = m.group(0)
            # Сплошные 11 цифр без форматирования (напр. 85460171573 — код КЛАДР) считаем телефоном
            # только если начинается с '+' или рядом есть телефонный контекст
            if val.isdigit() and not _has_context(text, m.start(), m.end(), _PHONE_CONTEXT_PATTERN):
                continue
            all_spans.append(("PHONE_NUMBER", m.start(), m.end(), 3))

        all_spans.extend(_find_regex_spans(URL_PATTERN, text, "URL", 3))
        all_spans.extend(_find_regex_spans(CADASTRAL_NUMBER_PATTERN, text, "CADASTRAL_NUMBER", 3))
        all_spans.extend(_find_regex_spans(DATE_PATTERN, text, "DATE", 3))

        all_spans.extend(
            (tag, m.start(), m.end(), 3)
            for tag, m in (
                ("PASSPORT", m) for m in PASSPORT_PATTERN.finditer(text)
                if _validate_passport(m.group(0))
            )
        )
        # ===== Priority 3 (с контекстным повышением до 2.5): Бланки ЗАГС (KR-REL-1) =====
        for m in _ZAGS_CERTIFICATE_PATTERN.finditer(text):
            start, end = m.start(1), m.end(1)
            tag, priority = _classify_zags_certificate(text, start, end)
            all_spans.append((tag, start, end, priority))
        all_spans.extend(
            (tag, m.start(), m.end(), 2.5)
            for tag, m in (
                ("KLADR", m) for m in KLADR_PATTERN.finditer(text)
                if _has_context(text, m.start(), m.end(), _KLADR_CONTEXT_PATTERN)
            )
        )
        all_spans.extend(
            (tag, m.start(1), m.end(1), 3)
            for pat, tag in ((JOURNAL_ODD_PATTERN, "JOURNAL_ODD"), (JOURNAL_EVEN_PATTERN, "JOURNAL_EVEN"))
            for m in pat.finditer(text)
        )

        # Если задан целевой список атрибутов (например, 10 атрибутов чётного варианта),
        # фильтруем спаны до разрешения коллизий, чтобы нецелевые теги не вытесняли целевые
        if target_tags is not None:
            all_spans = [s for s in all_spans if s[0] in target_tags]

        # Разрешение коллизий
        resolved = _resolve_overlaps(all_spans)
        return [[tag, start, end] for tag, start, end, _ in resolved]


def detect_pii(
    text: str,
    ner_detector=None,
    target_tags: set[str] | frozenset[str] | None = None,
) -> list[list]:
    """Функция-обертка для обнаружения ПДн с сохранением совместимости интерфейса."""
    detector = HybridDetector(ner_detector)
    return detector.detect(text, target_tags=target_tags)
