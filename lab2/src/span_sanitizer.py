"""
Модуль санитарии, верификации и детерминированного разрешения коллизий спанов (LR-ARC-3).

Обеспечивает:
- Проверку типов и числовых диапазонов координат: 0 <= start < end <= text_len.
- Безопасную обработку списков, кортежей и словарей.
- Детерминированное разрешение пересечений и наложений (Non-Maximum Suppression):
    1) Приоритет более длинному спану.
    2) При равной длине — более высокому приоритету класса.
    3) При равном классе — более раннему началу в тексте.
    4) При полном совпадении — алфавитный порядок тега.
- Гарантию соблюдения нормативного BIO-инварианта C3.
"""

from __future__ import annotations

from typing import Any, Sequence, Tuple, Union

# Таблица нормативных приоритетов классов для детерминированного арбитража
DEFAULT_TAG_PRIORITY: dict[str, int] = {
    # 1. Сущности со строгими математическими суммами (C4)
    "INN": 100,
    "BANK_CARD": 100,
    "OGRNIP": 100,
    "SNILS": 100,
    # 2. Документы государственного образца
    "PASSPORT": 90,
    "BIRTH_CERTIFICATE": 90,
    "MARRIAGE_CERTIFICATE": 90,
    "CADASTRAL_NUMBER": 85,
    # 3. Контактные и сетевые идентификаторы
    "EMAIL": 80,
    "PHONE_NUMBER": 80,
    "IP_ADDRESS": 80,
    "URL": 80,
    # 4. Календарные даты и адресные коды
    "DATE": 75,
    "KLADR": 75,
    "JOURNAL_ODD": 70,
    "JOURNAL_EVEN": 70,
    # 5. Составные именованные сущности
    "PER": 60,
    "ORG": 60,
    "ADDR": 60,
    "LOC": 55,
    # 6. Атомарные компоненты ФИО
    "SURNAME": 50,
    "NAME": 50,
    "PATRONYMIC": 50,
}


def sanitize_spans(
    spans: Sequence[Any],
    text_len: int,
    tag_priorities: dict[str, int] | None = None,
) -> list[tuple[str, int, int]]:
    """
    Санитизирует список спанов, отсекает невалидные границы и устраняет наложения.

    Args:
        spans: коллекция спанов (кортежи, списки или словари).
        text_len: длина исходного текста в символах.
        tag_priorities: опциональный словарь приоритетов тегов.

    Returns:
        Отсортированный по возрастанию start список непересекающихся кортежей (tag, start, end).
    """
    # Защита от перепутанных позиционных аргументов sanitize_spans(text_len, spans)
    if isinstance(spans, int) and isinstance(text_len, (list, tuple)):
        spans, text_len = text_len, spans

    if not spans or text_len <= 0:
        return []

    priorities = tag_priorities if tag_priorities is not None else DEFAULT_TAG_PRIORITY

    # Шаг 1: Валидация типов, нормализация и первичная фильтрация границ
    valid_candidates: list[tuple[str, int, int]] = []

    for item in spans:
        tag: str | None = None
        raw_start: Any = None
        raw_end: Any = None

        if isinstance(item, dict):
            tag = item.get("tag") or item.get("entity_type") or item.get("label")
            raw_start = item.get("start")
            raw_end = item.get("end")
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            tag = item[0]
            raw_start = item[1]
            raw_end = item[2]
        else:
            continue

        if tag is None:
            continue

        # В Python bool является подклассом int (True == 1), исключаем булевы типы
        if isinstance(raw_start, bool) or isinstance(raw_end, bool):
            continue

        try:
            start = int(raw_start)
            end = int(raw_end)
        except (ValueError, TypeError):
            continue

        # Проверка диапазонов координат
        if start < 0 or start >= text_len:
            continue
        if end <= start:
            continue

        # Безопасное усечение правой границы до длины текста
        end = min(end, text_len)
        valid_candidates.append((str(tag), start, end))

    if not valid_candidates:
        return []

    # Шаг 2: Детерминированная сортировка для жадного выбора (Greedy NMS)
    # Критерий:
    # 1. Длина спана (-length) — более длинные вперед
    # 2. Приоритет тега (-priority) — более важные вперед
    # 3. Начальная позиция (start) — более ранние вперед
    # 4. Имя тега (tag) — лексикографический tie-breaker
    valid_candidates.sort(
        key=lambda s: (
            -(s[2] - s[1]),
            -priorities.get(s[0], 10),
            s[1],
            s[0],
        )
    )

    # Шаг 3: Разрешение наложений через интервальную маску занятости
    occupied = bytearray(text_len)
    selected: list[tuple[str, int, int]] = []

    for tag, start, end in valid_candidates:
        # Проверяем, свободен ли весь интервал [start, end)
        # Если ни один байт не занят (значение 0)
        if not any(occupied[start:end]):
            occupied[start:end] = b"\x01" * (end - start)
            selected.append((tag, start, end))

    # Шаг 4: Упорядочивание по возрастанию позиции начала (канонический BIO-порядок)
    selected.sort(key=lambda s: s[1])
    return selected
