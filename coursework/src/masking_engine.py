"""
Модуль маскирования персональных данных (Masking Engine).

Реализует базовый метод обезличивания — замену символов сущности на символы маски ('*').
Используется для сравнительного анализа с методом синонимической замены
с помощью метрики BERTScore.

Требование из методички к КР:
«Маскирование можно реализовать одной функцией, которая получает на вход текст
и отдаёт на выход количество «*» равное количеству символов в исходном тексте».
"""

from __future__ import annotations


def mask_span(text: str, mask_char: str = "*") -> str:
    """
    Замаскировать строку, вернув строку из символов mask_char той же длины.

    Args:
        text: исходный текст сущности.
        mask_char: символ маскирования (по умолчанию '*').

    Returns:
        Строка из символов mask_char той же длины.
    """
    return mask_char * len(text)


def mask_text(
    text: str,
    spans: list[tuple[str, int, int] | list],
    mask_char: str = "*",
) -> str:
    """
    Замаскировать все указанные спаны ПДн в тексте.

    Args:
        text: исходный текст.
        spans: список спанов [(tag, start, end), ...].
        mask_char: символ маскирования.

    Returns:
        Текст с замаскированными сущностями.
    """
    sorted_spans = sorted(spans, key=lambda s: int(s[1]), reverse=True)
    result = text

    for tag, start, end in sorted_spans:
        original = text[int(start):int(end)]
        masked = mask_span(original, mask_char)
        result = result[:int(start)] + masked + result[int(end):]

    return result
