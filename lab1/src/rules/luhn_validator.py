"""
Валидация номера банковской карты по алгоритму Луна.

Алгоритм:
1. Нумерация цифр справа налево (последняя цифра — позиция 1).
2. Цифры на чётных позициях (2, 4, 6, ...) удваиваются; если результат > 9, вычитается 9.
3. Сумма всех цифр должна делиться на 10 без остатка.
"""


def validate_luhn(number: str) -> bool:
    """
    Проверить валидность номера по алгоритму Луна.

    Args:
        number: строка из цифр (пробелы и дефисы автоматически удаляются).

    Returns:
        True, если контрольный разряд по Луну корректен.
    """
    cleaned = number.replace(" ", "").replace("-", "")
    if not cleaned.isdigit() or len(cleaned) < 2:
        return False

    digits = [int(ch) for ch in cleaned]
    total = 0

    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:  # чётные позиции (0-indexed: 1, 3, 5, ...)
            d *= 2
            if d > 9:
                d -= 9
        total += d

    return total % 10 == 0


def generate_luhn_digit(prefix: str) -> str:
    """
    Рассчитать контрольную цифру Луна для заданного префикса.

    Args:
        prefix: строка из цифр без последней (контрольной) цифры.

    Returns:
        Полный номер с корректной контрольной цифрой.
    """
    cleaned = prefix.replace(" ", "").replace("-", "")
    if not cleaned.isdigit():
        raise ValueError(f"Префикс должен содержать только цифры: '{prefix}'")

    # Добавляем временную 0 на место контрольной цифры
    digits = [int(ch) for ch in cleaned] + [0]

    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d

    check = (10 - (total % 10)) % 10
    return cleaned + str(check)
