"""
Валидация ОГРНИП (Основной государственный регистрационный номер
индивидуального предпринимателя).

ОГРНИП — 15 цифр. Контрольное число: остаток от деления первых 14 цифр
на 13. Если остаток >= 10, берётся младший разряд (остаток % 10).
Результат сравнивается с 15-й цифрой.
"""


def validate_ogrnip(value: str) -> bool:
    """
    Проверить валидность ОГРНИП.

    Args:
        value: строка из 15 цифр.

    Returns:
        True, если контрольный разряд корректен.
    """
    stripped = value.strip()
    if not stripped.isdigit() or len(stripped) != 15:
        return False

    base14 = int(stripped[:14])
    remainder = base14 % 13
    check = remainder % 10
    return check == int(stripped[14])


def generate_ogrnip_check_digit(base14: str) -> str:
    """
    Рассчитать контрольную цифру ОГРНИП для первых 14 цифр.

    Args:
        base14: строка из 14 цифр.

    Returns:
        Полный 15-значный ОГРНИП с контрольным разрядом.
    """
    stripped = base14.strip()
    if not stripped.isdigit() or len(stripped) != 14:
        raise ValueError(f"База ОГРНИП должна быть 14 цифр: '{base14}'")

    remainder = int(stripped) % 13
    check = remainder % 10
    return stripped + str(check)
