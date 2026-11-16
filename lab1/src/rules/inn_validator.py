"""
Валидация ИНН (Идентификационный номер налогоплательщика).

ИНН-10 (юридическое лицо):
  - 10 цифр, последняя — контрольный разряд.
  - Коэффициенты: [2, 4, 10, 3, 5, 9, 4, 6, 8]
  - Контрольное число = (сумма произведений первых 9 цифр на коэффициенты) mod 11 mod 10

ИНН-12 (физическое лицо):
  - 12 цифр, последние 2 — контрольные разряды.
  - Этап 1 (11-я цифра): коэффициенты [7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
  - Этап 2 (12-я цифра): коэффициенты [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
"""


# Коэффициенты для расчёта контрольных разрядов ИНН
_COEFF_INN10 = [2, 4, 10, 3, 5, 9, 4, 6, 8]
_COEFF_INN12_D11 = [7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
_COEFF_INN12_D12 = [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8]


def _calc_check_digit(digits: list[int], coefficients: list[int]) -> int:
    """Рассчитать контрольный разряд ИНН по заданным коэффициентам."""
    total = sum(d * c for d, c in zip(digits, coefficients))
    remainder = total % 11
    return remainder % 10


def validate_inn(value: str) -> bool:
    """
    Проверить валидность ИНН (10 или 12 цифр).

    Args:
        value: строка, содержащая только цифры ИНН (без пробелов и разделителей).

    Returns:
        True, если ИНН валиден.
    """
    stripped = value.strip()
    if not stripped.isdigit():
        return False

    digits = [int(ch) for ch in stripped]

    if len(digits) == 10:
        expected = _calc_check_digit(digits[:9], _COEFF_INN10)
        return digits[9] == expected

    if len(digits) == 12:
        expected_d11 = _calc_check_digit(digits[:10], _COEFF_INN12_D11)
        if digits[10] != expected_d11:
            return False
        expected_d12 = _calc_check_digit(digits[:11], _COEFF_INN12_D12)
        return digits[11] == expected_d12

    return False


def generate_inn_check_digits(base: str) -> str:
    """
    Сгенерировать контрольные разряды для базовой части ИНН.

    Args:
        base: первые 10 цифр (для ИНН-12) или 9 цифр (для ИНН-10).

    Returns:
        Полный ИНН с контрольными разрядами.

    Raises:
        ValueError: если длина базы не 9 и не 10.
    """
    stripped = base.strip()
    if not stripped.isdigit():
        raise ValueError(f"База ИНН должна содержать только цифры: '{base}'")

    digits = [int(ch) for ch in stripped]

    if len(digits) == 9:
        # ИНН-10
        d10 = _calc_check_digit(digits, _COEFF_INN10)
        return stripped + str(d10)

    if len(digits) == 10:
        # ИНН-12
        d11 = _calc_check_digit(digits, _COEFF_INN12_D11)
        digits.append(d11)
        d12 = _calc_check_digit(digits, _COEFF_INN12_D12)
        return stripped + str(d11) + str(d12)

    raise ValueError(f"Длина базы должна быть 9 или 10, получено: {len(digits)}")
