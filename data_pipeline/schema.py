"""
Единая схема разметки тегов NER (Strict IOB2) и функции валидации.
"""

from __future__ import annotations
from typing import Sequence

# Строгий список тегов для проекта
LABEL_LIST: list[str] = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-ADDR",
    "I-ADDR",
]

LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL: dict[int, str] = {i: label for i, label in enumerate(LABEL_LIST)}

# Веса классов для борьбы с сильным дисбалансом тега "O" и редкого тега "ADDR"
CLASS_WEIGHTS: dict[str, float] = {
    "O": 0.25,
    "B-PER": 1.0,
    "I-PER": 1.0,
    "B-ORG": 1.0,
    "I-ORG": 1.0,
    "B-LOC": 1.0,
    "I-LOC": 1.0,
    "B-ADDR": 1.3,
    "I-ADDR": 1.3,
}

CLASS_WEIGHTS_LIST: list[float] = [CLASS_WEIGHTS[label] for label in LABEL_LIST]


def validate_iob2_sequence(tokens: Sequence[str], ner_tags: Sequence[int | str]) -> bool:
    """
    Проверить строгое соответствие последовательности схеме IOB2:
    1. Длина списка токенов равна длине списка меток.
    2. Все метки валидны.
    3. Метка I-X может следовать только за B-X или I-X того же типа X.
       Переходы O -> I-X или B-X -> I-Y запрещены.

    Returns:
        True, если последовательность строго валидна, иначе False.
    """
    if len(tokens) != len(ner_tags):
        return False

    prev_tag_type: str | None = None

    for raw_tag in ner_tags:
        if isinstance(raw_tag, int):
            if raw_tag not in ID2LABEL:
                return False
            tag = ID2LABEL[raw_tag]
        else:
            tag = raw_tag
            if tag not in LABEL2ID:
                return False

        if tag == "O":
            prev_tag_type = None
        elif tag.startswith("B-"):
            prev_tag_type = tag[2:]
        elif tag.startswith("I-"):
            current_type = tag[2:]
            if prev_tag_type != current_type:
                # Нарушение IOB2: I-X не предшествует B-X или I-X
                return False
            prev_tag_type = current_type
        else:
            return False

    return True


def repair_iob2_sequence(ner_tags: Sequence[int | str]) -> list[int]:
    """
    Исправить мелкие дефекты разметки открытых датасетов для приведения к строгому IOB2:
    - Если встречается I-X без предшествующего B-X или I-X, он превращается в B-X.
    - Возвращает список числовых ID меток.
    """
    repaired: list[int] = []
    prev_tag_type: str | None = None

    for raw_tag in ner_tags:
        if isinstance(raw_tag, int):
            tag = ID2LABEL.get(raw_tag, "O")
        else:
            tag = str(raw_tag)

        if tag == "O" or tag not in LABEL2ID:
            repaired.append(LABEL2ID["O"])
            prev_tag_type = None
        elif tag.startswith("B-"):
            curr_type = tag[2:]
            if curr_type in ("PER", "ORG", "LOC", "ADDR"):
                repaired.append(LABEL2ID[tag])
                prev_tag_type = curr_type
            else:
                repaired.append(LABEL2ID["O"])
                prev_tag_type = None
        elif tag.startswith("I-"):
            curr_type = tag[2:]
            if curr_type in ("PER", "ORG", "LOC", "ADDR"):
                if prev_tag_type == curr_type:
                    repaired.append(LABEL2ID[tag])
                else:
                    # Чиним: превращаем незаконный I-X в B-X
                    repaired.append(LABEL2ID[f"B-{curr_type}"])
                    prev_tag_type = curr_type
            else:
                repaired.append(LABEL2ID["O"])
                prev_tag_type = None

    return repaired
