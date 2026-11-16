"""
Пакет data_pipeline для сборки, валидации и генерации единого NER-датасета
для обучения модели ai-forever/ru-en-RoSBERTa.
"""

from data_pipeline.schema import (
    LABEL_LIST,
    LABEL2ID,
    ID2LABEL,
    CLASS_WEIGHTS,
    validate_iob2_sequence,
)

__all__ = [
    "LABEL_LIST",
    "LABEL2ID",
    "ID2LABEL",
    "CLASS_WEIGHTS",
    "validate_iob2_sequence",
]
