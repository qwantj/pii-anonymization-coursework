"""
Пакет training для дообучения ai-forever/ru-en-RoSBERTa на задаче Token Classification.
"""

from data_pipeline.schema import LABEL_LIST, LABEL2ID, ID2LABEL, CLASS_WEIGHTS

__all__ = ["LABEL_LIST", "LABEL2ID", "ID2LABEL", "CLASS_WEIGHTS"]
