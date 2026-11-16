"""
Адаптер обратной совместимости: класс RuBertNerDetector, перенаправляющий
вызовы на обновленный детектор RoSBERTaNerDetector (ai-forever/ru-en-RoSBERTa).
"""

from __future__ import annotations

import os
from lab1.src.ner.rosberta_detector import RoSBERTaNerDetector


class RuBertNerDetector(RoSBERTaNerDetector):
    """
    Фасад обратной совместимости со старым классом RuBertNerDetector.
    Использует современную архитектуру ai-forever/ru-en-RoSBERTa с точными
    символьными границами, поддержкой ADDR/ORG/LOC/PER и скользящим окном.
    """

    def __init__(self, model_name: str | None = None, min_score: float = 0.40, use_onnx: bool = False):
        # Если передан явный путь в конструктор или переменные окружения, используем его
        chosen = model_name or os.environ.get("RUBERT_MODEL_PATH") or os.environ.get("ROSBERTA_MODEL_PATH")
        super().__init__(model_name_or_path=chosen, min_score=min_score, use_onnx=use_onnx)
