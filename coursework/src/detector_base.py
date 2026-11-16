"""
Унификация архитектуры инференса и устранение скрытой дивергенции моделей (KR-ARC-1).

Обеспечивает:
1. Единый интерфейс BaseNerDetector.
2. Фабрику NerDetectorFactory с выбором бэкенда (rosberta, onnx, rubert, logreg).
3. 100% согласованность между этапом оценки F1 Token Classification и этапом обезличивания.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import os
from typing import List, Tuple, Optional, Any

logger = logging.getLogger(__name__)


class BaseNerDetector(ABC):
    """Единый абстрактный интерфейс детектора сущностей ПДн."""

    @abstractmethod
    def detect(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Обнаружить спаны ПДн в тексте.

        Returns:
            Список кортежей [(tag, start, end), ...].
        """
        pass


class RoSBERTaDetectorAdapter(BaseNerDetector):
    """Адаптер для модели RoSBERTa / ONNX."""

    def __init__(self, use_onnx: bool = False):
        from lab1.src.ner.rosberta_detector import RoSBERTaNerDetector
        self._detector = RoSBERTaNerDetector(use_onnx=use_onnx)

    def detect(self, text: str) -> List[Tuple[str, int, int]]:
        return self._detector.detect(text)


class RuBertDetectorAdapter(BaseNerDetector):
    """Адаптер для модели RuBERT."""

    def __init__(self, use_onnx: bool = False):
        from lab1.src.ner.rubert_detector import RuBertNerDetector
        self._detector = RuBertNerDetector(use_onnx=use_onnx)

    def detect(self, text: str) -> List[Tuple[str, int, int]]:
        return self._detector.detect(text)


class TrainedTokenClassifierAdapter(BaseNerDetector):
    """Адаптер для легковесного классификатора (Logistic Regression)."""

    def __init__(self):
        from lab1.src.ner.trained_detector import TrainedTokenClassifierDetector
        self._detector = TrainedTokenClassifierDetector()

    def detect(self, text: str) -> List[Tuple[str, int, int]]:
        return self._detector.detect(text)


class NerDetectorFactory:
    """Фабрика для создания детектора по имени бэкенда (KR-ARC-1)."""

    @classmethod
    def has_local_rosberta(cls, use_onnx: bool = False) -> bool:
        """Проверить наличие локальных весов RoSBERTa (PyTorch или INT8 ONNX)."""
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if use_onnx:
            return os.path.exists(os.path.join(root, "models", "rosberta_onnx", "model_int8.onnx"))
        return (
            os.path.exists(os.path.join(root, "models", "rosberta-ner-finetuned", "config.json"))
            or os.path.exists(os.path.join(root, "models", "rosberta_onnx", "model_int8.onnx"))
        )

    @classmethod
    def create_detector(
        cls,
        detector_type: str = "logreg",
        use_onnx: bool = False,
    ) -> Any:
        det_name = (detector_type or "logreg").lower()
        if det_name == "auto":
            if cls.has_local_rosberta(use_onnx=use_onnx):
                return RoSBERTaDetectorAdapter(use_onnx=use_onnx)
            return TrainedTokenClassifierAdapter()
        if det_name in ("logreg", "trained", "default"):
            return TrainedTokenClassifierAdapter()
        elif det_name in ("rosberta",):
            return RoSBERTaDetectorAdapter(use_onnx=use_onnx)
        elif det_name in ("onnx", "quantized"):
            return RoSBERTaDetectorAdapter(use_onnx=True)
        elif det_name == "rubert":
            return RuBertDetectorAdapter(use_onnx=use_onnx)
        else:
            raise ValueError(
                f"Неизвестный тип детектора '{detector_type}'. "
                "Допустимые значения: 'logreg', 'rosberta', 'onnx', 'rubert', 'auto'."
            )
