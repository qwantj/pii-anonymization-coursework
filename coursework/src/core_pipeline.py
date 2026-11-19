"""
Сквозной пайплайн Курсовой работы (Core Pipeline).

Объединяет:
1. Распознавание всех 10 атрибутов чётного варианта (ЛР1).
2. Унификацию инференса детектора NerDetectorFactory (KR-ARC-1).
3. Разрешение морфологической омонимии и контекстное определение рода (advanced_morphology).
4. Обеспечение идемпотентности замен (idempotency_manager).
5. Абстрактные стратегии обезличивания: синонимизация, маскирование, HMAC (KR-ARC-3).
"""

from __future__ import annotations

from typing import Optional, Tuple, List, Any, AbstractSet
from lab1.src.hybrid_pipeline import detect_pii, HybridDetector
from coursework.src.idempotency_manager import IdempotencyVault
from coursework.src.detector_base import NerDetectorFactory, BaseNerDetector
from coursework.src.strategies import (
    AnonymizationStrategy,
    SynonymicAnonymizationStrategy,
    CharacterMaskingStrategy,
    HmacAnonymizationStrategy,
)

# 10 обязательных атрибутов ПДн чётного варианта (вариант № 8)
EVEN_VARIANT_TAGS: frozenset[str] = frozenset({
    "SURNAME",
    "NAME",
    "INN",
    "PASSPORT",
    "BANK_CARD",
    "OGRNIP",
    "IP_ADDRESS",
    "EMAIL",
    "BIRTH_CERTIFICATE",
    "KLADR",
})


class CourseworkPipeline:
    """Интегрированный пайплайн распознавания и обезличивания ПДн."""

    def __init__(
        self,
        ner_detector: Any = None,
        use_onnx: bool = False,
        detector_type: str = "logreg",
        target_tags: Optional[AbstractSet[str]] = EVEN_VARIANT_TAGS,
    ):
        self.detector_type = "onnx" if (use_onnx and detector_type == "rosberta") else detector_type
        self.target_tags = target_tags

        if ner_detector is None:
            self._ner = NerDetectorFactory.create_detector(
                detector_type=detector_type,
                use_onnx=use_onnx,
            )
        else:
            self._ner = ner_detector

        self._hybrid = self._ner if isinstance(self._ner, HybridDetector) else HybridDetector(self._ner)

        # Реестр стратегий (KR-ARC-3)
        self._strategies: dict[str, AnonymizationStrategy] = {
            "synonymic": SynonymicAnonymizationStrategy(),
            "masking": CharacterMaskingStrategy(),
            "hmac": HmacAnonymizationStrategy(),
        }

    def get_detector(self) -> Any:
        """Возвращает инициализированный экземпляр детектора для устранения дивергенции (KR-ARC-1)."""
        return self._ner

    def set_strategy(self, mode: str, strategy: AnonymizationStrategy) -> None:
        """Зарегистрировать пользовательскую стратегию обезличивания."""
        self._strategies[mode] = strategy

    def process(
        self,
        text: str,
        mode: str = "synonymic",
        vault: Optional[IdempotencyVault] = None,
    ) -> Tuple[str, List[Any]]:
        """
        Обработать текст: обнаружить ПДн и обезличить выбранной стратегией.

        Args:
            text: входной текст.
            mode: 'synonymic' (синонимизация), 'masking' (маскирование '*') или 'hmac'.
            vault: экземпляр IdempotencyVault для сохранения связности между вызовами.

        Returns:
            (обезличенный_текст, список_найденных_спанов) (Инвариант C1).
        """
        if not text:
            return "", []

        # 1. Обнаружение спанов ПДн единым гибридным пайплайном (KR-ARC-1)
        spans = self._hybrid.detect(text, target_tags=self.target_tags)

        if not spans:
            return text, []

        # 2. Выбор стратегии трансформации текста (KR-ARC-3)
        strategy = self._strategies.get(mode, self._strategies["synonymic"])
        anonymized_text = strategy.execute(text, spans, vault=vault)

        return anonymized_text, spans


_DEFAULT_ANON_PIPELINE: Optional[CourseworkPipeline] = None


def anon(
    text: str,
    mode: str = "synonymic",
    vault: Optional[IdempotencyVault] = None,
) -> Tuple[str, List[Any]]:
    """
    Высокоуровневая функция распознавания и синонимического обезличивания ПДн.

    Пример:
        anon("Иван получил кредит") -> ("Пётр получил кредит", [["NAME", 0, 4]])
    """
    global _DEFAULT_ANON_PIPELINE
    if _DEFAULT_ANON_PIPELINE is None:
        _DEFAULT_ANON_PIPELINE = CourseworkPipeline(detector_type="logreg")
    return _DEFAULT_ANON_PIPELINE.process(text, mode=mode, vault=vault)
