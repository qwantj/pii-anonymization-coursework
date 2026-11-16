"""
Модуль оценки качества распознавания ПДн в рамках Курсовой работы.

Требование из задания на КР:
- Задача оценивается как задача token classification / NER.
- Соответственно каждый токен/слово текста может относиться только к одному
  из атрибутов ПДн или не относиться к ПДн, т.е. являться тегом O по схеме разметки BIO.
- Средневзвешенная F1-мера >= 0.85.
- Средняя (Macro) F1-мера >= 0.80.
- Метрика качества оценивается по всей программе в целом.
"""

from __future__ import annotations

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from lab1.src.metrics_evaluator import evaluate_token_classification, print_metrics
from lab1.src.hybrid_pipeline import detect_pii
from coursework.src.core_pipeline import EVEN_VARIANT_TAGS


def evaluate_dataset(
    dataset: list[dict],
    ner_detector=None,
    target_tags=EVEN_VARIANT_TAGS,
) -> dict:
    """
    Оценить качество распознавания на наборе данных по схеме token classification.

    Args:
        dataset: список словарей [{"text": "...", "entities": [["TAG", start, end], ...]}, ...]
        ner_detector: экземпляр детектора (или None для обученной AI-модели).
        target_tags: множество целевых атрибутов (по умолчанию 10 атрибутов чётного варианта).

    Returns:
        Словарь с метриками per_class, weighted_f1, macro_f1, overall.
    """
    texts = []
    true_spans_list = []
    pred_spans_list = []

    for item in dataset:
        text = item["text"]
        true_spans = [
            tuple(e) for e in item["entities"]
            if target_tags is None or e[0] in target_tags
        ]
        pred_spans = detect_pii(text, ner_detector, target_tags=target_tags)

        texts.append(text)
        true_spans_list.append(true_spans)
        pred_spans_list.append(pred_spans)

    return evaluate_token_classification(texts, true_spans_list, pred_spans_list)
