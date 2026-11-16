"""Проверка качества token classification на независимо сгенерированных данных."""

from __future__ import annotations

import json

from lab1.generate_data import generate_dataset
from lab1.src.hybrid_pipeline import detect_pii
from lab1.src.metrics_evaluator import evaluate_token_classification


def test_randomized_dataset_meets_f1_requirements(tmp_path):
    """Новые случайные значения должны проходить оба нормативных порога."""
    dataset_path = tmp_path / "random_dataset.json"
    generate_dataset(str(dataset_path), min_per_class=10, seed=20261004)

    with dataset_path.open(encoding="utf-8") as data_file:
        dataset = json.load(data_file)

    texts = [item["text"] for item in dataset]
    true_spans = [[tuple(entity) for entity in item["entities"]] for item in dataset]
    predicted_spans = [detect_pii(text) for text in texts]

    metrics = evaluate_token_classification(texts, true_spans, predicted_spans)

    assert metrics["weighted_f1"] >= 0.85, metrics
    assert metrics["macro_f1"] >= 0.80, metrics
