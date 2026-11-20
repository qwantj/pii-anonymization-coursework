"""Сохранение и загрузка артефактов оценки (метрики F1 и BERTScore) в JSON."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

TARGET_EVEN_ATTRS = {
    "SURNAME", "NAME", "INN", "PASSPORT", "BANK_CARD",
    "OGRNIP", "IP_ADDRESS", "EMAIL", "BIRTH_CERTIFICATE", "KLADR"
}


def save_evaluation_artifacts(
    artifacts_path: str,
    ner_metrics: dict,
    bert_metrics: dict,
    perf_metrics: dict | None = None,
    metadata: dict | None = None,
) -> None:
    """Сохранить вычисленные метрики в канонический JSON-файл артефактов."""
    os.makedirs(os.path.dirname(os.path.abspath(artifacts_path)), exist_ok=True)

    w_f1 = float(ner_metrics.get("weighted_f1", 0.0))
    m_f1 = float(ner_metrics.get("macro_f1", 0.0))

    per_class_filtered = {}
    for cls_name in sorted(TARGET_EVEN_ATTRS):
        if cls_name in ner_metrics.get("per_class", {}):
            m = ner_metrics["per_class"][cls_name]
            per_class_filtered[cls_name] = {
                "precision": round(float(m.get("precision", 0.0)), 4),
                "recall": round(float(m.get("recall", 0.0)), 4),
                "f1": round(float(m.get("f1", 0.0)), 4),
                "support": int(m.get("support", 0)),
            }
        else:
            per_class_filtered[cls_name] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}

    meta = dict(metadata) if metadata else {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "variant": 8,
        "variant_name": "even",
        "detector_type": "logreg",
        "device": "cpu",
        "dataset_path": "coursework/data/dataset.json",
        "total_texts": 246,
    }
    if "dataset_path" in meta and os.path.isabs(str(meta["dataset_path"])):
        meta["dataset_path"] = "coursework/data/dataset.json"

    bert_payload = {
        "backend": bert_metrics.get("backend", "bert-base-multilingual-cased"),
        "synonymic": {
            "precision": round(float(bert_metrics.get("synonymic", {}).get("precision", 0.0)), 4),
            "recall": round(float(bert_metrics.get("synonymic", {}).get("recall", 0.0)), 4),
            "f1": round(float(bert_metrics.get("synonymic", {}).get("f1", 0.0)), 4),
        },
        "masked": {
            "precision": round(float(bert_metrics.get("masked", {}).get("precision", 0.0)), 4),
            "recall": round(float(bert_metrics.get("masked", {}).get("recall", 0.0)), 4),
            "f1": round(float(bert_metrics.get("masked", {}).get("f1", 0.0)), 4),
        },
        "delta_f1": round(float(bert_metrics.get("delta_f1", 0.0)), 4),
    }
    if "isolated_entities" in bert_metrics:
        bert_payload["isolated_entities"] = bert_metrics["isolated_entities"]

    data = {
        "metadata": meta,
        "token_classification": {
            "weighted_f1": round(w_f1, 4),
            "macro_f1": round(m_f1, 4),
            "overall": {
                "precision": round(float(ner_metrics.get("overall", {}).get("precision", 0.0)), 4),
                "recall": round(float(ner_metrics.get("overall", {}).get("recall", 0.0)), 4),
                "f1": round(float(ner_metrics.get("overall", {}).get("f1", 0.0)), 4),
            },
            "compliance": {
                "weighted_f1_target": 0.85,
                "weighted_f1_passed": bool(w_f1 >= 0.85),
                "macro_f1_target": 0.80,
                "macro_f1_passed": bool(m_f1 >= 0.80),
            },
            "per_class": per_class_filtered,
        },
        "bertscore": bert_payload,
        "performance": perf_metrics or {},
    }

    with open(artifacts_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  [OK] Артефакты оценки сохранены: {artifacts_path}")


def load_evaluation_artifacts(artifacts_path: str) -> dict:
    """Загрузить метрики из JSON-файла артефактов."""
    if not os.path.exists(artifacts_path):
        raise FileNotFoundError(f"Файл артефактов не найден: {artifacts_path}")
    with open(artifacts_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
