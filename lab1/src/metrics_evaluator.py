"""
Модуль оценки качества распознавания ПДн в соответствии с требованиями ЛР1 и КР:

Требования к оценке:
- Задачу необходимо оценивать как задачу token classification / NER.
- Соответственно каждый токен/слово текста может относиться только к одному
  из атрибутов ПДн или не относиться к ПДн, т.е. являться тегом O по схеме разметки BIO.
- Средневзвешенная F1-мера >= 0.85.
- Средняя (Macro) F1-мера >= 0.80.
- Метрика качества оценивается по всей программе в целом.
"""

from __future__ import annotations

import re
from collections import defaultdict
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

# Паттерн токенизации на слова/числа/знаки
TOKEN_REGEX = re.compile(r'[а-яА-ЯёЁa-zA-Z0-9]+|[^\s\w]')


def tokenize_text(text: str) -> list[tuple[str, int, int]]:
    """Разбить текст на токены с символьными индексами."""
    tokens = []
    for m in TOKEN_REGEX.finditer(text):
        tokens.append((m.group(0), m.start(), m.end()))
    return tokens


def get_token_bio_tags(
    tokens: list[tuple[str, int, int]],
    spans: list[tuple[str, int, int] | list],
) -> list[str]:
    """
    Присвоить каждому токену BIO-тег.

    Каждый токен относится строго к ОДНОМУ атрибуту или O.
    """
    tags = ["O"] * len(tokens)
    # Сортируем спаны по началу
    sorted_spans = sorted(spans, key=lambda s: int(s[1]))

    for tag, start, end in sorted_spans:
        start = int(start)
        end = int(end)
        first_token = True

        for i, (tok, t_start, t_end) in enumerate(tokens):
            # Проверяем пересечение
            if max(start, t_start) < min(end, t_end):
                if first_token:
                    tags[i] = f"B-{tag}"
                    first_token = False
                else:
                    tags[i] = f"I-{tag}"

    return tags


def evaluate_token_classification(
    texts: list[str],
    true_spans_list: list[list[tuple[str, int, int] | list]],
    pred_spans_list: list[list[tuple[str, int, int] | list]],
) -> dict:
    """
    Вычислить метрики token classification по схеме BIO для всех текстов.

    Returns:
        Словарь с метриками:
        {
            "token_report": str,
            "weighted_f1": float,
            "macro_f1": float,
            "entity_metrics": dict,
            "overall": dict,
        }
    """
    all_true_bio: list[str] = []
    all_pred_bio: list[str] = []

    # Также собираем метки классов без BIO префиксов для агрегированных метрик по атрибутам
    all_true_entities: list[str] = []
    all_pred_entities: list[str] = []

    for text, true_spans, pred_spans in zip(texts, true_spans_list, pred_spans_list):
        tokens = tokenize_text(text)
        t_tags = get_token_bio_tags(tokens, true_spans)
        p_tags = get_token_bio_tags(tokens, pred_spans)

        all_true_bio.extend(t_tags)
        all_pred_bio.extend(p_tags)

        for t, p in zip(t_tags, p_tags):
            all_true_entities.append(t[2:] if t != "O" else "O")
            all_pred_entities.append(p[2:] if p != "O" else "O")

    # Уникальные классы атрибутов (без 'O') для дополнительного отчета.
    unique_attrs = sorted(
        (set(all_true_entities) | set(all_pred_entities)) - {"O"}
    )

    # Считаем per-attribute метрики (на уровне токенов)
    per_class = {}
    for attr in unique_attrs:
        p = precision_score(all_true_entities, all_pred_entities, labels=[attr], average="micro", zero_division=0)
        r = recall_score(all_true_entities, all_pred_entities, labels=[attr], average="micro", zero_division=0)
        f = f1_score(all_true_entities, all_pred_entities, labels=[attr], average="micro", zero_division=0)
        support = sum(1 for x in all_true_entities if x == attr)

        per_class[attr] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "support": support,
        }

    # Нормативные метрики считаются по исходным BIO-тегам всех токенов,
    # включая O: это непосредственно соответствует token classification.
    weighted_f1 = f1_score(
        all_true_bio, all_pred_bio, average="weighted", zero_division=0
    )
    macro_f1 = f1_score(
        all_true_bio, all_pred_bio, average="macro", zero_division=0
    )

    # Дополнительные агрегированные F1 по атрибутам ПДн (без O и BIO-префиксов)
    total_support = sum(per_class[a]["support"] for a in per_class)
    if total_support > 0:
        entity_weighted_f1 = sum(
            per_class[a]["f1"] * per_class[a]["support"] for a in per_class
        ) / total_support
    else:
        entity_weighted_f1 = 0.0

    entity_macro_f1 = (
        sum(per_class[a]["f1"] for a in per_class) / len(per_class)
        if per_class
        else 0.0
    )

    # Общие precision и recall
    overall_p = precision_score(all_true_entities, all_pred_entities, labels=unique_attrs, average="weighted", zero_division=0)
    overall_r = recall_score(all_true_entities, all_pred_entities, labels=unique_attrs, average="weighted", zero_division=0)
    overall_f1 = f1_score(all_true_entities, all_pred_entities, labels=unique_attrs, average="weighted", zero_division=0)

    # BIO-отчет
    bio_labels = sorted(list(set(all_true_bio) | set(all_pred_bio)))
    token_report = classification_report(all_true_bio, all_pred_bio, digits=4, zero_division=0)

    return {
        "per_class": per_class,
        "weighted_f1": round(weighted_f1, 4),
        "macro_f1": round(macro_f1, 4),
        "entity_weighted_f1": round(entity_weighted_f1, 4),
        "entity_macro_f1": round(entity_macro_f1, 4),
        "overall": {
            "precision": round(overall_p, 4),
            "recall": round(overall_r, 4),
            "f1": round(overall_f1, 4),
        },
        "token_report": token_report,
    }


def print_metrics(metrics: dict) -> None:
    """Красивый вывод метрик в консоль с проверкой порогов."""
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ TOKEN CLASSIFICATION (ПО ВСЕЙ ПРОГРАММЕ)")
    print("=" * 70)

    print(f"\n{'Класс':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 65)

    for cls, m in sorted(metrics["per_class"].items()):
        print(f"{cls:<25} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['support']:>10}")

    print("-" * 65)
    o = metrics["overall"]
    print(f"{'Overall (ПДн)':<25} {o['precision']:>10.4f} {o['recall']:>10.4f} {o['f1']:>10.4f}")
    print(f"\nСредневзвешенная F1-мера (Weighted F1): {metrics['weighted_f1']:.4f}")
    print(f"Средняя F1-мера (Macro F1):            {metrics['macro_f1']:.4f}")

    w_ok = metrics["weighted_f1"] >= 0.85
    m_ok = metrics["macro_f1"] >= 0.80

    print(f"\nТребование: Weighted F1 >= 0.85 -> {'[OK] ПРОЙДЕН' if w_ok else '[FAIL] НЕ ПРОЙДЕН'}")
    print(f"Требование: Macro F1    >= 0.80 -> {'[OK] ПРОЙДЕН' if m_ok else '[FAIL] НЕ ПРОЙДЕН'}")
    print("=" * 70)
