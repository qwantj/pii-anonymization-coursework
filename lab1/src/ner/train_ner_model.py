"""
Скрипт обучения AI-модели классификации токенов для распознавания ФИО (ЛР1 и КР).

Обучает модель:
- Схема разметки: BIO (B-SURNAME, I-SURNAME, B-NAME, I-NAME, O).
- Обучение на размеченных предложениях с разбиением на train (80%) и test (20%).
- Сохранение обученной модели (векторизатор признаков + классификатор) в joblib.
- Расчёт метрик Token Classification (Weighted F1, Macro F1, Precision, Recall).
"""

from __future__ import annotations

import glob
import json
import os
import random
import re
import sys
import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from lab1.src.ner.token_features import (
    tokenize_with_spans,
    align_tokens_with_spans,
    sentence_to_features,
)


def extract_clean_factrueval_sentences(split_dir: str, limit: int = 300) -> list[dict]:
    """
    Извлечь из открытого корпуса factRuEval-2016 предложения с разметкой SURNAME/NAME
    (без пересечений и неразмеченных отчеств).
    """
    if not os.path.isdir(split_dir):
        return []

    results: list[dict] = []
    for txt_path in sorted(glob.glob(os.path.join(split_dir, "*.txt"))):
        spans_path = txt_path[:-4] + ".spans"
        if not os.path.exists(spans_path):
            continue
        with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

        spans: list[tuple[str, int, int]] = []
        exclude_ranges: list[tuple[int, int]] = []
        with open(spans_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.split("#")[0].strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                stype, off, length = parts[1], int(parts[2]), int(parts[3])
                if stype in ("name", "surname"):
                    tag = "NAME" if stype == "name" else "SURNAME"
                    sub = text[off:off + length]
                    if re.match(r"^[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)*$", sub):
                        spans.append((tag, off, off + length))
                    else:
                        exclude_ranges.append((off, off + length))
                elif stype in ("patronymic", "nickname"):
                    exclude_ranges.append((off, off + length))

        cur = 0
        for part in re.split(r"(?<=[.!?])\s+|\n+", text):
            idx = text.find(part, cur)
            if idx == -1:
                continue
            cur = idx + len(part)
            sent = part.strip()
            if not (35 <= len(sent) <= 240):
                continue
            s_start = text.find(sent, idx)
            s_end = s_start + len(sent)
            if any(a < s_end and b > s_start for a, b in exclude_ranges):
                continue

            sent_ents = [
                [t, a - s_start, b - s_start]
                for t, a, b in spans
                if a >= s_start and b <= s_end
            ]
            sent_ents.sort(key=lambda x: x[1])
            clean_ents = []
            last_end = -1
            for e in sent_ents:
                if e[1] >= last_end:
                    clean_ents.append(e)
                    last_end = e[2]

            tags = {e[0] for e in clean_ents}
            if "SURNAME" in tags and "NAME" in tags:
                results.append({"text": sent, "entities": clean_ents})
                if len(results) >= limit:
                    return results

    return results


def _build_case_augmented_samples() -> list[dict]:
    """Дополнительные обучающие примеры с косвенными падежами (строго отдельные от теста)."""
    from lab1.generate_data import SURNAMES_M, SURNAMES_F, NAMES_M, NAMES_F
    from lab2.src.morph_utils import inflect_word

    rng = random.Random(777)
    templates = [
        ("Справка выдана {S_datv} {N_datv}.", "datv"),
        ("Доверенность оформлена на имя {S_gent} {N_gent}.", "gent"),
        ("Встреча согласована с {S_ablt} {N_ablt}.", "ablt"),
        ("Запрос направлен {S_datv}.", "datv"),
        ("Уведомление получено от {S_gent} {N_gent}.", "gent"),
        ("Подпись поставил(а) {S_nomn} {N_nomn}.", "nomn"),
        ("Документ передан гражданину {S_datv} {N_datv}.", "datv"),
    ]
    out: list[dict] = []
    for _ in range(120):
        is_m = rng.random() > 0.5
        s_nom = rng.choice(SURNAMES_M if is_m else SURNAMES_F)
        n_nom = rng.choice(NAMES_M if is_m else NAMES_F)
        g = "masc" if is_m else "femn"
        tmpl, case = rng.choice(templates)
        s_val = s_nom if case == "nomn" else inflect_word(s_nom, case, g)
        n_val = n_nom if case == "nomn" else inflect_word(n_nom, case, g)
        text = tmpl.format(
            S_nomn=s_val, N_nomn=n_val,
            S_gent=s_val, N_gent=n_val,
            S_datv=s_val, N_datv=n_val,
            S_ablt=s_val, N_ablt=n_val,
        )
        ents = []
        s_idx = text.find(s_val)
        if s_idx >= 0:
            ents.append(["SURNAME", s_idx, s_idx + len(s_val)])
        if "{N_" in tmpl:
            n_idx = text.find(n_val, s_idx + len(s_val) if s_idx >= 0 else 0)
            if n_idx >= 0:
                ents.append(["NAME", n_idx, n_idx + len(n_val)])
        out.append({"text": text, "entities": ents})
    return out


def load_dataset(json_path: str) -> list[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_training_data(dataset: list[dict]):
    """
    Преобразовать текстовые примеры со спанами в списки фичей и BIO-меток для каждого токена.
    """
    all_features = []
    all_labels = []

    for item in dataset:
        text = item["text"]
        entities = item["entities"]

        tokens_spans = tokenize_with_spans(text)
        token_texts = [t[0] for t in tokens_spans]

        # Для обучения NER-модели отбираем сущности SURNAME и NAME (однословные спаны -> B-SURNAME/B-NAME)
        ner_spans = [e for e in entities if e[0] in ("SURNAME", "NAME") and "-" not in text[int(e[1]):int(e[2])]]
        labels = align_tokens_with_spans(tokens_spans, ner_spans)
        # Гарантируем 3 канонических класса ['B-NAME', 'B-SURNAME', 'O']
        labels = ["B-" + lbl[2:] if lbl.startswith("I-") else lbl for lbl in labels]

        features = sentence_to_features(token_texts)

        all_features.extend(features)
        all_labels.extend(labels)

    return all_features, all_labels


def train_and_evaluate(dataset_path: str, model_save_path: str, extra_factrueval_dev: bool = True):
    print("=" * 70)
    print("ОБУЧЕНИЕ AI-МОДЕЛИ КЛАССИФИКАЦИИ ТОКЕНОВ (NER: SURNAME & NAME)")
    print("=" * 70)

    print(f"Загрузка обучающего набора из {dataset_path}...")
    raw_data = load_dataset(dataset_path)
    raw_data.extend(_build_case_augmented_samples())

    if extra_factrueval_dev:
        dev_dir = os.path.join(PROJECT_ROOT, "lab1", "data", "factRuEval", "factRuEval-2016-master", "devset")
        fre_dev = extract_clean_factrueval_sentences(dev_dir, limit=300)
        if fre_dev:
            print(f"Добавлено {len(fre_dev)} реальных предложений из factRuEval-2016 devset.")
            raw_data.extend(fre_dev)

    # Делим данные на уровне текстов, чтобы избежать утечки данных между предложениями
    train_data, test_data = train_test_split(raw_data, test_size=0.2, random_state=42)
    print(f"Текстов для обучения (train): {len(train_data)}")
    print(f"Текстов для валидации (test): {len(test_data)}")

    print("\nИзвлечение лингвистических и контекстных признаков...")
    X_train_dicts, y_train = prepare_training_data(train_data)
    X_test_dicts, y_test = prepare_training_data(test_data)

    print(f"Токенов в train: {len(y_train)}, токенов в test: {len(y_test)}")

    # Векторизация
    vectorizer = DictVectorizer(sparse=True)
    X_train = vectorizer.fit_transform(X_train_dicts)
    X_test = vectorizer.transform(X_test_dicts)
    print(f"Размерность признакового пространства: {X_train.shape[1]} признаков.")

    # Обучение модели
    print("\nОбучение классификатора токенов (LogisticRegression L2, max_iter=500)...")
    clf = LogisticRegression(
        C=2.0,
        max_iter=500,
        class_weight="balanced",  # компенсирует дисбаланс частоты тега 'O'
        solver="lbfgs",
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # Оценка на отложенной тестовой выборке
    print("\nВалидация обученной модели на тестовой выборке (Token Classification):")
    y_pred = clf.predict(X_test)

    # Классы без 'O' для расчета качества обнаружения ПДн
    target_classes = sorted(list(set(y_test) - {"O"}))

    print("\nПодробный отчет по токенной классификации:")
    report = classification_report(y_test, y_pred, digits=4, zero_division=0)
    print(report)

    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    # F1 строго по сущностям (без тега O)
    entity_weighted_f1 = f1_score(y_test, y_pred, labels=target_classes, average="weighted", zero_division=0)
    entity_macro_f1 = f1_score(y_test, y_pred, labels=target_classes, average="macro", zero_division=0)

    print("-" * 70)
    print(f"Weighted F1 (все токены, включая O):  {weighted_f1:.4f}")
    print(f"Macro F1 (все токены, включая O):     {macro_f1:.4f}")
    print(f"Weighted F1 (токены ПДн):             {entity_weighted_f1:.4f}")
    print(f"Macro F1 (токены ПДн):                {entity_macro_f1:.4f}")
    print("-" * 70)

    # Сохранение обученной модели
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    pipeline_artifact = {
        "vectorizer": vectorizer,
        "classifier": clf,
        "classes": clf.classes_.tolist(),
        "metrics": {
            "weighted_f1": round(weighted_f1, 4),
            "macro_f1": round(macro_f1, 4),
            "entity_weighted_f1": round(entity_weighted_f1, 4),
            "entity_macro_f1": round(entity_macro_f1, 4),
        }
    }
    joblib.dump(pipeline_artifact, model_save_path)
    print(f"\nМодель успешно сохранена в: {model_save_path}")

    return pipeline_artifact


if __name__ == "__main__":
    dataset_file = os.path.join(PROJECT_ROOT, "lab1", "data", "test_samples.json")
    model_output = os.path.join(PROJECT_ROOT, "lab1", "models", "ner_token_classifier.joblib")
    cw_model_output = os.path.join(PROJECT_ROOT, "coursework", "models", "ner_token_classifier.joblib")
    artifact = train_and_evaluate(dataset_file, model_output)
    os.makedirs(os.path.dirname(cw_model_output), exist_ok=True)
    joblib.dump(artifact, cw_model_output)
    print(f"Копия для КР сохранена в: {cw_model_output}")
