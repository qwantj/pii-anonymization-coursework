"""
Инференс-модуль для обученной AI-модели классификации токенов.

Загружает сохраненную модель из joblib, токенизирует входной текст,
извлекает контекстные признаки и реконструирует спаны сущностей
(SURNAME, NAME) из последовательности предсказанных BIO-тегов.
"""

from __future__ import annotations

import os
import re
import joblib

from lab1.src.ner.token_features import (
    tokenize_with_spans,
    sentence_to_features,
)

_LAB1_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models",
    "ner_token_classifier.joblib",
)
_CW_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "coursework",
    "models",
    "ner_token_classifier.joblib",
)
DEFAULT_MODEL_PATH = _CW_MODEL_PATH if os.path.exists(_CW_MODEL_PATH) else _LAB1_MODEL_PATH

_HYPHEN_RIGHT_RE = re.compile(r'^-[А-ЯЁA-Z][а-яёa-z]{1,}(?:-[А-ЯЁA-Z][а-яёa-z]{1,})*')
_HYPHEN_LEFT_RE = re.compile(r'(?:^|(?<=[\s«"(\[.,;:]))([А-ЯЁA-Z][а-яёa-z]{1,}(?:-[А-ЯЁA-Z][а-яёa-z]{1,})*)-$')


class TrainedTokenClassifierDetector:
    """Детектор сущностей на базе обученной AI-модели токенной классификации."""

    def __init__(self, model_path: str | None = None):
        candidates = [model_path] if model_path else [_CW_MODEL_PATH, _LAB1_MODEL_PATH]
        path = next((p for p in candidates if p and os.path.exists(p)), candidates[0])
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл обученной модели не найден: {path}. Сначала запустите train_ner_model.py.")

        artifact = joblib.load(path)
        self.vectorizer = artifact["vectorizer"]
        self.classifier = artifact["classifier"]
        self.metrics = artifact.get("metrics", {})

    @staticmethod
    def _stitch_hyphenated_entities(
        text: str,
        entities: list[tuple[str, int, int]],
    ) -> list[tuple[str, int, int]]:
        """Склеивать и расширять дефисные составные фамилии/имена (Петров-Водкин, Мамин-Сибиряк)."""
        if not entities:
            return entities

        # 1. Сначала объединяем смежные спаны, разделённые дефисом без пробелов
        merged: list[tuple[str, int, int]] = []
        i = 0
        while i < len(entities):
            tag, start, end = entities[i]
            while i + 1 < len(entities):
                n_tag, n_start, n_end = entities[i + 1]
                if text[end:n_start] == "-" and tag in ("SURNAME", "NAME") and n_tag in ("SURNAME", "NAME"):
                    tag = "SURNAME" if ("SURNAME" in (tag, n_tag)) else "NAME"
                    end = n_end
                    i += 1
                    continue
                break
            merged.append((tag, start, end))
            i += 1

        # 2. Если классификатор захватил только одну половину дефисного антропонима, расширяем до целого токена
        expanded: list[tuple[str, int, int]] = []
        for tag, start, end in merged:
            if tag in ("SURNAME", "NAME"):
                m_right = _HYPHEN_RIGHT_RE.match(text[end:])
                if m_right:
                    end = end + len(m_right.group(0))
                    tag = "SURNAME"
                m_left = _HYPHEN_LEFT_RE.search(text[:start])
                if m_left:
                    start = m_left.start(1)
                    tag = "SURNAME"
            expanded.append((tag, start, end))

        # Устраняем возможные перекрытия после расширения
        deduped: list[tuple[str, int, int]] = []
        for span in expanded:
            if deduped and span[1] < deduped[-1][2]:
                prev_tag, prev_s, prev_e = deduped[-1]
                deduped[-1] = (
                    "SURNAME" if ("SURNAME" in (prev_tag, span[0])) else prev_tag,
                    min(prev_s, span[1]),
                    max(prev_e, span[2]),
                )
            else:
                deduped.append(span)

        return deduped

    def detect(self, text: str) -> list[tuple[str, int, int]]:
        """
        Предсказать именованные сущности (SURNAME, NAME) в тексте.

        Returns:
            Список кортежей [(tag, start_idx, end_idx), ...].
        """
        if not text or not text.strip():
            return []

        tokens_spans = tokenize_with_spans(text)
        if not tokens_spans:
            return []

        token_texts = [t[0] for t in tokens_spans]
        features = sentence_to_features(token_texts)

        X = self.vectorizer.transform(features)
        y_pred = self.classifier.predict(X)

        # Реконструкция спанов из BIO-разметки
        entities: list[tuple[str, int, int]] = []
        current_tag = None
        current_start = 0
        current_end = 0

        for (token, t_start, t_end), bio_tag in zip(tokens_spans, y_pred):
            if bio_tag.startswith("B-"):
                if current_tag is not None:
                    entities.append((current_tag, current_start, current_end))
                current_tag = bio_tag[2:]
                current_start = t_start
                current_end = t_end
            elif bio_tag.startswith("I-"):
                tag = bio_tag[2:]
                if current_tag == tag:
                    current_end = t_end
                else:
                    if current_tag is not None:
                        entities.append((current_tag, current_start, current_end))
                    current_tag = tag
                    current_start = t_start
                    current_end = t_end
            else:  # O
                if current_tag is not None:
                    entities.append((current_tag, current_start, current_end))
                    current_tag = None

        if current_tag is not None:
            entities.append((current_tag, current_start, current_end))

        return self._stitch_hyphenated_entities(text, entities)
