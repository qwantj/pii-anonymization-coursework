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

_SURNAME_SUFFIXES = (
    "ов", "ова", "ев", "ева", "ёв", "ёва", "ин", "ина", "ын", "ына",
    "ский", "ская", "цкий", "цкая", "ых", "их", "ко", "чук", "юк", "енко",
)
_AMBIGUOUS_NAMES = {
    "саша", "женя", "валя", "слава", "лена", "паша", "миша", "ярослава",
    "ваня", "тоня", "шура", "даня", "алекса",
}
_NAME_CONTEXT_WORDS = {
    "документы", "справка", "заявитель", "клиент", "гражданин", "гражданка",
    "у", "для", "от", "к",
}
_KNOWN_NAMES_CACHE: set[str] | None = None


def _get_known_names() -> set[str]:
    """Лениво собрать множество известных русских имён в нижнем регистре."""
    global _KNOWN_NAMES_CACHE
    if _KNOWN_NAMES_CACHE is None:
        from lab2.src.anonymizer import _NAMES_M, _NAMES_F
        names = {n.lower().replace("ё", "е") for n in (_NAMES_M + _NAMES_F)}
        names.update(n.replace("ё", "е") for n in _AMBIGUOUS_NAMES)
        _KNOWN_NAMES_CACHE = names
    return _KNOWN_NAMES_CACHE


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

    @staticmethod
    def _refine_surname_name_tags(
        text: str,
        tokens_spans: list[tuple[str, int, int]],
        entities: list[tuple[str, int, int]],
    ) -> list[tuple[str, int, int]]:
        """Пост-коррекция тегов SURNAME/NAME и пропущенных унисекс-имён по морфологии."""
        from lab2.src.morphology_engine import MorphologyEngine

        morph = MorphologyEngine.get_analyzer()
        known_names = _get_known_names()

        refined: list[tuple[str, int, int]] = []
        for tag, start, end in entities:
            span_text = text[start:end]
            if "-" not in span_text and tag in ("SURNAME", "NAME"):
                parses = morph.parse(span_text)
                has_surn = any(p.is_known and "Surn" in p.tag for p in parses)
                has_name = any(p.is_known and "Name" in p.tag for p in parses)
                non_fixd_lemmas = {
                    p.normal_form.lower().replace("ё", "е")
                    for p in parses
                    if p.is_known and "Fixd" not in p.tag
                }
                in_known = bool(non_fixd_lemmas & known_names) or (
                    span_text.lower().replace("ё", "е") in known_names
                )
                low = span_text.lower()

                if tag == "SURNAME" and not low.endswith(_SURNAME_SUFFIXES) and not has_surn and (has_name or in_known):
                    tag = "NAME"
                elif tag == "NAME" and not in_known and has_surn and not has_name:
                    tag = "SURNAME"

            refined.append((tag, start, end))

        surname_ranges = [(s, e) for t, s, e in refined if t == "SURNAME"]

        def _is_covered(s: int, e: int) -> bool:
            return any(max(s, es) < min(e, ee) for _, es, ee in refined)

        def _is_surname_span(s: int, e: int) -> bool:
            return any(max(s, ss) < min(e, se) for ss, se in surname_ranges)

        def _is_trigger_neighbor(tok_idx: int) -> bool:
            if tok_idx < 0 or tok_idx >= len(tokens_spans):
                return False
            nb_word, nb_s, nb_e = tokens_spans[tok_idx]
            if _is_surname_span(nb_s, nb_e):
                return True
            if nb_word.lower() in _NAME_CONTEXT_WORDS:
                return True
            nb_parses = morph.parse(nb_word)
            return bool(nb_parses and "VERB" in nb_parses[0].tag and "past" in nb_parses[0].tag)

        for idx, (word, t_start, t_end) in enumerate(tokens_spans):
            if not word or not word[0].isupper() or not word.isalpha():
                continue
            if _is_covered(t_start, t_end):
                continue

            parses = morph.parse(word)
            is_ambig_name = any(
                "Name" in p.tag and p.normal_form.lower() in _AMBIGUOUS_NAMES
                for p in parses
            )
            if is_ambig_name and (_is_trigger_neighbor(idx - 1) or _is_trigger_neighbor(idx + 1)):
                refined.append(("NAME", t_start, t_end))

        refined.sort(key=lambda s: s[1])
        return refined

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

        entities = self._refine_surname_name_tags(text, tokens_spans, entities)
        return self._stitch_hyphenated_entities(text, entities)
