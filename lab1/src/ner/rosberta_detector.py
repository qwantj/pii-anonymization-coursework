"""
Детектор именованных сущностей (NER) на базе модели ai-forever/ru-en-RoSBERTa.

Особенности реализации:
1. Byte-Level BPE токенизатор с параметром add_prefix_space=True.
2. Точное восстановление символьных границ (offset_mapping) с очисткой ведущих/замыкающих пробелов (Ġ).
3. Пост-обработка дефисов (Санкт-Петербург, Петрова-Водкина) и инициалов (А. С. Пушкин).
4. Скользящее окно (Sliding Window: max_length=512, stride=64) для текстов произвольной длины.
5. Поддержка локального чекпоинта, ONNX Runtime и фоллбэка на базовую модель Hugging Face.
"""

from __future__ import annotations

import os
import re
from typing import Sequence

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

from data_pipeline.schema import ID2LABEL, LABEL2ID, LABEL_LIST


class RoSBERTaNerDetector:
    """Детектор ПДн на базе билингвальной модели ai-forever/ru-en-RoSBERTa."""

    DEFAULT_MODEL = "ai-forever/ru-en-RoSBERTa"
    LOCAL_CHECKPOINT_DIRS = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "rosberta-ner-finetuned")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "rubert_pii_final")),
    ]

    DEFAULT_ONNX_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "rosberta_onnx", "model_int8.onnx"))

    def __init__(
        self,
        model_name_or_path: str | None = None,
        min_score: float = 0.40,
        device: str | None = None,
        use_onnx: bool = False,
        onnx_model_path: str | None = None,
    ):
        self.min_score = min_score
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_onnx = use_onnx or (os.environ.get("USE_ONNX", "").lower() in ("1", "true", "yes"))
        self.onnx_session = None
        self.onnx_path = None

        # 1. Поиск лучшего доступного чекпоинта
        chosen_path = model_name_or_path or os.environ.get("ROSBERTA_MODEL_PATH")
        if not chosen_path:
            for local_dir in self.LOCAL_CHECKPOINT_DIRS:
                if os.path.exists(os.path.join(local_dir, "config.json")):
                    chosen_path = local_dir
                    break

        if not chosen_path:
            chosen_path = self.DEFAULT_MODEL

        self.model_path = chosen_path

        # 2. Инициализация ONNX Runtime сессии (INT8 / FP32)
        if self.use_onnx:
            target_onnx = onnx_model_path or os.environ.get("ONNX_MODEL_PATH") or self.DEFAULT_ONNX_PATH
            if not os.path.exists(target_onnx):
                alt_onnx = target_onnx.replace("_int8.onnx", ".onnx")
                if os.path.exists(alt_onnx):
                    target_onnx = alt_onnx

            if os.path.exists(target_onnx):
                try:
                    import onnxruntime as ort
                    sess_options = ort.SessionOptions()
                    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    self.onnx_session = ort.InferenceSession(target_onnx, sess_options, providers=["CPUExecutionProvider"])
                    self.onnx_path = target_onnx
                except Exception:
                    self.use_onnx = False
                    self.onnx_session = None
            else:
                self.use_onnx = False

        self._init_model(chosen_path)

    def _init_model(self, path: str):
        """Инициализация токенизатора и весов модели."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(path, add_prefix_space=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(self.DEFAULT_MODEL, add_prefix_space=True)

        if not self.use_onnx:
            try:
                self.model = AutoModelForTokenClassification.from_pretrained(path)
            except Exception:
                config = AutoConfig.from_pretrained(
                    self.DEFAULT_MODEL,
                    num_labels=len(LABEL_LIST),
                    id2label=ID2LABEL,
                    label2id=LABEL2ID,
                )
                self.model = AutoModelForTokenClassification.from_pretrained(self.DEFAULT_MODEL, config=config)

            self.model.to(self.device)
            self.model.eval()
        else:
            self.model = None

    def detect(self, text: str) -> list[tuple[str, int, int]]:
        """
        Обнаружить сущности в тексте произвольной длины с помощью Sliding Window.

        Returns:
            Список кортежей [(tag, start, end), ...] с точными символьными границами.
        """
        if not text or not text.strip():
            return []

        # 1. Скользящее окно токенизации (max_length=512, stride=64)
        max_length = 512
        stride = 64
        
        # Получаем полную BPE токенизацию с оффсетами
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            max_length=max_length,
            stride=stride,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        offset_mappings = encoding["offset_mapping"]
        input_ids_batch = encoding["input_ids"].to(self.device)
        attention_mask_batch = encoding["attention_mask"].to(self.device)

        raw_spans_with_score: list[tuple[str, int, int, float]] = []

        chunk_batch_size = 4 if self.device == "cpu" else 16
        num_chunks = input_ids_batch.size(0)
        all_preds: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []

        if self.use_onnx and self.onnx_session is not None:
            input_ids_np = input_ids_batch.cpu().numpy()
            attention_mask_np = attention_mask_batch.cpu().numpy()
            for chunk_start in range(0, num_chunks, chunk_batch_size):
                chunk_end = min(chunk_start + chunk_batch_size, num_chunks)
                sub_inputs = input_ids_np[chunk_start:chunk_end]
                sub_mask = attention_mask_np[chunk_start:chunk_end]
                ort_outs = self.onnx_session.run(None, {
                    "input_ids": sub_inputs,
                    "attention_mask": sub_mask,
                })
                sub_logits = ort_outs[0]
                exp_logits = np.exp(sub_logits - np.max(sub_logits, axis=-1, keepdims=True))
                probs_np = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
                all_preds.append(np.argmax(probs_np, axis=-1))
                all_scores.append(np.max(probs_np, axis=-1))
            preds_batch_np = np.concatenate(all_preds, axis=0) if all_preds else np.empty((0, max_length))
            scores_batch_np = np.concatenate(all_scores, axis=0) if all_scores else np.empty((0, max_length))
        else:
            with torch.no_grad():
                for chunk_start in range(0, num_chunks, chunk_batch_size):
                    chunk_end = min(chunk_start + chunk_batch_size, num_chunks)
                    sub_inputs = input_ids_batch[chunk_start:chunk_end].to(self.device)
                    sub_mask = attention_mask_batch[chunk_start:chunk_end].to(self.device)
                    outputs = self.model(input_ids=sub_inputs, attention_mask=sub_mask)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1)
                    sub_scores, sub_preds = torch.max(probs, dim=-1)
                    all_preds.append(sub_preds.cpu().numpy())
                    all_scores.append(sub_scores.cpu().numpy())
            preds_batch_np = np.concatenate(all_preds, axis=0) if all_preds else np.empty((0, max_length))
            scores_batch_np = np.concatenate(all_scores, axis=0) if all_scores else np.empty((0, max_length))

        # 2. Извлечение спанов из каждого окна
        for chunk_idx in range(len(offset_mappings)):
            offsets = offset_mappings[chunk_idx].cpu().numpy()
            preds = preds_batch_np[chunk_idx]
            scores = scores_batch_np[chunk_idx]

            current_tag: str | None = None
            span_start: int = 0
            span_end: int = 0
            span_scores: list[float] = []

            for token_idx, (start, end) in enumerate(offsets):
                # Пропускаем спецтокены (offset (0, 0))
                if start == 0 and end == 0:
                    if current_tag is not None:
                        avg_score = sum(span_scores) / len(span_scores) if span_scores else 0.0
                        raw_spans_with_score.append((current_tag, span_start, span_end, avg_score))
                        current_tag = None
                        span_scores = []
                    continue

                label_id = int(preds[token_idx])
                score = float(scores[token_idx])
                label_name = ID2LABEL.get(label_id, "O")

                if label_name.startswith("B-"):
                    if current_tag is not None:
                        avg_score = sum(span_scores) / len(span_scores) if span_scores else 0.0
                        raw_spans_with_score.append((current_tag, span_start, span_end, avg_score))
                    current_tag = label_name[2:]
                    span_start = int(start)
                    span_end = int(end)
                    span_scores = [score]
                elif label_name.startswith("I-"):
                    tag_type = label_name[2:]
                    if current_tag == tag_type:
                        span_end = int(end)
                        span_scores.append(score)
                    else:
                        if current_tag is not None:
                            avg_score = sum(span_scores) / len(span_scores) if span_scores else 0.0
                            raw_spans_with_score.append((current_tag, span_start, span_end, avg_score))
                        current_tag = tag_type
                        span_start = int(start)
                        span_end = int(end)
                        span_scores = [score]
                else:
                    if current_tag is not None:
                        avg_score = sum(span_scores) / len(span_scores) if span_scores else 0.0
                        raw_spans_with_score.append((current_tag, span_start, span_end, avg_score))
                        current_tag = None
                        span_scores = []

            if current_tag is not None:
                avg_score = sum(span_scores) / len(span_scores) if span_scores else 0.0
                raw_spans_with_score.append((current_tag, span_start, span_end, avg_score))

        # 3. Дедупликация на стыках окон по максимальному score
        cleaned_spans = self._deduplicate_and_clean_spans(text, raw_spans_with_score)

        # 4. Пост-обработка: склейка дефисов и инициалов
        stitched_spans = self._stitch_hyphens_and_initials(text, cleaned_spans)

        # 5. Разделение PER на SURNAME и NAME для полной обратной совместимости с ЛР1/ЛР2
        expanded_spans: list[tuple[str, int, int]] = []
        for tag, start, end in stitched_spans:
            if tag == "PER":
                # Декомпозиция PER на SURNAME / NAME (или fallback на PER при невалидных токенах)
                sub_parts = self._split_per_span(text, start, end)
                expanded_spans.extend(sub_parts)
            else:
                expanded_spans.append((tag, start, end))

        # Убираем возможные дубликаты
        unique_spans = sorted(list(set(expanded_spans)), key=lambda s: s[1])
        return unique_spans

    def _deduplicate_and_clean_spans(
        self,
        text: str,
        spans_with_score: list[tuple[str, int, int, float]],
    ) -> list[tuple[str, int, int]]:
        """
        Очистить границы токенов от BPE-пробелов (Ġ) и дедуплицировать пересечения по score.
        """
        cleaned: list[tuple[str, int, int, float]] = []

        for tag, start, end, score in spans_with_score:
            if score < self.min_score:
                continue

            # Коррекция offset_mapping: срез ведущих и завершающих пробелов
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1

            if start >= end:
                continue

            cleaned.append((tag, start, end, score))

        # Сортируем по убыванию score
        cleaned.sort(key=lambda s: -s[3])
        accepted: list[tuple[str, int, int]] = []
        occupied: set[int] = set()

        for tag, start, end, _ in cleaned:
            span_range = set(range(start, end))
            # Если перекрытие больше 50%, выбираем спан с большим score (уже первым в списке)
            if len(span_range & occupied) > 0.5 * len(span_range):
                continue
            accepted.append((tag, start, end))
            occupied.update(span_range)

        accepted.sort(key=lambda s: s[1])
        return accepted

    def _stitch_hyphens_and_initials(
        self,
        text: str,
        spans: list[tuple[str, int, int]],
    ) -> list[tuple[str, int, int]]:
        """
        Склейка соседних спанов, разделенных дефисом (Санкт-Петербург, Петрова-Водкина)
        или инициалами (А. С. Пушкин, И. И. Иванов).
        """
        if len(spans) < 2:
            return spans

        stitched: list[tuple[str, int, int]] = []
        i = 0
        while i < len(spans):
            current_tag, current_start, current_end = spans[i]

            # Проверяем склейку цепочки со следующими спанами
            while i + 1 < len(spans):
                next_tag, next_start, next_end = spans[i + 1]
                if current_tag != next_tag:
                    break

                between = text[current_end:next_start]
                curr_str = text[current_start:current_end].strip()
                next_str = text[next_start:next_end].strip()

                # 1. Дефис: "Санкт-Петербург", "Ростов-на-Дону", "Петрова-Водкина"
                if between.strip() in ("-", "-на-"):
                    current_end = next_end
                    i += 1
                    continue

                # 2. Инициалы и составные ФИО: "А." + "С." или "С." + "Пушкин"
                is_curr_initial = bool(re.search(r"[А-ЯЁA-Z]\.?$", curr_str))
                is_next_initial = bool(re.match(r"^[А-ЯЁA-Z]\.?", next_str))
                is_space = between in (" ", "  ", "")

                if current_tag == "PER" and is_space and (is_curr_initial or is_next_initial):
                    current_end = next_end
                    i += 1
                    continue

                break

            stitched.append((current_tag, current_start, current_end))
            i += 1

        return stitched

    @classmethod
    def _is_valid_name_token(cls, word: str) -> bool:
        """
        Проверка валидности токена как антропонима (имя / фамилия / отчество).
        Отсекает небуквенный мусор, цифры, пунктуацию и аббревиатуры (ООО, МВД).
        """
        clean = word.strip(".,;:!?\"'()«»—–")
        if not clean or len(clean) < 2:
            return False
        # Запрет цифр
        if any(ch.isdigit() for ch in clean):
            return False
        # Только буквы алфавита (допускается дефис между частями имени/фамилии)
        if not re.match(r'^[A-Za-zА-Яа-яЁё]+(-[A-Za-zА-Яа-яЁё]+)*$', clean):
            return False
        # Отсечение аббревиатур капсом (ООО, ЗАО, НИИ, МВД)
        if clean.isupper() and len(clean) > 1:
            return False
        # Антропоним в русском языке начинается с заглавной буквы
        if not clean[0].isupper():
            return False
        return True

    @classmethod
    def _is_likely_surname(cls, word: str) -> bool:
        """
        Эвристика типичных признаков фамилий:
        1. Морфологический разбор через pymorphy3 ('Surn' in tag).
        2. Поддержка дефисных составных фамилий (Мамин-Сибиряк, Римский-Корсаков).
        3. Расширенный список славянских фамильных суффиксов (включая -ко, -енко, -ук, -юк, -як).
        """
        clean = word.strip(".,;:!?\"'()«»—–")
        if not clean or any(ch.isdigit() for ch in clean):
            return False

        # 1. Морфологический анализ pymorphy3 (переиспользование синглтона из token_features или get_morph)
        morph = None
        try:
            from lab1.src.ner.token_features import get_morph
            morph = get_morph()
        except Exception:
            try:
                from lab1.src.ner.token_features import _morph
                morph = _morph
            except Exception:
                try:
                    import pymorphy3
                    morph = pymorphy3.MorphAnalyzer()
                except Exception:
                    morph = None

        if morph is not None:
            parts = clean.split("-")
            for part in parts:
                try:
                    parses = morph.parse(part)
                    if any('Surn' in p.tag for p in parses):
                        return True
                except Exception:
                    pass

        # 2. Суффиксная эвристика с поддержкой частей дефисных фамилий
        suffixes = (
            "ов", "ова", "ев", "ева", "ин", "ина",
            "ский", "ская", "цкий", "цкая", "ых", "их",
            "ко", "енко", "ук", "юк", "як"
        )
        parts = clean.split("-")
        for part in parts:
            low = part.lower()
            if any(low.endswith(s) for s in suffixes):
                return True
        return False

    def _split_per_span(self, text: str, start: int, end: int) -> list[tuple[str, int, int]]:
        """
        Разделение общего тега PER на специализированные сущности SURNAME и NAME (KR-NER-1).
        
        Инварианты C1, C3:
        - Исключает родительский тег PER при успешном выделении антропонимов,
          предотвращая вытеснение дочерних спанов в _resolve_overlaps().
        - Честный fallback: если токен невалиден (знаки препинания, цифры, аббревиатуры)
          или разделение невозможно, возвращается исходный спан [("PER", start, end)].
        - Нормализует пунктуацию с точным пересчетом символьных смещений.
        - Корректно определяет порядок следования для 2- и 3-компонентных ФИО.
        - Исключает инверсию ролей для дефисных фамилий ("Дмитрий Мамин-Сибиряк").
        """
        span_text = text[start:end]
        if not span_text.strip():
            return [("PER", start, end)]

        raw_words = span_text.split()
        if not raw_words:
            return [("PER", start, end)]

        # Выделение токенов с нормализацией пунктуации и расчетом точных смещений
        token_entries: list[tuple[str, int, int]] = []
        cursor = start
        for word in raw_words:
            w_start = text.find(word, cursor)
            if w_start == -1 or w_start >= end:
                w_start = cursor
            w_end = w_start + len(word)
            cursor = w_end

            # Точный расчет смещений без внешних знаков препинания
            lead_strip = len(word) - len(word.lstrip(".,;:!?\"'()«»—–"))
            trail_strip = len(word) - len(word.rstrip(".,;:!?\"'()«»—–"))
            clean_start = w_start + lead_strip
            clean_end = w_end - trail_strip
            clean_word = text[clean_start:clean_end]

            if clean_word and self._is_valid_name_token(clean_word):
                token_entries.append((clean_word, clean_start, clean_end))

        # Честный fallback: если ни один токен не прошел валидацию антропонима
        if not token_entries:
            return [("PER", start, end)]

        # Одиночный токен (валидный антропоним)
        if len(token_entries) == 1:
            clean_word, c_start, c_end = token_entries[0]
            tag = "SURNAME" if self._is_likely_surname(clean_word) else "NAME"
            return [(tag, c_start, c_end)]

        # Двухкомпонентное именование: прямой и обратный порядок
        if len(token_entries) == 2:
            first_word, f_start, f_end = token_entries[0]
            second_word, s_start, s_end = token_entries[1]
            first_is_surname = self._is_likely_surname(first_word)
            second_is_surname = self._is_likely_surname(second_word)

            if not first_is_surname and second_is_surname:
                # "Имя Фамилия" (например, "Иван Иванов", "Дмитрий Мамин-Сибиряк")
                return [("NAME", f_start, f_end), ("SURNAME", s_start, s_end)]
            else:
                # "Фамилия Имя" (например, "Иванов Иван", "Мамин-Сибиряк Дмитрий")
                return [("SURNAME", f_start, f_end), ("NAME", s_start, s_end)]

        # Трехкомпонентное именование (ФИО): прямой и обратный порядок
        if len(token_entries) == 3:
            w0, s0, e0 = token_entries[0]
            w1, s1, e1 = token_entries[1]
            w2, s2, e2 = token_entries[2]

            first_is_surname = self._is_likely_surname(w0)
            third_is_surname = self._is_likely_surname(w2)

            if not first_is_surname and third_is_surname:
                # Обратный порядок: "Иван Иванович Иванов" -> NAME, NAME, SURNAME
                return [("NAME", s0, e0), ("NAME", s1, e1), ("SURNAME", s2, e2)]
            else:
                # Прямой порядок: "Иванов Иван Иванович" -> SURNAME, NAME, NAME
                return [("SURNAME", s0, e0), ("NAME", s1, e1), ("NAME", s2, e2)]

        # Общий случай для N >= 4 токенов
        first_is_surname = self._is_likely_surname(token_entries[0][0])
        last_is_surname = self._is_likely_surname(token_entries[-1][0])
        results: list[tuple[str, int, int]] = []
        if not first_is_surname and last_is_surname:
            for idx, (_, c_start, c_end) in enumerate(token_entries):
                tag = "SURNAME" if idx == len(token_entries) - 1 else "NAME"
                results.append((tag, c_start, c_end))
        else:
            for idx, (_, c_start, c_end) in enumerate(token_entries):
                tag = "SURNAME" if idx == 0 else "NAME"
                results.append((tag, c_start, c_end))
        return results
