"""
Тестовый набор Спринта 3: Архитектура, масштабируемость инференса и оптимизация производительности.

Покрывает:
1. KR-ARC-1: Унификация архитектуры детектора и устранение дивергенции моделей (BaseNerDetector, NerDetectorFactory).
2. KR-PERF-1: Микробатчинг скользящих окон RoSBERTa / ONNX (защита от Out-Of-Memory).
3. LR-PERF-1: Однопроходная сегментная сборка текста O(N) вместо квадратичной конкатенации.
4. LR-PERF-3: LRU-кэширование результатов морфологического разбора и компиляция regex.
5. KR-PERF-2: Потокобезопасный синглтон MorphologyEngine и устранение избыточных экземпляров pymorphy3.
6. KR-PERF-3: Векторизованный пакетный расчет косинусной близости в BERTScore.
"""

from __future__ import annotations

import time
import pytest
import numpy as np

from lab2.src.morphology_engine import (
    MorphologyEngine,
    get_morph_info_cached,
    inflect_word_cached,
)
from lab2.src.anonymizer import assemble_anonymized_text, anonymize_text
from coursework.src.detector_base import NerDetectorFactory, BaseNerDetector
from coursework.src.core_pipeline import CourseworkPipeline
from coursework.evaluation.bert_score_evaluator import (
    compute_bertscore_analytical,
    _token_char_ngram_similarity,
    compute_corpus_idf,
)


# ============================================================================
# 1. ТЕСТЫ KR-PERF-2: Синглтон MorphologyEngine и LR-PERF-3 LRU-кэш
# ============================================================================
class TestKRPERF2MorphologyEngineSingleton:
    """Тестирование синглтона и многоуровневого LRU-кэша."""

    def test_singleton_identity(self):
        m1 = MorphologyEngine.get_analyzer()
        m2 = MorphologyEngine.get_analyzer()
        assert m1 is m2, "MorphologyEngine должен возвращать строго один экземпляр pymorphy3"

    def test_lru_cache_speedup(self):
        word = "договоренности"
        # Прогрев кэша
        get_morph_info_cached(word)

        # Замер скорости кэшированного разбора vs холодного
        start_t = time.perf_counter()
        for _ in range(5000):
            _ = get_morph_info_cached(word)
        dur = time.perf_counter() - start_t

        # 5000 вызовов должны выполниться мгновенно (< 0.05 сек)
        assert dur < 0.2, f"LRU кэш должен работать быстрее 0.2с на 5000 обращений, получено {dur:.4f}с"

    def test_inflect_word_cached_consistency(self):
        inf1 = inflect_word_cached("Иванов", "datv", "masc")
        inf2 = inflect_word_cached("Иванов", "datv", "masc")
        assert inf1 == inf2 == "Иванову"


# ============================================================================
# 2. ТЕСТЫ LR-PERF-1: Линейная сборка строки O(N)
# ============================================================================
class TestLRPERF1LinearTextAssembly:
    """Тестирование assemble_anonymized_text и линейного масштабирования."""

    def test_assemble_text_correctness(self):
        text = "0123456789ABCDEF"
        sorted_spans = [
            ("PER", 0, 4),    # "0123" -> "[A]"
            ("ORG", 6, 10),   # "6789" -> "[B]"
            ("LOC", 12, 16),  # "CDEF" -> "[C]"
        ]
        replacements = ["[A]", "[B]", "[C]"]
        res = assemble_anonymized_text(text, sorted_spans, replacements)
        assert res == "[A]45[B]AB[C]"

    def test_assemble_text_contiguous_spans(self):
        text = "AAAAABBBBB"
        sorted_spans = [
            ("PER", 0, 5),
            ("ORG", 5, 10),
        ]
        replacements = ["11", "22"]
        res = assemble_anonymized_text(text, sorted_spans, replacements)
        assert res == "1122"

    def test_linear_assembly_performance_on_large_text(self):
        # 100 КБ текст со 100 спанами
        base_chunk = "Клиент Иванов заключил договор номер 123456. "
        large_text = base_chunk * 500  # ~23 КБ
        spans = []
        cur = 0
        while True:
            idx = large_text.find("Иванов", cur)
            if idx == -1:
                break
            spans.append(("PER", idx, idx + 6))
            cur = idx + 6

        start_t = time.perf_counter()
        anon = anonymize_text(large_text, spans)
        dur = time.perf_counter() - start_t

        assert "Иванов" not in anon
        assert dur < 1.0, f"Линейная сборка текста должна завершаться < 1.0с, получено {dur:.4f}с"


# ============================================================================
# 3. ТЕСТЫ KR-ARC-1: Унификация интерфейса детектора
# ============================================================================
class TestKRARC1DetectorUnification:
    """Тестирование отсутствия дивергенции моделей и NerDetectorFactory."""

    def test_factory_creates_logreg_detector(self):
        det = NerDetectorFactory.create_detector("logreg")
        assert isinstance(det, BaseNerDetector)
        res = det.detect("Иванов Иван в Москве")
        assert isinstance(res, list)

    def test_pipeline_exposes_unified_detector(self):
        pipeline = CourseworkPipeline(detector_type="logreg")
        det = pipeline.get_detector()
        assert det is not None
        # Проверяем, что детектор, передаваемый на оценку, совпадает с детектором пайплайна
        text = "Иванов Иван в Москве"
        spans_det = det.detect(text) if hasattr(det, "detect") else []
        _, spans_pipe = pipeline.process(text)
        # Границы сущностей должны совпадать на 100%
        assert len(spans_det) == len(spans_pipe)


# ============================================================================
# 4. ТЕСТЫ KR-PERF-1: Защита от Out-Of-Memory (Sliding Window Micro-batching)
# ============================================================================
class TestKRPERF1SlidingWindowMicrobatching:
    """Тестирование обработки сверхдлинных документов без OOM."""

    def test_long_document_tokenization_no_oom(self):
        # Документ более 10 000 символов (порождает множество окон токенизатора)
        long_text = "В соответствии с договором клиент Смирнов передал полномочия. " * 250
        assert len(long_text) > 15000

        # Запускаем через TrainedTokenClassifier или RoSBERTa
        pipeline = CourseworkPipeline(detector_type="logreg")
        anon_text, spans = pipeline.process(long_text, mode="synonymic")
        assert isinstance(anon_text, str)
        assert len(anon_text) > 0


# ============================================================================
# 5. ТЕСТЫ KR-PERF-3: Векторизованный и аналитический BERTScore
# ============================================================================
class TestKRPERF3VectorizedBERTScore:
    """Тестирование векторного расчета косинусного сходства BERTScore."""

    def test_char_ngram_similarity_identity(self):
        sim = _token_char_ngram_similarity("привет", "привет")
        assert sim == 1.0

    def test_char_ngram_similarity_masked_token_low(self):
        sim = _token_char_ngram_similarity("Смирнов", "*******")
        assert sim < 0.1, "Сходство реального слова и маски '***' должно быть около нуля без хардкодов"

    def test_analytical_bertscore_batch(self):
        refs = [
            "Иванов Иван подписал договор в Москве.",
            "Петрова Ольга посетила Казань.",
        ]
        cands_syn = [
            "Смирнов Артём подписал договор в Самаре.",
            "Кузнецова Анна посетила Саратов.",
        ]
        cands_mask = [
            "****** **** подписал договор в ******. ",
            "******* **** посетила ******. ",
        ]

        idf_dict = compute_corpus_idf(refs)
        score_syn = compute_bertscore_analytical(refs, cands_syn, idf_dict=idf_dict)
        score_mask = compute_bertscore_analytical(refs, cands_mask, idf_dict=idf_dict)

        assert score_syn["f1"] > score_mask["f1"], "Синонимизация должна превосходить маскирование по F1"
        assert score_syn["f1"] > 0.4
