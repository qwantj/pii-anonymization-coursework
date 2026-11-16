"""
Тесты токенизации ai-forever/ru-en-RoSBERTa и алгоритмов детекции:
1. Выравнивание меток (align_labels_with_tokens) с BPE subwords.
2. Проверка точного совпадения границ text[start:end] без ведущих пробелов (Ġ).
3. Склейка дефисных слов (Мамин-Сибиряк, Ростов-на-Дону) и инициалов (А. С. Пушкин).
4. Проверка Sliding Window на длинном тексте (> 1500 слов).
"""

import pytest
from transformers import AutoTokenizer

from training.train_rosberta_ner import align_labels_with_tokens
from coursework.src.detector_base import NerDetectorFactory
from lab1.src.ner.rosberta_detector import RoSBERTaNerDetector
from data_pipeline.schema import LABEL2ID, ID2LABEL


def _get_algo_detector(min_score: float = 0.1) -> RoSBERTaNerDetector:
    if NerDetectorFactory.has_local_rosberta():
        return RoSBERTaNerDetector(min_score=min_score)
    stub = RoSBERTaNerDetector.__new__(RoSBERTaNerDetector)
    stub.min_score = min_score
    return stub


def test_align_labels_with_tokens():
    """Тест выравнивания меток слов по сабтокенам Byte-Level BPE."""
    # Слово "электрокардиограмма" и спецтокены
    word_ids = [None, 0, 0, 0, 1, 2, 2, None]
    # Метки: 0 -> B-ORG, 1 -> O, 2 -> B-LOC
    labels = [LABEL2ID["B-ORG"], LABEL2ID["O"], LABEL2ID["B-LOC"]]

    aligned = align_labels_with_tokens(labels, word_ids)

    # 1. Спецтокены должны получить -100
    assert aligned[0] == -100
    assert aligned[-1] == -100

    # 2. Первый сабтокен слова 0 получает B-ORG, остальные - I-ORG
    assert aligned[1] == LABEL2ID["B-ORG"]
    assert aligned[2] == LABEL2ID["I-ORG"]
    assert aligned[3] == LABEL2ID["I-ORG"]

    # 3. Слово 1 получает O
    assert aligned[4] == LABEL2ID["O"]

    # 4. Слово 2: первый сабтокен B-LOC, второй I-LOC
    assert aligned[5] == LABEL2ID["B-LOC"]
    assert aligned[6] == LABEL2ID["I-LOC"]


def test_span_cleaning_and_boundary_alignment():
    """Тест точного совпадения границ спана без лишних BPE-пробелов."""
    detector = _get_algo_detector(min_score=0.1)

    text = "   Клиент   Петрова-Водкина Ольга   проживает в городе   Ростов-на-Дону . "
    raw_spans = [
        ("PER", 12, 27, 0.95),  # "Петрова-Водкина"
        ("PER", 28, 33, 0.95),  # "Ольга"
        ("LOC", 57, 71, 0.95),  # "Ростов-на-Дону"
    ]

    cleaned = detector._deduplicate_and_clean_spans(text, raw_spans)
    stitched = detector._stitch_hyphens_and_initials(text, cleaned)

    for tag, start, end in stitched:
        sub = text[start:end]
        assert not sub.startswith(" "), f"Спан '{sub}' содержит ведущий пробел!"
        assert not sub.endswith(" "), f"Спан '{sub}' содержит замыкающий пробел!"
        assert len(sub.strip()) == len(sub), f"Спан '{sub}' не очищен от пробелов!"

    # Проверка склейки дефисного города
    loc_spans = [text[s[1]:s[2]] for s in stitched if s[0] == "LOC"]
    assert "Ростов-на-Дону" in loc_spans


def test_initials_and_hyphens_stitching():
    """Тест склейки инициалов и дефисных фамилий."""
    detector = _get_algo_detector(min_score=0.1)
    text = "Произведение написал А. С. Пушкин в сотрудничестве с Мамин-Сибиряк ."

    # Имитируем разбитые BPE спаны
    spans = [
        ("PER", text.find("А."), text.find("А.") + 2),
        ("PER", text.find("С."), text.find("С.") + 2),
        ("PER", text.find("Пушкин"), text.find("Пушкин") + len("Пушкин")),
        ("PER", text.find("Мамин"), text.find("Мамин") + len("Мамин")),
        ("PER", text.find("Сибиряк"), text.find("Сибиряк") + len("Сибиряк")),
    ]

    stitched = detector._stitch_hyphens_and_initials(text, spans)
    extracted = [text[s[1]:s[2]] for s in stitched]

    assert "А. С. Пушкин" in extracted
    assert "Мамин-Сибиряк" in extracted


def test_sliding_window_long_document():
    """Тест работы Sliding Window на тексте более 1500 слов."""
    if not NerDetectorFactory.has_local_rosberta():
        pytest.skip("Веса RoSBERTa не загружены локально (запустите python scripts/download_models.py)")
    detector = RoSBERTaNerDetector(min_score=0.1)

    # Генерируем длинный документ с сущностями в начале, середине и конце
    filler = "Обычный контекст без персональных данных для заполнения окна модели . " * 300
    long_text = f"В начале документа упомянут Иванов Иван . {filler} В середине документа директор Петров Петр . {filler} В конце документа город Москва ."

    assert len(long_text.split()) > 1500, "Текст должен содержать более 1500 слов"

    # Запуск детекции
    results = detector.detect(long_text)

    # Проверяем, что все спаны имеют корректные неотрицательные индексы в пределах длины текста
    for tag, start, end in results:
        assert 0 <= start < end <= len(long_text)
        sub = long_text[start:end]
        assert sub.strip() == sub, f"Спан '{sub}' имеет некорректные границы"
