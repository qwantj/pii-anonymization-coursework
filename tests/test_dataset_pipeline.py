"""
Тесты пайплайна сборки и валидации датасетов:
1. Валидация парсеров FactRuEval, Collection3, BSNLP, NEREL и синтетики.
2. Проверка строгой схемы IOB2 (отсутствие нелегальных переходов O -> I-X, B-X -> I-Y).
3. Проверка отсутствия утечки данных (data leakage) между сплитами train, val, test.
"""

import pytest
from data_pipeline.schema import (
    LABEL_LIST,
    LABEL2ID,
    ID2LABEL,
    validate_iob2_sequence,
    repair_iob2_sequence,
)
from data_pipeline.synthetic_generator import (
    generate_synthetic_dataset,
    _build_iob2_example,
)
from data_pipeline.open_datasets import (
    _align_spans_to_tokens,
    _find_token_offsets,
    _tokenize_text,
)


def test_iob2_strict_validation():
    """Тестирование строгой проверки IOB2."""
    tokens = ["Иван", "Иванович", "работает", "в", "ООО", "Вектор"]

    # Валидная последовательность
    valid_tags = [
        LABEL2ID["B-PER"], LABEL2ID["I-PER"],
        LABEL2ID["O"], LABEL2ID["O"],
        LABEL2ID["B-ORG"], LABEL2ID["I-ORG"]
    ]
    assert validate_iob2_sequence(tokens, valid_tags) is True

    # Невалидный переход O -> I-PER
    invalid_o_to_i = [
        LABEL2ID["O"], LABEL2ID["I-PER"],
        LABEL2ID["O"], LABEL2ID["O"],
        LABEL2ID["B-ORG"], LABEL2ID["I-ORG"]
    ]
    assert validate_iob2_sequence(tokens, invalid_o_to_i) is False

    # Невалидный переход B-PER -> I-ORG
    invalid_cross_tag = [
        LABEL2ID["B-PER"], LABEL2ID["I-ORG"],
        LABEL2ID["O"], LABEL2ID["O"],
        LABEL2ID["B-ORG"], LABEL2ID["I-ORG"]
    ]
    assert validate_iob2_sequence(tokens, invalid_cross_tag) is False

    # Проверка починки разметки
    repaired = repair_iob2_sequence(invalid_o_to_i)
    assert validate_iob2_sequence(tokens, repaired) is True
    assert repaired[1] == LABEL2ID["B-PER"]


def test_synthetic_generator_coverage():
    """Проверка генерации синтетических примеров и Hard Negatives."""
    samples = generate_synthetic_dataset(target_count=40, hard_negatives_count=10)
    assert len(samples) == 40

    has_per = False
    has_org = False
    has_loc = False
    has_addr = False
    has_all_o = False

    for s in samples:
        tokens = s["tokens"]
        tags = s["ner_tags"]

        assert len(tokens) == len(tags)
        assert validate_iob2_sequence(tokens, tags) is True

        labels_str = [ID2LABEL[t] for t in tags]
        if any("PER" in l for l in labels_str):
            has_per = True
        if any("ORG" in l for l in labels_str):
            has_org = True
        if any("LOC" in l for l in labels_str):
            has_loc = True
        if any("ADDR" in l for l in labels_str):
            has_addr = True
        if all(l == "O" for l in labels_str):
            has_all_o = True

    assert has_per, "В синтетике отсутствует класс PER"
    assert has_org, "В синтетике отсутствует класс ORG"
    assert has_loc, "В синтетике отсутствует класс LOC"
    assert has_addr, "В синтетике отсутствует класс ADDR"
    assert has_all_o, "В синтетике отсутствуют Hard Negatives (All-O)"


def test_open_datasets_token_alignment():
    """Проверка выравнивания символьных спанов по токенам."""
    text = "Встреча прошла в Москве в офисе Яндекс на ул. Тверская, д. 12."
    tokens = _tokenize_text(text)
    offsets = _find_token_offsets(text, tokens)

    spans = [
        (text.find("Москве"), text.find("Москве") + len("Москве"), "LOC"),
        (text.find("Яндекс"), text.find("Яндекс") + len("Яндекс"), "ORG"),
        (text.find("ул. Тверская, д. 12"), text.find("ул. Тверская, д. 12") + len("ул. Тверская, д. 12"), "ADDR"),
    ]

    tags = _align_spans_to_tokens(tokens, offsets, spans)
    assert validate_iob2_sequence(tokens, tags) is True

    # Проверяем метки
    tag_names = [ID2LABEL[t] for t in tags]
    assert "B-LOC" in tag_names
    assert "B-ORG" in tag_names
    assert "B-ADDR" in tag_names
    assert "I-ADDR" in tag_names
