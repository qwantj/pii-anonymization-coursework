"""
Тесты идемпотентности и сквозного выполнения (End-to-End):
1. Проверка идемпотентности: одно и то же лицо в разных падежах получает один псевдоним.
2. Проверка семантической близости BERTScore >= 0.85 при синонимическом обезличивании.
3. Сквозной запуск точек входа run_lab1, run_lab2 и run_coursework без ошибок.
"""

import pytest
from coursework.src.core_pipeline import CourseworkPipeline
from coursework.src.idempotency_manager import IdempotencyVault
from coursework.evaluation.bert_score_evaluator import evaluate_bertscore_comparison


def test_idempotency_cross_case_consistency():
    """
    Тест идемпотентности: одно и то же лицо в 4 разных падежах внутри документа
    должно получить одинаковое фейковое имя во всех вхождениях.
    """
    pipeline = CourseworkPipeline()
    vault = IdempotencyVault()

    text = (
        "Иванов подписал договор. Документы Иванова проверены. "
        "Справка выдана Иванову. Копия согласована с Ивановым."
    )

    anon_text, spans = pipeline.process(text, mode="synonymic", vault=vault)

    # Проверяем, что исходный "Иванов" полностью исчез
    assert "Иванов" not in anon_text

    # Разбиваем на предложения и находим сгенерированные фамилии
    words = anon_text.split()
    first_surname = words[0]  # именительный: напр. "Смирнов"

    # Проверяем, что корень/основа фамилии совпадает во всех предложениях
    base_stem = first_surname[:4].lower()

    # Второе предложение: "Документы Смирнова проверены."
    assert base_stem in words[4].lower(), f"Несогласованная замена во 2-м предложении: {words[4]}"
    # Третье предложение: "Справка выдана Смирнову."
    assert base_stem in words[8].lower(), f"Несогласованная замена в 3-м предложении: {words[8]}"
    # Четвертое предложение: "Копия согласована с Смирновым."
    assert base_stem in words[12].lower(), f"Несогласованная замена в 4-м предложении: {words[12]}"


def test_bertscore_quality_threshold():
    """Тест достижения порога BERTScore >= 0.85 при синонимизации."""
    originals = [
        "Клиент Иванов Сергей подал заявку на кредит в г. Москва.",
        "Индивидуальный предприниматель Петрова Ольга зарегистрирована с ОГРНИП 304500116000157.",
        "Договор подряда заключен между ООО Вектор и директором Сидоровым в Санкт-Петербурге.",
    ]

    pipeline = CourseworkPipeline()
    synonymized = []
    masked = []

    for text in originals:
        syn, _ = pipeline.process(text, mode="synonymic")
        mask, _ = pipeline.process(text, mode="masking")
        synonymized.append(syn)
        masked.append(mask)

    metrics = evaluate_bertscore_comparison(originals, synonymized, masked)

    syn_f1 = metrics["synonymic"]["f1"]
    mask_f1 = metrics["masked"]["f1"]
    delta = metrics["delta_f1"]

    assert syn_f1 >= 0.85, f"BERTScore синонимизации {syn_f1} ниже требуемого порога 0.85!"
    assert syn_f1 > mask_f1, f"Синонимизация ({syn_f1}) должна сохранять больше смысла, чем маскирование ({mask_f1})!"
    assert delta > 0.05, f"Преимущество синонимизации над маскированием должно быть значимым (получено {delta})"


def test_entry_points_execution():
    """Проверка запуска функций точек входа ЛР1, ЛР2 и КР без исключений."""
    # 1. ЛР1 (с проверкой hybrid_pipeline)
    from lab1.src.hybrid_pipeline import detect_pii
    sample_text = "Иванов Иван, ИНН 770698959438, email ivan@mail.ru"
    spans_l1 = detect_pii(sample_text)
    assert len(spans_l1) >= 2

    # 2. ЛР2 (с проверкой anonymize_text)
    from lab2.src.anonymizer import anonymize_text
    anon_l2 = anonymize_text(sample_text, spans_l1)
    assert "770698959438" not in anon_l2
    assert "ivan@mail.ru" not in anon_l2

    # 3. КР (с проверкой CourseworkPipeline)
    cw_pipeline = CourseworkPipeline()
    cw_anon, cw_spans = cw_pipeline.process(sample_text, mode="synonymic")
    assert len(cw_spans) >= 2
    assert "770698959438" not in cw_anon
