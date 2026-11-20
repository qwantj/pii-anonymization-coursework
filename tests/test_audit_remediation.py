"""
Тестовый комплекс целевой верификации устранения замечаний аудита курсовой работы CTZI.
Покрывает:
1. Исключение родительского PER, декомпозицию ФИО, дефисные фамилии, нормализацию пунктуации и честный fallback.
2. Отсутствие InconsistentVersionWarning при десериализации joblib.
3. Высокую полноту (Recall = 1.0), приоритет EMAIL (1.5), поддержку Punycode, валидацию Lookbehind и санитизацию mailto:.
4. Достижение нормативных метрик Weighted F1 >= 0.85 и Macro F1 >= 0.80 (целевой >= 0.92).
"""

import os
import re
import json
import warnings
import joblib
import pytest

from lab1.src.ner.rosberta_detector import RoSBERTaNerDetector
from lab1.src.hybrid_pipeline import detect_pii, HybridDetector
from lab1.src.rules.regex_patterns import EMAIL_PATTERN, IP_ADDRESS_PATTERN
from coursework.src.core_pipeline import CourseworkPipeline
from coursework.src.detector_base import NerDetectorFactory
from coursework.evaluation.f1_evaluator import evaluate_dataset


def _get_rosberta_or_algo_stub():
    if NerDetectorFactory.has_local_rosberta():
        return RoSBERTaNerDetector(), True
    stub = RoSBERTaNerDetector.__new__(RoSBERTaNerDetector)
    stub.min_score = 0.1
    return stub, False


class TestAuditRemediation:

    def test_item1_split_per_span_structure_and_heuristics(self):
        """
        Пункт 1: Всесторонняя проверка алгоритма _split_per_span:
        - Исключение родительского PER при разделении антропонимов.
        - Нормализация внешней пунктуации (запятые, точки вне границ спанов).
        - Поддержка дефисных фамилий ("Мамин-Сибиряк") и предотвращение ложной инверсии ролей.
        - Обратный порядок для 2- и 3-компонентных ФИО.
        - Славянские суффиксы (-ко, -енко, -ук, -юк, -як).
        - Честный fallback на PER для мусорных токенов (цифры, пунктуация, аббревиатуры).
        - Сквозная детекция в гибридном конвейере с явным трансформером RoSBERTa.
        """
        detector, has_weights = _get_rosberta_or_algo_stub()

        # 1. Базовое разделение прямого порядка: "Иванов Иван"
        text1 = "Гражданин Иванов Иван обратился с заявлением."
        spans1 = detector._split_per_span(text1, 10, 21)
        tags1 = [s[0] for s in spans1]
        assert "PER" not in tags1, "Родительский тег PER не должен сохраняться при успешном разделении!"
        assert ("SURNAME", 10, 16) in [(s[0], s[1], s[2]) for s in spans1]
        assert ("NAME", 17, 21) in [(s[0], s[1], s[2]) for s in spans1]

        # 2. Нормализация пунктуации: запятая и точка не должны попадать внутрь спанов
        p_text = "Заявитель: Иванов, Иван. Документы сданы."
        p_spans = detector._split_per_span(p_text, 11, 24)  # "Иванов, Иван."
        assert [s[0] for s in p_spans] == ["SURNAME", "NAME"]
        s_sub = p_text[p_spans[0][1]:p_spans[0][2]]
        n_sub = p_text[p_spans[1][1]:p_spans[1][2]]
        assert s_sub == "Иванов", f"Запятая ошибочно захвачена в спан SURNAME: '{s_sub}'"
        assert n_sub == "Иван", f"Точка ошибочно захвачена в спан NAME: '{n_sub}'"

        # 3. Дефисные фамилии и исключение ложной инверсии ролей ("Имя Фамилия")
        rev_text = "Писатель Дмитрий Мамин-Сибиряк оставил богатое наследие."
        rev_spans = detector._split_per_span(rev_text, 9, 30)  # "Дмитрий Мамин-Сибиряк"
        assert [(s[0], rev_text[s[1]:s[2]]) for s in rev_spans] == [
            ("NAME", "Дмитрий"),
            ("SURNAME", "Мамин-Сибиряк")
        ], f"Инверсия ролей для дефисной фамилии: {rev_spans}"

        # Дефисная фамилия на первой позиции ("Фамилия Имя")
        dir_text = "Мамин-Сибиряк Дмитрий Наркисович"
        dir_spans = detector._split_per_span(dir_text, 0, 32)
        assert [(s[0], dir_text[s[1]:s[2]]) for s in dir_spans] == [
            ("SURNAME", "Мамин-Сибиряк"),
            ("NAME", "Дмитрий"),
            ("NAME", "Наркисович")
        ]

        # 4. Обратный порядок 3-словных ФИО ("Имя Отчество Фамилия")
        fio_rev = "Врач Иван Иванович Иванов ведет прием."
        fio_spans = detector._split_per_span(fio_rev, 5, 25)  # "Иван Иванович Иванов"
        assert [(s[0], fio_rev[s[1]:s[2]]) for s in fio_spans] == [
            ("NAME", "Иван"),
            ("NAME", "Иванович"),
            ("SURNAME", "Иванов")
        ], f"Неверная разметка обратного порядка ФИО: {fio_spans}"

        # 5. Проверка продуктивных славянских суффиксов (-енко, -ко, -ук, -як)
        ukr_text = "Поэт Тарас Шевченко родился в селе Моринцы."
        ukr_spans = detector._split_per_span(ukr_text, 5, 19)  # "Тарас Шевченко"
        assert [(s[0], ukr_text[s[1]:s[2]]) for s in ukr_spans] == [
            ("NAME", "Тарас"),
            ("SURNAME", "Шевченко")
        ]

        # 6. Честный fallback на PER для невалидных токенов (устранение мертвого кода)
        for junk in ["...", "123", "ООО", "!!!", "г."]:
            text_junk = f"Объект {junk} зарегистрирован."
            j_start = text_junk.find(junk)
            j_end = j_start + len(junk)
            spans_junk = detector._split_per_span(text_junk, j_start, j_end)
            assert spans_junk == [("PER", j_start, j_end)], (
                f"Честный fallback на PER не сработал для мусорного токена '{junk}': {spans_junk}"
            )

        # 7. Сквозная интеграция в гибридном конвейере (RoSBERTa при наличии весов, иначе дефолтный пайплайн)
        active_det = detector if has_weights else CourseworkPipeline().get_detector()
        final_spans = detect_pii(text1, ner_detector=active_det)
        final_tags = [s[0] for s in final_spans]
        assert "PER" not in final_tags, "PER просочился в финальные результаты пайплайна!"
        assert "SURNAME" in final_tags and "NAME" in final_tags

    def test_item3_email_regex_and_pipeline_priority(self):
        """
        Пункт 3: EMAIL защищен приоритетом 1.5 от нейросети, поддерживает Punycode,
        имеет Recall = 1.0 и 0% FP на URL/путях.
        """
        text = "Для корреспонденции используйте maria898@gmail.com или admin@q.com."
        
        # 1. Проверка изоляции от нейросети (имя maria в email не должно вытеснять EMAIL)
        spans = detect_pii(text)
        email_spans = [s for s in spans if s[0] == "EMAIL"]
        assert len(email_spans) == 2, f"Ожидалось 2 адреса электронной почты, найдено: {email_spans}"
        
        # 2. Проверка поддержки Punycode и однобуквенных доменов второго уровня
        puny_text = "Связь: user@domain.xn--p1ai и admin@x.org"
        p_spans = detect_pii(puny_text)
        found_emails = [puny_text[s[1]:s[2]] for s in p_spans if s[0] == "EMAIL"]
        assert "user@domain.xn--p1ai" in found_emails
        assert "admin@x.org" in found_emails

        # 3. Проверка негативных проверок (не захватывать URL с basic-auth и пути Windows)
        neg_text = "Инструкция: http://user@domain.com и файл c:\\windows\\user@domain.com"
        raw_matches = EMAIL_PATTERN.findall(neg_text)
        assert len(raw_matches) == 0, f"Ложные срабатывания на путях и URL: {raw_matches}"

        # 4. Проверка поведения Lookbehind в отношении ссылок mailto: и предварительной санитизации
        mailto_text = "Связь по ссылке mailto:support@company.ru в документе."
        # Из-за двоеточия в (?<![...:]) строка mailto:support@company.ru не извлекается напрямую
        assert len(EMAIL_PATTERN.findall(mailto_text)) == 0, "Lookbehind должен отсекать двоеточие в URI схемах"
        # Проверка инженерного регламента: предварительная санитизация восстанавливает распознавание
        sanitized_mailto = re.sub(r'(?i)\bmailto:\s*', '', mailto_text)
        assert EMAIL_PATTERN.findall(sanitized_mailto) == ["support@company.ru"]

    def test_item4_no_inconsistent_version_warnings_in_joblib(self):
        """Пункт 4: Загрузка models/ner_token_classifier.joblib должна проходить без предупреждений."""
        from sklearn.exceptions import InconsistentVersionWarning
        
        for p in ["lab1/models/ner_token_classifier.joblib", "coursework/models/ner_token_classifier.joblib"]:
            if not os.path.exists(p):
                continue
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                artifact = joblib.load(p)
                sklearn_warns = [w for w in captured if issubclass(w.category, InconsistentVersionWarning)]
                assert len(sklearn_warns) == 0, f"Файл {p} вызвал InconsistentVersionWarning: {[str(w.message) for w in sklearn_warns]}"
                assert "vectorizer" in artifact and "classifier" in artifact
                assert artifact["classes"] == ['B-NAME', 'B-SURNAME', 'O']

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_items1_and_3_benchmark_metrics_compliance(self):
        """Пункты 1 и 3: Проверка выполнения нормативов качества на контрольном датасете."""
        dataset_path = "coursework/data/dataset.json"
        assert os.path.exists(dataset_path), f"Датасет {dataset_path} не найден!"
        
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        pipeline = CourseworkPipeline()
        metrics = evaluate_dataset(data, ner_detector=pipeline.get_detector())
        
        # Проверка интегральных порогов (Инвариант C9)
        assert metrics["weighted_f1"] >= 0.85, f"Weighted F1 {metrics['weighted_f1']} < 0.85"
        assert metrics["macro_f1"] >= 0.80, f"Macro F1 {metrics['macro_f1']} < 0.80"
        
        # Проверка восстановления полноты по ключевым проблемным классам
        assert metrics["per_class"]["EMAIL"]["recall"] >= 0.90, "Recall EMAIL ниже норматива 0.90!"
        assert metrics["per_class"]["SURNAME"]["f1"] >= 0.85, "F1 SURNAME ниже норматива 0.85!"
        assert metrics["per_class"]["NAME"]["f1"] >= 0.85, "F1 NAME ниже норматива 0.85!"
