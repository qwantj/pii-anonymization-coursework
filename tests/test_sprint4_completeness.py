"""
Тестовый набор Спринта 4: Полнота решений, энтропия генерации и академическая валидация.

Покрывает:
1. KR-COMP-1: Масштабирование словарей замен (250+ единиц) и контроль коллизий псевдонимов.
2. KR-COMP-2: Реализация канонического IDF-взвешивания в BERTScore без искусственных констант.
3. LR-COMP-3: Реалистичные структуры email и расширенные пулы организаций/топонимов.
4. LR-COMP-4: Модуль автоматизированного аудита качества деидентификации (DeidentificationAuditor).
5. KR-COMP-3: Адаптивное предикативное согласование зависимых глаголов и причастий (PredicateAgreementRepair).
7. LR-ARC-1: Реестр провайдеров Strategy и безопасное fallback-маскирование неизвестных тегов.
8. KR-ARC-2: Персистентность сессий, управление жизненным циклом и экспорт карты замен IdempotencyVault.
9. KR-ARC-3: Стратегии обезличивания в конвейере (Synonymic, Masking, HMAC).
10. LR-PERF-2: Потоковый интерфейс пакетной обработки для больших массивов (BatchProcessor).
11. LR-REL-4: Политика обработки невалидных контрольных сумм InvalidChecksumPolicy.
"""

from __future__ import annotations

import json
import os
import tempfile
import pytest

from lab2.src.faker_providers import (
    InvalidChecksumPolicy,
    set_invalid_checksum_policy,
    anonymize_inn,
    anonymize_bank_card,
    anonymize_ogrnip,
    anonymize_email_realistic,
    anonymize_org,
    anonymize_location,
    _ORG_NAMES_RU,
    _LOCATIONS_RU,
)
from lab2.src.registry import ProviderRegistry, FallbackMaskingStrategy
from lab2.src.anonymizer import anonymize_text
from lab2.src.quality_evaluator import DeidentificationAuditor
from lab2.src.batch_processor import process_jsonl_dataset, anonymize_stream
from coursework.src.idempotency_manager import IdempotencyVault
from coursework.src.predicate_agreement import PredicateAgreementRepair
from coursework.src.core_pipeline import CourseworkPipeline
from coursework.src.strategies import (
    SynonymicAnonymizationStrategy,
    CharacterMaskingStrategy,
    HmacAnonymizationStrategy,
)
from coursework.evaluation.bert_score_evaluator import (
    compute_corpus_idf,
    compute_bertscore_analytical,
    evaluate_bertscore_comparison,
)


# ============================================================================
# 1. ТЕСТЫ LR-REL-4: Политика невалидных контрольных сумм
# ============================================================================
class TestLRREL4InvalidChecksumPolicy:
    """Тестирование обработки дефектных контрольных сумм (FORCE_VALID, PRESERVE_DEFECT, PASS_THROUGH)."""

    def test_inn_force_valid_default(self):
        # Битый ИНН-10 (последняя цифра неверна)
        bad_inn = "7706123450"
        anon = anonymize_inn(bad_inn, policy=InvalidChecksumPolicy.FORCE_VALID)
        clean = anon.replace("-", "").replace(" ", "")
        from lab1.src.rules.inn_validator import validate_inn
        assert validate_inn(clean), "В режиме FORCE_VALID должен генерироваться валидный ИНН"

    def test_inn_pass_through(self):
        bad_inn = "7706123450"
        anon = anonymize_inn(bad_inn, policy=InvalidChecksumPolicy.PASS_THROUGH)
        assert anon == bad_inn, "В режиме PASS_THROUGH битый ИНН должен остаться без изменений"

    def test_inn_preserve_defect(self):
        bad_inn = "7706123450"
        anon = anonymize_inn(bad_inn, policy=InvalidChecksumPolicy.PRESERVE_DEFECT)
        clean = anon.replace("-", "").replace(" ", "")
        from lab1.src.rules.inn_validator import validate_inn
        assert not validate_inn(clean), "В режиме PRESERVE_DEFECT выходной ИНН должен оставаться с дефектом"


# ============================================================================
# 2. ТЕСТЫ LR-ARC-1: Реестр Strategy и Fallback-маскирование неизвестных тегов
# ============================================================================
class TestLRARC1ProviderRegistryAndFallback:
    """Тестирование безопасного маскирования неизвестных тегов для исключения утечек ПДн."""

    def test_unknown_tag_is_masked_safely(self):
        text = "Секретный код доступа: SECRET_TOKEN_12345."
        # Передаем неизвестный системе тег "CUSTOM_SECRET"
        spans = [("CUSTOM_SECRET", 23, 41)]
        anon = anonymize_text(text, spans)
        # Утечки быть не должно: токен должен быть замаскирован
        assert "SECRET_TOKEN_12345" not in anon
        assert "*" * 18 in anon, "Неизвестный тег должен быть замаскирован звездочками соответствующей длины"

    def test_registry_custom_tag_registration(self):
        @ProviderRegistry.register("CUSTOM_PASSPORT", strategy_name="synonymic")
        def custom_handler(val: str, *args, **kwargs) -> str:
            return "CUSTOM_ANON"

        text = "Тест кастомного обработчика: ABC-999."
        spans = [("CUSTOM_PASSPORT", 29, 36)]
        anon = anonymize_text(text, spans)
        assert "CUSTOM_ANON" in anon


# ============================================================================
# 3. ТЕСТЫ LR-COMP-3: Реалистичные email и пулы организаций
# ============================================================================
class TestLRCOMP3RealisticEmailAndPools:
    """Тестирование генерации осмысленных email и расширенных словарей."""

    def test_email_with_person_name_transliteration(self):
        orig_email = "ivan.ivanov@mail.ru"
        anon_email = anonymize_email_realistic(orig_email, current_person_nom="Артём Смирнов")
        assert "@mail.ru" in anon_email
        assert "artem.smirnov" in anon_email

    def test_email_fallback_plausible_name(self):
        orig_email = "test@yandex.ru"
        anon = anonymize_email_realistic(orig_email)
        assert "@yandex.ru" in anon
        assert "." in anon.split("@")[0]

    def test_expanded_organizations_and_locations_pools(self):
        assert len(_ORG_NAMES_RU) >= 50, "Пул российских организаций должен содержать >= 50 единиц"
        assert len(_LOCATIONS_RU) >= 50, "Пул российских локаций должен содержать >= 50 единиц"


# ============================================================================
# 4. ТЕСТЫ KR-COMP-1: Масштабирование словарей и контроль коллизий
# ============================================================================
class TestKRCOMP1DictionaryScalingAndCollisionControl:
    """Тестирование биективного маппинга без дублирования псевдонимов разным людям."""

    def test_two_different_persons_get_different_pseudonyms(self):
        vault = IdempotencyVault()
        # Два разных человека в документе
        rep1 = vault.get_or_create_name_replacement("Иванов", "SURNAME")
        rep2 = vault.get_or_create_name_replacement("Петров", "SURNAME")
        assert rep1 != rep2, "Два разных физических лица не должны получать одинаковые фамилии"


# ============================================================================
# 5. ТЕСТЫ KR-ARC-2: Управление сессиями и экспорт карты замен
# ============================================================================
class TestKRARC2SessionPersistenceAndExport:
    """Тестирование экспорта translation map и управления сессиями."""

    def test_export_translation_map_json(self):
        vault = IdempotencyVault(session_id="session_test_001")
        vault.get_or_create_name_replacement("Иванов", "SURNAME")
        vault.get_or_create_structured_replacement("770698959438", "INN")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            vault.export_translation_map(tmp_path)
            assert os.path.exists(tmp_path)
            with open(tmp_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data["session_id"] == "session_test_001"
            assert len(data["names"]) >= 1
            assert len(data["structured"]) >= 1
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


# ============================================================================
# 6. ТЕСТЫ KR-ARC-3: Стратегии обезличивания в CourseworkPipeline
# ============================================================================
class TestKRARC3PipelineStrategies:
    """Тестирование переключения стратегий (synonymic, masking, hmac)."""

    def test_pipeline_synonymic_strategy(self):
        pipe = CourseworkPipeline(detector_type="logreg")
        text = "Иванов заключил договор."
        res, spans = pipe.process(text, mode="synonymic")
        assert isinstance(res, str)
        assert len(res) > 0

    def test_pipeline_masking_strategy(self):
        pipe = CourseworkPipeline(detector_type="logreg")
        text = "Иванов заключил договор."
        res, spans = pipe.process(text, mode="masking")
        assert "*" in res or len(spans) == 0

    def test_pipeline_hmac_strategy(self):
        pipe = CourseworkPipeline(detector_type="logreg")
        text = "Иванов заключил договор."
        res, spans = pipe.process(text, mode="hmac")
        assert isinstance(res, str)


# ============================================================================
# 7. ТЕСТЫ KR-COMP-3: Предикативное согласование зависимых глаголов
# ============================================================================
class TestKRCOMP3PredicateAgreementRepair:
    """Тестирование гармонизации рода сказуемого и подлежащего."""

    def test_repair_verb_gender_agreement(self):
        # Если исходное предложение имело глагол женского рода, а субъект стал мужчиной
        text = "Артём подала заявление на субсидию."
        # Артём: [0, 5], target_gender = 'masc'
        repaired = PredicateAgreementRepair.repair_sentence_clause(
            text=text,
            subject_start=0,
            subject_end=5,
            target_gender="masc",
        )
        assert "подал" in repaired
        assert "подала" not in repaired


# ============================================================================
# 8. ТЕСТЫ LR-COMP-4: Модуль автоматизированного аудита (DeidentificationAuditor)
# ============================================================================
class TestLRCOMP4DeidentificationAuditor:
    """Тестирование расчета метрик FIR, CVR, ZLR, GAR."""

    def test_auditor_dataset_evaluation(self):
        samples = [
            {
                "text": "Клиент Иванов Иван. ИНН 770698959438.",
                "entities": [("PER", 7, 18), ("INN", 24, 36)],
            },
            {
                "text": "Карта 4276 3800 1234 5678 выдана Смирнову.",
                "entities": [("BANK_CARD", 6, 25), ("PER", 33, 41)],
            },
        ]
        auditor = DeidentificationAuditor()
        summary = auditor.evaluate_dataset(samples)

        assert summary["total_documents"] == 2
        assert summary["total_entities"] == 4
        assert summary["zero_leakage_rate"] == 1.0
        assert summary["format_integrity_rate"] == 1.0
        assert summary["checksum_validity_rate"] == 1.0
        assert summary["status"] == "PASSED (100% compliance)"


# ============================================================================
# 9. ТЕСТЫ LR-PERF-2: Потоковая пакетная обработка больших массивов
# ============================================================================
class TestLRPERF2BatchProcessor:
    """Тестирование потокового генератора и обработки .jsonl файлов."""

    def test_stream_anonymizer_batches(self):
        items = [
            {"id": i, "text": f"Клиент {i} Иванов подал заявление.", "entities": [("PER", len(f"Клиент {i} "), len(f"Клиент {i} Иванов"))]}
            for i in range(25)
        ]
        results = list(anonymize_stream(items, max_workers=2, chunk_size=10))
        assert len(results) == 25
        for r in results:
            assert "anonymized_text" in r
            assert "Иванов" not in r["anonymized_text"]

    def test_process_jsonl_dataset_io(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f_in:
            for i in range(15):
                f_in.write(json.dumps({"id": i, "text": "Иванов передал документы.", "spans": [("PER", 0, 6)]}, ensure_ascii=False) + "\n")
            in_path = f_in.name

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f_out:
            out_path = f_out.name

        try:
            count = process_jsonl_dataset(in_path, out_path, max_workers=2, chunk_size=5)
            assert count == 15
            with open(out_path, "r", encoding="utf-8") as res_f:
                lines = [json.loads(line) for line in res_f]
            assert len(lines) == 15
            assert all("anonymized_text" in item for item in lines)
        finally:
            if os.path.exists(in_path):
                os.remove(in_path)
            if os.path.exists(out_path):
                os.remove(out_path)


# ============================================================================
# 10. ТЕСТЫ KR-COMP-2: Каноническое IDF-взвешивание в BERTScore
# ============================================================================
class TestKRCOMP2CanonicalIDFWeighting:
    """Тестирование расчета IDF и взвешенного BERTScore."""

    def test_compute_corpus_idf_stops_vs_content(self):
        corpus = [
            "договор был подписан в городе москва",
            "акт был составлен в городе самара",
            "соглашение было заключено в городе казань",
        ]
        idf = compute_corpus_idf(corpus)
        # Стоп-слово 'в' и 'городе' встречаются во всех документах -> их IDF должен быть ниже редких слов 'самара', 'казань'
        assert idf["в"] < idf["самара"]
        assert idf["городе"] < idf["москва"]

    def test_bertscore_comparison_with_idf(self):
        originals = ["Иванов Иван подписал кредитный договор в Москве."]
        synonymized = ["Смирнов Артём подписал кредитный договор в Самаре."]
        masked = ["****** **** подписал кредитный договор в ******. "]

        res = evaluate_bertscore_comparison(originals, synonymized, masked, use_idf=True)
        assert res["delta_f1"] > 0.0, "Синонимизация должна превосходить маскирование"
        assert res["synonymic"]["f1"] > res["masked"]["f1"]


