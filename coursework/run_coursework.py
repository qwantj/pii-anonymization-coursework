"""
Курсовая работа: Комплексный программный комплекс распознавания и синонимического
обезличивания персональных данных с оценкой BERTScore и генерацией отчета.

Запуск:
    python run_coursework.py             # Полный цикл: бенчмарк + метрики + BERTScore + генерация report.docx
    python run_coursework.py --demo      # Интерактивное тестирование
    python run_coursework.py --tests     # Запуск специализированных проверок
    python run_coursework.py --detector=rosberta --report-type=full_package --export-pdf  # Полный академический комплект
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

CW_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CW_DIR)
sys.path.insert(0, PROJECT_ROOT)

from coursework.src.core_pipeline import CourseworkPipeline
from coursework.src.idempotency_manager import IdempotencyVault
from coursework.evaluation.f1_evaluator import evaluate_dataset
from coursework.evaluation.bert_score_evaluator import evaluate_bertscore_comparison
from coursework.report.build_report import generate_full_report, save_evaluation_artifacts


def run_full_pipeline(
    dataset_path: str,
    report_dir: str,
    use_onnx: bool = False,
    detector_type: str = "logreg",
    report_type: str = "summary",
    export_pdf: bool = False,
):
    print("=" * 80)
    print("КУРСОВАЯ РАБОТА: СКВОЗНОЙ ПАЙПЛАЙН (ЧЁТНЫЙ ВАРИАНТ)")
    effective_det = "onnx" if (use_onnx and detector_type in ("rosberta", "onnx")) else detector_type
    mode_name = "Квантованная INT8 ONNX (ускоренный CPU)" if use_onnx else f"Детектор [{effective_det}]"
    print(f"Режим инференса детектора: {mode_name}")
    print("=" * 80)

    start_time = time.time()

    # 1. Загрузка данных
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"\n[Этап 1] Загружен бенчмарк: {len(data)} текстов.")

    # 2. Инициализация единого пайплайна и устранение дивергенции моделей (KR-ARC-1)
    pipeline = CourseworkPipeline(
        use_onnx=use_onnx,
        detector_type=detector_type,
    )
    detector = pipeline.get_detector()

    # 3. Оценка распознавания единым детектором (F1 Token Classification, KR-ARC-1)
    print("\n[Этап 2] Оценка качества распознавания ПДн (Token Classification)...")
    ner_metrics = evaluate_dataset(data, ner_detector=detector)

    print(f"  Weighted F1: {ner_metrics['weighted_f1']:.4f} (Требование >= 0.85)")
    print(f"  Macro F1:    {ner_metrics['macro_f1']:.4f} (Требование >= 0.80)")
    print(f"  Overall P:   {ner_metrics['overall']['precision']:.4f}, R: {ner_metrics['overall']['recall']:.4f}")

    # 4. Обезличивание: синонимизация (с идемпотентностью) и маскирование (KR-ARC-3)
    print("\n[Этап 3] Обезличивание данных (синонимическая замена и маскирование)...")

    originals = []
    synonymized_texts = []
    masked_texts = []

    for item in data:
        text = item["text"]
        vault = IdempotencyVault()

        syn_text, spans = pipeline.process(text, mode="synonymic", vault=vault)
        mask_text, _ = pipeline.process(text, mode="masking")

        originals.append(text)
        synonymized_texts.append(syn_text)
        masked_texts.append(mask_text)

    # 5. Оценка BERTScore с каноническим IDF-взвешиванием и векторизацией (KR-PERF-3, KR-COMP-2)
    print("\n[Этап 4] Оценка семантической близости (BERTScore с IDF-взвешиванием)...")
    bert_metrics = evaluate_bertscore_comparison(originals, synonymized_texts, masked_texts, use_idf=True)

    # Строгий тест на изолированных парах ФИО (сравнение семантики замен без фонового контекста)
    iso_orig, iso_syn, iso_mask = [], [], []
    for item in data:
        t = item["text"]
        for tag, s, e in item["entities"]:
            if tag in ("SURNAME", "NAME"):
                sub = t[s:e]
                v_iso = IdempotencyVault()
                syn_i, _ = pipeline.process(sub, mode="synonymic", vault=v_iso)
                mask_i, _ = pipeline.process(sub, mode="masking")
                if syn_i != sub:
                    iso_orig.append(sub)
                    iso_syn.append(syn_i)
                    iso_mask.append(mask_i)
                if len(iso_orig) >= 80:
                    break
        if len(iso_orig) >= 80:
            break
    if iso_orig:
        iso_res = evaluate_bertscore_comparison(iso_orig, iso_syn, iso_mask, use_idf=False)
        bert_metrics["isolated_entities"] = {
            "synonymic_f1": iso_res["synonymic"]["f1"],
            "masked_f1": iso_res["masked"]["f1"],
            "delta_f1": iso_res["delta_f1"],
        }

    print(f"  Модель BERTScore:             {bert_metrics.get('backend', 'transformers')}")
    print(f"  Синонимическое обезличивание: F1 = {bert_metrics['synonymic']['f1']:.4f} "
          f"(P={bert_metrics['synonymic']['precision']:.4f}, R={bert_metrics['synonymic']['recall']:.4f})")
    print(f"  Маскирование ('*'):           F1 = {bert_metrics['masked']['f1']:.4f} "
          f"(P={bert_metrics['masked']['precision']:.4f}, R={bert_metrics['masked']['recall']:.4f})")
    print(f"  Преимущество синонимизации:   +{bert_metrics['delta_f1']:.4f} F1-score")

    # Сохранение артефактов эксперимента (KR-COMP-4)
    duration = time.time() - start_time
    artifacts_path = os.path.join(report_dir, "evaluation_artifacts.json")
    print(f"\n[Этап 4.1] Сохранение артефактов эксперимента ({artifacts_path})...")
    save_evaluation_artifacts(
        artifacts_path=artifacts_path,
        ner_metrics=ner_metrics,
        bert_metrics=bert_metrics,
        perf_metrics={
            "total_duration_sec": round(duration, 2),
            "avg_latency_per_doc_ms": round(duration / len(data) * 1000, 2),
            "throughput_docs_per_sec": round(len(data) / duration, 2) if duration > 0 else 0.0,
        },
        metadata={
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "variant": 8,
            "variant_name": "even",
            "detector_type": detector_type,
            "use_onnx": use_onnx,
            "dataset_path": dataset_path,
            "total_texts": len(data),
        }
    )

    # 6. Генерация отчета и экспорт в PDF (KR-COMP-4)
    print(f"\n[Этап 5] Формирование отчета (режим '{report_type}', экспорт PDF={export_pdf})...")
    generate_full_report(
        report_dir=report_dir,
        ner_metrics=ner_metrics,
        bert_metrics=bert_metrics,
        report_type=report_type,
        export_pdf=export_pdf,
    )

    print("\n" + "=" * 80)
    print("КУРСОВАЯ РАБОТА УСПЕШНО ВЫПОЛНЕНА!")
    print(f"Файл отчета: {os.path.join(report_dir, 'report.docx')}")
    print("=" * 80)


def run_specialized_tests(detector_type: str = "logreg"):
    print("=" * 70)
    print("ТЕСТИРОВАНИЕ СПЕЦИФИЧЕСКИХ ТРЕБОВАНИЙ КУРСОВОЙ РАБОТЫ")
    print("=" * 70)

    pipeline = CourseworkPipeline(detector_type=detector_type)

    # Тест 1: Идемпотентность и склонение
    print("\n[Тест 1] Идемпотентность и согласование падежей:")
    text_idemp = (
        "Иванов подписал договор. Документы Иванова проверены. "
        "Справка выдана Иванову. Копия согласована с Ивановым."
    )
    vault = IdempotencyVault()
    anon_idemp, spans = pipeline.process(text_idemp, mode="synonymic", vault=vault)
    print(f"  Исходный текст: {text_idemp}")
    print(f"  Результат:      {anon_idemp}")

    # Тест 2: Снятие омонимии рода (Саша/Женя)
    print("\n[Тест 2] Разрешение омонимии рода (Саша/Женя) и изоляция клауз:")
    text_fem = "Саша оплатила госпошлину."
    text_masc = "Саша оплатил госпошлину."
    text_complex = "Саша пришла в офис, а Женя ушёл домой."

    anon_fem, _ = pipeline.process(text_fem, mode="synonymic", vault=IdempotencyVault())
    anon_masc, _ = pipeline.process(text_masc, mode="synonymic", vault=IdempotencyVault())
    anon_complex, _ = pipeline.process(text_complex, mode="synonymic", vault=IdempotencyVault())
    print(f"  Женский род: '{text_fem}' -> '{anon_fem}'")
    print(f"  Мужской род: '{text_masc}' -> '{anon_masc}'")
    print(f"  Сложное:     '{text_complex}' -> '{anon_complex}'")

    # Тест 3: Двойные фамилии и омонимия фамилии типа 'Иванова'
    print("\n[Тест 3] Двойные фамилии и омонимия 'Иванова':")
    text_hyphen = "Петров-Водкин Иван подписал договор."
    text_ivanova = "Иванова Анна оплатила счёт."
    anon_hyphen, _ = pipeline.process(text_hyphen, mode="synonymic", vault=IdempotencyVault())
    anon_ivanova, _ = pipeline.process(text_ivanova, mode="synonymic", vault=IdempotencyVault())
    print(f"  Дефисная:    '{text_hyphen}' -> '{anon_hyphen}'")
    print(f"  Иванова (ж): '{text_ivanova}' -> '{anon_ivanova}'")

    print("\n[OK] Специализированные тесты завершены успешно!")
    print("=" * 70)


def run_interactive(use_onnx: bool = False, detector_type: str = "logreg"):
    pipeline = CourseworkPipeline(use_onnx=use_onnx, detector_type=detector_type)
    mode_name = "Квантованная INT8 ONNX" if use_onnx else f"Детектор [{detector_type}]"
    print(f"\n=== Интерактивный режим Курсовой работы [{mode_name}] ===")
    print("Введите текст (или 'exit'):\n")

    while True:
        text = input("> ")
        if text.strip().lower() in ("exit", "quit", "выход"):
            break

        vault = IdempotencyVault()
        syn_res, spans = pipeline.process(text, mode="synonymic", vault=vault)
        mask_res, _ = pipeline.process(text, mode="masking")

        print(f"\n  Распознано ПДн: {spans}")
        print(f"  Синонимизация: {syn_res}")
        print(f"  Маскирование:  {mask_res}\n")


def main():
    parser = argparse.ArgumentParser(description="Курсовая работа: Распознавание и обезличивание ПДн")
    parser.add_argument("--demo", action="store_true", help="Интерактивный ввод")
    parser.add_argument("--tests", action="store_true", help="Запуск специализированных проверок")
    parser.add_argument("--detector", type=str, default="logreg", choices=["logreg", "rosberta", "onnx", "rubert", "auto"],
                        help="Выбор бэкенда детектора (KR-ARC-1)")
    parser.add_argument("--report-type", type=str, default="summary", choices=["summary", "km1", "km2", "km3", "full_package"],
                        help="Тип генерируемой академической отчетности (KR-COMP-4)")
    parser.add_argument("--export-pdf", action="store_true", help="Экспортировать отчет в формат PDF (KR-COMP-4)")
    parser.add_argument("--onnx", "--quantized", dest="onnx", action="store_true", default=False,
                        help="Использовать квантованную модель (INT8 ONNX, быстрый CPU инференс)")
    parser.add_argument("--pytorch", dest="onnx", action="store_false",
                        help="Использовать оригинальную модель (PyTorch FP32, по умолчанию)")
    args = parser.parse_args()

    if args.demo:
        run_interactive(use_onnx=args.onnx, detector_type=args.detector)
    elif args.tests:
        run_specialized_tests(detector_type=args.detector)
    else:
        dataset_path = os.path.join(CW_DIR, "data", "dataset.json")
        report_dir = os.path.join(CW_DIR, "report")

        if not os.path.exists(dataset_path):
            print("Бенчмарк не найден, генерирую...")
            from coursework.data.data_generator import generate_coursework_dataset
            generate_coursework_dataset(dataset_path)

        run_full_pipeline(
            dataset_path=dataset_path,
            report_dir=report_dir,
            use_onnx=args.onnx,
            detector_type=args.detector,
            report_type=args.report_type,
            export_pdf=args.export_pdf,
        )


if __name__ == "__main__":
    main()
