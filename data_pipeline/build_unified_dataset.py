"""
Сборка единого NER-датасета:
1. Объединение FactRuEval, Collection3, BSNLP, NEREL, WikiANN и синтетики 152-ФЗ.
2. Дедупликация по SHA-256 хешу нормализованного текста.
3. Балансировка all-O предложений (не более 20%).
4. Стратифицированный сплит 85% Train / 7.5% Val / 7.5% Test (с учетом ADDR).
5. Строгая IOB2-валидация каждого примера.
6. Сохранение DatasetDict на диск и экспорт в архив для Google Colab.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import tarfile
from collections import Counter
from typing import Any

from datasets import Dataset, DatasetDict

from data_pipeline.open_datasets import (
    load_factrueval_corpus,
    load_collection3_corpus,
    load_bsnlp_corpus,
    load_nerel_corpus,
    load_wikiann_en_corpus,
)
from data_pipeline.schema import (
    ID2LABEL,
    LABEL2ID,
    validate_iob2_sequence,
    repair_iob2_sequence,
)
from data_pipeline.synthetic_generator import generate_synthetic_dataset

_rng = random.Random(42)


def _compute_sentence_hash(tokens: list[str]) -> str:
    """Вычислить SHA-256 хеш нормализованной последовательности токенов."""
    normalized = " ".join(t.lower().strip() for t in tokens if t.strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _has_tag(ner_tags: list[int], tag_name: str) -> bool:
    """Проверить, содержит ли последовательность меток указанный класс."""
    b_id = LABEL2ID.get(f"B-{tag_name}")
    i_id = LABEL2ID.get(f"I-{tag_name}")
    return any(t in (b_id, i_id) for t in ner_tags)


def build_and_validate_unified_dataset(
    factrueval_dir: str = "lab1/data/factRuEval/factRuEval-2016-master",
    synthetic_count: int = 20000,
    wikiann_count: int = 12000,
    output_dir: str = "data/unified_ner_dataset",
    archive_path: str = "data/unified_ner_dataset.tar.gz",
    max_all_o_ratio: float = 0.20,
) -> DatasetDict:
    """Собрать, дедуплицировать, разделить и сохранить единый датасет."""
    print("=" * 70)
    print("СБОРКА ЕДИНОГО УНИФИЦИРОВАННОГО NER-ДАТАСЕТА (Strict IOB2)")
    print("=" * 70)

    raw_pool: list[dict] = []

    # 1. Загрузка FactRuEval-2016
    print("\n[1/6] Парсинг FactRuEval-2016...")
    try:
        fre_samples = load_factrueval_corpus(factrueval_dir)
        print(f"  Успешно загружено {len(fre_samples)} предложений из FactRuEval.")
        raw_pool.extend(fre_samples)
    except Exception as e:
        print(f"  [WARN] Ошибка загрузки FactRuEval: {e}")

    # 2. Загрузка Collection3
    print("\n[2/6] Загрузка Collection3 (Persons-1000)...")
    try:
        col3_samples = load_collection3_corpus()
        print(f"  Успешно загружено {len(col3_samples)} предложений из Collection3.")
        raw_pool.extend(col3_samples)
    except Exception as e:
        print(f"  [WARN] Ошибка загрузки Collection3: {e}")

    # 3. Загрузка BSNLP
    print("\n[3/6] Загрузка BSNLP (ru/en)...")
    try:
        bsnlp_samples = load_bsnlp_corpus()
        print(f"  Успешно загружено {len(bsnlp_samples)} предложений из BSNLP.")
        raw_pool.extend(bsnlp_samples)
    except Exception as e:
        print(f"  [WARN] Ошибка загрузки BSNLP: {e}")

    # 4. Загрузка NEREL
    print("\n[4/6] Загрузка NEREL (с выделением ADDR)...")
    try:
        nerel_samples = load_nerel_corpus()
        print(f"  Успешно загружено {len(nerel_samples)} предложений из NEREL.")
        raw_pool.extend(nerel_samples)
    except Exception as e:
        print(f"  [WARN] Ошибка загрузки NEREL: {e}")

    # 5. Загрузка WikiANN English
    print(f"\n[5/6] Загрузка WikiANN en ({wikiann_count} примеров)...")
    try:
        wiki_samples = load_wikiann_en_corpus(target_count=wikiann_count)
        print(f"  Успешно загружено {len(wiki_samples)} англоязычных предложений.")
        raw_pool.extend(wiki_samples)
    except Exception as e:
        print(f"  [WARN] Ошибка загрузки WikiANN: {e}")

    # 6. Генерация доменной синтетики по 152-ФЗ
    print(f"\n[6/6] Генерация {synthetic_count} синтетических примеров по 152-ФЗ...")
    synth_samples = generate_synthetic_dataset(target_count=synthetic_count, hard_negatives_count=2500)
    print(f"  Сгенерировано {len(synth_samples)} примеров синтетики.")
    raw_pool.extend(synth_samples)

    print(f"\nВсего собрано сырых примеров: {len(raw_pool)}")

    # -------------------------------------------------------------------------
    # Дедупликация и строгая IOB2-валидация
    # -------------------------------------------------------------------------
    print("\nДедупликация и строгая валидация разметки IOB2...")
    seen_hashes: set[str] = set()
    valid_with_entities: list[dict] = []
    valid_all_o: list[dict] = []

    for item in raw_pool:
        tokens = item["tokens"]
        tags = item["ner_tags"]

        if not tokens or len(tokens) != len(tags):
            continue

        repaired_tags = repair_iob2_sequence(tags)
        if not validate_iob2_sequence(tokens, repaired_tags):
            continue

        h = _compute_sentence_hash(tokens)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        clean_item = {
            "tokens": tokens,
            "ner_tags": repaired_tags,
            "source": item.get("source", "unknown"),
        }

        # Разделяем на предложения с сущностями и all-O
        if any(t != LABEL2ID["O"] for t in repaired_tags):
            valid_with_entities.append(clean_item)
        else:
            valid_all_o.append(clean_item)

    print(f"  Уникальных с сущностями: {len(valid_with_entities)}")
    print(f"  Уникальных all-O:        {len(valid_all_o)}")

    # Балансировка all-O: не более max_all_o_ratio от общего числа
    max_o_count = int(len(valid_with_entities) * max_all_o_ratio / (1 - max_all_o_ratio))
    _rng.shuffle(valid_all_o)
    selected_all_o = valid_all_o[:max_o_count]
    print(f"  Отобрано all-O (баланс {max_all_o_ratio * 100:.0f}%): {len(selected_all_o)}")

    unified_samples = valid_with_entities + selected_all_o
    _rng.shuffle(unified_samples)
    print(f"Итоговый размер единого датасета: {len(unified_samples)} примеров.")

    # -------------------------------------------------------------------------
    # Стратифицированное разбиение (85% train / 7.5% val / 7.5% test)
    # -------------------------------------------------------------------------
    print("\nСтратифицированное разбиение на train (85%), validation (7.5%), test (7.5%)...")

    # Страты:
    # 1. Примеры с редким тегом ADDR
    # 2. Примеры с сущностями без ADDR
    # 3. Примеры all-O
    stratum_addr = [s for s in unified_samples if _has_tag(s["ner_tags"], "ADDR")]
    stratum_other = [s for s in unified_samples if not _has_tag(s["ner_tags"], "ADDR") and any(t != 0 for t in s["ner_tags"])]
    stratum_o = [s for s in unified_samples if all(t == 0 for t in s["ner_tags"])]

    print(f"  Страта ADDR:      {len(stratum_addr)}")
    print(f"  Страта PER/ORG/LOC: {len(stratum_other)}")
    print(f"  Страта All-O:     {len(stratum_o)}")

    def split_stratum(items: list[dict], train_r=0.85, val_r=0.075):
        _rng.shuffle(items)
        n = len(items)
        n_train = int(n * train_r)
        n_val = int(n * val_r)
        return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]

    train_addr, val_addr, test_addr = split_stratum(stratum_addr)
    train_other, val_other, test_other = split_stratum(stratum_other)
    train_o, val_o, test_o = split_stratum(stratum_o)

    train_samples = train_addr + train_other + train_o
    val_samples = val_addr + val_other + val_o
    test_samples = test_addr + test_other + test_o

    _rng.shuffle(train_samples)
    _rng.shuffle(val_samples)
    _rng.shuffle(test_samples)

    print(f"  Train:      {len(train_samples)} примеров ({len(train_samples)/len(unified_samples)*100:.1f}%)")
    print(f"  Validation: {len(val_samples)} примеров ({len(val_samples)/len(unified_samples)*100:.1f}%)")
    print(f"  Test:       {len(test_samples)} примеров ({len(test_samples)/len(unified_samples)*100:.1f}%)")

    # Контроль отсутствия утечки (Leakage Check)
    train_hashes = set(_compute_sentence_hash(s["tokens"]) for s in train_samples)
    val_hashes = set(_compute_sentence_hash(s["tokens"]) for s in val_samples)
    test_hashes = set(_compute_sentence_hash(s["tokens"]) for s in test_samples)

    assert not (train_hashes & val_hashes), "ОШИБКА: Обнаружена утечка данных между Train и Validation!"
    assert not (train_hashes & test_hashes), "ОШИБКА: Обнаружена утечка данных между Train и Test!"
    assert not (val_hashes & test_hashes), "ОШИБКА: Обнаружена утечка данных между Validation и Test!"
    print("  [OK] Контроль утечек пройден: 0% пересечения между сплитами.")

    # -------------------------------------------------------------------------
    # Сохранение в формате DatasetDict и экспорт
    # -------------------------------------------------------------------------
    dataset_dict = DatasetDict({
        "train": Dataset.from_list(train_samples),
        "validation": Dataset.from_list(val_samples),
        "test": Dataset.from_list(test_samples),
    })

    os.makedirs(output_dir, exist_ok=True)
    dataset_dict.save_to_disk(output_dir)
    print(f"\nДатасет успешно сохранен на диск: {output_dir}")

    # Создаем tar.gz архив для Colab
    os.makedirs(os.path.dirname(archive_path) or ".", exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(output_dir, arcname=os.path.basename(output_dir))
    print(f"Архив для Google Colab создан: {archive_path}")

    # Статистика распределения тегов
    tag_counts = Counter()
    for s in unified_samples:
        for t in s["ner_tags"]:
            tag_counts[ID2LABEL[t]] += 1

    print("\n--- Распределение меток в объединенном датасете ---")
    for lbl, count in tag_counts.most_common():
        print(f"  {lbl:8s}: {count:8d} ({count / sum(tag_counts.values()) * 100:.2f}%)")

    return dataset_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сборка единого NER-датасета")
    parser.add_argument("--quick-validate", action="store_true", help="Быстрая сборка мини-версии для тестов")
    args = parser.parse_args()

    if args.quick_validate:
        build_and_validate_unified_dataset(
            synthetic_count=1500,
            wikiann_count=1000,
            output_dir="data/unified_ner_dataset_mini",
            archive_path="data/unified_ner_dataset_mini.tar.gz",
        )
    else:
        build_and_validate_unified_dataset()
