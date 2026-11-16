"""
Модуль потоковой пакетной обработки больших массивов документов (LR-PERF-2).

Обеспечивает:
1. Масштабируемое обезличивание терабайтных массивов (.jsonl / .csv) без переполнения RAM.
2. Параллельную обработку через ThreadPoolExecutor / ProcessPoolExecutor.
3. Потоковый генератор с чанкингом памяти (chunk_size).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
from typing import Any, Callable, Dict, Generator, Iterable, Iterator, List, Optional

from lab2.src.anonymizer import anonymize_text

logger = logging.getLogger(__name__)


def anonymize_batch_worker(item: Dict[str, Any]) -> Dict[str, Any]:
    """Обработчик одного элемента пакета."""
    text = item.get("text", "")
    spans = item.get("spans", item.get("entities", []))
    anon_text = anonymize_text(text, spans)
    res = dict(item)
    res["anonymized_text"] = anon_text
    return res


def anonymize_stream(
    items: Iterable[Dict[str, Any]],
    max_workers: int = 4,
    chunk_size: int = 250,
) -> Generator[Dict[str, Any], None, None]:
    """
    Потоковый генератор обезличенных документов с фиксированным размером чанка в памяти.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        batch: List[Dict[str, Any]] = []
        for item in items:
            batch.append(item)
            if len(batch) >= chunk_size:
                results = executor.map(anonymize_batch_worker, batch)
                for res in results:
                    yield res
                batch.clear()

        if batch:
            results = executor.map(anonymize_batch_worker, batch)
            for res in results:
                yield res


def process_jsonl_dataset(
    input_path: str,
    output_path: str,
    max_workers: int = 4,
    chunk_size: int = 250,
) -> int:
    """
    Пакетная обработка JSONL файла с записью результата в output_path.

    Returns:
        Количество успешно обработанных документов.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    count = 0

    def line_generator() -> Generator[Dict[str, Any], None, None]:
        with open(input_path, "r", encoding="utf-8") as f_in:
            for line in f_in:
                stripped = line.strip()
                if stripped:
                    try:
                        yield json.loads(stripped)
                    except json.JSONDecodeError:
                        continue

    with open(output_path, "w", encoding="utf-8") as f_out:
        for res in anonymize_stream(line_generator(), max_workers=max_workers, chunk_size=chunk_size):
            f_out.write(json.dumps(res, ensure_ascii=False) + "\n")
            count += 1

    return count
