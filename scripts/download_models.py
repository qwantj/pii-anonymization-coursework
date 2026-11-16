"""
Скрипт загрузки квантованной нейросетевой модели RoSBERTa INT8 ONNX (~414 МБ)
из GitHub Releases репозитория qwantj/pii-anonymization-coursework.

Лёгкий ML-детектор (logreg, coursework/models/ner_token_classifier.joblib)
уже включён в репозиторий и работает сразу после git clone.
"""

from __future__ import annotations

import os
import sys
import urllib.request

RELEASE_BASE = "https://github.com/qwantj/pii-anonymization-coursework/releases/download/v1.0.0-models"
FILES = [
    "model_int8.onnx",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX_DIR = os.path.join(PROJECT_ROOT, "models", "rosberta_onnx")
HF_DIR = os.path.join(PROJECT_ROOT, "models", "rosberta-ner-finetuned")


def download_file(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        print(f"  [SKIP] Уже существует: {dest}")
        return
    print(f"  [GET]  {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def main() -> int:
    print("Загрузка квантованной модели RoSBERTa INT8 ONNX из GitHub Releases...")
    try:
        for fname in FILES:
            url = f"{RELEASE_BASE}/{fname}"
            if fname == "model_int8.onnx":
                download_file(url, os.path.join(ONNX_DIR, fname))
            else:
                download_file(url, os.path.join(HF_DIR, fname))
                download_file(url, os.path.join(ONNX_DIR, fname))
        print("\n[OK] Модель RoSBERTa INT8 ONNX готова.")
        print("Запуск с ней: python coursework/run_coursework.py --detector onnx --onnx")
        return 0
    except Exception as exc:
        print(f"\n[ERR] Не удалось скачать веса ({exc}).", file=sys.stderr)
        print("Проверьте доступ к GitHub Releases или используйте детектор по умолчанию (--detector logreg).")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
