"""
Скрипт экспорта дообученной модели ai-forever/ru-en-RoSBERTa в ONNX Runtime
и выполнения динамического квантования (INT8) для ускорения CPU-инференса в 3-4 раза.
"""

from __future__ import annotations

import argparse
import os
import sys
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from data_pipeline.schema import ID2LABEL, LABEL2ID


def export_to_onnx(
    model_path: str,
    onnx_output_path: str,
    quantize: bool = True,
    quantized_output_path: str | None = None,
) -> str:
    """
    Экспортировать модель Token Classification в ONNX и опционально квантовать в INT8.
    """
    print(f"Экспорт модели из {model_path} в ONNX...")
    os.makedirs(os.path.dirname(onnx_output_path) or ".", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval()

    dummy_text = "Пример текста для инициализации графа вычислений"
    inputs = tokenizer(dummy_text, return_tensors="pt")

    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"},
    }

    try:
        torch.onnx.export(
            model,
            (inputs["input_ids"], inputs["attention_mask"]),
            onnx_output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=14,
            do_constant_folding=True,
            dynamo=False,
        )
    except TypeError:
        torch.onnx.export(
            model,
            (inputs["input_ids"], inputs["attention_mask"]),
            onnx_output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=14,
            do_constant_folding=True,
        )
    print(f"Базовая ONNX модель сохранена: {onnx_output_path}")

    final_path = onnx_output_path

    if quantize:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            if not quantized_output_path:
                quantized_output_path = onnx_output_path.replace(".onnx", "_int8.onnx")

            print(f"Выполнение динамического квантования (INT8) в {quantized_output_path}...")
            local_tmp = os.path.join(os.path.dirname(quantized_output_path) or ".", ".tmp_quant")
            os.makedirs(local_tmp, exist_ok=True)
            try:
                quantize_dynamic(
                    model_input=onnx_output_path,
                    model_output=quantized_output_path,
                    weight_type=QuantType.QInt8,
                    extra_options={"temp_folder": local_tmp},
                )
            finally:
                if os.path.exists(local_tmp):
                    import shutil
                    shutil.rmtree(local_tmp, ignore_errors=True)
            print(f"Квантованная модель успешно сохранена: {quantized_output_path}")
            final_path = quantized_output_path
        except ImportError:
            print("[WARN] onnxruntime не установлен. Квантование пропущено, используется базовая ONNX модель.")
        except Exception as e:
            print(f"[WARN] Ошибка квантования: {e}")

    return final_path


class FastTokenClassifierInference:
    """
    Унифицированный движок инференса:
    - Использует ONNX Runtime (если доступен)
    - Или PyTorch модель с автоматическим фоллбэком
    """

    def __init__(self, model_path_or_onnx: str, tokenizer_path: str | None = None):
        self.is_onnx = model_path_or_onnx.endswith(".onnx") and os.path.exists(model_path_or_onnx)
        tok_path = tokenizer_path or (os.path.dirname(model_path_or_onnx) if self.is_onnx else model_path_or_onnx)
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path, add_prefix_space=True)

        if self.is_onnx:
            try:
                import onnxruntime as ort
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = 4
                self.session = ort.InferenceSession(model_path_or_onnx, sess_options)
            except Exception:
                self.is_onnx = False

        if not self.is_onnx:
            self.model = AutoModelForTokenClassification.from_pretrained(tok_path)
            self.model.eval()

    def predict(self, tokens: list[str]) -> list[str]:
        """Предсказать список меток для последовательности слов."""
        inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="np" if self.is_onnx else "pt",
            truncation=True,
            max_length=512,
        )

        if self.is_onnx:
            ort_inputs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            }
            ort_outs = self.session.run(None, ort_inputs)
            logits = ort_outs[0]
            preds = logits.argmax(axis=-1)[0]
        else:
            with torch.no_grad():
                outputs = self.model(**inputs)
                preds = outputs.logits.argmax(dim=-1)[0].cpu().numpy()

        word_ids = self.tokenizer(tokens, is_split_into_words=True).word_ids()
        word_labels = ["O"] * len(tokens)
        seen_words = set()

        for idx, word_id in enumerate(word_ids):
            if word_id is not None and word_id not in seen_words:
                pred_label_id = int(preds[idx])
                word_labels[word_id] = ID2LABEL.get(pred_label_id, "O")
                seen_words.add(word_id)

        return word_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Экспорт модели в ONNX")
    parser.add_argument("--model", "--model_path", dest="model", type=str, default="models/rosberta-ner-finetuned", help="Путь к PyTorch чекпоинту")
    parser.add_argument("--output", "--output_dir", dest="output", type=str, default="models/rosberta-ner-finetuned/model.onnx", help="Путь ONNX")
    parser.add_argument("--quantize", action="store_true", help="Выполнить динамическое квантование INT8")
    args = parser.parse_args()

    # Если передан каталог, сохраняем внутри него model.onnx
    out_path = args.output
    if os.path.isdir(out_path) or not out_path.endswith(".onnx"):
        out_path = os.path.join(out_path, "model.onnx")

    if os.path.exists(os.path.join(args.model, "config.json")):
        export_to_onnx(args.model, out_path, quantize=args.quantize)
    else:
        print(f"Чекпоинт {args.model} не найден. Сначала выполните дообучение.")
