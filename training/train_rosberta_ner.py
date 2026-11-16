"""
Скрипт дообучения ai-forever/ru-en-RoSBERTa для задачи Token Classification (NER).
Включает:
- Токенизацию с add_prefix_space=True и выравнивание меток по сабтокенам
- Кастомный WeightedTrainer с весами классов (CrossEntropyLoss)
- Дифференцированный Learning Rate (2e-5 backbone / 1e-4 classifier head)
- Строгую IOB2-оценку метрик через seqeval (overall + per-class: PER, ORG, LOC, ADDR)
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from datasets import DatasetDict, load_from_disk
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from seqeval.scheme import IOB2

from data_pipeline.schema import (
    LABEL_LIST,
    LABEL2ID,
    ID2LABEL,
    CLASS_WEIGHTS_LIST,
)


def align_labels_with_tokens(labels: list[int], word_ids: list[int | None]) -> list[int]:
    """
    Выравнивание меток слов по Byte-Level BPE сабтокенам:
    - Спецтокенам (<s>, </s>, <pad>) назначается -100.
    - Первому сабтокену слова назначается оригинальная метка (B-X или I-X).
    - Последующим сабтокенам того же слова назначается метка продолжения I-X.
    """
    new_labels: list[int] = []
    current_word = None

    for word_id in word_ids:
        if word_id is None:
            new_labels.append(-100)
        elif word_id != current_word:
            current_word = word_id
            new_labels.append(labels[word_id])
        else:
            orig_label_id = labels[word_id]
            orig_label = ID2LABEL.get(orig_label_id, "O")
            if orig_label.startswith("B-"):
                sub_label = "I-" + orig_label[2:]
                new_labels.append(LABEL2ID.get(sub_label, orig_label_id))
            else:
                new_labels.append(orig_label_id)

    return new_labels


def tokenize_and_align_dataset(dataset_dict: DatasetDict, tokenizer, max_length: int = 512):
    """Применить токенизацию и выравнивание меток ко всем сплитам датасета."""
    def tokenize_function(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
        )
        all_labels = examples["ner_tags"]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            aligned = align_labels_with_tokens(labels, word_ids)
            new_labels.append(aligned)

        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs

    return dataset_dict.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset_dict["train"].column_names,
    )


class WeightedTrainer(Trainer):
    """
    Кастомный Trainer с взвешенной кросс-энтропией и дифференцированным LR:
    - Пониженный вес для 'O' (0.25)
    - Повышенный вес для редкого 'ADDR' (1.3)
    - 2e-5 для backbone RoSBERTa / 1e-4 для classifier
    """

    def __init__(self, class_weights: torch.Tensor | None = None, *args, **kwargs):
        # Автоматическая адаптация tokenizer -> processing_class для новейших версий transformers
        if "tokenizer" in kwargs and "processing_class" not in kwargs:
            try:
                import inspect
                sig = inspect.signature(Trainer.__init__)
                if "processing_class" in sig.parameters and "tokenizer" not in sig.parameters:
                    kwargs["processing_class"] = kwargs.pop("tokenizer")
            except Exception:
                pass
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        """Дифференцированный Learning Rate: backbone (1x) vs classifier head (5x)."""
        if self.optimizer is None:
            decay_params = []
            no_decay_params = []
            classifier_params = []
            no_decay = ["bias", "LayerNorm.weight"]

            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if "classifier" in name:
                    classifier_params.append(param)
                elif any(nd in name for nd in no_decay):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

            optimizer_grouped_parameters = [
                {
                    "params": decay_params,
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": no_decay_params,
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": classifier_params,
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate * 5.0,  # 1e-4 при базовом 2e-5
                },
            ]
            self.optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                eps=self.args.adam_epsilon,
            )
        return self.optimizer


def build_compute_metrics():
    """Сборка функции оценки seqeval в строгом режиме IOB2."""
    def compute_metrics(p):
        predictions, labels = p
        preds = np.argmax(predictions, axis=2)

        true_predictions = [
            [LABEL_LIST[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(preds, labels)
        ]
        true_labels = [
            [LABEL_LIST[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(preds, labels)
        ]

        p_score = precision_score(true_labels, true_predictions, mode="strict", scheme=IOB2, zero_division=0)
        r_score = recall_score(true_labels, true_predictions, mode="strict", scheme=IOB2, zero_division=0)
        f_score = f1_score(true_labels, true_predictions, mode="strict", scheme=IOB2, zero_division=0)

        # Детальный отчет seqeval
        report = classification_report(true_labels, true_predictions, mode="strict", scheme=IOB2, zero_division=0, output_dict=True)

        res = {
            "precision": float(p_score),
            "recall": float(r_score),
            "f1": float(f_score),
        }

        for ent_type in ("PER", "ORG", "LOC", "ADDR"):
            if ent_type in report:
                res[f"{ent_type}_f1"] = float(report[ent_type]["f1-score"])

        return res

    return compute_metrics


def train_rosberta(
    model_name_or_path: str = "ai-forever/ru-en-RoSBERTa",
    dataset_dir: str = "data/unified_ner_dataset_mini",
    output_dir: str = "models/rosberta-ner-finetuned",
    num_train_epochs: int = 4,
    per_device_batch_size: int = 8,
    learning_rate: float = 2e-5,
    dry_run: bool = False,
):
    """Главная функция запуска дообучения."""
    print("=" * 70)
    print("ДООБУЧЕНИЕ МОДЕЛИ: ai-forever/ru-en-RoSBERTa (Token Classification)")
    print("=" * 70)

    # 1. Загрузка токенизатора
    print(f"\n[1/5] Загрузка токенизатора {model_name_or_path} (add_prefix_space=True)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, add_prefix_space=True)

    # 2. Загрузка и подготовка датасета
    print(f"\n[2/5] Загрузка датасета из {dataset_dir}...")
    dataset = load_from_disk(dataset_dir)

    if dry_run:
        print("  [DRY-RUN] Обрезка выборки до 16 примеров для тестового прогона...")
        dataset["train"] = dataset["train"].select(range(min(16, len(dataset["train"]))))
        dataset["validation"] = dataset["validation"].select(range(min(8, len(dataset["validation"]))))
        dataset["test"] = dataset["test"].select(range(min(8, len(dataset["test"]))))

    tokenized_dataset = tokenize_and_align_dataset(dataset, tokenizer)

    # 3. Инициализация модели
    print(f"\n[3/5] Инициализация архитектуры модели с {len(LABEL_LIST)} классами...")
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_name_or_path,
        config=config,
    )

    # 4. Настройка параметров обучения
    print("\n[4/5] Конфигурация TrainingArguments...")
    class_weights_tensor = torch.tensor(CLASS_WEIGHTS_LIST, dtype=torch.float)

    epochs = 1 if dry_run else num_train_epochs
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=1 if dry_run else 4,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=2 if dry_run else 50,
        load_best_model_at_end=not dry_run,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available() and not dry_run,
        save_total_limit=2,
        report_to="none",
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    trainer = WeightedTrainer(
        class_weights=class_weights_tensor,
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(),
    )

    # 5. Запуск обучения и сохранение
    print("\n[5/5] Запуск цикла обучения Trainer.train()...")
    train_result = trainer.train()

    eval_metrics = trainer.evaluate()
    print("\nМетрики на валидационном сплите:")
    for k, v in eval_metrics.items():
        print(f"  {k}: {v}")

    # Сохранение финальной модели
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nМодель и токенизатор успешно сохранены в: {output_dir}")

    return eval_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Дообучение ru-en-RoSBERTa NER")
    parser.add_argument("--dry-run", action="store_true", help="Быстрый прогон на 16 примерах для валидации пайплайна")
    parser.add_argument("--epochs", type=int, default=4, help="Количество эпох")
    parser.add_argument("--batch-size", type=int, default=8, help="Размер батча")
    parser.add_argument("--dataset", type=str, default="data/unified_ner_dataset_mini", help="Путь к DatasetDict")
    parser.add_argument("--output", type=str, default="models/rosberta-ner-finetuned", help="Директория сохранения")
    args = parser.parse_args()

    train_rosberta(
        dataset_dir=args.dataset,
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_batch_size=args.batch_size,
        dry_run=args.dry_run,
    )
