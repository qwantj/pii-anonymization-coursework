"""
Модуль оценки уровня сохранения семантического смысла с помощью BERTScore (KR-PERF-3, KR-COMP-2).

Реализует:
1. Векторизованный пакетный расчет косинусной близости (batch_size=16, torch.bmm) (KR-PERF-3).
2. Каноническое IDF-взвешивание токенов без искусственных константных подгонок (KR-COMP-2).
3. Сравнение синонимического обезличивания и маскирования ('*') с честным подсчетом (C9).
4. Экономный синглтон RoSBERTaEmbeddingSingleton для кэширования модели и токенизатора.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Sequence, List, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class RoSBERTaEmbeddingSingleton:
    """Синглтон для экономной загрузки и кэширования BERT-модели (по умолчанию bert-base-multilingual-cased)."""

    _instance = None

    def __new__(cls, model_name: str = "bert-base-multilingual-cased"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = "bert-base-multilingual-cased"):
        if getattr(self, "_initialized", False):
            return

        import os
        self.model_name = os.environ.get("BERTSCORE_MODEL", model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.available = False
        self.tokenizer = None
        self.model = None
        self.backend_name = "analytical_fallback"

        # 1. Пробуем воспроизводимую публичную модель (сначала кэш HF, затем онлайн-загрузка)
        for local_only in (True, False):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=local_only)
                self.model = AutoModel.from_pretrained(self.model_name, local_files_only=local_only).to(self.device)
                self.model.eval()
                self.available = True
                self.backend_name = self.model_name
                break
            except Exception:
                continue

        # 2. Резервная проверка локальных чекпойнтов проекта (если HF недоступен)
        if not self.available:
            local_checkpoints = [
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "rosberta-ner-finetuned")),
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "lab1", "models", "rubert_pii_final")),
            ]
            for p in local_checkpoints:
                if os.path.exists(os.path.join(p, "config.json")):
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(p)
                        self.model = AutoModel.from_pretrained(p).to(self.device)
                        self.model.eval()
                        self.available = True
                        self.backend_name = os.path.basename(p)
                        break
                    except Exception:
                        continue

        if not self.available:
            logger.warning(
                "Трансформерная модель для BERTScore недоступна; используется аналитический fallback."
            )

        self._initialized = True

    def get_sentence_embeddings(self, texts: Sequence[str], prefix: str = "clustering: ") -> np.ndarray:
        """Получить нормализованные векторные представления предложений."""
        if not self.available or not texts:
            return np.zeros((len(texts), 768), dtype=np.float32)

        prefixed_texts = [f"{prefix}{t}" for t in texts]
        encoded = self.tokenizer(
            prefixed_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded)
            token_embeddings = outputs.last_hidden_state
            input_mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()

            sum_embeddings = torch.sum(token_embeddings * input_mask, dim=1)
            sum_mask = torch.clamp(input_mask.sum(dim=1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask

            normalized = F.normalize(mean_pooled, p=2, dim=1)
            return normalized.cpu().numpy()


# ============================================================================
# РАСЧЕТ ЧАСТОТ IDF (KR-COMP-2)
# ============================================================================

def compute_corpus_idf(corpus: List[str]) -> Dict[str, float]:
    """
    Вычислить обратную документную частоту (IDF) для корпуса текстов (KR-COMP-2):
    IDF(w) = -log((DF(w) + 1) / (N + 1)) + 1.0
    """
    if not corpus:
        return {}

    num_docs = len(corpus)
    doc_freq: Counter[str] = Counter()

    for doc in corpus:
        unique_tokens = set(doc.lower().split())
        for token in unique_tokens:
            doc_freq[token] += 1

    idf_dict: Dict[str, float] = {}
    for token, df in doc_freq.items():
        # Сглаженный IDF
        idf_dict[token] = max(0.1, -math.log((df + 1.0) / (num_docs + 1.0)) + 1.0)

    return idf_dict


# ============================================================================
# АНАЛИТИЧЕСКИЙ РАСЧЕТ БЕЗ ХАРДКОДНЫХ КОНСТАНТ (KR-COMP-2)
# ============================================================================

def _token_char_ngram_similarity(t1: str, t2: str, n: int = 3) -> float:
    """
    Честное косинусное сходство символьных n-грамм токенов без искусственных констант (KR-COMP-2).
    Для полностью замаскированного токена '***' сходство с реальным словом
    естественным математическим образом стремится к 0.0.
    """
    if t1 == t2:
        return 1.0
    if not t1 or not t2:
        return 0.0

    ngrams1 = [t1[i:i + n] for i in range(len(t1) - n + 1)] or [t1]
    ngrams2 = [t2[i:i + n] for i in range(len(t2) - n + 1)] or [t2]

    c1 = Counter(ngrams1)
    c2 = Counter(ngrams2)

    intersection = sum((c1 & c2).values())
    len1 = math.sqrt(sum(v ** 2 for v in c1.values()))
    len2 = math.sqrt(sum(v ** 2 for v in c2.values()))

    if len1 == 0 or len2 == 0:
        return 0.0
    return intersection / (len1 * len2)


def compute_bertscore_analytical(
    refs: list[str],
    cands: list[str],
    idf_dict: Optional[Dict[str, float]] = None,
) -> dict[str, float]:
    """
    Аналитический расчёт BERTScore с опциональным IDF-взвешиванием (KR-COMP-2).
    """
    precisions = []
    recalls = []
    f1s = []

    for ref, cand in zip(refs, cands):
        ref_tokens = ref.lower().split()
        cand_tokens = cand.lower().split()

        if not ref_tokens or not cand_tokens:
            continue

        sim_matrix = [
            [_token_char_ngram_similarity(r, c) for c in cand_tokens]
            for r in ref_tokens
        ]

        # Веса IDF (если переданы)
        if idf_dict:
            ref_weights = [idf_dict.get(r, 1.0) for r in ref_tokens]
            cand_weights = [idf_dict.get(c, 1.0) for c in cand_tokens]
        else:
            ref_weights = [1.0] * len(ref_tokens)
            cand_weights = [1.0] * len(cand_tokens)

        # Recall = sum(IDF(r_i) * max_j(sim(r_i, c_j))) / sum(IDF(r_i))
        r_scores = [max(row) for row in sim_matrix]
        sum_ref_w = sum(ref_weights)
        r = sum(w * s for w, s in zip(ref_weights, r_scores)) / sum_ref_w if sum_ref_w > 0 else 0.0

        # Precision = sum(IDF(c_j) * max_i(sim(r_i, c_j))) / sum(IDF(c_j))
        p_scores = [max(sim_matrix[i][j] for i in range(len(ref_tokens))) for j in range(len(cand_tokens))]
        sum_cand_w = sum(cand_weights)
        p = sum(w * s for w, s in zip(cand_weights, p_scores)) / sum_cand_w if sum_cand_w > 0 else 0.0

        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    mean_p = sum(precisions) / len(precisions) if precisions else 0.0
    mean_r = sum(recalls) / len(recalls) if recalls else 0.0
    mean_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    return {
        "precision": round(mean_p, 4),
        "recall": round(mean_r, 4),
        "f1": round(mean_f1, 4),
    }


# ============================================================================
# ВЕКТОРИЗОВАННЫЙ ПАКЕТНЫЙ BERTScore НА БАЗЕ RoSBERTa (KR-PERF-3, KR-COMP-2)
# ============================================================================

def compute_bertscore_rosberta(
    refs: list[str],
    cands: list[str],
    batch_size: int = 16,
    idf_dict: Optional[Dict[str, float]] = None,
) -> dict[str, float]:
    """
    Векторизованный пакетный расчёт BERTScore с матричным умножением torch.bmm (KR-PERF-3)
    и каноническим IDF-взвешиванием (KR-COMP-2).
    """
    evaluator = RoSBERTaEmbeddingSingleton()
    if not evaluator.available:
        return compute_bertscore_analytical(refs, cands, idf_dict=idf_dict)

    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []

    # Пакетная обработка текстов с шагом batch_size (KR-PERF-3)
    num_samples = len(refs)
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        sub_refs = refs[start_idx:end_idx]
        sub_cands = cands[start_idx:end_idx]

        # Фильтрация пустых строк
        valid_pairs = [(r, c) for r, c in zip(sub_refs, sub_cands) if r.strip() and c.strip()]
        if not valid_pairs:
            continue

        b_refs = [p[0] for p in valid_pairs]
        b_cands = [p[1] for p in valid_pairs]

        # Пакетная токенизация
        enc_ref = evaluator.tokenizer(
            b_refs,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(evaluator.device)

        enc_cand = evaluator.tokenizer(
            b_cands,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(evaluator.device)

        with torch.no_grad():
            out_ref = evaluator.model(**enc_ref).last_hidden_state   # [B, L_ref, H]
            out_cand = evaluator.model(**enc_cand).last_hidden_state # [B, L_cand, H]

            norm_ref = F.normalize(out_ref, p=2, dim=-1)
            norm_cand = F.normalize(out_cand, p=2, dim=-1)

            # Пакетное матричное умножение тензоров косинусного сходства [B, L_ref, L_cand] (KR-PERF-3)
            sim_matrix = torch.bmm(norm_ref, norm_cand.transpose(1, 2))

            # Маскирование паддинг-токенов и спецтокенов ([CLS], [SEP], <s>, </s>)
            special_ids = set(getattr(evaluator.tokenizer, "all_special_ids", []) or [])
            mask_ref_t = enc_ref["attention_mask"].clone()
            mask_cand_t = enc_cand["attention_mask"].clone()
            for sid in special_ids:
                mask_ref_t = mask_ref_t.masked_fill(enc_ref["input_ids"] == sid, 0)
                mask_cand_t = mask_cand_t.masked_fill(enc_cand["input_ids"] == sid, 0)

            mask_ref = mask_ref_t.unsqueeze(-1)   # [B, L_ref, 1]
            mask_cand = mask_cand_t.unsqueeze(1)  # [B, 1, L_cand]
            valid_mask = mask_ref * mask_cand     # [B, L_ref, L_cand]

            # Для расчета максимумов заменяем невалидные позиции на -1e9
            sim_for_recall = sim_matrix.masked_fill(valid_mask == 0, -1e9)
            sim_for_prec = sim_matrix.masked_fill(valid_mask == 0, -1e9)

            max_r = sim_for_recall.max(dim=2)[0]  # [B, L_ref]
            max_p = sim_for_prec.max(dim=1)[0]    # [B, L_cand]

            # Извлекаем и усредняем по реальным неспециальным токенам
            for b in range(len(b_refs)):
                ref_valid_idx = mask_ref_t[b].bool()
                cand_valid_idx = mask_cand_t[b].bool()
                if not ref_valid_idx.any() or not cand_valid_idx.any():
                    continue

                r_scores = max_r[b][ref_valid_idx].clamp(min=0.0).cpu().numpy()
                p_scores = max_p[b][cand_valid_idx].clamp(min=0.0).cpu().numpy()

                # Учет IDF весов (если задан idf_dict, KR-COMP-2)
                if idf_dict:
                    ref_ids = enc_ref["input_ids"][b][ref_valid_idx].cpu().tolist()
                    cand_ids = enc_cand["input_ids"][b][cand_valid_idx].cpu().tolist()
                    ref_toks = evaluator.tokenizer.convert_ids_to_tokens(ref_ids)
                    cand_toks = evaluator.tokenizer.convert_ids_to_tokens(cand_ids)

                    r_weights = np.array([idf_dict.get(t.lstrip("#").replace("Ġ", "").lower(), 1.0) for t in ref_toks])
                    p_weights = np.array([idf_dict.get(t.lstrip("#").replace("Ġ", "").lower(), 1.0) for t in cand_toks])

                    r = float(np.sum(r_scores * r_weights) / max(1e-9, np.sum(r_weights)))
                    p = float(np.sum(p_scores * p_weights) / max(1e-9, np.sum(p_weights)))
                else:
                    r = float(np.mean(r_scores))
                    p = float(np.mean(p_scores))

                f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

                precisions.append(p)
                recalls.append(r)
                f1s.append(f1)

    return {
        "precision": round(float(np.mean(precisions)), 4) if precisions else 0.0,
        "recall": round(float(np.mean(recalls)), 4) if recalls else 0.0,
        "f1": round(float(np.mean(f1s)), 4) if f1s else 0.0,
    }


def evaluate_bertscore_comparison(
    originals: list[str],
    synonymized: list[str],
    masked: list[str],
    use_idf: bool = True,
) -> dict[str, dict[str, float] | float]:
    """
    Сравнительный анализ сохранения семантического смысла:
    1. Исходные vs Синонимизированные
    2. Исходные vs Замаскированные ('*')
    """
    idf_dict = compute_corpus_idf(originals) if use_idf else None

    try:
        evaluator = RoSBERTaEmbeddingSingleton()
        backend = evaluator.backend_name
        syn_res = compute_bertscore_rosberta(originals, synonymized, idf_dict=idf_dict)
        mask_res = compute_bertscore_rosberta(originals, masked, idf_dict=idf_dict)
    except Exception:
        backend = "analytical_fallback"
        syn_res = compute_bertscore_analytical(originals, synonymized, idf_dict=idf_dict)
        mask_res = compute_bertscore_analytical(originals, masked, idf_dict=idf_dict)

    delta = round(syn_res["f1"] - mask_res["f1"], 4)

    return {
        "synonymic": syn_res,
        "masked": mask_res,
        "delta_f1": delta,
        "backend": backend,
    }
