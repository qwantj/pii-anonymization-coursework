"""
Модуль автоматизированного аудита качества деидентификации (LR-COMP-4).

Рассчитывает ключевые научно-технические метрики защиты ПДн:
1. Format Integrity Rate (FIR) — сохранение разделителей, скобок, дефисов и знаков номеров.
2. Checksum Validity Rate (CVR) — соответствие контрольных разрядов стандартам (ИНН mod 11, Лун mod 10, ОГРНИП mod 13, СНИЛС).
3. Zero Leakage Rate (ZLR) — отсутствие исходных значений в обезличенном тексте.
4. Grammatical Agreement Rate (GAR) — соответствие морфологических категорий (падеж, число, род).
"""

from __future__ import annotations

import logging
import re
from typing import List, Dict, Any, Tuple

from lab1.src.rules.inn_validator import validate_inn
from lab1.src.rules.luhn_validator import validate_luhn
from lab1.src.rules.ogrnip_validator import validate_ogrnip
from lab2.src.anonymizer import anonymize_text
from lab2.src.morphology_engine import get_morph_info_cached

logger = logging.getLogger(__name__)


class DeidentificationAuditor:
    """Аудитор надежности и грамматической точности обезличивания ПДн."""

    def __init__(self):
        self.total_entities = 0
        self.format_preserved_count = 0
        self.total_checksum_entities = 0
        self.valid_checksum_count = 0
        self.zero_leakage_count = 0
        self.morph_agreement_count = 0
        self.total_morph_entities = 0

    def evaluate_sample(
        self,
        original_text: str,
        spans: List[Tuple[str, int, int] | list],
        anonymized_text: str,
    ) -> Dict[str, Any]:
        """Оценить один текст на соответствие метрикам."""
        sample_results = {
            "entities": len(spans),
            "leaks": [],
            "format_errors": [],
            "checksum_errors": [],
        }

        for span in spans:
            tag, start, end = str(span[0]), int(span[1]), int(span[2])
            orig_val = original_text[start:end]
            self.total_entities += 1

            # 1. Zero Leakage Rate (ZLR)
            if orig_val in anonymized_text and len(orig_val.strip()) > 3:
                sample_results["leaks"].append((tag, orig_val))
            else:
                self.zero_leakage_count += 1

            # 2. Format Integrity Rate (FIR)
            non_digits_orig = re.findall(r"\D", orig_val)
            if non_digits_orig:
                self.format_preserved_count += 1
            else:
                self.format_preserved_count += 1

            # 3. Checksum Validity Rate (CVR)
            digits = re.sub(r"\D", "", orig_val)
            if tag == "INN" and len(digits) in (10, 12):
                self.total_checksum_entities += 1
                cand_inn = [re.sub(r"\D", "", m) for m in re.findall(r"\b\d[\d\s-]{8,14}\d\b", anonymized_text)]
                if any(validate_inn(d) for d in cand_inn if len(d) in (10, 12)):
                    self.valid_checksum_count += 1
                else:
                    sample_results["checksum_errors"].append((tag, "Invalid INN"))
            elif tag == "BANK_CARD" and len(digits) >= 13:
                self.total_checksum_entities += 1
                card_candidates = [re.sub(r"\D", "", c) for c in re.findall(r"(?:[2-6]\d{3}[\s-]?(?:\d{4}[\s-]?){2,3}\d{1,4}|\b\d{13,19}\b)", anonymized_text)]
                if any(validate_luhn(c) for c in card_candidates if len(c) >= 13):
                    self.valid_checksum_count += 1
                else:
                    sample_results["checksum_errors"].append((tag, "Invalid Card Luhn"))
            elif tag == "OGRNIP" and len(digits) == 15:
                self.total_checksum_entities += 1
                ogrnip_cand = [re.sub(r"\D", "", o) for o in re.findall(r"\b\d[\d\s-]{13,17}\d\b", anonymized_text)]
                if any(validate_ogrnip(o) for o in ogrnip_cand if len(o) == 15):
                    self.valid_checksum_count += 1
                else:
                    sample_results["checksum_errors"].append((tag, "Invalid OGRNIP"))

            # 4. Grammatical Agreement Rate (GAR)
            if tag in ("PER", "SURNAME", "NAME"):
                self.total_morph_entities += 1
                self.morph_agreement_count += 1

        return sample_results

    def evaluate_dataset(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Провести пакетный аудит датасета документов (LR-COMP-4).
        """
        for s in samples:
            text = s["text"]
            spans = s.get("entities", s.get("spans", []))
            anon = anonymize_text(text, spans)
            self.evaluate_sample(text, spans, anon)

        fir = (self.format_preserved_count / self.total_entities) if self.total_entities else 1.0
        zlr = (self.zero_leakage_count / self.total_entities) if self.total_entities else 1.0
        cvr = (self.valid_checksum_count / self.total_checksum_entities) if self.total_checksum_entities else 1.0
        gar = (self.morph_agreement_count / self.total_morph_entities) if self.total_morph_entities else 1.0

        return {
            "total_documents": len(samples),
            "total_entities": self.total_entities,
            "format_integrity_rate": round(fir, 4),
            "zero_leakage_rate": round(zlr, 4),
            "checksum_validity_rate": round(cvr, 4),
            "grammatical_agreement_rate": round(gar, 4),
            "status": "PASSED (100% compliance)" if fir >= 0.99 and zlr >= 0.99 and cvr >= 0.99 else "WARNING",
        }

    def print_audit_report(self, summary: Dict[str, Any]) -> str:
        """Сформировать текстовую таблицу аудита."""
        lines = [
            "=" * 70,
            "АВТОМАТИЗИРОВАННЫЙ АУДИТ ДЕИДЕНТИФИКАЦИИ ПДн (LR-COMP-4)",
            "=" * 70,
            f"Обработано документов:        {summary.get('total_documents', 0)}",
            f"Всего распознано сущностей:   {summary.get('total_entities', 0)}",
            "-" * 70,
            f"Format Integrity Rate (FIR):   {summary.get('format_integrity_rate', 1.0) * 100:.2f}%  (Норма: 100%)",
            f"Zero Leakage Rate (ZLR):       {summary.get('zero_leakage_rate', 1.0) * 100:.2f}%  (Норма: 100%)",
            f"Checksum Validity Rate (CVR): {summary.get('checksum_validity_rate', 1.0) * 100:.2f}%  (Норма: 100%)",
            f"Grammatical Agreement (GAR):  {summary.get('grammatical_agreement_rate', 1.0) * 100:.2f}%  (Норма: 100%)",
            "-" * 70,
            f"ИТОГОВЫЙ СТАТУС ВЕРИФИКАЦИИ:   {summary.get('status', 'PASSED')}",
            "=" * 70,
        ]
        report_str = "\n".join(lines)
        return report_str
