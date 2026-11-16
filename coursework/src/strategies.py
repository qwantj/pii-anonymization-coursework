"""
Абстрактные стратегии обезличивания для Курсовой работы (KR-ARC-3).

Реализует паттерн Strategy:
- SynonymicAnonymizationStrategy: синонимическая замена с сохранением морфологии и идемпотентности.
- CharacterMaskingStrategy: замена символов сущности на заданный символ маски ('*').
- HmacAnonymizationStrategy: криптографическое псевдонимирование на базе HMAC-SHA256.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import hmac
from typing import List, Tuple, Any, Optional

from coursework.src.idempotency_manager import IdempotencyVault
from coursework.src.masking_engine import mask_text
from lab2.src.anonymizer import assemble_anonymized_text


class AnonymizationStrategy(ABC):
    """Базовый абстрактный класс стратегии обезличивания текста."""

    @abstractmethod
    def execute(
        self,
        text: str,
        spans: List[Tuple[str, int, int] | list],
        vault: Optional[IdempotencyVault] = None,
    ) -> str:
        """
        Выполнить обезличивание спанов в тексте.

        Returns:
            Результирующий текст.
        """
        pass


class SynonymicAnonymizationStrategy(AnonymizationStrategy):
    """Стратегия синонимического обезличивания с контролем идемпотентности (C5, C6, C7)."""

    def execute(
        self,
        text: str,
        spans: List[Tuple[str, int, int] | list],
        vault: Optional[IdempotencyVault] = None,
    ) -> str:
        if not text or not spans:
            return text

        v = vault or IdempotencyVault()

        # Нормализация спанов
        valid_spans: List[Tuple[str, int, int]] = []
        for s in spans:
            valid_spans.append((str(s[0]), int(s[1]), int(s[2])))

        # Сортировка по возрастанию для сборки O(N) (LR-PERF-1)
        valid_spans.sort(key=lambda s: s[1])

        replacements: List[str] = []
        for tag, start, end in valid_spans:
            orig_val = text[start:end]
            if tag in ("SURNAME", "NAME", "PER", "FIO"):
                rep = v.get_or_create_name_replacement(
                    original_word=orig_val,
                    tag=tag,
                    text=text,
                    start=start,
                    end=end,
                )
            else:
                rep = v.get_or_create_structured_replacement(
                    original_value=orig_val,
                    tag=tag,
                )
            replacements.append(rep)

        return assemble_anonymized_text(text, valid_spans, replacements)


class CharacterMaskingStrategy(AnonymizationStrategy):
    """Стратегия посимвольного маскирования (замена на '*')."""

    def __init__(self, mask_char: str = "*"):
        self.mask_char = mask_char

    def execute(
        self,
        text: str,
        spans: List[Tuple[str, int, int] | list],
        vault: Optional[IdempotencyVault] = None,
    ) -> str:
        return mask_text(text, spans, mask_char=self.mask_char)


class HmacAnonymizationStrategy(AnonymizationStrategy):
    """Стратегия криптографического псевдонимирования на базе HMAC-SHA256."""

    def __init__(self, secret_key: bytes = b"CTZI-2026-SECRET-KEY"):
        self.secret_key = secret_key

    def execute(
        self,
        text: str,
        spans: List[Tuple[str, int, int] | list],
        vault: Optional[IdempotencyVault] = None,
    ) -> str:
        if not text or not spans:
            return text

        valid_spans = [(str(s[0]), int(s[1]), int(s[2])) for s in spans]
        valid_spans.sort(key=lambda s: s[1])

        replacements: List[str] = []
        for tag, start, end in valid_spans:
            orig = text[start:end]
            h = hmac.new(self.secret_key, orig.encode("utf-8"), hashlib.sha256).hexdigest()[:len(orig)]
            replacements.append(f"[{tag}_{h}]")

        return assemble_anonymized_text(text, valid_spans, replacements)
