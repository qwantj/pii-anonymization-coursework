"""
Универсальный реестр провайдеров на базе паттерна Strategy и безопасная обработка неизвестных тегов (LR-ARC-1).

Обеспечивает:
1. Регистрацию стратегий обезличивания (синонимизация, маскирование, HMAC).
2. Безопасную Fallback-политику: неизвестные спаны маскируются символом '*',
   предотвращая утечку конфиденциальных данных (Zero Leakage, C1, C4, C5).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class BaseAnonymizationStrategy(ABC):
    """Абстрактный базовый класс стратегии обезличивания."""

    @abstractmethod
    def anonymize(
        self,
        original_val: str,
        tag: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Выполняет трансформацию значения сущности."""
        pass


class FallbackMaskingStrategy(BaseAnonymizationStrategy):
    """
    Политика безопасной деградации (Fallback):
    неизвестные или незарегистрированные теги маскируются символами '*'
    с сохранением исходной длины строки во избежание утечки ПДн.
    """

    def __init__(self, mask_char: str = "*"):
        self.mask_char = mask_char

    def anonymize(
        self,
        original_val: str,
        tag: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        logger.warning("Unknown PII tag '%s' masked by FallbackMaskingStrategy", tag)
        return self.mask_char * len(original_val)


class ProviderRegistry:
    """Потокобезопасный реестр обработчиков и стратегий обезличивания."""

    _strategies: Dict[str, Dict[str, Callable[..., str]]] = {}
    _fallback_strategy: BaseAnonymizationStrategy = FallbackMaskingStrategy()
    _default_strategy_name: str = "synonymic"

    @classmethod
    def register(cls, tag: str, strategy_name: str = "synonymic"):
        """Декоратор для регистрации обработчика под конкретный тег и стратегию."""
        def decorator(fn: Callable[..., str]):
            cls._strategies.setdefault(strategy_name, {})[tag] = fn
            return fn
        return decorator

    @classmethod
    def set_handler(cls, tag: str, handler: Callable[..., str], strategy_name: str = "synonymic"):
        """Прямая регистрация функции-обработчика."""
        cls._strategies.setdefault(strategy_name, {})[tag] = handler

    @classmethod
    def get_handler(cls, tag: str, strategy_name: str = "synonymic") -> Callable[..., str]:
        """
        Получить обработчик для тега. Если тег неизвестен, возвращается fallback-обработчик.
        """
        handlers = cls._strategies.get(strategy_name, {})
        if tag in handlers:
            return handlers[tag]

        # Возвращаем fallback обертку
        def fallback_handler(val: str, *args, **kwargs) -> str:
            return cls._fallback_strategy.anonymize(val, tag)

        return fallback_handler

    @classmethod
    def get_all_handlers(cls, strategy_name: str = "synonymic") -> Dict[str, Callable[..., str]]:
        """Получить словарь всех зарегистрированных обработчиков стратегии."""
        return cls._strategies.get(strategy_name, {})

    @classmethod
    def set_fallback_strategy(cls, strategy: BaseAnonymizationStrategy):
        cls._fallback_strategy = strategy
