"""
Модуль токенизации и извлечения контекстных признаков для обучения
AI-модели классификации токенов (Token Classification / NER).

Разбивает текст на токены и извлекает богатый вектор лингвистических признаков:
- Лексические свойства (слово, строчная форма, длина, суффиксы и префиксы).
- Орфографические флаги (заглавная буква, все заглавные, цифры, дефисы).
- Морфологические характеристики из pymorphy3 (POS, одушевленность, род, падеж, словарные теги Surn, Name, Patr).
- Контекстные признаки соседних слов в окне [-2, +2].
"""

from __future__ import annotations

import re
import pymorphy3

_morph = pymorphy3.MorphAnalyzer()

# Паттерн токенизации: слова, числа и отдельные знаки препинания
TOKEN_PATTERN = re.compile(r'[а-яА-ЯёЁa-zA-Z0-9]+|[^\s\w]')


def tokenize_with_spans(text: str) -> list[tuple[str, int, int]]:
    """
    Разбить текст на токены с сохранением точных символьных смещений (start, end).

    Returns:
        Список кортежей (token_text, start_idx, end_idx).
    """
    tokens = []
    for m in TOKEN_PATTERN.finditer(text):
        tokens.append((m.group(0), m.start(), m.end()))
    return tokens


def align_tokens_with_spans(
    tokens: list[tuple[str, int, int]],
    char_spans: list[tuple[str, int, int] | list],
) -> list[str]:
    """
    Сопоставить токены с разметкой спанов ПДн и сгенерировать BIO-теги.

    Схема BIO:
    - B-TAG: первый токен сущности.
    - I-TAG: последующие токены той же сущности.
    - O: токен вне сущности.
    """
    bio_tags = ["O"] * len(tokens)

    # Сортируем спаны по началу
    sorted_spans = sorted(char_spans, key=lambda s: int(s[1]))

    for tag, start, end in sorted_spans:
        start = int(start)
        end = int(end)
        first_token = True

        for i, (tok, t_start, t_end) in enumerate(tokens):
            # Токен перекрывается со спаном
            if max(start, t_start) < min(end, t_end):
                if first_token:
                    bio_tags[i] = f"B-{tag}"
                    first_token = False
                else:
                    bio_tags[i] = f"I-{tag}"

    return bio_tags


def extract_token_features(tokens: list[str], index: int) -> dict[str, str | int | bool]:
    """
    Извлечь вектор признаков для токена на позиции index в списке tokens.
    """
    word = tokens[index]
    lower = word.lower()

    # Морфологический анализ
    parses = _morph.parse(word)
    best = parses[0] if parses else None
    tag_str = str(best.tag) if best else "UNKNOWN"
    pos = str(best.tag.POS) if (best and best.tag.POS) else "OTHER"
    top_tags = [str(p.tag) for p in parses[:4]] if parses else []

    features: dict[str, str | int | bool] = {
        "bias": 1.0,
        "word.lower": lower,
        "word.len": len(word),
        "word.isupper": word.isupper(),
        "word.istitle": word.istitle(),
        "word.isdigit": word.isdigit(),
        "word.pos": pos,
        "word.is_surn": ("Surn" in tag_str),
        "word.is_name": ("Name" in tag_str),
        "word.any_surn": any("Surn" in t for t in top_tags),
        "word.any_name": any("Name" in t for t in top_tags),
        "word.is_patr": ("Patr" in tag_str),
        "word.is_anim": ("anim" in tag_str),
        "word.gender": (best.tag.gender or "none") if best else "none",
        "word.case": (best.tag.case or "none") if best else "none",
        "word.suffix2": lower[-2:] if len(lower) >= 2 else lower,
        "word.suffix3": lower[-3:] if len(lower) >= 3 else lower,
        "word.suffix4": lower[-4:] if len(lower) >= 4 else lower,
        "word.prefix2": lower[:2] if len(lower) >= 2 else lower,
    }

    # Контекст слева: предыдущее слово (index - 1)
    if index > 0:
        prev_word = tokens[index - 1]
        p_parses = _morph.parse(prev_word)
        p_best = p_parses[0] if p_parses else None
        p_tag = str(p_best.tag) if p_best else ""
        p_top = [str(p.tag) for p in p_parses[:4]] if p_parses else []
        features.update({
            "-1:word.lower": prev_word.lower(),
            "-1:word.istitle": prev_word.istitle(),
            "-1:word.isupper": prev_word.isupper(),
            "-1:word.pos": str(p_best.tag.POS) if (p_best and p_best.tag.POS) else "OTHER",
            "-1:is_surn": ("Surn" in p_tag) or any("Surn" in t for t in p_top),
            "-1:is_name": ("Name" in p_tag) or any("Name" in t for t in p_top),
        })
    else:
        features["BOS"] = True  # Beginning of sentence

    # Контекст слева: пред-предыдущее слово (index - 2)
    if index > 1:
        prev2_word = tokens[index - 2]
        features.update({
            "-2:word.lower": prev2_word.lower(),
            "-2:word.istitle": prev2_word.istitle(),
        })

    # Контекст справа: следующее слово (index + 1)
    if index < len(tokens) - 1:
        next_word = tokens[index + 1]
        n_parses = _morph.parse(next_word)
        n_best = n_parses[0] if n_parses else None
        n_tag = str(n_best.tag) if n_best else ""
        n_top = [str(p.tag) for p in n_parses[:4]] if n_parses else []
        features.update({
            "+1:word.lower": next_word.lower(),
            "+1:word.istitle": next_word.istitle(),
            "+1:word.isupper": next_word.isupper(),
            "+1:word.pos": str(n_best.tag.POS) if (n_best and n_best.tag.POS) else "OTHER",
            "+1:is_surn": ("Surn" in n_tag) or any("Surn" in t for t in n_top),
            "+1:is_name": ("Name" in n_tag) or any("Name" in t for t in n_top),
        })
    else:
        features["EOS"] = True  # End of sentence

    # Контекст справа: после-следующее слово (index + 2)
    if index < len(tokens) - 2:
        next2_word = tokens[index + 2]
        features.update({
            "+2:word.lower": next2_word.lower(),
            "+2:word.istitle": next2_word.istitle(),
        })

    return features


def sentence_to_features(tokens: list[str]) -> list[dict]:
    """Извлечь вектор признаков для каждого токена предложения."""
    return [extract_token_features(tokens, i) for i in range(len(tokens))]
