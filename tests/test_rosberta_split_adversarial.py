"""
Состязательный стресс-тест декомпозиции антропонимов (_split_per_span) детектора RoSBERTaNerDetector.
Автономный набор проверок граничных условий, продуктивных славянских суффиксов,
дефисных фамилий, нормализации пунктуации, честного fallback на PER и исключения родительского тега.
"""

import pytest
from coursework.src.detector_base import NerDetectorFactory
from lab1.src.ner.rosberta_detector import RoSBERTaNerDetector


@pytest.fixture(scope="module")
def detector():
    if NerDetectorFactory.has_local_rosberta():
        return RoSBERTaNerDetector()
    stub = RoSBERTaNerDetector.__new__(RoSBERTaNerDetector)
    stub.min_score = 0.1
    return stub


PUNCTUATION_CHARS = set(".,;:!?\"'()«»—–")


def _verify_clean_boundaries(text: str, spans: list[tuple[str, int, int]]) -> None:
    """Инвариант: границы спанов не должны содержать внешнюю пунктуацию и пробелы."""
    for tag, start, end in spans:
        sub = text[start:end]
        assert sub == sub.strip(), f"Спан содержит внешние пробелы: '{sub}' ({tag}, {start}, {end})"
        assert not any(sub.startswith(p) for p in PUNCTUATION_CHARS), (
            f"Спан '{sub}' ({tag}) начинается с пунктуации!"
        )
        assert not any(sub.endswith(p) for p in PUNCTUATION_CHARS), (
            f"Спан '{sub}' ({tag}) заканчивается пунктуацией!"
        )


class TestHyphenatedCompoundSurnames:
    """1. Дефисные составные фамилии: Мамин-Сибиряк, Салтыков-Щедрин, Римский-Корсаков."""

    def test_mamin_sibiryak_solo(self, detector):
        text = "Мамин-Сибиряк"
        spans = detector._split_per_span(text, 0, len(text))
        assert spans == [("SURNAME", 0, len(text))]

    def test_mamin_sibiryak_direct_order(self, detector):
        text = "Мамин-Сибиряк Дмитрий Наркисович"
        spans = detector._split_per_span(text, 0, len(text))
        assert "PER" not in [s[0] for s in spans]
        assert [(s[0], text[s[1]:s[2]]) for s in spans] == [
            ("SURNAME", "Мамин-Сибиряк"),
            ("NAME", "Дмитрий"),
            ("NAME", "Наркисович"),
        ]
        _verify_clean_boundaries(text, spans)

    def test_mamin_sibiryak_reverse_order(self, detector):
        text = "Дмитрий Мамин-Сибиряк"
        spans = detector._split_per_span(text, 0, len(text))
        assert "PER" not in [s[0] for s in spans]
        assert [(s[0], text[s[1]:s[2]]) for s in spans] == [
            ("NAME", "Дмитрий"),
            ("SURNAME", "Мамин-Сибиряк"),
        ]
        _verify_clean_boundaries(text, spans)

    def test_mamin_sibiryak_reverse_order_with_patronymic(self, detector):
        text = "Дмитрий Наркисович Мамин-Сибиряк"
        spans = detector._split_per_span(text, 0, len(text))
        assert "PER" not in [s[0] for s in spans]
        assert [(s[0], text[s[1]:s[2]]) for s in spans] == [
            ("NAME", "Дмитрий"),
            ("NAME", "Наркисович"),
            ("SURNAME", "Мамин-Сибиряк"),
        ]
        _verify_clean_boundaries(text, spans)

    def test_saltykov_shchedrin_solo(self, detector):
        text = "Салтыков-Щедрин"
        spans = detector._split_per_span(text, 0, len(text))
        assert spans == [("SURNAME", 0, len(text))]

    def test_saltykov_shchedrin_direct_order(self, detector):
        text = "Салтыков-Щедрин Михаил Евграфович"
        spans = detector._split_per_span(text, 0, len(text))
        assert "PER" not in [s[0] for s in spans]
        assert [(s[0], text[s[1]:s[2]]) for s in spans] == [
            ("SURNAME", "Салтыков-Щедрин"),
            ("NAME", "Михаил"),
            ("NAME", "Евграфович"),
        ]
        _verify_clean_boundaries(text, spans)

    def test_saltykov_shchedrin_reverse_order(self, detector):
        text = "Михаил Салтыков-Щедрин"
        spans = detector._split_per_span(text, 0, len(text))
        assert "PER" not in [s[0] for s in spans]
        assert [(s[0], text[s[1]:s[2]]) for s in spans] == [
            ("NAME", "Михаил"),
            ("SURNAME", "Салтыков-Щедрин"),
        ]
        _verify_clean_boundaries(text, spans)

    def test_saltykov_shchedrin_reverse_with_patronymic(self, detector):
        text = "Михаил Евграфович Салтыков-Щедрин"
        spans = detector._split_per_span(text, 0, len(text))
        assert "PER" not in [s[0] for s in spans]
        assert [(s[0], text[s[1]:s[2]]) for s in spans] == [
            ("NAME", "Михаил"),
            ("NAME", "Евграфович"),
            ("SURNAME", "Салтыков-Щедрин"),
        ]
        _verify_clean_boundaries(text, spans)

    def test_rimsky_korsakov_solo(self, detector):
        text = "Римский-Корсаков"
        spans = detector._split_per_span(text, 0, len(text))
        assert spans == [("SURNAME", 0, len(text))]

    def test_rimsky_korsakov_direct_order(self, detector):
        text = "Римский-Корсаков Николай Андреевич"
        spans = detector._split_per_span(text, 0, len(text))
        assert "PER" not in [s[0] for s in spans]
        assert [(s[0], text[s[1]:s[2]]) for s in spans] == [
            ("SURNAME", "Римский-Корсаков"),
            ("NAME", "Николай"),
            ("NAME", "Андреевич"),
        ]
        _verify_clean_boundaries(text, spans)

    def test_rimsky_korsakov_reverse_order(self, detector):
        text = "Николай Римский-Корсаков"
        spans = detector._split_per_span(text, 0, len(text))
        assert "PER" not in [s[0] for s in spans]
        assert [(s[0], text[s[1]:s[2]]) for s in spans] == [
            ("NAME", "Николай"),
            ("SURNAME", "Римский-Корсаков"),
        ]
        _verify_clean_boundaries(text, spans)

    def test_rimsky_korsakov_reverse_with_patronymic(self, detector):
        text = "Николай Андреевич Римский-Корсаков"
        spans = detector._split_per_span(text, 0, len(text))
        assert "PER" not in [s[0] for s in spans]
        assert [(s[0], text[s[1]:s[2]]) for s in spans] == [
            ("NAME", "Николай"),
            ("NAME", "Андреевич"),
            ("SURNAME", "Римский-Корсаков"),
        ]
        _verify_clean_boundaries(text, spans)


class TestDirectAndReverseOrderArbitration:
    """2. Прямой и обратный порядок следования компонентов ФИО."""

    @pytest.mark.parametrize("first,last", [
        ("Иванов", "Иван"),
        ("Петров", "Петр"),
        ("Сидоров", "Алексей"),
        ("Смирнов", "Сергей"),
    ])
    def test_direct_two_tokens(self, detector, first, last):
        text = f"{first} {last}"
        spans = detector._split_per_span(text, 0, len(text))
        assert "PER" not in [s[0] for s in spans]
        assert [(s[0], text[s[1]:s[2]]) for s in spans] == [
            ("SURNAME", first),
            ("NAME", last),
        ]
        _verify_clean_boundaries(text, spans)

    @pytest.mark.parametrize("first,last", [
        ("Иван", "Иванов"),
        ("Петр", "Петров"),
        ("Алексей", "Сидоров"),
        ("Сергей", "Смирнов"),
    ])
    def test_reverse_two_tokens(self, detector, first, last):
        text = f"{first} {last}"
        spans = detector._split_per_span(text, 0, len(text))
        assert "PER" not in [s[0] for s in spans]
        assert [(s[0], text[s[1]:s[2]]) for s in spans] == [
            ("NAME", first),
            ("SURNAME", last),
        ]
        _verify_clean_boundaries(text, spans)

    @pytest.mark.parametrize("surn,name,patr", [
        ("Иванов", "Иван", "Иванович"),
        ("Петров", "Петр", "Петрович"),
        ("Сидоров", "Алексей", "Сергеевич"),
    ])
    def test_direct_three_tokens(self, detector, surn, name, patr):
        text = f"{surn} {name} {patr}"
        spans = detector._split_per_span(text, 0, len(text))
        assert "PER" not in [s[0] for s in spans]
        assert [(s[0], text[s[1]:s[2]]) for s in spans] == [
            ("SURNAME", surn),
            ("NAME", name),
            ("NAME", patr),
        ]
        _verify_clean_boundaries(text, spans)

    @pytest.mark.parametrize("name,patr,surn", [
        ("Иван", "Иванович", "Иванов"),
        ("Петр", "Петрович", "Петров"),
        ("Алексей", "Сергеевич", "Сидоров"),
    ])
    def test_reverse_three_tokens(self, detector, name, patr, surn):
        text = f"{name} {patr} {surn}"
        spans = detector._split_per_span(text, 0, len(text))
        assert "PER" not in [s[0] for s in spans]
        assert [(s[0], text[s[1]:s[2]]) for s in spans] == [
            ("NAME", name),
            ("NAME", patr),
            ("SURNAME", surn),
        ]
        _verify_clean_boundaries(text, spans)

    def test_four_tokens_direct(self, detector):
        text = "Иванов Иван Иванович Младший"
        spans = detector._split_per_span(text, 0, len(text))
        assert "PER" not in [s[0] for s in spans]
        assert [(s[0], text[s[1]:s[2]]) for s in spans] == [
            ("SURNAME", "Иванов"),
            ("NAME", "Иван"),
            ("NAME", "Иванович"),
            ("NAME", "Младший"),
        ]


class TestSlavicProductiveSuffixes:
    """3. Продуктивные славянские суффиксы: -ко, -енко, -ук, -юк, -як, -ых/-их, -ский/-ская."""

    @pytest.mark.parametrize("surname,name,patr", [
        ("Шевченко", "Тарас", "Григорьевич"),  # -енко
        ("Кличко", "Виталий", "Владимирович"),  # -ко
        ("Бондарчук", "Сергей", "Федорович"),   # -ук
        ("Костюк", "Андрей", "Николаевич"),    # -юк
        ("Поздняк", "Иван", "Михайлович"),     # -як
        ("Черных", "Александр", "Сергеевич"),  # -ых
        ("Долгих", "Владимир", "Иванович"),    # -их
    ])
    def test_slavic_suffix_direct_and_reverse(self, detector, surname, name, patr):
        # 1. Solo surname
        solo_spans = detector._split_per_span(surname, 0, len(surname))
        assert solo_spans == [("SURNAME", 0, len(surname))], f"Solo surname check failed: {surname}"

        # 2. Direct 2-token: "Фамилия Имя"
        text_dir2 = f"{surname} {name}"
        spans_dir2 = detector._split_per_span(text_dir2, 0, len(text_dir2))
        assert [(s[0], text_dir2[s[1]:s[2]]) for s in spans_dir2] == [
            ("SURNAME", surname),
            ("NAME", name),
        ]

        # 3. Reverse 2-token: "Имя Фамилия"
        text_rev2 = f"{name} {surname}"
        spans_rev2 = detector._split_per_span(text_rev2, 0, len(text_rev2))
        assert [(s[0], text_rev2[s[1]:s[2]]) for s in spans_rev2] == [
            ("NAME", name),
            ("SURNAME", surname),
        ]

        # 4. Direct 3-token: "Фамилия Имя Отчество"
        text_dir3 = f"{surname} {name} {patr}"
        spans_dir3 = detector._split_per_span(text_dir3, 0, len(text_dir3))
        assert [(s[0], text_dir3[s[1]:s[2]]) for s in spans_dir3] == [
            ("SURNAME", surname),
            ("NAME", name),
            ("NAME", patr),
        ]

        # 5. Reverse 3-token: "Имя Отчество Фамилия"
        text_rev3 = f"{name} {patr} {surname}"
        spans_rev3 = detector._split_per_span(text_rev3, 0, len(text_rev3))
        assert [(s[0], text_rev3[s[1]:s[2]]) for s in spans_rev3] == [
            ("NAME", name),
            ("NAME", patr),
            ("SURNAME", surname),
        ]


class TestPunctuationNormalization:
    """4. Точная нормализация пунктуации: знаки .,;:!?'()«»—– вне символьных границ спанов."""

    def test_russian_guillemets_and_exclamation(self, detector):
        text = "«Иванов, Иван!»"
        spans = detector._split_per_span(text, 0, len(text))
        assert "PER" not in [s[0] for s in spans]
        assert [(s[0], text[s[1]:s[2]]) for s in spans] == [
            ("SURNAME", "Иванов"),
            ("NAME", "Иван"),
        ]
        _verify_clean_boundaries(text, spans)

    def test_parentheses_and_trailing_period(self, detector):
        text = "(Шевченко Тарас Григорьевич.)"
        spans = detector._split_per_span(text, 0, len(text))
        assert "PER" not in [s[0] for s in spans]
        assert [(s[0], text[s[1]:s[2]]) for s in spans] == [
            ("SURNAME", "Шевченко"),
            ("NAME", "Тарас"),
            ("NAME", "Григорьевич"),
        ]
        _verify_clean_boundaries(text, spans)

    def test_em_dashes_en_dashes_and_quotes(self, detector):
        text = "— Бондарчук Андрей —"
        spans = detector._split_per_span(text, 0, len(text))
        assert "PER" not in [s[0] for s in spans]
        assert [(s[0], text[s[1]:s[2]]) for s in spans] == [
            ("SURNAME", "Бондарчук"),
            ("NAME", "Андрей"),
        ]
        _verify_clean_boundaries(text, spans)

    def test_complex_punctuation_nested_in_sentence(self, detector):
        full_text = "В архиве найдена запись: «Салтыков-Щедрин, Михаил Евграфович!», подтверждающая факт."
        start = full_text.index("«Салтыков-Щедрин, Михаил Евграфович!»")
        end = start + len("«Салтыков-Щедрин, Михаил Евграфович!»")
        spans = detector._split_per_span(full_text, start, end)
        assert "PER" not in [s[0] for s in spans]
        assert [(s[0], full_text[s[1]:s[2]]) for s in spans] == [
            ("SURNAME", "Салтыков-Щедрин"),
            ("NAME", "Михаил"),
            ("NAME", "Евграфович"),
        ]
        _verify_clean_boundaries(full_text, spans)

    def test_semicolon_and_colon_delimiters(self, detector):
        text = "Римский-Корсаков: Николай Андреевич;"
        spans = detector._split_per_span(text, 0, len(text))
        assert [(s[0], text[s[1]:s[2]]) for s in spans] == [
            ("SURNAME", "Римский-Корсаков"),
            ("NAME", "Николай"),
            ("NAME", "Андреевич"),
        ]
        _verify_clean_boundaries(text, spans)


class TestHonestFallbackOnPER:
    """5. Честный fallback на PER: небуквенный мусор, цифры, многоточия, аббревиатуры."""

    @pytest.mark.parametrize("junk", [
        "...",
        "???",
        "---",
        "«»",
        ",,,",
        "!?",
        "123",
        "42",
        "2026",
        "007",
        "ООО",
        "МВД",
        "ЗАО",
        "НИИ",
        "ФСБ",
        "ПАО",
        "ОАО",
        "123...",
        "ООО 123",
        "??? !!!",
        "иванов иван",  # нижний регистр не является валидным антропонимом
        "человек",
        "г.",
    ])
    def test_junk_tokens_fallback_to_per(self, detector, junk):
        text = f"Тестовая строка: {junk} в тексте."
        start = text.index(junk)
        end = start + len(junk)
        spans = detector._split_per_span(text, start, end)
        assert spans == [("PER", start, end)], (
            f"Ожидался честный fallback [('PER', {start}, {end})], получено: {spans}"
        )

    def test_empty_and_whitespace_spans(self, detector):
        text = "   "
        spans = detector._split_per_span(text, 0, len(text))
        assert spans == [("PER", 0, len(text))]

    def test_single_letter_tokens_fallback(self, detector):
        text = "А Б"
        spans = detector._split_per_span(text, 0, len(text))
        assert spans == [("PER", 0, len(text))]


class TestParentPERExclusion:
    """6. Исключение родительского PER: ни в одном случае успешного разделения PER не присутствует."""

    @pytest.mark.parametrize("name_text", [
        "Иванов Иван",
        "Иван Иванов",
        "Иванов Иван Иванович",
        "Иван Иванович Иванов",
        "Мамин-Сибиряк",
        "Дмитрий Мамин-Сибиряк",
        "Мамин-Сибиряк Дмитрий Наркисович",
        "Салтыков-Щедрин",
        "Михаил Салтыков-Щедрин",
        "Римский-Корсаков Николай Андреевич",
        "Тарас Шевченко",
        "Виталий Кличко",
        "Сергей Бондарчук",
        "Андрей Костюк",
        "Иван Поздняк",
        "«Иванов, Иван!»",
        "(Шевченко Тарас Григорьевич.)",
    ])
    def test_no_parent_per_in_output(self, detector, name_text):
        spans = detector._split_per_span(name_text, 0, len(name_text))
        tags = [s[0] for s in spans]
        assert "PER" not in tags, f"Родительский тег PER обнаружен в выдаче для '{name_text}': {spans}"
        assert all(t in ("SURNAME", "NAME") for t in tags), f"Неожиданные теги в выдаче: {tags}"


class TestAdversarialStressScenarios:
    """7. Стресс-тесты: пробелы, повторы слов в контексте, смешанный мусор."""

    def test_internal_multiple_spaces(self, detector):
        text = "Иванов     Иван"
        spans = detector._split_per_span(text, 0, len(text))
        assert [(s[0], text[s[1]:s[2]]) for s in spans] == [
            ("SURNAME", "Иванов"),
            ("NAME", "Иван"),
        ]
        _verify_clean_boundaries(text, spans)

    def test_repeated_tokens_in_context_cursor_accuracy(self, detector):
        text = "Иван пришел к Иван Иванов в гости."
        # целевой спан — второй "Иван Иванов"
        start = text.index("Иван Иванов")
        end = start + len("Иван Иванов")
        spans = detector._split_per_span(text, start, end)
        assert spans == [
            ("NAME", start, start + 4),
            ("SURNAME", start + 5, end),
        ]
        assert text[spans[0][1]:spans[0][2]] == "Иван"
        assert text[spans[1][1]:spans[1][2]] == "Иванов"

    def test_mixed_junk_with_valid_token(self, detector):
        # "ООО Иванов" -> ООО отсекается валидатором, остается одиночный валидный токен Иванов
        text = "ООО Иванов"
        spans = detector._split_per_span(text, 0, len(text))
        assert spans == [("SURNAME", 4, 10)]
        assert text[spans[0][1]:spans[0][2]] == "Иванов"

    def test_idempotence_and_deterministic_output(self, detector):
        text = "Мамин-Сибиряк Дмитрий Наркисович"
        run1 = detector._split_per_span(text, 0, len(text))
        run2 = detector._split_per_span(text, 0, len(text))
        run3 = detector._split_per_span(text, 0, len(text))
        assert run1 == run2 == run3
