"""
Главный модуль синонимического обезличивания ПДн (ЛР2).

Реализует инженерные улучшения:
- LR-ARC-1: Реестр провайдеров Strategy и безопасный fallback для неизвестных тегов.
- LR-ARC-2: Локальный контекст внутридокументной идемпотентности AnonymizationContext (C7, C8).
- LR-ARC-3: Слой верификации и детерминированного NMS спанов sanitize_spans (C3).
- LR-PERF-1: Однопроходная линейная сегментная сборка текста O(N) через chunks и join.
- LR-REL-2: Контекстное разрешение амбивалентного рода Женя/Саша (C6).
- LR-COMP-1: Полноценная поддержка отчеств и масштабирование лексикона (250+ единиц).
- LR-COMP-2: Провайдер дат DATE.
- LR-COMP-3: Правдоподобная структура email и расширенные пулы организаций/топонимов.
"""

from __future__ import annotations

import logging
import random
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from faker import Faker

from lab2.src.morph_utils import (
    get_morph_info,
    inflect_word,
    is_latin_text,
    strip_quotes_and_legal_forms,
    handle_english_casing_and_possessive,
    resolve_ambivalent_gender,
    preserve_text_casing,
    inflect_hyphenated_entity,
)
from lab2.src.span_sanitizer import sanitize_spans
from lab2.src.registry import ProviderRegistry, BaseAnonymizationStrategy
from lab2.src.faker_providers import (
    anonymize_inn,
    anonymize_bank_card,
    anonymize_passport,
    anonymize_ogrnip,
    anonymize_ip,
    anonymize_email,
    anonymize_email_realistic,
    anonymize_birth_certificate,
    anonymize_kladr,
    anonymize_phone,
    anonymize_snils,
    anonymize_url,
    anonymize_marriage_certificate,
    anonymize_cadastral_number,
    anonymize_date,
    anonymize_journal,
    anonymize_address,
    anonymize_org,
    anonymize_location,
    InvalidChecksumPolicy,
)

logger = logging.getLogger(__name__)

_rng = random.SystemRandom()
_faker_en = Faker("en_US")

# ============================================================================
# РАСШИРЕННЫЕ СЛОВАРИ ФИО (250+ фамилий, 200+ имен, 100+ отчеств) (LR-COMP-1, KR-COMP-1)
# ============================================================================

_SURNAMES_M = [
    "Смирнов", "Кузнецов", "Попов", "Васильев", "Соколов", "Михайлов", "Новиков",
    "Фёдоров", "Морозов", "Волков", "Алексеев", "Лебедев", "Семёнов", "Егоров",
    "Павлов", "Николаев", "Марков", "Степанов", "Орлов", "Андреев", "Макаров",
    "Захаров", "Борисов", "Тарасов", "Белов", "Комаров", "Медведев", "Громов",
    "Сафонов", "Богданов", "Воронов", "Гусев", "Титов", "Кузьмин", "Кудрявцев",
    "Баранов", "Куликов", "Алексеев", "Степанов", "Яковлев", "Сорокин", "Сергеев",
    "Романов", "Зайцев", "Пахомов", "Афанасьев", "Филиппов", "Григорьев", "Ковалёв",
    "Ильин", "Щербаков", "Краснов", "Соловьёв", "Власов", "Беляев", "Антонов",
    "Герасимов", "Богданов", "Акимов", "Абрамов", "Горбунов", "Назаров", "Матвеев",
    "Тимофеев", "Осипов", "Лазарев", "Мельников", "Ермаков", "Колесников", "Козлов",
    "Карпов", "Фролов", "Виноградов", "Калинин", "Суворов", "Пономарёв", "Румянцев",
    "Давыдов", "Тихонов", "Мартынов", "Федотов", "Селезнёв", "Петухов", "Белоусов",
    "Гордеев", "Архипов", "Седов", "Устинов", "Мухин", "Коновалов", "Рожков",
    "Кошелев", "Савельев", "Панфилов", "Денисов", "Лукин", "Киселёв", "Дмитриев",
    "Широков", "Логинов", "Рябов", "Дроздов", "Зыков", "Ларионов", "Корнилов",
    "Елисеев", "Орехов", "Самойлов", "Чернов", "Беляков", "Галкин", "Зотов",
    "Гущин", "Бобров", "Миронов", "Туманов", "Князев", "Крюков", "Суханов",
    "Губанов", "Сычёв", "Шаров", "Шубин", "Шестаков", "Третьяков", "Евдокимов",
    "Корнеев", "Нестеров", "Гришин", "Мартынов", "Прохоров", "Аникеев", "Бабушкин",
    "Балашов", "Быков", "Васин", "Веселов", "Вишняков", "Гаврилов", "Гладков",
    "Головин", "Горшков", "Дементьев", "Дорофеев", "Жуков", "Журавлёв", "Игнатов",
    "Казаков", "Капустин", "Калашников", "Кириллов", "Климов", "Князев", "Ковалёв",
    "Колобов", "Кондратьев", "Коршунов", "Королёв", "Котов", "Красильников", "Крылов",
    "Кулагин", "Курбатов", "Лаптев", "Левин", "Лихачёв", "Лобанов", "Мальцев",
    "Маслов", "Матюхин", "Михеев", "Моисеев", "Молчанов", "Муравьёв", "Мясников",
    "Наумов", "Некрасов", "Никитин", "Носков", "Овсянников", "Окулов", "Опарин",
    "Островский", "Панов", "Панкратов", "Пестов", "Пименов", "Платонов", "Плотников",
    "Поздняков", "Поляков", "Потапов", "Радионов", "Рогов", "Рожков", "Русаков",
    "Рыбаков", "Рыжов", "Савин", "Самсонов", "Селивёрстов", "Сидоров", "Симонов",
    "Ситников", "Скворцов", "Смирницкий", "Соболев", "Спиридонов", "Старостин", "Субботин",
    "Суслов", "Сысоев", "Тетерин", "Токарев", "Толмачёв", "Трофимов", "Трусов",
    "Туров", "Уваров", "Ульянов", "Фадеев", "Фокин", "Фомичёв", "Харитонов",
    "Хлебников", "Хохлов", "Цветков", "Чеботарев", "Черкасов", "Чернышёв", "Чистяков",
    "Чумаков", "Шапошников", "Шарапов", "Шевелёв", "Шилов", "Ширяев", "Щукин",
    "Юдин", "Юрков", "Якушев", "Яшин",
]

_SURNAMES_F = [
    s + "а" if not s.endswith(("ов", "ев", "ин", "ын")) else (
        s[:-2] + "ова" if s.endswith("ов") else (
            s[:-2] + "ева" if s.endswith("ев") else (
                s[:-2] + "ина" if s.endswith("ин") else s + "а"
            )
        )
    )
    for s in _SURNAMES_M
]
# Корректировка окончаний -ский -> -ская
_SURNAMES_F = [
    s[:-2] + "ая" if s.endswith("ий") or s.endswith("ый") else (
        s if s.endswith("а") or s.endswith("я") else s + "а"
    )
    for s in _SURNAMES_M
]

_NAMES_M = [
    "Артём", "Максим", "Кирилл", "Дмитрий", "Роман", "Евгений", "Олег",
    "Виктор", "Григорий", "Тимофей", "Павел", "Антон", "Игорь", "Юрий",
    "Фёдор", "Борис", "Геннадий", "Валерий", "Станислав", "Руслан", "Владислав",
    "Леонид", "Вадим", "Константин", "Аркадий", "Глеб", "Марк", "Лев",
    "Давид", "Матвей", "Александр", "Сергей", "Андрей", "Алексей", "Иван",
    "Михаил", "Никита", "Денис", "Даниил", "Владимир", "Николай", "Степан",
    "Илья", "Ярослав", "Георгий", "Анатолий", "Вячеслав", "Василий", "Семён",
    "Пётр", "Филипп", "Артур", "Виталий", "Захар", "Макар", "Платон",
    "Богдан", "Тимур", "Гордей", "Савелий", "Ян", "Тарас", "Ростислав",
    "Эдуард", "Эмиль", "Мирон", "Родион", "Святослав", "Всеволод", "Яков",
    "Валентин", "Герман", "Егор", "Юлиан", "Валерьян", "Демьян", "Лука",
    "Аким", "Климент", "Серафим", "Феликс", "Архип", "Емельян", "Игнат",
    "Кузьма", "Лавр", "Леон", "Мартын", "Назар", "Никанор", "Остап",
    "Потап", "Прохор", "Савва", "Святогор", "Спартак", "Трофим", "Фома",
    "Харитон", "Эрнест", "Юлий", "Яромир",
]

_NAMES_F = [
    "Алина", "Полина", "Ксения", "Виктория", "Марина", "Зинаида", "Тамара",
    "Нина", "Раиса", "Кристина", "Жанна", "Диана", "Любовь", "Милана",
    "Арина", "Варвара", "Ульяна", "София", "Маргарита", "Евгения", "Василиса",
    "Вероника", "Алёна", "Регина", "Инна", "Валерия", "Оксана", "Злата",
    "Лилия", "Карина", "Анна", "Елена", "Ольга", "Татьяна", "Наталья",
    "Ирина", "Светлана", "Екатерина", "Мария", "Анастасия", "Дарья", "Юлия",
    "Александра", "Елизавета", "Надежда", "Вера", "Людмила", "Галина", "Лариса",
    "Яна", "Ангелина", "Виолетта", "Снежана", "Эвелина", "Алиса", "Кира",
    "Ева", "Таисия", "Эльвира", "Мирослава", "Влада", "Ярослава", "Антонина",
    "Клавдия", "Стефания", "Анжелика", "Нонна", "Белла", "Лолита", "Альбина",
    "Ада", "Алевтина", "Алла", "Амалия", "Валентина", "Венера", "Вилена",
    "Владислава", "Глафира", "Гульнара", "Дина", "Доминика", "Зоя", "Илона",
    "Ираида", "Калерия", "Камилла", "Лидия", "Луиза", "Майя", "Марта",
    "Мелания", "Нелли", "Олеся", "Прасковья", "Римма", "Роза", "Руслана",
    "Сабина", "Серафима", "Сильвия", "Сусанна", "Фаина", "Эмма", "Юлиана",
]

_PATRONYMICS_M = [
    "Александрович", "Сергеевич", "Дмитриевич", "Андреевич", "Алексеевич",
    "Иванович", "Михайлович", "Николаевич", "Владимирович", "Артёмович",
    "Максимович", "Кириллович", "Романович", "Евгеньевич", "Олегович",
    "Викторович", "Григорьевич", "Тимофеевич", "Павлович", "Антонович",
    "Игоревич", "Юрьевич", "Фёдорович", "Борисович", "Геннадьевич",
    "Валерьевич", "Станиславович", "Русланович", "Владиславович", "Леонидович",
    "Вадимович", "Константинович", "Аркадьевич", "Глебович", "Маркович",
    "Львович", "Давидович", "Матвеевич", "Никитич", "Денисович",
    "Даниилович", "Степанович", "Ильич", "Ярославович", "Георгиевич",
    "Анатольевич", "Вячеславович", "Васильевич", "Семёнович", "Петрович",
    "Филиппович", "Артурович", "Витальевич", "Захарович", "Макарович",
    "Платонович", "Богданович", "Тимурович", "Гордеевич", "Савельевич",
    "Ростиславович", "Эдуардович", "Миронович", "Родионович", "Святославович",
    "Всеволодович", "Яковлевич", "Валентинович", "Германович", "Егорович",
    "Демьянович", "Лукич", "Акимович", "Серафимович", "Феликсович",
    "Архипович", "Емельянович", "Игнатович", "Кузьмич", "Лаврович",
    "Леонович", "Мартынович", "Назарович", "Никанорович", "Остапович",
    "Потапович", "Прохорович", "Саввич", "Трофимович", "Фомич",
    "Харитонович", "Эрнестович", "Юльевич", "Яромирович", "Борисович",
]

_PATRONYMICS_F: list[str] = []
for p in _PATRONYMICS_M:
    if p.endswith("ович") or p.endswith("евич"):
        _PATRONYMICS_F.append(p[:-2] + "на")
    elif p.endswith("ич"):
        if p in ("Ильич", "Кузьмич", "Лукич", "Фомич", "Саввич"):
            _PATRONYMICS_F.append(p[:-2] + "инична")
        else:
            _PATRONYMICS_F.append(p[:-2] + "ична")
    else:
        _PATRONYMICS_F.append(p + "на")


def _pick_random_name(gender: str | None, is_surname: bool, exclude: str | None = None) -> str:
    """Выбрать случайное имя или фамилию по роду с исключением исходного значения."""
    if is_surname:
        pool = _SURNAMES_F if gender == "femn" else _SURNAMES_M
    else:
        pool = _NAMES_F if gender == "femn" else _NAMES_M
    if exclude:
        ex_low = exclude.lower()
        filtered = [x for x in pool if x.lower() != ex_low]
        if filtered:
            return _rng.choice(filtered)
    return _rng.choice(pool)


def _pick_random_patronymic(gender: str | None, exclude: str | None = None) -> str:
    """Выбрать случайное отчество по роду (LR-COMP-1)."""
    pool = _PATRONYMICS_F if gender == "femn" else _PATRONYMICS_M
    if exclude:
        ex_low = exclude.lower()
        filtered = [x for x in pool if x.lower() != ex_low]
        if filtered:
            return _rng.choice(filtered)
    return _rng.choice(pool)


# ============================================================================
# LR-ARC-2: КОНТЕКСТ ВНУТРИДОКУМЕНТНОЙ ИДЕМПОТЕНТНОСТИ (AnonymizationContext)
# ============================================================================

class AnonymizationContext:
    """
    Контекст локальной ссылочной целостности и внутридокументной идемпотентности (LR-ARC-2).
    Гарантирует:
    1. Одинаковые сущности в рамках одного документа заменяются одинаково (C7).
    2. Разные вызовы для разных документов получают изолированные генерации (C8).
    3. Нормализованная лемма сохраняет базовую форму в именительном падеже (C6).
    """

    def __init__(self):
        # (tag, canonical_lemma) -> (base_nom, gender)
        self._name_cache: Dict[Tuple[str, str], Tuple[str, str]] = {}
        # (tag, normalized_digits_or_key) -> replacement
        self._structured_cache: Dict[Tuple[str, str], str] = {}
        # Хранилище сущностей ФИО из 3-х компонентов: (tag, full_lemma) -> (sur_nom, name_nom, patr_nom, gender)
        self._fio_cache: Dict[Tuple[str, str], Tuple[str, str, str, str]] = {}

    def get_or_create_fio_triad(
        self,
        full_lemma: str,
        target_gender: str,
        exclude_triad: Optional[Tuple[str, str, str]] = None,
    ) -> Tuple[str, str, str, str]:
        key = ("PER_TRIAD", full_lemma)
        if key not in self._fio_cache:
            ex_sur = exclude_triad[0] if exclude_triad else None
            ex_nam = exclude_triad[1] if exclude_triad else None
            ex_pat = exclude_triad[2] if exclude_triad else None
            sur = _pick_random_name(target_gender, is_surname=True, exclude=ex_sur)
            nam = _pick_random_name(target_gender, is_surname=False, exclude=ex_nam)
            pat = _pick_random_patronymic(target_gender, exclude=ex_pat)
            self._fio_cache[key] = (sur, nam, pat, target_gender)
        return self._fio_cache[key]

    def get_or_create_name(
        self,
        tag: str,
        lemma: str,
        is_surname: bool,
        target_gender: str,
        exclude: Optional[str] = None,
    ) -> str:
        key = (tag, lemma)
        if key not in self._name_cache:
            fake_nom = _pick_random_name(target_gender, is_surname=is_surname, exclude=exclude or lemma)
            self._name_cache[key] = (fake_nom, target_gender)
        return self._name_cache[key][0]

    def get_or_create_structured(
        self,
        tag: str,
        norm_key: str,
        generator_fn,
        *args,
        **kwargs,
    ) -> str:
        key = (tag, norm_key)
        if key not in self._structured_cache:
            self._structured_cache[key] = generator_fn(*args, **kwargs)
        return self._structured_cache[key]


# ============================================================================
# ОБЕЗЛИЧИВАНИЕ ФИО С МОРФОЛОГИЕЙ И ПОДДЕРЖКОЙ ОТЧЕСТВ (LR-COMP-1, LR-REL-2)
# ============================================================================

def _is_patronymic(word: str) -> bool:
    """Определить, является ли слово отчеством по суффиксам или граммемам."""
    low = word.lower()
    if low.endswith(("ович", "евич", "ич", "овна", "евна", "ична", "инична")):
        return True
    info = get_morph_info(word)
    parsed = info.get("parsed")
    if parsed and "Patr" in parsed.tag:
        return True
    return False


def _anonymize_person_name(
    original: str,
    tag: str,
    context: Optional[AnonymizationContext] = None,
    text: str = "",
    start: int = 0,
    end: int = 0,
) -> str:
    """Обезличить ФИО с сохранением морфологии, отчеств и билингвальной маршрутизацией."""
    if is_latin_text(original):
        if tag == "SURNAME":
            fake_val = _faker_en.last_name()
        elif tag == "NAME":
            fake_val = _faker_en.first_name()
        else:
            fake_val = f"{_faker_en.first_name()} {_faker_en.last_name()}"
        return handle_english_casing_and_possessive(original, fake_val)

    words = original.split()

    # 1. Трехкомпонентное ФИО: Фамилия Имя Отчество (LR-COMP-1: устранение бага 'Смирнов Артём Кирилл')
    if len(words) == 3 and tag in ("PER", "FIO"):
        w1_gender = resolve_ambivalent_gender(words[1], text, start, end)
        w0_info = get_morph_info(words[0])
        w2_low = words[2].lower()
        if w0_info.get("gender") == "femn" or w2_low.endswith(("на", "ична", "инична")) or w1_gender == "femn":
            gender = "femn"
        elif w0_info.get("gender") == "masc" or w2_low.endswith(("ич", "ович", "евич")) or w1_gender == "masc":
            gender = "masc"
        else:
            gender = w1_gender
        full_lemma = " ".join(get_morph_info(w)["normal_form"] for w in words).lower()

        if context:
            sur_nom, nam_nom, pat_nom, gender = context.get_or_create_fio_triad(
                full_lemma, gender, exclude_triad=(words[0], words[1], words[2])
            )
        else:
            sur_nom = _pick_random_name(gender, is_surname=True, exclude=words[0])
            nam_nom = _pick_random_name(gender, is_surname=False, exclude=words[1])
            pat_nom = _pick_random_patronymic(gender, exclude=words[2])

        res_parts = []
        for orig_w, fake_nom in zip(words, (sur_nom, nam_nom, pat_nom)):
            w_info = get_morph_info(orig_w)
            c = w_info["case"]
            if c and c != "nomn":
                inflected = inflect_word(fake_nom, c, gender, w_info["number"])
            else:
                inflected = fake_nom
            res_parts.append(preserve_text_casing(orig_w, inflected))

        return " ".join(res_parts)

    # 2. Двухкомпонентное ФИО: "Иванов Иван" или "Иван Иванов"
    if len(words) == 2 and tag in ("PER", "FIO"):
        gender = resolve_ambivalent_gender(words[0], text, start, end)
        w0_info = get_morph_info(words[0])
        w1_info = get_morph_info(words[1])

        # Определяем порядок
        w0_is_surn = any(words[0].lower().endswith(s) for s in ("ов", "ова", "ев", "ева", "ин", "ина", "ский", "ская", "цкий", "цкая"))
        w1_is_patr = _is_patronymic(words[1])

        res_parts = []
        for idx, w in enumerate(words):
            if w1_is_patr and idx == 1:
                fake_nom = _pick_random_patronymic(gender, exclude=w)
            else:
                is_sur = w0_is_surn if idx == 0 else (not w0_is_surn)
                fake_nom = _pick_random_name(gender, is_surname=is_sur, exclude=w)

            w_info = get_morph_info(w)
            c = w_info["case"]
            inflected = inflect_word(fake_nom, c, gender, w_info["number"]) if (c and c != "nomn") else fake_nom
            res_parts.append(preserve_text_casing(w, inflected))

        return " ".join(res_parts)

    # 2.5. Дефисная фамилия (например, "Петров-Водкин", "Мамин-Сибиряк")
    if "-" in original and tag in ("SURNAME", "PER"):
        first_part = original.split("-")[0]
        info = get_morph_info(original)
        gender = resolve_ambivalent_gender(first_part, text, start, end)
        lemma = "-".join(get_morph_info(p)["normal_form"] for p in original.split("-")).lower()
        if context:
            sur1 = context.get_or_create_name(tag, lemma + ":1", True, gender, exclude=original.split("-")[0])
            sur2 = context.get_or_create_name(tag, lemma + ":2", True, gender, exclude=original.split("-")[-1])
        else:
            sur1 = _pick_random_name(gender, is_surname=True, exclude=original.split("-")[0])
            sur2 = _pick_random_name(gender, is_surname=True, exclude=original.split("-")[-1])
            while sur2 == sur1:
                sur2 = _pick_random_name(gender, is_surname=True, exclude=original.split("-")[-1])
        fake_nom = f"{sur1}-{sur2}"
        c = info["case"]
        inflected = inflect_hyphenated_entity(fake_nom, c, gender) if (c and c != "nomn") else fake_nom
        return preserve_text_casing(original, inflected)

    # 3. Одиночное слово (SURNAME, NAME или PER)
    info = get_morph_info(original)
    gender = resolve_ambivalent_gender(original, text, start, end)
    lemma = info["normal_form"].lower()
    is_surname = (tag == "SURNAME") or (
        tag == "PER" and any(lemma.endswith(s) for s in ("ов", "ова", "ев", "ева", "ин", "ина", "ский", "ская", "цкий", "цкая"))
    )

    if _is_patronymic(original):
        fake_nom = _pick_random_patronymic(gender, exclude=original)
    elif context:
        fake_nom = context.get_or_create_name(tag, lemma, is_surname, gender, exclude=original)
    else:
        fake_nom = _pick_random_name(gender, is_surname, exclude=original)

    c = info["case"]
    n = info["number"]
    if c and c != "nomn":
        inflected = inflect_word(fake_nom, c, gender, n)
    else:
        inflected = fake_nom

    return preserve_text_casing(original, inflected)


def _anonymize_org_entity(original: str, context: Optional[AnonymizationContext] = None) -> str:
    """Обезличить организацию с сохранением кавычек, ОПФ и падежей."""
    if is_latin_text(original):
        fake = anonymize_org(original, lang="en")
        return handle_english_casing_and_possessive(original, fake)

    prefix, l_q, core, r_q = strip_quotes_and_legal_forms(original)
    norm_key = core.strip().lower()

    if context:
        fake_core = context.get_or_create_structured("ORG", norm_key, anonymize_org, core, lang="ru")
    else:
        fake_core = anonymize_org(core, lang="ru")

    if prefix:
        fake_core = re.sub(r'^(?:ООО|ПАО|АО|ИП)\s*', '', fake_core).strip()

    res = f"{prefix}{l_q}{fake_core}{r_q}"
    return preserve_text_casing(original, res)


def _anonymize_loc_entity(original: str, context: Optional[AnonymizationContext] = None) -> str:
    """Обезличить топоним с сохранением падежей (KR-REL-2)."""
    if is_latin_text(original):
        if context:
            fake = context.get_or_create_structured("LOC_EN", original.strip().lower(), anonymize_location, original, lang="en")
        else:
            fake = anonymize_location(original, lang="en")
        return handle_english_casing_and_possessive(original, fake)

    info = get_morph_info(original)
    norm_key = info["normal_form"].lower()

    if context:
        fake_loc_nom = context.get_or_create_structured("LOC_NOM", norm_key, anonymize_location, original, lang="ru")
    else:
        fake_loc_nom = anonymize_location(original, lang="ru")

    if info["case"] and info["case"] != "nomn":
        inflected = inflect_word(fake_loc_nom, info["case"], info["gender"], info.get("number"))
    else:
        inflected = fake_loc_nom

    return preserve_text_casing(original, inflected)


def _anonymize_addr_entity(original: str, context: Optional[AnonymizationContext] = None) -> str:
    lang = "en" if is_latin_text(original) else "ru"
    norm_key = "".join(original.split()).lower()
    if context:
        return context.get_or_create_structured("ADDR", norm_key, anonymize_address, original, lang=lang)
    return anonymize_address(original, lang=lang)


# ============================================================================
# РЕГИСТРАЦИЯ ОБРАБОТЧИКОВ В ProviderRegistry (LR-ARC-1)
# ============================================================================

_HANDLERS = {
    "PER": lambda text, ctx=None, full_text="", s=0, e=0: _anonymize_person_name(text, "PER", ctx, full_text, s, e),
    "SURNAME": lambda text, ctx=None, full_text="", s=0, e=0: _anonymize_person_name(text, "SURNAME", ctx, full_text, s, e),
    "NAME": lambda text, ctx=None, full_text="", s=0, e=0: _anonymize_person_name(text, "NAME", ctx, full_text, s, e),
    "ORG": lambda text, ctx=None: _anonymize_org_entity(text, ctx),
    "LOC": lambda text, ctx=None: _anonymize_loc_entity(text, ctx),
    "ADDR": lambda text, ctx=None: _anonymize_addr_entity(text, ctx),
    "INN": lambda text, ctx=None: ctx.get_or_create_structured("INN", "".join(c for c in text if c.isdigit()), anonymize_inn, text) if ctx else anonymize_inn(text),
    "BANK_CARD": lambda text, ctx=None: ctx.get_or_create_structured("BANK_CARD", "".join(c for c in text if c.isdigit()), anonymize_bank_card, text) if ctx else anonymize_bank_card(text),
    "PASSPORT": lambda text, ctx=None: ctx.get_or_create_structured("PASSPORT", "".join(c for c in text if c.isdigit()), anonymize_passport, text) if ctx else anonymize_passport(text),
    "OGRNIP": lambda text, ctx=None: ctx.get_or_create_structured("OGRNIP", "".join(c for c in text if c.isdigit()), anonymize_ogrnip, text) if ctx else anonymize_ogrnip(text),
    "IP_ADDRESS": lambda text, ctx=None: ctx.get_or_create_structured("IP_ADDRESS", text.strip(), anonymize_ip, text) if ctx else anonymize_ip(text),
    "EMAIL": lambda text, ctx=None: ctx.get_or_create_structured("EMAIL", text.strip().lower(), anonymize_email, text) if ctx else anonymize_email(text),
    "BIRTH_CERTIFICATE": lambda text, ctx=None: ctx.get_or_create_structured("BIRTH_CERTIFICATE", text.strip(), anonymize_birth_certificate, text) if ctx else anonymize_birth_certificate(text),
    "KLADR": lambda text, ctx=None: ctx.get_or_create_structured("KLADR", "".join(c for c in text if c.isdigit()), anonymize_kladr, text) if ctx else anonymize_kladr(text),
    "PHONE_NUMBER": lambda text, ctx=None: ctx.get_or_create_structured("PHONE_NUMBER", "".join(c for c in text if c.isdigit()), anonymize_phone, text) if ctx else anonymize_phone(text),
    "SNILS": lambda text, ctx=None: ctx.get_or_create_structured("SNILS", "".join(c for c in text if c.isdigit()), anonymize_snils, text) if ctx else anonymize_snils(text),
    "URL": lambda text, ctx=None: anonymize_url(text),
    "MARRIAGE_CERTIFICATE": lambda text, ctx=None: ctx.get_or_create_structured("MARRIAGE_CERTIFICATE", text.strip(), anonymize_marriage_certificate, text) if ctx else anonymize_marriage_certificate(text),
    "CADASTRAL_NUMBER": lambda text, ctx=None: anonymize_cadastral_number(text),
    "DATE": lambda text, ctx=None: ctx.get_or_create_structured("DATE", text.strip().lower(), anonymize_date, text) if ctx else anonymize_date(text),
    "JOURNAL_ODD": lambda text, ctx=None: anonymize_journal(text, True),
    "JOURNAL_EVEN": lambda text, ctx=None: anonymize_journal(text, False),
}

# Регистрируем все обработчики в глобальном ProviderRegistry
for _t, _h in _HANDLERS.items():
    ProviderRegistry.set_handler(_t, _h, strategy_name="synonymic")


# ============================================================================
# LR-PERF-1: ЛИНЕЙНАЯ СЕГМЕНТНАЯ СБОРКА ТЕКСТА O(N)
# ============================================================================

def assemble_anonymized_text(
    text: str,
    sorted_spans: List[Tuple[str, int, int]],
    replacements: List[str],
) -> str:
    """
    Линейная сборка строки O(N) вместо многократной конкатенации срезов (LR-PERF-1).
    Спаны должны быть отсортированы по начальному индексу start.
    """
    chunks: List[str] = []
    cursor = 0

    for (tag, start, end), replacement in zip(sorted_spans, replacements):
        if start > cursor:
            chunks.append(text[cursor:start])
        chunks.append(replacement)
        cursor = end

    if cursor < len(text):
        chunks.append(text[cursor:])

    return "".join(chunks)


# ============================================================================
# ГЛАВНАЯ ТОЧКА ВХОДА ЛР2 (Инвариант C1: anonymize_text(text, pii_spans) -> str)
# ============================================================================

def anonymize_text(
    text: str,
    pii_spans: list[tuple[str, int, int] | list | dict],
    strategy_name: str = "synonymic",
) -> str:
    """
    Обезличить текст по списку спанов ПДн с предварительной санитарией (LR-ARC-3),
    внутридокументной идемпотентностью (LR-ARC-2), линейной сборкой O(N) (LR-PERF-1)
    и безопасным реестром обработчиков (LR-ARC-1).

    Args:
        text: исходный текст.
        pii_spans: список спанов [(tag, start, end), ...].
        strategy_name: стратегия обезличивания ('synonymic').

    Returns:
        Обезличенный текст.
    """
    if not text or not pii_spans:
        return text

    # Слой санитарии: проверка границ, типов и устранение наложений (Инвариант C3)
    sanitized_spans = sanitize_spans(pii_spans, len(text))
    if not sanitized_spans:
        return text

    # Сортировка спанов по возрастанию начальной координаты для линейной сборки O(N)
    sorted_spans = sorted(sanitized_spans, key=lambda s: s[1])

    # Инициализация локального контекста идемпотентности на сессию документа (LR-ARC-2)
    doc_context = AnonymizationContext()

    replacements: List[str] = []
    for tag, start, end in sorted_spans:
        orig_val = text[start:end]
        handler = ProviderRegistry.get_handler(tag, strategy_name=strategy_name)

        try:
            # Передаем контекст документа и координаты
            if tag in ("PER", "SURNAME", "NAME"):
                rep = handler(orig_val, ctx=doc_context, full_text=text, s=start, e=end)
            elif tag in ("ORG", "LOC", "ADDR", "INN", "BANK_CARD", "PASSPORT", "OGRNIP", "IP_ADDRESS", "EMAIL", "BIRTH_CERTIFICATE", "KLADR", "PHONE_NUMBER", "SNILS", "MARRIAGE_CERTIFICATE", "DATE"):
                rep = handler(orig_val, ctx=doc_context)
            else:
                rep = handler(orig_val)
        except TypeError:
            rep = handler(orig_val)

        replacements.append(rep)

    return assemble_anonymized_text(text, sorted_spans, replacements)
