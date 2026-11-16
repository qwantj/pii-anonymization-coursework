"""
Генератор синтетических данных NER по 152-ФЗ для обучения ai-forever/ru-en-RoSBERTa.
Генерирует 20 000 примеров по 6 категориям + 2 500 Hard Negatives.
"""

from __future__ import annotations

import random
import re
from typing import Generator

import pymorphy3
from faker import Faker

from data_pipeline.schema import LABEL2ID, validate_iob2_sequence, repair_iob2_sequence

_morph = pymorphy3.MorphAnalyzer()
_faker_ru = Faker("ru_RU")
_faker_en = Faker("en_US")
_rng = random.Random(42)

CASES = ["nomn", "gent", "datv", "accs", "ablt", "loct"]

# -----------------------------------------------------------------------------
# Базовые словари сущностей
# -----------------------------------------------------------------------------

FIRST_NAMES_M = [
    "Александр", "Дмитрий", "Максим", "Сергей", "Андрей", "Алексей", "Артём",
    "Илья", "Кирилл", "Михаил", "Никита", "Матвей", "Роман", "Егор", "Арсений",
    "Иван", "Денис", "Евгений", "Даниил", "Тимофей", "Владислав", "Игорь", "Владимир",
    "Павел", "Руслан", "Константин", "Олег", "Виктор", "Юрий", "Григорий",
]

FIRST_NAMES_F = [
    "Анастасия", "Мария", "Дарья", "Анна", "Полина", "Виктория", "Екатерина",
    "Ксения", "Александра", "Елизавета", "Алиса", "София", "Валерия", "Кристина",
    "Ольга", "Юлия", "Татьяна", "Елена", "Наталья", "Ирина", "Светлана", "Марина",
    "Алина", "Вероника", "Диана", "Евгения", "Олеся", "Яна", "Надежда", "Любовь",
]

LAST_NAMES_M = [
    "Иванов", "Смирнов", "Кузнецов", "Попов", "Васильев", "Петров", "Соколов",
    "Михайлов", "Новиков", "Фёдоров", "Морозов", "Волков", "Алексеев", "Лебедев",
    "Семёнов", "Егоров", "Павлов", "Козлов", "Степанов", "Николаев", "Орлов",
    "Андреев", "Макаров", "Никитин", "Захаров", "Зайцев", "Соловьёв", "Борисов",
]

LAST_NAMES_F = [
    "Иванова", "Смирнова", "Кузнецова", "Попова", "Васильева", "Петрова", "Соколова",
    "Михайлова", "Новикова", "Фёдорова", "Морозова", "Волкова", "Алексеева", "Лебедева",
    "Семёнова", "Егорова", "Павлова", "Козлова", "Степанова", "Николаева", "Орлова",
    "Андреева", "Макарова", "Никитина", "Захарова", "Зайцева", "Соловьёва", "Борисова",
]

DOUBLE_LAST_NAMES = [
    ("Салтыков-Щедрин", "masc"),
    ("Мамин-Сибиряк", "masc"),
    ("Соколова-Михайлова", "femn"),
    ("Петрова-Водкина", "femn"),
    ("Мусин-Пушкин", "masc"),
    ("Римский-Корсаков", "masc"),
    ("Оболенская-Ильина", "femn"),
]

MID_NAMES_M = [
    "Александрович", "Дмитриевич", "Сергеевич", "Андреевич", "Алексеевич",
    "Иванович", "Михайлович", "Николаевич", "Владимирович", "Павлович",
]

MID_NAMES_F = [
    "Александровна", "Дмитриевна", "Сергеевна", "Андреевна", "Алексеевна",
    "Ивановна", "Михайловна", "Николаевна", "Владимировна", "Павловна",
]

COMPANIES_RU = [
    "ООО «Вектор»", "ПАО «Сбербанк»", "АО «Тинькофф Банк»", "ООО «Газпром трансгаз»",
    "ПАО «НК «Роснефть»", "АО «Альфа-Банк»", "ООО «Яндекс Технологии»",
    "ПАО «ГМК «Норильский никель»", "АО «Лаборатория Касперского»", "ООО «Мэйл.Ру Финанс»",
    "ПАО «МегаФон»", "ПАО «МТС»", "ПАО «ВымпелКом»", "АО «СОГАЗ»", "ООО «Озон Холдинг»",
    "ООО «Вайлдберриз»", "АО «Почта России»", "ПАО «Аэрофлот»", "ООО «РТК-ИТ»",
    "ИП Смирнов А.В.", "ИП Кузнецова Е.Н.", "ИП Попов М.С.", "ИП Лебедев И.Д.",
]

COMPANIES_EN = [
    "Apple Inc.", "Microsoft Corp.", "Google LLC", "Amazon.com Inc.",
    "Tesla Inc.", "Oracle Corporation", "Meta Platforms Inc.", "IBM Corporation",
    "Intel Corp.", "Cisco Systems", "Salesforce Inc.", "Adobe Systems",
]

CITIES_RU = [
    "Москва", "Санкт-Петербург", "Новосибирск", "Екатеринбург", "Казань",
    "Нижний Новгород", "Челябинск", "Самара", "Омск", "Ростов-на-Дону",
    "Уфа", "Красноярск", "Воронеж", "Пермь", "Волгоград", "Краснодар",
    "Саратов", "Тюмень", "Тольятти", "Ижевск", "Барнаул", "Иркутск",
]

CITIES_EN = [
    "New York", "London", "San Francisco", "Chicago", "Boston", "Los Angeles",
    "Berlin", "Paris", "Toronto", "Sydney", "Singapore", "Dublin",
]

STREETS_RU = [
    "ул. Тверская", "Ленинский проспект", "ул. Арбат", "Невский проспект",
    "ул. Малая Бронная", "Кутузовский проспект", "ул. Баумана", "ул. Красная",
    "ул. Гагарина", "ул. Пушкина", "ул. Советская", "пр-т Мира", "ул. Кирова",
    "ул. Ленина", "ул. Мира", "ул. Садовая", "ул. Набережная", "ул. Комсомольская",
]


def _inflect_word(word: str, case: str) -> str:
    """Просклонять слово в заданный падеж через pymorphy3."""
    parses = _morph.parse(word)
    if not parses:
        return word
    inflected = parses[0].inflect({case})
    return inflected.word.capitalize() if inflected else word


def _generate_fio(case: str = "nomn", format_type: str = "full") -> tuple[str, str]:
    """Сгенерировать ФИО в заданном падеже и формате. Возвращает (текст, род)."""
    gender = _rng.choice(["masc", "femn"])

    if format_type == "double":
        cand, gen = _rng.choice(DOUBLE_LAST_NAMES)
        surname = cand
        name = _rng.choice(FIRST_NAMES_M if gen == "masc" else FIRST_NAMES_F)
        mid = _rng.choice(MID_NAMES_M if gen == "masc" else MID_NAMES_F)
        gender = gen
    else:
        surname = _rng.choice(LAST_NAMES_M if gender == "masc" else LAST_NAMES_F)
        name = _rng.choice(FIRST_NAMES_M if gender == "masc" else FIRST_NAMES_F)
        mid = _rng.choice(MID_NAMES_M if gender == "masc" else MID_NAMES_F)

    # Склонение
    if case != "nomn":
        surname = _inflect_word(surname, case)
        name = _inflect_word(name, case)
        mid = _inflect_word(mid, case)

    if format_type == "full":
        return f"{surname} {name} {mid}", gender
    elif format_type == "name_surname":
        return f"{name} {surname}", gender
    elif format_type == "initials_last":
        return f"{surname} {name[0]}. {mid[0]}.", gender
    elif format_type == "initials_first":
        return f"{name[0]}. {mid[0]}. {surname}", gender
    elif format_type == "double":
        return f"{surname} {name[0]}. {mid[0]}.", gender
    else:
        return f"{surname} {name}", gender


def _generate_address() -> tuple[str, str, str]:
    """Сгенерировать почтовый адрес. Возвращает (индекс, город, адрес_улицы)."""
    zip_code = f"{_rng.randint(100000, 699999)}"
    city = _rng.choice(CITIES_RU)
    street = _rng.choice(STREETS_RU)
    house = _rng.randint(1, 150)
    building = f", корп. {_rng.randint(1, 5)}" if _rng.random() > 0.6 else ""
    apt = f", кв. {_rng.randint(1, 300)}" if _rng.random() > 0.4 else ""
    street_addr = f"{street}, д. {house}{building}{apt}"
    return zip_code, city, street_addr


# -----------------------------------------------------------------------------
# 150+ шаблонов по 6 категориям
# -----------------------------------------------------------------------------

TEMPLATES_LEGAL = [
    "Договор подряда № {num} заключен между {org} и гражданином {fio_gent}, проживающим по адресу: {zip}, г. {city}, {addr}.",
    "Настоящая доверенность выдана {fio_datv}, паспорт гражданина РФ {passport}, зарегистрированному по адресу: {zip}, г. {city}, {addr}.",
    "Акт приема-передачи оказанных услуг подписан со стороны заказчика {org} в лице генерального директора {fio_gent}.",
    "Стороны соглашения: {org} (Исполнитель) и {fio_nomn} (Заказчик, адрес регистрации: г. {city}, {addr}).",
    "Претензия направлена в адрес {org} от имени {fio_gent}, проживающего в г. {city} по адресу {addr}.",
    "Дополнительное соглашение к трудовому договору с сотрудником {fio_ablt} подписано в офисе {org}.",
    "Арбитражный суд города {city} рассмотрел иск {org} к индивидуальному предпринимателю {fio_datv}.",
    "Право подписи финансовых документов предоставлено главному бухгалтеру {fio_datv} приказом по {org}.",
    "Уведомление о расторжении контракта вручено {fio_datv} представителем компании {org} в г. {city}.",
    "Справка выдана {fio_datv} для предъявления по месту требования по месту жительства: г. {city}, {addr}.",
    "Нотариус города {city} удостоверил подлинность подписи гражданина {fio_gent}.",
    "Согласие на обработку персональных данных предоставлено {fio_ablt} в пользу организации {org}.",
    "В соответствии с решением общего собрания акционеров {org}, генеральным директором избран {fio_nomn}.",
    "Договор купли-продажи недвижимости заключен между продавцом {fio_ablt} и покупателем {fio_ablt} в г. {city}.",
    "Судебный пристав-исполнитель возбудил исполнительное производство в отношении должника {fio_gent}.",
]

TEMPLATES_BANKING_TAX_HR = [
    "Заявление на открытие текущего счета в {org} от клиента {fio_gent}, ИНН {inn}, проживающего: {zip}, г. {city}, {addr}.",
    "Анкета заемщика: заявитель {fio_nomn}, паспорт серии {passport_series} номер {passport_num}, выдан ОВД г. {city}, СНИЛС {snils}.",
    "Справка о доходах по форме 2-НДФЛ сотрудника {fio_gent}, табельный номер {tab_num}, выдана работодателем {org}.",
    "Платежное поручение № {num} на перевод средств в пользу ИП {fio_gent}, ОГРНИП {ogrnip}, ИНН {inn}.",
    "Уведомление налогового органа г. {city} направлено налогоплательщику {fio_datv} по адресу {addr}.",
    "Заявка на выпуск зарплатной карты {card_num} для работника {fio_gent} передана в {org}.",
    "Трудовая книжка работника {fio_gent} хранится в отделе кадров предприятия {org}.",
    "Личная карточка работника формы Т-2: ФИО {fio_nomn}, адрес фактического проживания: г. {city}, {addr}.",
    "Удержание по исполнительному листу из заработной платы {fio_gent} перечислено на расчетный счет банка {org}.",
    "Клиент {fio_nomn} подал распоряжение на закрытие вклада и перевод остатка в {org}.",
    "Поручение брокеру {org} на совершение сделки с ценными бумагами отдано клиентом {fio_ablt}.",
    "Регистрация в качестве плательщика налога на профессиональный доход оформлена на гражданина {fio_accs}.",
    "В налоговую инспекцию г. {city} представлена декларация 3-НДФЛ от имени {fio_gent}.",
    "Кредитный договор № {num} между {org} и заемщиком {fio_ablt} на сумму {amount} рублей.",
    "Анкета соискателя на должность ведущего специалиста в {org}: кандидат {fio_nomn}, контакты: тел. {phone}.",
]

TEMPLATES_MEDICAL = [
    "Медицинская карта амбулаторного больного № {num}: пациент {fio_nomn}, проживает: г. {city}, {addr}.",
    "Консультативное заключение врача-кардиолога поликлиники {org}: осмотрен больной {fio_nomn}, 1985 г.р.",
    "Направление на госпитализацию в клиническую больницу г. {city} выдано лечащим врачом {fio_ablt}.",
    "Эпикриз из истории болезни пациента {fio_gent}: поступил по скорой помощи с адреса {addr}.",
    "Листок нетрудоспособности № {num} выдан гражданке {fio_datv} медицинским центром {org}.",
    "Результаты лабораторных исследований пациента {fio_gent} заверены заведующим лабораторией {fio_ablt}.",
    "Рецептурный бланк выписан врачом {fio_ablt} для пациента {fio_gent}, адрес: г. {city}, {addr}.",
    "Протокол ультразвукового исследования больного {fio_gent} в диагностическом отделении клиники {org}.",
    "Выписка из стационара № {num}: пациентка {fio_nomn} выписана в удовлетворительном состоянии в г. {city}.",
    "Справка о прохождении предварительного медицинского осмотра выдана {fio_datv} для работы в {org}.",
]

TEMPLATES_TECH_LOGS_CSV = [
    '{{"timestamp": "{iso_time}", "user": "{fio_nomn}", "company": "{org}", "location": "{city}", "ip": "{ip}", "status": "200 OK"}}',
    'audit_log: user "{fio_nomn}" modified customer record at address "{addr}", city "{city}", organization "{org}".',
    '{id},{fio_nomn},{org},{city},{addr},{phone}',
    'POST /api/v2/users/create payload: {{"fullName": "{fio_nomn}", "office": "{city}", "branch": "{org}", "registeredAddr": "{addr}"}}',
    'DB_DUMP: INSERT INTO users (name, company, city, address) VALUES (\'{fio_nomn}\', \'{org}\', \'{city}\', \'{addr}\');',
    'SecurityAlert: Suspicious login for user account "{fio_nomn}" from IP address {ip} in {city}.',
    'ExportCSV row: "{num}";"{fio_nomn}";"{org}";"{city}";"{addr}";"{email}"',
    'BillingTransaction: customer="{fio_nomn}", merchant="{org}", city="{city}", ref="{num}".',
    'CRM_SYNC: Contact updated by manager {fio_nomn} in branch {org} at {city}.',
    'Syslog [Auth]: User {fio_nomn} logged out from terminal session in {org} head office.',
]

TEMPLATES_INFORMAL_CHATS = [
    "здравствуйте меня зовут {fio_lower} живу в городе {city_lower} на {addr_lower} хочу оформить доставку",
    "привет подскажите где забрать заказ на имя {fio_lower} я из города {city_lower}",
    "добрый день компания {org_lower} может прислать курьера по адресу {addr_lower} для {fio_lower_gent}",
    "напишите пожалуйста на почту {email} клиенту {fio_lower_datv} по поводу заявки в {org_lower}",
    "меня зовут {fio_lower} мой адрес {addr_lower} в городе {city_lower} когда приедет мастер",
    "здравствуйте переведите пожалуйста оператора на {fio_lower_accs} из филиала в {city_lower}",
    "отправьте квитанцию на имя {fio_lower_gent} проживающего по адресу {addr_lower}",
    "алло это {fio_lower} я оставлял обращение в поддержку {org_lower} вчера",
    "подтверждаю заявку от гражданина {fio_lower_gent} адрес проживания {city_lower} {addr_lower}",
    "пишу от имени {fio_lower_gent} договор оформлен на компанию {org_lower}",
]

TEMPLATES_MIXED_RU_EN = [
    "International invoice № {num}: Vendor {org_en} (Silicon Valley), Client: {fio_ru_nomn} from {city_ru}, address: {addr_ru}.",
    "Resume summary: Candidate {fio_ru_nomn} worked as Senior Engineer at {org_en} in {city_en} under manager {fio_en}.",
    "Cross-border transfer from {fio_ru_gent} in Moscow to {fio_en} at {org_en} London branch.",
    "Contract between Russian representative {fio_ru_ablt} ({org_ru}) and regional director {fio_en} ({org_en}).",
    "Meeting minutes: Attendees included {fio_ru_nomn} from Yandex and {fio_en} representing {org_en} in {city_en}.",
    "Shipping manifest: Consignee {fio_ru_nomn}, {addr_ru}, {city_ru}; Carrier: {org_en}, Dispatcher: {fio_en}.",
    "Service agreement between {org_ru} represented by {fio_ru_ablt} and independent consultant {fio_en}.",
    "Audit report prepared by {fio_en} from Deloitte for Russian subsidiary {org_ru} in {city_ru}.",
    "NDA signed by employee {fio_ru_nomn} with foreign corporation {org_en} head office in {city_en}.",
    "Software license delivered to {fio_ru_datv} by distributor {org_en} via partner {org_ru}.",
]

TEMPLATES_HARD_NEGATIVES = [
    "В соответствии со статьей 152 Федерального закона Российской Федерации оператор обязан обеспечить конфиденциальность.",
    "Трудовой кодекс Российской Федерации устанавливает порядок предоставления ежегодного оплачиваемого отпуска.",
    "Генеральный директор издал приказ о назначении ответственного за пожарную безопасность на предприятии.",
    "Главный врач утвердил график дежурств медицинского персонала на праздничные дни.",
    "Отдел кадров совместно с Юридическим управлением подготовил проект типового трудового соглашения.",
    "Служба безопасности провела внеплановую проверку пропускного режима на контрольно-пропускном пункте.",
    "Бухгалтерия напоминает о необходимости своевременной сдачи авансовых отчетов до конца текущего месяца.",
    "Департамент информационных технологий завершил плановые регламентные работы на серверах баз данных.",
    "Согласно правилам внутреннего трудового распорядка рабочий день начинается в девять часов утра.",
    "Министерство финансов опубликовало проект основных направлений бюджетной и налоговой политики.",
    "Конституция гарантирует каждому гражданину право на неприкосновенность частной жизни и тайну переписки.",
    "Совет директоров утвердил годовой бюджет и инвестиционную программу развития производственных мощностей.",
    "Управление Федеральной антимонопольной службы вынесло решение о соблюдении законодательства о конкуренции.",
    "Федеральная служба по надзору в сфере связи осуществляет контроль за соблюдением правил обработки информации.",
    "Правительство утвердило план мероприятий по цифровизации государственного сектора экономики.",
]


def _build_iob2_example(text: str, entities: list[tuple[str, str]]) -> dict | None:
    """Создать и провалидировать словарь токенов и меток IOB2."""
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    offsets = []
    cursor = 0
    for tok in tokens:
        idx = text.find(tok, cursor)
        if idx == -1:
            idx = cursor
        end = idx + len(tok)
        offsets.append((idx, end))
        cursor = end

    labels = [LABEL2ID["O"]] * len(tokens)

    # Ищем каждую сущность и размечаем
    for ent_text, ent_tag in entities:
        if not ent_text:
            continue
        idx = text.find(ent_text)
        if idx == -1:
            continue
        end = idx + len(ent_text)

        first = True
        b_tag = LABEL2ID.get(f"B-{ent_tag}", LABEL2ID["O"])
        i_tag = LABEL2ID.get(f"I-{ent_tag}", LABEL2ID["O"])

        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_end > idx and tok_start < end:
                if first:
                    labels[i] = b_tag
                    first = False
                else:
                    labels[i] = i_tag

    labels = repair_iob2_sequence(labels)
    if validate_iob2_sequence(tokens, labels):
        return {
            "tokens": tokens,
            "ner_tags": labels,
            "source": "Synthetic-152FZ",
        }
    return None


def generate_synthetic_dataset(
    target_count: int = 20000,
    hard_negatives_count: int = 2500,
) -> list[dict]:
    """
    Сгенерировать целевой объем синтетических данных (20 000 примеров)
    с морфологическим разнообразием и Hard Negatives.
    """
    dataset: list[dict] = []

    # 1. Генерация Hard Negatives (размеченных как O)
    hn_generated = 0
    while hn_generated < hard_negatives_count:
        tpl = _rng.choice(TEMPLATES_HARD_NEGATIVES)
        example = _build_iob2_example(tpl, [])
        if example:
            dataset.append(example)
            hn_generated += 1

    # 2. Генерация доменных синтетических примеров (остаток до target_count)
    remaining = target_count - len(dataset)
    fio_formats = ["full", "name_surname", "initials_last", "initials_first", "double"]

    while len(dataset) < target_count:
        cat = _rng.choice([1, 2, 3, 4, 5, 6])

        if cat == 1:
            # Юридические договоры
            tpl = _rng.choice(TEMPLATES_LEGAL)
            case = _rng.choice(CASES)
            fmt = _rng.choice(fio_formats)
            fio_str, _ = _generate_fio(case, fmt)
            org = _rng.choice(COMPANIES_RU)
            zip_c, city, addr = _generate_address()
            num = f"{_rng.randint(100, 9999)}/{_rng.randint(20, 26)}"

            # Подставляем в шаблон
            text = tpl.format(
                num=num, org=org, fio_nomn=fio_str, fio_gent=fio_str,
                fio_datv=fio_str, fio_ablt=fio_str, fio_accs=fio_str,
                zip=zip_c, city=city, addr=addr, passport=f"{_rng.randint(10, 99)} {_rng.randint(10, 99)} {_rng.randint(100000, 999999)}",
            )
            entities = [
                (org, "ORG"),
                (fio_str, "PER"),
                (city, "LOC"),
                (zip_c, "ADDR"),
                (addr, "ADDR"),
            ]

        elif cat == 2:
            # Банковские, кадровые и налоговые анкеты
            tpl = _rng.choice(TEMPLATES_BANKING_TAX_HR)
            case = _rng.choice(CASES)
            fmt = _rng.choice(fio_formats)
            fio_str, _ = _generate_fio(case, fmt)
            org = _rng.choice(COMPANIES_RU)
            zip_c, city, addr = _generate_address()
            # Валидные и случайные номера (размечаются как O!)
            inn = f"{_rng.randint(100000000000, 999999999999)}"
            ogrnip = f"{_rng.randint(100000000000000, 999999999999999)}"
            snils = f"{_rng.randint(100, 999)}-{_rng.randint(100, 999)}-{_rng.randint(100, 999)} {_rng.randint(10, 99)}"
            card = f"4276 {_rng.randint(1000, 9999)} {_rng.randint(1000, 9999)} {_rng.randint(1000, 9999)}"
            phone = f"+7 (9{_rng.randint(10, 99)}) {_rng.randint(100, 999)}-{_rng.randint(10, 99)}-{_rng.randint(10, 99)}"

            text = tpl.format(
                num=_rng.randint(1000, 99999), org=org, fio_nomn=fio_str, fio_gent=fio_str,
                fio_datv=fio_str, fio_ablt=fio_str, fio_accs=fio_str, zip=zip_c, city=city,
                addr=addr, inn=inn, ogrnip=ogrnip, snils=snils, card_num=card, phone=phone,
                passport_series=f"{_rng.randint(10, 99)} {_rng.randint(10, 99)}",
                passport_num=f"{_rng.randint(100000, 999999)}", tab_num=_rng.randint(100, 9999),
                amount=f"{_rng.randint(10000, 5000000):,}",
            )
            # Внимание: номера документов НЕ входят в entities (получают тег O)
            entities = [
                (org, "ORG"),
                (fio_str, "PER"),
                (city, "LOC"),
                (zip_c, "ADDR"),
                (addr, "ADDR"),
            ]

        elif cat == 3:
            # Медицинские записи
            tpl = _rng.choice(TEMPLATES_MEDICAL)
            case = _rng.choice(CASES)
            fmt = _rng.choice(fio_formats)
            fio_str, _ = _generate_fio(case, fmt)
            doctor_str, _ = _generate_fio("ablt", "name_surname")
            org = _rng.choice(COMPANIES_RU)
            zip_c, city, addr = _generate_address()
            num = f"{_rng.randint(100, 99999)}/{_rng.randint(1, 12)}"

            text = tpl.format(
                num=num, org=org, fio_nomn=fio_str, fio_gent=fio_str,
                fio_datv=fio_str, fio_ablt=doctor_str, fio_accs=fio_str,
                city=city, addr=addr,
            )
            entities = [
                (org, "ORG"),
                (fio_str, "PER"),
                (doctor_str, "PER"),
                (city, "LOC"),
                (addr, "ADDR"),
            ]

        elif cat == 4:
            # Технические логи, дампы JSON/CSV
            tpl = _rng.choice(TEMPLATES_TECH_LOGS_CSV)
            fio_str, _ = _generate_fio("nomn", "name_surname")
            org = _rng.choice(COMPANIES_RU)
            zip_c, city, addr = _generate_address()
            ip = f"192.168.{_rng.randint(1, 254)}.{_rng.randint(1, 254)}"
            email = f"user_{_rng.randint(100, 999)}@corp.ru"

            text = tpl.format(
                iso_time="2026-10-18T01:15:00Z", fio_nomn=fio_str, org=org,
                city=city, addr=addr, ip=ip, num=_rng.randint(1000, 99999),
                id=_rng.randint(1, 9999), phone=f"+79{_rng.randint(100000000, 999999999)}",
                email=email,
            )
            entities = [
                (org, "ORG"),
                (fio_str, "PER"),
                (city, "LOC"),
                (addr, "ADDR"),
            ]

        elif cat == 5:
            # Неформальные чаты и email в нижнем регистре
            tpl = _rng.choice(TEMPLATES_INFORMAL_CHATS)
            case = _rng.choice(["nomn", "gent", "datv", "accs"])
            fio_str, _ = _generate_fio(case, "name_surname")
            fio_lower = fio_str.lower()
            org = _rng.choice(COMPANIES_RU)
            org_lower = org.lower()
            zip_c, city, addr = _generate_address()
            city_lower = city.lower()
            addr_lower = addr.lower()
            email = f"support_{_rng.randint(10, 99)}@example.com"

            text = tpl.format(
                fio_lower=fio_lower, fio_lower_gent=fio_lower,
                fio_lower_datv=fio_lower, fio_lower_accs=fio_lower,
                org_lower=org_lower, city_lower=city_lower,
                addr_lower=addr_lower, email=email,
            )
            entities = [
                (org_lower, "ORG"),
                (fio_lower, "PER"),
                (city_lower, "LOC"),
                (addr_lower, "ADDR"),
            ]

        else:
            # Смешанные RU-EN тексты
            tpl = _rng.choice(TEMPLATES_MIXED_RU_EN)
            fio_ru, _ = _generate_fio("nomn", "full")
            fio_ru_g, _ = _generate_fio("gent", "full")
            fio_ru_ab, _ = _generate_fio("ablt", "full")
            fio_ru_d, _ = _generate_fio("datv", "full")
            fio_en = f"{_faker_en.first_name()} {_faker_en.last_name()}"
            org_ru = _rng.choice(COMPANIES_RU)
            org_en = _rng.choice(COMPANIES_EN)
            city_ru = _rng.choice(CITIES_RU)
            city_en = _rng.choice(CITIES_EN)
            zip_c, _, addr_ru = _generate_address()

            text = tpl.format(
                num=_rng.randint(1000, 99999), org_ru=org_ru, org_en=org_en,
                fio_ru_nomn=fio_ru, fio_ru_gent=fio_ru_g, fio_ru_ablt=fio_ru_ab,
                fio_ru_datv=fio_ru_d, fio_en=fio_en, city_ru=city_ru,
                city_en=city_en, addr_ru=addr_ru,
            )
            entities = [
                (org_ru, "ORG"),
                (org_en, "ORG"),
                (fio_ru, "PER"),
                (fio_ru_g, "PER"),
                (fio_ru_ab, "PER"),
                (fio_ru_d, "PER"),
                (fio_en, "PER"),
                (city_ru, "LOC"),
                (city_en, "LOC"),
                (addr_ru, "ADDR"),
            ]

        example = _build_iob2_example(text, entities)
        if example:
            dataset.append(example)

    return dataset[:target_count]
