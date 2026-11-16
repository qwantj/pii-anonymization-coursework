"""
Состязательный стресс-тест EMAIL_PATTERN и гибридного конвейера детекции (Milestone 1).
Разработан EMPIRICAL CHALLENGER (critic, specialist).

Покрывает:
1. Однобуквенные домены второго уровня (SLD): admin@q.com, user@x.org, test@z.ru и др.
2. Punycode домены верхнего уровня (TLD): user@domain.xn--p1ai, info@company.xn--80akhbyknj4f и др.
3. Негативные сценарии (0 False Positives):
   - URL с basic-auth/встроенным логином: http://user@domain.com, https://, ftp://
   - Пути Windows и файловых систем: c:\\users\\user@domain.com, file:///d:/path/user@domain.com
   - Схема mailto: блокировка при наличии двоеточия и корректное распознавание при санитизации
   - Синтаксически невалидные почтовые адреса
4. Обработка пунктуации и границ слов (кавычки, скобки, точки, запятые, знаки восклицания/вопроса).
5. Сквозная приоритизация в гибридном конвейере (detect_pii):
   - Защита email от каннибализации нейросетью при наличии антропонимов в local-part
     (maria898@gmail.com, petr779@inbox.ru, dmitry38@rambler.ru).
   - Сохранение детекции внешних имен (Мария, Петр Иванов) без захвата подстрок email.
6. Генеративное стресс-тестирование случайными валидными и негативными комбинациями.
"""

from __future__ import annotations

import re
import pytest

from lab1.src.rules.regex_patterns import EMAIL_PATTERN
from lab1.src.hybrid_pipeline import detect_pii, HybridDetector


# ============================================================================
# 1. Однобуквенные домены второго уровня (SLD)
# ============================================================================
class TestSingleLetterSLD:
    """Проверка распознавания однобуквенных доменов второго уровня (admin@q.com, etc.)."""

    @pytest.mark.parametrize("email", [
        "admin@q.com",
        "user@x.org",
        "test@z.ru",
        "support@a.net",
        "info@1.io",
        "a@b.co",
        "x@y.zz",
        "contact@i.travel",
        "dev@k.info",
    ])
    def test_single_letter_sld_isolated(self, email: str):
        matches = EMAIL_PATTERN.findall(email)
        assert matches == [email], f"Ожидался адрес {email}, получено: {matches}"

    @pytest.mark.parametrize("email", [
        "admin@sub.q.com",
        "user@a.b.c.org",
        "deep@1.2.3.net",
        "multi@x.y.z.ru",
    ])
    def test_single_letter_subdomains_and_sld(self, email: str):
        matches = EMAIL_PATTERN.findall(email)
        assert matches == [email], f"Ожидался адрес с субдоменами {email}, получено: {matches}"

    def test_single_letter_sld_in_full_text(self):
        text = "Для обратной связи пишите на admin@q.com, либо резервный user@x.org или test@z.ru."
        matches = EMAIL_PATTERN.findall(text)
        assert matches == ["admin@q.com", "user@x.org", "test@z.ru"]


# ============================================================================
# 2. Punycode домены верхнего уровня (TLD)
# ============================================================================
class TestPunycodeTLD:
    """Проверка доменов с интернационализированными Punycode TLD (xn--...)."""

    @pytest.mark.parametrize("email", [
        "user@domain.xn--p1ai",             # .рф
        "info@company.xn--80akhbyknj4f",     # .рус
        "test@site.xn--j1amh",               # .укр
        "contact@org.xn--d1alf",             # .мск
        "user@sub.domain.xn--p1ai",          # поддомен + .рф
        "admin@xn--d1acj3b.xn--p1ai",        # домен.рф в punycode целиком
        "service@q.xn--p1ai",                # 1-буквенный SLD + punycode TLD
    ])
    def test_punycode_tld_isolated(self, email: str):
        matches = EMAIL_PATTERN.findall(email)
        assert matches == [email], f"Punycode адрес {email} не распознан или обрезан: {matches}"

    def test_punycode_boundary_no_truncation(self):
        # Критическая проверка: regex не должен обрезать xn--p1ai до xn
        text = "user@domain.xn--p1ai"
        m = EMAIL_PATTERN.search(text)
        assert m is not None
        assert m.group(0) == "user@domain.xn--p1ai"
        assert m.group(0).endswith("xn--p1ai"), "TLD был усечен!"

    def test_punycode_in_russian_sentence(self):
        text = "Пишите: user@domain.xn--p1ai или info@company.xn--80akhbyknj4f для оформления."
        matches = EMAIL_PATTERN.findall(text)
        assert matches == ["user@domain.xn--p1ai", "info@company.xn--80akhbyknj4f"]


# ============================================================================
# 3. Негативные сценарии (0 False Positives)
# ============================================================================
class TestNegativeScenariosZeroFalsePositives:
    """Проверка строгого отсутствия ложных срабатываний (FP = 0)."""

    @pytest.mark.parametrize("uri", [
        "http://user@domain.com",
        "https://user@domain.com",
        "http://user:password@domain.com",
        "https://admin:secret123@sub.domain.org/path",
        "ftp://user@domain.com",
        "sftp://user@remote.server.net/dir",
        "ssh://git@github.com",
        "file:///d:/path/user@domain.com",
        "file://c:/users/user@domain.com",
    ])
    def test_no_match_in_url_with_login_or_file_uri(self, uri: str):
        matches = EMAIL_PATTERN.findall(uri)
        assert len(matches) == 0, f"Ложное срабатывание в URI '{uri}': {matches}"

    @pytest.mark.parametrize("path", [
        r"c:\users\user@domain.com",
        r"C:\Windows\user@domain.com",
        r"d:\path\to\user@domain.com",
        r"\\server\share\user@domain.com",
        r"/home/user@domain.com",
        r"./user@domain.com",
        r"../user@domain.com",
        r"C:\Program Files\user@domain.com\run.exe",
    ])
    def test_no_match_in_file_paths(self, path: str):
        matches = EMAIL_PATTERN.findall(path)
        assert len(matches) == 0, f"Ложное срабатывание в пути '{path}': {matches}"

    def test_mailto_behavior_raw_vs_sanitized(self):
        # 1. При наличии двоеточия напрямую не захватывать
        raw_mailto = "mailto:user@domain.com"
        assert len(EMAIL_PATTERN.findall(raw_mailto)) == 0, "Lookbehind должен отсекать mailto: с двоеточием"

        raw_mailto_complex = "Связь: mailto:admin@q.com или mailto:user@domain.xn--p1ai"
        assert len(EMAIL_PATTERN.findall(raw_mailto_complex)) == 0

        # 2. При наличии пробела после двоеточия захватывать
        spaced_mailto = "mailto: user@domain.com"
        assert EMAIL_PATTERN.findall(spaced_mailto) == ["user@domain.com"]

        # 3. При предварительной регламентной санитизации распознавать корректно
        sanitized = re.sub(r'(?i)\bmailto:\s*', '', raw_mailto_complex)
        assert EMAIL_PATTERN.findall(sanitized) == ["admin@q.com", "user@domain.xn--p1ai"]

    @pytest.mark.parametrize("invalid", [
        "@domain.com",             # нет имени пользователя
        "user@",                   # нет домена
        "user@domain",             # нет TLD
        "user@.com",               # пустой домен второго уровня
        "user@domain.",            # точка без TLD
        "user@-domain.com",        # дефис в начале домена
        "user@domain-.com",        # дефис в конце домена
        "user@domain..com",        # две точки подряд в домене
        "user@domain.c",           # TLD из одной буквы (недопустимо)
        "user@@domain.com",        # двойная собака
        "user space@domain.com",   # пробел внутри
    ])
    def test_syntactically_invalid_emails(self, invalid: str):
        matches = EMAIL_PATTERN.findall(invalid)
        # Ни одна строка не должна быть полностью признана валидным адресом
        assert invalid not in matches, f"Невалидная строка '{invalid}' признана email: {matches}"


# ============================================================================
# 4. Обработка пунктуации и границ
# ============================================================================
class TestPunctuationAndWordBoundaries:
    """Проверка чистоты извлечения без захвата внешней пунктуации."""

    @pytest.mark.parametrize("wrapped, expected", [
        ("admin@q.com.", "admin@q.com"),
        ("(admin@q.com)", "admin@q.com"),
        ("[admin@q.com]", "admin@q.com"),
        ("{admin@q.com}", "admin@q.com"),
        ("<admin@q.com>", "admin@q.com"),
        ('"admin@q.com"', "admin@q.com"),
        ("'admin@q.com'", "admin@q.com"),
        ("«admin@q.com»", "admin@q.com"),
        ("admin@q.com,", "admin@q.com"),
        ("admin@q.com;", "admin@q.com"),
        ("admin@q.com!", "admin@q.com"),
        ("admin@q.com?", "admin@q.com"),
        ("admin@q.com:", "admin@q.com"),
    ])
    def test_punctuation_stripping(self, wrapped: str, expected: str):
        matches = EMAIL_PATTERN.findall(wrapped)
        assert matches == [expected], f"В '{wrapped}' извлечено: {matches}, ожидалось: {[expected]}"


# ============================================================================
# 5. Сквозная приоритизация в гибридном конвейере (detect_pii)
# ============================================================================
class TestEndToEndHybridPipelinePrioritization:
    """
    Проверка арбитража приоритетов (Приоритет 1.5 для EMAIL против Приоритета 2 для нейросети).
    Гарантирует, что имена в local-part email не фрагментируют и не каннибализируют email.
    """

    @pytest.mark.parametrize("email", [
        "maria898@gmail.com",
        "petr779@inbox.ru",
        "dmitry38@rambler.ru",
        "anna123@yandex.ru",
        "alexander.ivanov@mail.ru",
        "elena_smirnova@domain.xn--p1ai",
        "sergey@x.org",
        "admin@q.com",
    ])
    def test_single_email_not_cannibalized_by_ner(self, email: str):
        text = f"Контакты: {email} для связи."
        spans = detect_pii(text)
        email_spans = [s for s in spans if s[0] == "EMAIL"]
        assert len(email_spans) == 1, f"EMAIL не обнаружен или фрагментирован: {spans}"
        assert text[email_spans[0][1]:email_spans[0][2]] == email

        # Проверка отсутствия ложных тегов NAME/SURNAME/PER внутри спана email
        e_start, e_end = email_spans[0][1], email_spans[0][2]
        internal_spans = [s for s in spans if s[0] != "EMAIL" and (s[1] >= e_start and s[2] <= e_end)]
        assert len(internal_spans) == 0, f"Внутри email обнаружены паразитные спаны: {internal_spans}"

    def test_composite_sentence_with_real_names_and_email(self):
        text = "Пользователь Мария (maria898@gmail.com) отправила отчет."
        spans = detect_pii(text)

        email_spans = [s for s in spans if s[0] == "EMAIL"]
        assert len(email_spans) == 1
        assert text[email_spans[0][1]:email_spans[0][2]] == "maria898@gmail.com"

        name_spans = [s for s in spans if s[0] == "NAME"]
        name_texts = [text[s[1]:s[2]] for s in name_spans]
        assert "Мария" in name_texts
        assert "maria" not in name_texts, "Подстрока maria внутри email не должна быть отдельным NAME!"

    def test_composite_sentence_fio_and_email(self):
        text = "Петр Иванов petr779@inbox.ru позвонил Дмитрию."
        spans = detect_pii(text)

        email_spans = [s for s in spans if s[0] == "EMAIL"]
        assert len(email_spans) == 1
        assert text[email_spans[0][1]:email_spans[0][2]] == "petr779@inbox.ru"

        extracted_map = {text[s[1]:s[2]]: s[0] for s in spans}
        assert extracted_map.get("Петр") == "NAME"
        assert extracted_map.get("Иванов") == "SURNAME"
        assert extracted_map.get("petr779@inbox.ru") == "EMAIL"
        assert "petr" not in extracted_map

    def test_explicit_rosberta_detector_prioritization(self):
        """Проверка при явной передаче детектора RoSBERTa."""
        from coursework.src.detector_base import NerDetectorFactory
        if not NerDetectorFactory.has_local_rosberta():
            pytest.skip("Веса RoSBERTa не загружены локально (запустите python scripts/download_models.py)")
        try:
            from lab1.src.ner.rosberta_detector import RoSBERTaNerDetector
            rosberta = RoSBERTaNerDetector()
        except Exception as exc:
            pytest.skip(f"RoSBERTaNerDetector недоступен: {exc}")

        text = "Сотрудник Алексей Смирнов (alexander.ivanov@mail.ru) подтвердил адрес admin@q.com."
        spans = detect_pii(text, ner_detector=rosberta)

        emails = [text[s[1]:s[2]] for s in spans if s[0] == "EMAIL"]
        assert "alexander.ivanov@mail.ru" in emails
        assert "admin@q.com" in emails
        assert len(emails) == 2

        # Убедимся, что alexander внутри почты не создал отдельный спан
        names = [text[s[1]:s[2]] for s in spans if s[0] in ("NAME", "SURNAME", "PER")]
        assert "alexander" not in [n.lower() for n in names]


# ============================================================================
# 6. Генеративное / стресс-тестирование
# ============================================================================
class TestGenerativeStress:
    """Генеративный стресс-тест на устойчивость regex и пайплайна."""

    def test_generative_valid_emails(self):
        slds = ["q", "x", "z", "mail", "company-corp", "sub123", "a-b-c"]
        tlds = ["com", "org", "net", "ru", "xn--p1ai", "xn--80akhbyknj4f", "info", "tech"]
        users = ["user", "maria898", "petr.ivanov", "test+tag", "dev_ops", "a", "12345"]

        count = 0
        for u in users:
            for s in slds:
                for t in tlds:
                    addr = f"{u}@{s}.{t}"
                    matches = EMAIL_PATTERN.findall(addr)
                    assert matches == [addr], f"Генеративный адрес {addr} не распознан: {matches}"
                    count += 1
        assert count >= 100, f"Выполнено {count} генеративных проверок"

    def test_generative_negative_paths_and_urls(self):
        schemes = ["http://", "https://", "ftp://", "file:///", "c:\\", "d:\\users\\"]
        emails = ["admin@q.com", "user@domain.xn--p1ai", "petr779@inbox.ru"]

        for scheme in schemes:
            for e in emails:
                bogus = f"{scheme}{e}"
                matches = EMAIL_PATTERN.findall(bogus)
                assert len(matches) == 0, f"Ложное срабатывание на '{bogus}': {matches}"
