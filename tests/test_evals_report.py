"""Тесты отчёта приёмки.

Отчёт — это то, что читает человек, когда отвечает на два вопроса M2 и пишет
ADR. Ошибка здесь не роняет прогон и не красит гейт: она молча даёт неверное
число, на которое потом ссылается решение. Поэтому проверяются не формулировки,
а арифметика и правило учёта молчания.

Ключевое: **продукт без ответа системы входит в описательную статистику как
ноль**, ровно как в метриках (ADR-022). Если бы среднее считалось только по
отвеченным, система, промолчавшая на трудных составах, выглядела бы аккуратнее
той, что ответила на всех, — и вывод по вопросу 1 получился бы обратный.

Сети и БД здесь нет: словарь собирается вручную, файлы лежат в `tmp_path`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nutri_radar.config import EvalsSettings, Settings
from nutri_radar.evals.metrics import score_system
from nutri_radar.evals.report import (
    GOLD_LABEL,
    SugarProfile,
    build_report,
    format_language_gap,
    format_sample_note,
    format_zero_sugar,
    profile_gold,
    profile_system,
    read_all_predictions,
    write_report,
)
from nutri_radar.evals.schemas import GoldRecord, PredictionRecord, write_jsonl
from nutri_radar.extract.normalize import ANY_LANG, AliasEntry, AliasIndex
from nutri_radar.extract.schemas import ExtractionResult, Ingredient, IngredientKind

SYSTEM = "test-system"


@pytest.fixture
def index() -> AliasIndex:
    return AliasIndex(
        [
            AliasEntry("sugar", ANY_LANG, "sugar", IngredientKind.SUGAR),
            AliasEntry("сахар", "ru", "sugar", IngredientKind.SUGAR),
            AliasEntry("glucose syrup", ANY_LANG, "glucose syrup", IngredientKind.SUGAR),
        ]
    )


def _extraction(*items: tuple[str, IngredientKind]) -> ExtractionResult:
    return ExtractionResult(
        ingredients=[Ingredient(canonical_name=name, kind=kind) for name, kind in items]
    )


def _gold(
    code: str, lang: str, *items: tuple[str, IngredientKind], assisted: bool = False
) -> GoldRecord:
    return GoldRecord(
        code=code,
        lang=lang,
        ingredients_text="состав для теста",
        extraction=_extraction(*items),
        annotator="tester",
        assisted=assisted,
    )


def _prediction(
    code: str, *items: tuple[str, IngredientKind], system: str = SYSTEM
) -> PredictionRecord:
    return PredictionRecord(code=code, system=system, extraction=_extraction(*items))


@pytest.fixture
def report_settings(settings: Settings) -> Settings:
    """Порог различимости задаётся явно — иначе тест зависит от `.env`."""
    return settings.model_copy(
        update={"evals": EvalsSettings(gold_size=4, significant_diff_share=0.20)}
    )


class TestПрофильСахара:
    def test_среднее_и_доля_нулей(self):
        profile = SugarProfile()
        profile.add(2)
        profile.add(0)

        assert profile.products == 2
        assert profile.mean == 1.0
        assert profile.zero_share == 0.5

    def test_пустой_профиль_не_делит_на_ноль(self):
        profile = SugarProfile()

        assert profile.mean == 0.0
        assert profile.zero_share == 0.0


class TestПрофильЭталона:
    def test_считается_по_языкам(self, index: AliasIndex):
        gold = [
            _gold("1", "ru", ("сахар", IngredientKind.SUGAR)),
            _gold("2", "ru"),
            _gold("3", "de", ("sugar", IngredientKind.SUGAR)),
        ]

        profile = profile_gold(gold, index)

        assert profile["ru"].products == 2
        assert profile["ru"].mean == 0.5
        assert profile["de"].mean == 1.0

    def test_имена_канонизируются_словарём(self, index: AliasIndex):
        """Три написания одного сиропа — одна форма, а не три."""
        gold = [
            _gold(
                "1",
                "ru",
                ("сахар", IngredientKind.SUGAR),
                ("sugar", IngredientKind.SUGAR),
            )
        ]

        assert profile_gold(gold, index)["ru"].mean == 1.0


class TestПрофильСистемы:
    def test_молчание_считается_нулём(self, index: AliasIndex):
        """Иначе промолчавшая на трудных составах система выглядела бы лучше."""
        gold = [
            _gold("1", "ru", ("сахар", IngredientKind.SUGAR)),
            _gold("2", "ru", ("сахар", IngredientKind.SUGAR)),
        ]
        predictions = [_prediction("1", ("sugar", IngredientKind.SUGAR))]

        profile = profile_system(gold, predictions, index)

        assert profile["ru"].products == 2
        assert profile["ru"].mean == 0.5
        assert profile["ru"].zero == 1

    def test_лишние_предсказания_вне_эталона_не_учитываются(self, index: AliasIndex):
        gold = [_gold("1", "ru", ("сахар", IngredientKind.SUGAR))]
        predictions = [
            _prediction("1", ("sugar", IngredientKind.SUGAR)),
            _prediction("999", ("sugar", IngredientKind.SUGAR)),
        ]

        assert profile_system(gold, predictions, index)["ru"].products == 1

    def test_тип_берётся_у_системы(self, index: AliasIndex):
        """Словарь знает, что это sugar; система назвала ароматизатором."""
        gold = [_gold("1", "ru", ("сахар", IngredientKind.SUGAR))]
        predictions = [_prediction("1", ("sugar", IngredientKind.FLAVOURING))]

        assert profile_system(gold, predictions, index)["ru"].mean == 0.0


class TestЧтениеПредсказаний:
    def test_ключ_берётся_из_записи_а_не_из_имени_файла(self, tmp_path: Path):
        """Переименование файла не должно переименовывать систему в отчёте."""
        write_jsonl(tmp_path / "как-угодно.jsonl", [_prediction("1", system="настоящее-имя")])

        assert set(read_all_predictions(tmp_path)) == {"настоящее-имя"}

    def test_пустой_файл_пропускается(self, tmp_path: Path):
        write_jsonl(tmp_path / "живая.jsonl", [_prediction("1")])
        (tmp_path / "пустая.jsonl").write_text("", encoding="utf-8")

        assert set(read_all_predictions(tmp_path)) == {SYSTEM}

    def test_отсутствующий_каталог_не_роняет_отчёт(self, tmp_path: Path):
        assert read_all_predictions(tmp_path / "нет-такого") == {}


class TestРазмерВыборки:
    def test_незаконченная_разметка_названа_честно(self, report_settings: Settings):
        note = format_sample_note([_gold("1", "ru")], report_settings)

        assert "1 продуктов" in note
        assert "не закончена" in note

    def test_подсказка_помечается_отдельно(self, report_settings: Settings):
        """Разметка с подсказкой слабее слепой — это должно быть видно."""
        gold = [_gold("1", "ru", assisted=True), _gold("2", "de")]

        note = format_sample_note(gold, report_settings)

        assert "с подсказкой модели: 1" in note
        assert "завышены" in note

    def test_слепая_разметка_предупреждения_не_получает(self, report_settings: Settings):
        note = format_sample_note([_gold("1", "ru")], report_settings)

        assert "подсказкой" not in note

    def test_языки_перечисляются_с_числами(self, report_settings: Settings):
        note = format_sample_note(
            [_gold("1", "ru"), _gold("2", "de"), _gold("3", "de")], report_settings
        )

        assert "de: 2" in note
        assert "ru: 1" in note


class TestВопрос1:
    def test_ноль_вместо_найденного_человеком_виден_отдельно(self, index: AliasIndex):
        """Это и есть прямой ответ: предел системы или свойство корпуса."""
        gold = [
            _gold("1", "ru", ("сахар", IngredientKind.SUGAR)),
            _gold("2", "ru"),
        ]
        predictions = [_prediction("1"), _prediction("2")]
        results = {SYSTEM: score_system(gold, predictions, index)}
        profiles = {SYSTEM: profile_system(gold, predictions, index)}

        text = format_zero_sugar(profile_gold(gold, index), profiles, results)

        # Человек нашёл ноль на одном продукте из двух.
        assert "1 из 2" in text
        # Система вернула ноль на обоих, но «вместо N» — только на одном.
        assert "| 2 | 100.0% | 1 |" in text

    def test_система_без_продуктов_не_делит_на_ноль(self, index: AliasIndex):
        gold = [_gold("1", "ru", ("сахар", IngredientKind.SUGAR))]
        predictions = [_prediction("1")]
        results = {SYSTEM: score_system(gold, predictions, index)}

        text = format_zero_sugar({}, {SYSTEM: {}}, results)

        assert "—" in text


class TestВопрос2:
    @pytest.fixture
    def text(self, index: AliasIndex, report_settings: Settings) -> str:
        gold = [
            _gold("1", "ru"),
            _gold("2", "de", ("sugar", IngredientKind.SUGAR)),
        ]
        predictions = [
            _prediction("1", ("sugar", IngredientKind.SUGAR)),
            _prediction("2", ("sugar", IngredientKind.SUGAR)),
        ]
        return format_language_gap(
            profile_gold(gold, index),
            {SYSTEM: profile_system(gold, predictions, index)},
            report_settings,
        )

    def test_эталон_стоит_рядом_с_системой_на_каждом_языке(self, text: str):
        """Вопрос ставится к разнице «система против человека», а не между систем."""
        rows = [line for line in text.splitlines() if line.startswith("| ru |")]

        assert [GOLD_LABEL in row for row in rows] == [True, False]
        assert any(SYSTEM in row for row in rows)
        # По каждому языку эталон ровно один раз — иначе строки задвоены.
        assert sum(GOLD_LABEL in line for line in text.splitlines() if line.startswith("|")) == 2

    def test_размер_выборки_рядом_с_каждым_числом(self, text: str):
        """На 20 продуктах интервал шире разницы — число без знаменателя врёт."""
        assert "| ru | 1 |" in text
        assert "| de | 1 |" in text

    def test_порог_различимости_берётся_из_настроек(self, text: str):
        assert "20%" in text
        assert "значимой называть нельзя" in text


class TestСборкаОтчёта:
    def test_без_эталона_отчёт_честно_говорит_что_считать_нечего(
        self, index: AliasIndex, report_settings: Settings
    ):
        """Разметку ведёт человек — до её конца отчёт не выдумывает чисел."""
        text = build_report([], {SYSTEM: [_prediction("1")]}, index, report_settings)

        assert "Эталон пуст" in text
        assert "evals annotate" in text

    def test_без_предсказаний_отчёт_называет_следующую_команду(
        self, index: AliasIndex, report_settings: Settings
    ):
        text = build_report([_gold("1", "ru")], {}, index, report_settings)

        assert "evals predict" in text

    def test_полный_отчёт_содержит_все_разделы(self, index: AliasIndex, report_settings: Settings):
        gold = [
            _gold("1", "ru", ("сахар", IngredientKind.SUGAR)),
            _gold("2", "de", ("sugar", IngredientKind.SUGAR)),
        ]
        predictions = {SYSTEM: [_prediction("1", ("sugar", IngredientKind.SUGAR))]}

        text = build_report(gold, predictions, index, report_settings)

        assert "## Сравнение систем" in text
        assert "## Разбивка по языкам" in text
        assert "### Вопрос 1" in text
        assert "### Вопрос 2" in text
        assert SYSTEM in text

    def test_системы_идут_в_детерминированном_порядке(
        self, index: AliasIndex, report_settings: Settings
    ):
        """Иначе diff отчёта между прогонами показывает перестановку строк."""
        gold = [_gold("1", "ru", ("сахар", IngredientKind.SUGAR))]
        predictions = {
            "яяя-система": [_prediction("1", system="яяя-система")],
            "ааа-система": [_prediction("1", system="ааа-система")],
        }

        text = build_report(gold, predictions, index, report_settings)

        assert text.index("ааа-система") < text.index("яяя-система")


class TestЗаписьОтчёта:
    def test_файл_пишется_в_utf8(self, tmp_path: Path):
        """Отчёт открывают на Windows, где консоль в cp1251, а файл — нет."""
        path = write_report("# Отчёт\n\nКириллица", tmp_path / "отчёт.md")

        assert path.read_text(encoding="utf-8").startswith("# Отчёт")

    def test_каталог_создаётся(self, tmp_path: Path):
        path = write_report("текст", tmp_path / "новый" / "отчёт.md")

        assert path.exists()
