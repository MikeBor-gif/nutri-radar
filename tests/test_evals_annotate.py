"""Тесты ручной разметки эталона.

Главное, что здесь проверяется, — **возобновляемость**. Сто составов руками
это часы работы, разметка почти наверняка идёт в несколько заходов, и потеря
прогресса стоит не теста, а живого рабочего дня владельца проекта. Поэтому
проверяется не «функция вызывается», а факт на диске: после прерывания
размеченное лежит в файле, а следующая сессия продолжает с неразмеченного.

Второе — **слепота к предсказаниям**. Она метод, а не удобство: увидев ответ
модели, человек начинает его править, и эталон незаметно становится копией
предсказания. Флаг `assisted` обязан попадать в каждую запись, иначе смещение
растворится в устном «вроде смотрел».

Ввод-вывод здесь подставной, терминала нет — правило 4 брифа.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from nutri_radar.evals.annotate import (
    QUIT_TOKEN,
    UNREADABLE_TOKEN,
    annotate_session,
    build_record,
    parse_ingredients,
    pending_items,
)
from nutri_radar.evals.schemas import GoldRecord, SampleItem, read_jsonl, write_jsonl
from nutri_radar.extract.schemas import IngredientKind

ANNOTATOR = "tester"


class ПрерываниеСессииError(Exception):
    """Изображает Ctrl-C посреди сессии."""


def _item(code: str, lang: str = "ru", text: str = "сахар, вода") -> SampleItem:
    return SampleItem(code=code, lang=lang, ingredients_text=text)


def _scripted(answers: list[str]) -> Callable[[str], str]:
    """Разметчик, отвечающий по списку. Ответы кончились — значит Ctrl-C."""
    queue = list(answers)

    def ask(_prompt: str) -> str:
        if not queue:
            raise ПрерываниеСессииError("ответы кончились")
        return queue.pop(0)

    return ask


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    path = tmp_path / "sample.jsonl"
    write_jsonl(path, [_item("1"), _item("2", "de", "Zucker, Wasser"), _item("3")])
    return path


@pytest.fixture
def gold_file(tmp_path: Path) -> Path:
    return tmp_path / "gold.jsonl"


class TestРазборСтроки:
    def test_имя_и_тип_через_двоеточие(self):
        ingredients = parse_ingredients("sugar:sugar, water:base, e322:additive")

        assert [i.canonical_name for i in ingredients] == ["sugar", "water", "e322"]
        assert ingredients[0].kind is IngredientKind.SUGAR
        assert ingredients[2].kind is IngredientKind.ADDITIVE

    def test_тип_по_умолчанию_основа(self):
        """Большинство ингредиентов — основа; писать это сто раз незачем."""
        assert parse_ingredients("water")[0].kind is IngredientKind.BASE

    def test_регистр_и_пробелы_не_мешают(self):
        ingredients = parse_ingredients("  Glucose Syrup : SUGAR ")

        assert ingredients[0].canonical_name == "glucose syrup"
        assert ingredients[0].kind is IngredientKind.SUGAR

    def test_пустые_куски_пропускаются(self):
        assert len(parse_ingredients("sugar, , water,")) == 2

    def test_пустая_строка_даёт_пустой_список(self):
        assert parse_ingredients("") == []

    def test_неизвестный_тип_отбивается_и_называет_допустимые(self):
        """Молча подставить `base` нельзя — это тихо испортило бы эталон."""
        with pytest.raises(ValueError, match="неизвестный тип"):
            parse_ingredients("sugar:сладкое")

    def test_сообщение_об_ошибке_перечисляет_типы(self):
        with pytest.raises(ValueError, match="flavouring"):
            parse_ingredients("sugar:нечто")


class TestЧтоОсталось:
    def test_размеченное_исключается_по_коду(self):
        sample = [_item("1"), _item("2"), _item("3")]
        done = [
            build_record(
                _item("2"), [], allergens=[], unreadable=False, annotator="x", assisted=False
            )
        ]

        assert [item.code for item in pending_items(sample, done)] == ["1", "3"]

    def test_порядок_выборки_не_важен(self):
        """Выборку могли пересортировать, а размеченное — нет."""
        sample = [_item("3"), _item("1")]
        done = [
            build_record(
                _item("1"), [], allergens=[], unreadable=False, annotator="x", assisted=False
            )
        ]

        assert [item.code for item in pending_items(sample, done)] == ["3"]

    def test_ничего_не_размечено_значит_всё_впереди(self):
        sample = [_item("1"), _item("2")]

        assert len(pending_items(sample, [])) == 2

    def test_фильтр_по_языку_оставляет_только_свой(self):
        """Выборка перемешана: без фильтра «двадцать русских» дают двадцать случайных."""
        sample = [_item("1", "ru"), _item("2", "de"), _item("3", "ru")]

        assert [item.code for item in pending_items(sample, [], lang="ru")] == ["1", "3"]

    def test_фильтр_по_языку_учитывает_уже_размеченное(self):
        sample = [_item("1", "ru"), _item("2", "de"), _item("3", "ru")]
        done = [
            build_record(
                _item("1", "ru"), [], allergens=[], unreadable=False, annotator="x", assisted=False
            )
        ]

        assert [item.code for item in pending_items(sample, done, lang="ru")] == ["3"]


class TestЗапись:
    def test_уверенность_человека_всегда_единица(self):
        """Поле существует ради модели; сомнение выражается флагом unreadable."""
        record = build_record(
            _item("1"),
            parse_ingredients("sugar:sugar"),
            allergens=[],
            unreadable=False,
            annotator=ANNOTATOR,
            assisted=False,
        )

        assert record.extraction.model_confidence == 1.0

    def test_формы_сахара_считаются_тем_же_кодом_что_у_модели(self):
        """Ради этого эталон и хранится типом `ExtractionResult`."""
        record = build_record(
            _item("1"),
            parse_ingredients("sugar:sugar, glucose syrup:sugar, water:base"),
            allergens=[],
            unreadable=False,
            annotator=ANNOTATOR,
            assisted=False,
        )

        assert record.distinct_sugar_forms == 2

    def test_исходный_текст_хранится_вместе_с_разметкой(self):
        """Без него нельзя перепроверить спорную метку: дельты M1 меняют состав."""
        record = build_record(
            _item("1", text="Сахар, вода"),
            [],
            allergens=[],
            unreadable=True,
            annotator=ANNOTATOR,
            assisted=False,
        )

        assert record.ingredients_text == "Сахар, вода"


class TestСессияРазметки:
    def test_размеченное_попадает_в_файл(self, sample_file: Path, gold_file: Path):
        added = annotate_session(
            annotator=ANNOTATOR,
            ask=_scripted(["sugar:sugar, water:base", "молоко"]),
            show=lambda _: None,
            sample_path=sample_file,
            gold_path=gold_file,
            limit=1,
        )

        records = read_jsonl(gold_file, GoldRecord)
        assert added == 1
        assert records[0].code == "1"
        assert records[0].annotator == ANNOTATOR
        assert records[0].extraction.distinct_sugar_forms == 1
        assert records[0].extraction.allergens == ["молоко"]

    def test_прогресс_пишется_после_каждого_продукта(self, sample_file: Path, gold_file: Path):
        """Обрыв на втором продукте не должен стоить первого."""
        with pytest.raises(ПрерываниеСессииError):
            annotate_session(
                annotator=ANNOTATOR,
                ask=_scripted(["sugar:sugar", ""]),  # хватает ровно на один продукт
                show=lambda _: None,
                sample_path=sample_file,
                gold_path=gold_file,
            )

        assert [record.code for record in read_jsonl(gold_file, GoldRecord)] == ["1"]

    def test_следующая_сессия_продолжает_с_неразмеченного(self, sample_file: Path, gold_file: Path):
        annotate_session(
            annotator=ANNOTATOR,
            ask=_scripted(["sugar:sugar", ""]),
            show=lambda _: None,
            sample_path=sample_file,
            gold_path=gold_file,
            limit=1,
        )

        added = annotate_session(
            annotator=ANNOTATOR,
            ask=_scripted(["water:base", ""]),
            show=lambda _: None,
            sample_path=sample_file,
            gold_path=gold_file,
            limit=1,
        )

        assert added == 1
        # Второй продукт, а не повтор первого, и первый на месте.
        assert [record.code for record in read_jsonl(gold_file, GoldRecord)] == ["1", "2"]

    def test_выход_сохраняет_размеченное(self, sample_file: Path, gold_file: Path):
        added = annotate_session(
            annotator=ANNOTATOR,
            ask=_scripted(["sugar:sugar", "", QUIT_TOKEN]),
            show=lambda _: None,
            sample_path=sample_file,
            gold_path=gold_file,
        )

        assert added == 1
        assert len(read_jsonl(gold_file, GoldRecord)) == 1

    def test_выход_на_аллергенах_не_теряет_предыдущее(self, sample_file: Path, gold_file: Path):
        added = annotate_session(
            annotator=ANNOTATOR,
            ask=_scripted(["sugar:sugar", "", "water:base", QUIT_TOKEN]),
            show=lambda _: None,
            sample_path=sample_file,
            gold_path=gold_file,
        )

        # Второй продукт брошен на полпути и не записан — но первый цел.
        assert added == 1
        assert [record.code for record in read_jsonl(gold_file, GoldRecord)] == ["1"]

    def test_нечитаемый_состав_помечается_флагом(self, sample_file: Path, gold_file: Path):
        annotate_session(
            annotator=ANNOTATOR,
            ask=_scripted([UNREADABLE_TOKEN, ""]),
            show=lambda _: None,
            sample_path=sample_file,
            gold_path=gold_file,
            limit=1,
        )

        record = read_jsonl(gold_file, GoldRecord)[0]
        assert record.extraction.unreadable is True
        assert record.extraction.ingredients == []

    def test_битый_ввод_переспрашивается_а_не_чинится(self, sample_file: Path, gold_file: Path):
        """Тихая подстановка типа испортила бы эталон незаметно для человека."""
        shown: list[str] = []

        annotate_session(
            annotator=ANNOTATOR,
            ask=_scripted(["sugar:сладкое", "sugar:sugar", ""]),
            show=shown.append,
            sample_path=sample_file,
            gold_path=gold_file,
            limit=1,
        )

        assert any("Не понял" in line for line in shown)
        assert read_jsonl(gold_file, GoldRecord)[0].extraction.ingredients[0].kind is (
            IngredientKind.SUGAR
        )

    def test_лимит_останавливает_сессию(self, sample_file: Path, gold_file: Path):
        added = annotate_session(
            annotator=ANNOTATOR,
            ask=_scripted(["sugar:sugar", "", "water:base", "", "water:base", ""]),
            show=lambda _: None,
            sample_path=sample_file,
            gold_path=gold_file,
            limit=2,
        )

        assert added == 2

    def test_язык_берётся_из_выборки(self, sample_file: Path, gold_file: Path):
        """На разбивке по языкам держится ответ на второй вопрос M2."""
        annotate_session(
            annotator=ANNOTATOR,
            ask=_scripted(["sugar:sugar", "", "sugar:sugar", ""]),
            show=lambda _: None,
            sample_path=sample_file,
            gold_path=gold_file,
            limit=2,
        )

        assert [record.lang for record in read_jsonl(gold_file, GoldRecord)] == ["ru", "de"]

    def test_отсутствие_выборки_отбивается_понятно(self, tmp_path: Path, gold_file: Path):
        with pytest.raises(FileNotFoundError, match="evals sample"):
            annotate_session(
                annotator=ANNOTATOR,
                ask=_scripted([]),
                show=lambda _: None,
                sample_path=tmp_path / "нет-такой.jsonl",
                gold_path=gold_file,
            )


class TestРежимСПодсказкой:
    def test_по_умолчанию_разметка_слепая(self, sample_file: Path, gold_file: Path):
        annotate_session(
            annotator=ANNOTATOR,
            ask=_scripted(["sugar:sugar", ""]),
            show=lambda _: None,
            sample_path=sample_file,
            gold_path=gold_file,
            limit=1,
        )

        assert read_jsonl(gold_file, GoldRecord)[0].assisted is False

    def test_подсказка_пишется_в_запись(self, sample_file: Path, gold_file: Path):
        """Смещение должно быть видно в отчёте, а не остаться в памяти человека."""
        annotate_session(
            annotator=ANNOTATOR,
            ask=_scripted(["sugar:sugar", ""]),
            show=lambda _: None,
            sample_path=sample_file,
            gold_path=gold_file,
            assisted=True,
            limit=1,
        )

        assert read_jsonl(gold_file, GoldRecord)[0].assisted is True

    def test_человека_предупреждают_в_момент_разметки(self, sample_file: Path, gold_file: Path):
        shown: list[str] = []

        annotate_session(
            annotator=ANNOTATOR,
            ask=_scripted(["sugar:sugar", ""]),
            show=shown.append,
            sample_path=sample_file,
            gold_path=gold_file,
            assisted=True,
            limit=1,
        )

        assert any("assisted=true" in line for line in shown)


class TestРазметкаОдногоЯзыка:
    """Языковой срез — не удобство, а способ ответить на второй вопрос M2.

    Разница ru 1,28 против de 2,42 различается только на эталоне, а размечать
    сто составов разом никто не станет. Значит, заход по одному языку обязан
    быть штатным режимом, а не ручной вознёй с выборкой.
    """

    def test_размечается_только_запрошенный_язык(self, sample_file: Path, gold_file: Path):
        annotate_session(
            annotator=ANNOTATOR,
            ask=_scripted(["sugar:sugar", "", "water:base", ""]),
            show=lambda _: None,
            sample_path=sample_file,
            gold_path=gold_file,
            lang="ru",
        )

        records = read_jsonl(gold_file, GoldRecord)
        assert [record.code for record in records] == ["1", "3"]
        assert {record.lang for record in records} == {"ru"}

    def test_лимит_действует_внутри_языка(self, sample_file: Path, gold_file: Path):
        added = annotate_session(
            annotator=ANNOTATOR,
            ask=_scripted(["sugar:sugar", ""]),
            show=lambda _: None,
            sample_path=sample_file,
            gold_path=gold_file,
            lang="ru",
            limit=1,
        )

        assert added == 1
        assert [record.code for record in read_jsonl(gold_file, GoldRecord)] == ["1"]

    def test_следующий_заход_продолжает_с_неразмеченного_в_этом_языке(
        self, sample_file: Path, gold_file: Path
    ):
        annotate_session(
            annotator=ANNOTATOR,
            ask=_scripted(["sugar:sugar", ""]),
            show=lambda _: None,
            sample_path=sample_file,
            gold_path=gold_file,
            lang="ru",
            limit=1,
        )
        annotate_session(
            annotator=ANNOTATOR,
            ask=_scripted(["water:base", ""]),
            show=lambda _: None,
            sample_path=sample_file,
            gold_path=gold_file,
            lang="ru",
        )

        assert [record.code for record in read_jsonl(gold_file, GoldRecord)] == ["1", "3"]

    def test_другой_язык_не_трогается(self, sample_file: Path, gold_file: Path):
        """Разметив ru, немецкие обязаны остаться нетронутыми для следующего захода."""
        annotate_session(
            annotator=ANNOTATOR,
            ask=_scripted(["sugar:sugar", "", "water:base", ""]),
            show=lambda _: None,
            sample_path=sample_file,
            gold_path=gold_file,
            lang="ru",
        )
        sample = read_jsonl(sample_file, SampleItem)
        done = read_jsonl(gold_file, GoldRecord)

        assert [item.code for item in pending_items(sample, done, lang="de")] == ["2"]

    def test_опечатка_в_языке_отбивается_а_не_даёт_пустую_сессию(
        self, sample_file: Path, gold_file: Path
    ):
        """«Размечено 0» читается как «всё сделано» и молча съедает заход."""
        with pytest.raises(ValueError, match="de, ru"):
            annotate_session(
                annotator=ANNOTATOR,
                ask=_scripted([]),
                show=lambda _: None,
                sample_path=sample_file,
                gold_path=gold_file,
                lang="py",
            )
