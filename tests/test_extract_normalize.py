"""Тесты канонизации имён ингредиентов.

Проверяется то, ради чего словарь существует: три написания одного сиропа —
это **одна** форма сахара, а не три. Плюс подсчёт неизвестных имён — наш
аналог `unknown_ingredients_n` у Open Food Facts.

Отдельно проверяется настоящий seed-файл из `data/dictionaries/`: он ведётся
руками, и опечатка в нём тихо сломала бы главную метрику проекта.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from nutri_radar.errors import DataSourceError
from nutri_radar.extract.normalize import (
    ANY_LANG,
    DICTIONARIES_DIR,
    SUGAR_SEED_FILE,
    AliasEntry,
    AliasIndex,
    NormalizationStats,
    distinct_sugar_forms,
    distinct_sugar_forms_by_dictionary,
    distinct_sugar_forms_grounded,
    format_unknown_report,
    load_seed_index,
    mentions_alias,
    normalize_ingredients,
    normalize_key,
    sugar_forms_by_dictionary,
    sugar_forms_grounded,
)
from nutri_radar.extract.schemas import Ingredient, IngredientKind

SEED_LINE = (
    '{"canonical_name": "glucose syrup", "kind": "sugar", '
    '"aliases": {"en": ["glucose syrup"], "ru": ["глюкозный сироп", "сироп глюкозы"]}}'
)


def _ingredient(name: str, kind: IngredientKind = IngredientKind.BASE) -> Ingredient:
    return Ingredient(canonical_name=name, kind=kind)


@pytest.fixture
def index() -> AliasIndex:
    return AliasIndex(
        [
            AliasEntry(
                "glucose-fructose syrup", ANY_LANG, "glucose-fructose syrup", IngredientKind.SUGAR
            ),
            AliasEntry(
                "глюкозно-фруктозный сироп", "ru", "glucose-fructose syrup", IngredientKind.SUGAR
            ),
            AliasEntry("sugar", ANY_LANG, "sugar", IngredientKind.SUGAR),
            AliasEntry("сахар", "ru", "sugar", IngredientKind.SUGAR),
            AliasEntry("sorbitol", ANY_LANG, "sorbitol", IngredientKind.SWEETENER),
        ]
    )


class TestКлючСопоставления:
    def test_регистр_и_дефисы_схлопываются(self):
        assert normalize_key("Glucose-Fructose Syrup") == normalize_key("glucose fructose syrup")

    def test_двойные_пробелы_и_пунктуация_убираются(self):
        assert normalize_key("  Sugar,   white!  ") == "sugar white"

    def test_диакритика_снимается(self):
        assert normalize_key("Rübenzucker") == normalize_key("Rubenzucker")

    def test_ё_и_е_совпадают(self):
        """В составах встречаются оба написания слова «мёд»."""
        assert normalize_key("мёд") == normalize_key("мед")

    def test_пустое_имя_даёт_пустой_ключ(self):
        assert normalize_key("   ") == ""


class TestСопоставление:
    def test_разные_написания_сводятся_к_одному_имени(self, index: AliasIndex):
        ingredients = [
            _ingredient("Glucose-Fructose Syrup", IngredientKind.SUGAR),
            _ingredient("glucose fructose syrup", IngredientKind.SUGAR),
            _ingredient("глюкозно-фруктозный сироп", IngredientKind.FLAVOURING),
        ]

        normalized = normalize_ingredients(ingredients, index, lang="ru")

        assert {item.name for item in normalized} == {"glucose-fructose syrup"}
        # И, как следствие, это ОДНА форма сахара, а не три.
        assert distinct_sugar_forms(normalized) == 1

    def test_тип_из_словаря_побеждает_тип_модели(self, index: AliasIndex):
        """Короткий промпт относит к sugar всё сладкое, включая подсластители."""
        normalized = normalize_ingredients([_ingredient("sorbitol", IngredientKind.SUGAR)], index)

        assert normalized[0].kind is IngredientKind.SWEETENER
        assert distinct_sugar_forms(normalized) == 0

    def test_поиск_без_языка_работает(self, index: AliasIndex):
        """Промпт требует английских имён даже на русском составе."""
        normalized = normalize_ingredients([_ingredient("sugar")], index, lang="ru")

        assert normalized[0].name == "sugar"
        assert normalized[0].known

    def test_неизвестное_имя_остаётся_как_есть(self, index: AliasIndex):
        normalized = normalize_ingredients([_ingredient("Mystery Crunch Bits")], index)

        assert not normalized[0].known
        assert normalized[0].name == "mystery crunch bits"
        assert normalized[0].original == "mystery crunch bits"


class TestКонфликтыСловаря:
    def test_один_алиас_на_два_имени_фиксируется(self, caplog):
        with caplog.at_level(logging.WARNING, logger="nutri_radar.extract.normalize"):
            index = AliasIndex(
                [
                    AliasEntry("syrup", "en", "glucose syrup"),
                    AliasEntry("syrup", "en", "sugar syrup"),
                ]
            )
            index.log_summary()

        assert index.conflicts
        assert any("двумя каноническими" in record.message for record in caplog.records)

    def test_повтор_одного_и_того_же_не_конфликт(self):
        index = AliasIndex(
            [
                AliasEntry("sugar", "en", "sugar"),
                AliasEntry("Sugar", "de", "sugar"),
            ]
        )

        assert index.conflicts == []


class TestЧтениеSeedФайла:
    def test_комментарии_и_пустые_строки_пропускаются(self, tmp_path: Path):
        path = tmp_path / "seed.jsonl"
        path.write_text(f"// комментарий\n\n{SEED_LINE}\n", encoding="utf-8")

        index = load_seed_index(tmp_path, files=["seed.jsonl"])

        assert index.lookup("сироп глюкозы").canonical_name == "glucose syrup"

    def test_каноническое_имя_тоже_алиас(self, tmp_path: Path):
        path = tmp_path / "seed.jsonl"
        path.write_text(SEED_LINE, encoding="utf-8")

        index = load_seed_index(tmp_path, files=["seed.jsonl"])

        assert index.lookup("Glucose Syrup").canonical_name == "glucose syrup"

    def test_отсутствующий_файл_отбивается_понятно(self, tmp_path: Path):
        with pytest.raises(DataSourceError, match="ведётся руками"):
            load_seed_index(tmp_path, files=["нет-такого.jsonl"])

    def test_битая_строка_отбивается_с_номером(self, tmp_path: Path):
        path = tmp_path / "seed.jsonl"
        path.write_text(f"{SEED_LINE}\nэто не json\n", encoding="utf-8")

        with pytest.raises(DataSourceError, match="строка 2"):
            load_seed_index(tmp_path, files=["seed.jsonl"])

    def test_запись_без_канонического_имени_отбивается(self, tmp_path: Path):
        path = tmp_path / "seed.jsonl"
        path.write_text('{"aliases": {"en": ["sugar"]}}', encoding="utf-8")

        with pytest.raises(DataSourceError, match="пустое canonical_name"):
            load_seed_index(tmp_path, files=["seed.jsonl"])


class TestНастоящийСловарьПроекта:
    """Файл ведётся руками — опечатка тихо сломала бы главную метрику."""

    @pytest.fixture
    def real_index(self) -> AliasIndex:
        return load_seed_index(DICTIONARIES_DIR, files=[SUGAR_SEED_FILE])

    def test_загружается_без_конфликтов(self, real_index: AliasIndex):
        assert real_index.conflicts == []
        assert real_index.size > 100

    @pytest.mark.parametrize(
        ("written", "canonical"),
        [
            ("Сахар", "sugar"),
            ("Zucker", "sugar"),
            ("Sucre", "sugar"),
            ("Glucose-Fructose Syrup", "glucose-fructose syrup"),
            ("глюкозно-фруктозный сироп", "glucose-fructose syrup"),
            ("Glukose-Fruktose-Sirup", "glucose-fructose syrup"),
            ("мёд", "honey"),
            ("Melasse", "molasses"),
            ("maltodekstryna", "maltodextrin"),
        ],
    )
    def test_формы_сахара_на_разных_языках_сводятся(
        self, real_index: AliasIndex, written: str, canonical: str
    ):
        entry = real_index.lookup(written)

        assert entry is not None, f"нет в словаре: {written}"
        assert entry.canonical_name == canonical
        assert entry.kind is IngredientKind.SUGAR

    def test_подсластители_помечены_своим_видом(self, real_index: AliasIndex):
        for name in ("сорбит", "aspartame", "ksylitol", "stevia"):
            entry = real_index.lookup(name)
            assert entry is not None, f"нет в словаре: {name}"
            assert entry.kind is IngredientKind.SWEETENER


class TestСтатистикаНеизвестных:
    def test_доля_и_топ_неизвестных(self, index: AliasIndex):
        stats = NormalizationStats()
        stats.add(
            normalize_ingredients(
                [_ingredient("sugar"), _ingredient("вкусняшка"), _ingredient("вкусняшка")],
                index,
            )
        )
        stats.add(normalize_ingredients([_ingredient("хрустяшка")], index))

        assert stats.total == 4
        assert stats.known == 1
        assert stats.unknown == 3
        assert stats.unknown_share == 0.75
        assert stats.distinct_unknown == 2
        assert stats.top_unknown(1) == [("вкусняшка", 2)]

    def test_топ_детерминирован_при_равной_частоте(self, index: AliasIndex):
        stats = NormalizationStats()
        stats.add(normalize_ingredients([_ingredient("яяя"), _ingredient("ааа")], index))

        assert stats.top_unknown(2) == [("ааа", 1), ("яяя", 1)]

    def test_отчёт_содержит_числа_и_имена(self, index: AliasIndex):
        stats = NormalizationStats()
        stats.add(normalize_ingredients([_ingredient("sugar"), _ingredient("нечто")], index))

        report = format_unknown_report(stats, 10)

        assert "нечто" in report
        assert "50.0%" in report

    def test_пустая_статистика_не_делит_на_ноль(self):
        stats = NormalizationStats()

        assert stats.unknown_share == 0.0
        assert stats.top_unknown(10) == []


class TestСчётСахараПоСловарю:
    """Регрессия на дефект из ADR-035.

    Хранимое число форм сахара считалось по типу от модели, и на живых данных
    `qwen2.5:3b` проставила `kind=sugar` овсу, соли, молоку и списку
    аллергенов подряд — у одного продукта вышло «30 форм сахара», из которых
    сахаром не была ни одна. Словарь этот вердикт больше не наследует.
    """

    def test_вердикт_модели_не_делает_ингредиент_сахаром(self, index: AliasIndex):
        ингредиенты = [
            _ingredient("oats", IngredientKind.SUGAR),
            _ingredient("salt", IngredientKind.SUGAR),
            _ingredient("milk", IngredientKind.SUGAR),
        ]

        assert distinct_sugar_forms_by_dictionary(ингредиенты, index) == 0

    def test_словарь_подтверждает_настоящий_сахар(self, index: AliasIndex):
        ингредиенты = [_ingredient("sugar"), _ingredient("oats", IngredientKind.SUGAR)]

        assert sugar_forms_by_dictionary(ингредиенты, index) == {"sugar"}

    def test_разные_написания_считаются_одной_формой(self, index: AliasIndex):
        ингредиенты = [
            _ingredient("глюкозно-фруктозный сироп"),
            _ingredient("glucose-fructose syrup"),
        ]

        assert distinct_sugar_forms_by_dictionary(ингредиенты, index, lang="ru") == 1

    def test_разные_формы_считаются_по_отдельности(self, index: AliasIndex):
        ингредиенты = [_ingredient("сахар"), _ingredient("глюкозно-фруктозный сироп")]

        assert distinct_sugar_forms_by_dictionary(ингредиенты, index, lang="ru") == 2

    def test_подсластитель_не_считается_формой_сахара(self, index: AliasIndex):
        """`sorbitol` в словаре помечен `sweetener`, и это не сахар."""
        assert distinct_sugar_forms_by_dictionary([_ingredient("sorbitol")], index) == 0

    def test_старый_счёт_на_тех_же_данных_даёт_завышенное_число(self, index: AliasIndex):
        """Прямое сравнение двух способов — то самое расхождение из ADR-035."""
        ингредиенты = [
            _ingredient("oats", IngredientKind.SUGAR),
            _ingredient("salt", IngredientKind.SUGAR),
            _ingredient("sugar", IngredientKind.SUGAR),
        ]

        по_модели = distinct_sugar_forms(normalize_ingredients(ингредиенты, index))
        по_словарю = distinct_sugar_forms_by_dictionary(ингредиенты, index)

        assert по_модели == 3
        assert по_словарю == 1

    def test_пустой_состав_даёт_ноль(self, index: AliasIndex):
        assert distinct_sugar_forms_by_dictionary([], index) == 0


class TestНастоящийСловарьЗакрываетНайденныеДыры:
    """Имена, которых словарю не хватило на живых данных (ADR-035)."""

    @pytest.fixture
    def настоящий(self) -> AliasIndex:
        return load_seed_index()

    @pytest.mark.parametrize(
        ("имя", "язык"),
        [
            ("fructose syrup", "en"),
            ("karamellzuckersirup", "de"),
            ("rohrohrzucker", "de"),
            ("organic cane sugar", "en"),
            ("laktoza z mleka", "pl"),
        ],
    )
    def test_форма_сахара_из_живых_данных_теперь_известна(
        self, настоящий: AliasIndex, имя: str, язык: str
    ):
        assert distinct_sugar_forms_by_dictionary([_ingredient(имя)], настоящий, lang=язык) == 1

    @pytest.mark.parametrize("имя", ["milk", "salt", "cocoa butter", "wheat flour", "gluten"])
    def test_не_сахар_из_живых_данных_сахаром_не_становится(self, настоящий: AliasIndex, имя: str):
        """Эти имена модель называла сахаром чаще всего."""
        ингредиент = _ingredient(имя, IngredientKind.SUGAR)

        assert distinct_sugar_forms_by_dictionary([ингредиент], настоящий) == 0


class TestСверкаСТекстомСостава:
    """Вторая проверка из ADR-035: форма обязана встречаться в составе.

    Словарь отвечает «это сахар», текст — «он тут есть». Измерено, что без
    второй проверки 38,1% посчитанных форм не имеют опоры в исходном тексте:
    модель дописывает правдоподобное.
    """

    СОСТАВ = "Мука пшеничная, сахар, масло растительное, глюкозно-фруктозный сироп"

    def test_форма_из_текста_засчитывается(self, index: AliasIndex):
        формы = sugar_forms_grounded(
            [_ingredient("сахар")], index, source_text=self.СОСТАВ, lang="ru"
        )

        assert формы == {"sugar"}

    def test_выдуманная_форма_не_засчитывается(self, index: AliasIndex):
        """Ровно случай йогурта «ZERO SUGAR», которому достались мёд и патока."""
        формы = sugar_forms_grounded(
            [_ingredient("glucose-fructose syrup")],
            index,
            source_text="Молоко цельное, закваска",
            lang="ru",
        )

        assert формы == set()

    def test_опора_ищется_по_любому_языку_словаря(self, index: AliasIndex):
        """Модель отвечает по-английски даже на русский состав — это норма."""
        формы = sugar_forms_grounded(
            [_ingredient("glucose-fructose syrup")], index, source_text=self.СОСТАВ, lang="ru"
        )

        assert формы == {"glucose-fructose syrup"}

    def test_пустой_текст_состава_не_даёт_опоры_никому(self, index: AliasIndex):
        assert distinct_sugar_forms_grounded([_ingredient("сахар")], index, source_text="") == 0

    def test_сверка_строже_словаря_на_тех_же_данных(self, index: AliasIndex):
        ингредиенты = [_ingredient("сахар"), _ingredient("глюкозно-фруктозный сироп")]
        текст = "Мука пшеничная, сахар, соль"

        по_словарю = distinct_sugar_forms_by_dictionary(ингредиенты, index, lang="ru")
        со_сверкой = distinct_sugar_forms_grounded(ингредиенты, index, source_text=текст, lang="ru")

        assert по_словарю == 2
        assert со_сверкой == 1

    def test_отрицание_проверка_не_различает(self, index: AliasIndex):
        """Честная фиксация границы: «без сахара» содержит слово «сахар».

        То же ограничение измерено у поиска на запросах «без пальмового
        масла» (ADR-029). Тест закрепляет известное поведение, а не желаемое:
        если оно изменится, это должно быть решением, а не случайностью.
        """
        формы = sugar_forms_grounded(
            [_ingredient("сахар")], index, source_text="Напиток без сахара", lang="ru"
        )

        assert формы == {"sugar"}


class TestГраницыСлов:
    def test_совпадение_идёт_с_начала_слова(self):
        """`сахар` находит `сахара`, но не находит `несахар`."""
        assert mentions_alias("мука сахара соль", ["сахар"]) is True
        assert mentions_alias("подсахар", ["сахар"]) is False

    def test_одиночная_проверка_не_разделяет_вложенные_имена(self):
        """`glucose` найдётся внутри `glucose fructose syrup` — и это ожидаемо.

        Разделение таких случаев делает `sugar_forms_grounded`, разбирая
        длинные формы первыми; у одиночной проверки такой задачи нет.
        """
        assert mentions_alias("glucose fructose syrup", ["glucose"]) is True

    def test_отдельное_слово_находится(self):
        assert mentions_alias("sugar salt water", ["sugar"]) is True

    def test_составной_алиас_находится(self):
        assert mentions_alias("wheat flour glucose syrup salt", ["glucose syrup"]) is True

    def test_пустой_алиас_не_даёт_ложной_опоры(self):
        assert mentions_alias("sugar", [""]) is False


class TestВложенныеИменаРазделяются:
    """Длинные формы разбираются первыми и вычёркивают найденное."""

    @pytest.fixture
    def индекс(self) -> AliasIndex:
        return AliasIndex(
            [
                AliasEntry("glucose", ANY_LANG, "glucose", IngredientKind.SUGAR),
                AliasEntry(
                    "glucose-fructose syrup",
                    ANY_LANG,
                    "glucose-fructose syrup",
                    IngredientKind.SUGAR,
                ),
            ]
        )

    def test_длинная_форма_забирает_своё_вхождение(self, индекс: AliasIndex):
        """В составе одна форма, а модель назвала две — засчитается длинная."""
        формы = sugar_forms_grounded(
            [_ingredient("glucose"), _ingredient("glucose-fructose syrup")],
            индекс,
            # Текст по-английски намеренно: у этого индекса русских алиасов нет,
            # а проверяется здесь разбор вложенных имён, а не многоязычность.
            source_text="Water, glucose-fructose syrup",
        )

        assert формы == {"glucose-fructose syrup"}

    def test_обе_формы_засчитываются_когда_обе_в_составе(self, индекс: AliasIndex):
        формы = sugar_forms_grounded(
            [_ingredient("glucose"), _ingredient("glucose-fructose syrup")],
            индекс,
            source_text="Glucose-fructose syrup, water, glucose",
        )

        assert формы == {"glucose", "glucose-fructose syrup"}


class TestОбратныйПорядокСлов:
    """Второй проход сверки — реальная потеря на живых данных.

    У печенья Oreo в составе «сироп глюкозно-фруктозный», а в словаре
    «глюкозно-фруктозный сироп». До второго прохода настоящая форма сахара
    не засчитывалась.
    """

    def test_обратный_порядок_слов_находится(self, index: AliasIndex):
        формы = sugar_forms_grounded(
            [_ingredient("glucose-fructose syrup")],
            index,
            source_text="Мука пшеничная, сахар, сироп глюкозно-фруктозный",
            lang="ru",
        )

        assert формы == {"glucose-fructose syrup"}

    def test_одного_слова_из_двух_не_хватает(self, index: AliasIndex):
        """Требуются ВСЕ слова имени, иначе опора засчиталась бы от «сиропа»."""
        формы = sugar_forms_grounded(
            [_ingredient("glucose-fructose syrup")],
            index,
            source_text="Вода, сироп из топинамбура",
            lang="ru",
        )

        assert формы == set()
