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
    format_unknown_report,
    load_seed_index,
    normalize_ingredients,
    normalize_key,
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
