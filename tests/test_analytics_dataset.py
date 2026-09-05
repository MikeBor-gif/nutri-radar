"""Тесты выгрузки и сплита.

Проверяется не «функция вызывается», а три свойства, от которых зависят
все числа майлстоуна.

**Воспроизводимость.** Сплит с тем же seed обязан давать то же разбиение.
Иначе два прогона посчитаны на разных множествах, и разницу их точности
нельзя приписать ни модели, ни данным.

**Непересечение train и test.** Продукт, попавший в оба, — это утечка,
которая поднимает точность и ничего не значит.

**Отсутствие нутриентов.** `nutriscore_grade` считается по ним, и попади
они в признаки, задача выродилась бы в пересчёт формулы. Проверяется
структурно: список колонок закрыт.

Базы здесь нет — всё на синтетическом датафрейме.
"""

from __future__ import annotations

import pandas as pd
import pytest

from nutri_radar.analytics.dataset import (
    COLUMNS,
    TARGETS,
    common_subset,
    load_dataset,
    majority_baseline,
    prepare,
    save_dataset,
    split,
)
from nutri_radar.config import AnalyticsSettings, Settings

GRADES = ["a", "b", "c", "d", "e"]


def _frame(rows: int = 500, langs: tuple[str, ...] = ("ru", "de", "fr")) -> pd.DataFrame:
    """Синтетический набор с управляемым распределением классов.

    Классы намеренно неравные: на равномерных данных стратификация
    неотличима от случайного сплита, и тест ничего бы не проверял.
    """
    grades = [GRADES[min(i % 9, 4)] for i in range(rows)]
    return pd.DataFrame(
        {
            "code": [str(100000 + i) for i in range(rows)],
            "ingredients_text": [f"сахар, вода, продукт номер {i}" for i in range(rows)],
            "lang": [langs[i % len(langs)] for i in range(rows)],
            "nutriscore_grade": grades,
            "nova_group": [1 if i % 7 == 0 else 4 for i in range(rows)],
        }
    )


@pytest.fixture
def analytics_settings(settings: Settings) -> Settings:
    """Настройки с явными значениями — тест не должен зависеть от `.env`."""
    return settings.model_copy(
        update={
            "analytics": AnalyticsSettings(
                random_seed=42, test_size=0.2, llm_subset_size=50, embed_train_size=100
            )
        }
    )


class TestКолонки:
    def test_нутриентов_в_наборе_нет(self):
        """Структурная защита от утечки: список колонок закрыт."""
        forbidden = {"sugars_100g", "energy_kcal_100g", "salt_100g", "nutrition_score_fr_100g"}

        assert not forbidden & set(COLUMNS)

    def test_набор_колонок_ровно_тот_что_нужен(self):
        assert set(COLUMNS) == {
            "code",
            "ingredients_text",
            "lang",
            "nutriscore_grade",
            "nova_group",
        }


class TestПодготовка:
    def test_строки_без_метки_отбрасываются(self):
        frame = _frame(10)
        frame.loc[0:2, "nutriscore_grade"] = None

        assert len(prepare(frame, "nutriscore_grade")) == 7

    def test_целая_метка_не_становится_дробной(self):
        """`nova_group` в базе целое; «4.0» — артефакт хранения NaN в float."""
        frame = _frame(10)
        frame["nova_group"] = frame["nova_group"].astype("float64")

        assert set(prepare(frame, "nova_group")["nova_group"]) <= {"1", "4"}

    def test_неизвестная_метка_отбивается(self):
        with pytest.raises(ValueError, match="Неизвестная метка"):
            prepare(_frame(10), "sugars_100g")

    def test_все_метки_из_TARGETS_поддержаны(self):
        frame = _frame(50)

        for target in TARGETS:
            assert not prepare(frame, target).empty


class TestСплит:
    def test_воспроизводим_по_seed(self, analytics_settings: Settings):
        """Два прогона с тем же seed обязаны дать то же разбиение."""
        frame = prepare(_frame(), "nutriscore_grade")

        first, _ = split(frame, "nutriscore_grade", analytics_settings)
        second, _ = split(frame, "nutriscore_grade", analytics_settings)

        assert list(first["code"]) == list(second["code"])

    def test_train_и_test_не_пересекаются(self, analytics_settings: Settings):
        """Продукт в обоих множествах — утечка, поднимающая точность даром."""
        frame = prepare(_frame(), "nutriscore_grade")

        train, test = split(frame, "nutriscore_grade", analytics_settings)

        assert not set(train["code"]) & set(test["code"])

    def test_ничего_не_теряется(self, analytics_settings: Settings):
        frame = prepare(_frame(), "nutriscore_grade")

        train, test = split(frame, "nutriscore_grade", analytics_settings)

        assert len(train) + len(test) == len(frame)

    def test_доли_классов_сохраняются(self, analytics_settings: Settings):
        """Смысл стратификации: доля редкого класса не должна гулять."""
        frame = prepare(_frame(), "nutriscore_grade")
        expected = frame["nutriscore_grade"].value_counts(normalize=True)

        _, test = split(frame, "nutriscore_grade", analytics_settings)
        actual = test["nutriscore_grade"].value_counts(normalize=True)

        for label in expected.index:
            assert abs(expected[label] - actual[label]) < 0.02

    def test_вырожденный_класс_не_роняет_прогон(self, analytics_settings: Settings):
        """nova_group=2 встречается 25 раз на 138 тысяч — падать из-за него
        значило бы не считать nova вовсе."""
        frame = prepare(_frame(200), "nutriscore_grade")
        frame.loc[0, "nutriscore_grade"] = "z"

        train, test = split(frame, "nutriscore_grade", analytics_settings)

        assert "z" not in set(train["nutriscore_grade"]) | set(test["nutriscore_grade"])
        assert len(train) + len(test) == len(frame) - 1


class TestОбщаяПодвыборка:
    def test_размер_берётся_из_настроек(self, analytics_settings: Settings):
        frame = prepare(_frame(), "nutriscore_grade")
        _, test = split(frame, "nutriscore_grade", analytics_settings)

        assert len(common_subset(test, "nutriscore_grade", analytics_settings)) == 50

    def test_воспроизводима_по_seed(self, analytics_settings: Settings):
        """На ней меряются все три подхода — состав обязан совпадать
        независимо от порядка запуска."""
        frame = prepare(_frame(), "nutriscore_grade")
        _, test = split(frame, "nutriscore_grade", analytics_settings)

        first = common_subset(test, "nutriscore_grade", analytics_settings)
        second = common_subset(test, "nutriscore_grade", analytics_settings)

        assert list(first["code"]) == list(second["code"])

    def test_подвыборка_лежит_внутри_теста(self, analytics_settings: Settings):
        frame = prepare(_frame(), "nutriscore_grade")
        _, test = split(frame, "nutriscore_grade", analytics_settings)

        subset = common_subset(test, "nutriscore_grade", analytics_settings)

        assert set(subset["code"]) <= set(test["code"])

    def test_тест_меньше_запрошенного_берётся_целиком(self, analytics_settings: Settings):
        frame = prepare(_frame(60), "nutriscore_grade")
        _, test = split(frame, "nutriscore_grade", analytics_settings)

        assert len(common_subset(test, "nutriscore_grade", analytics_settings)) == len(test)


class TestБазлайн:
    def test_возвращает_самый_частый_класс_и_долю(self):
        frame = pd.DataFrame({"nutriscore_grade": ["e"] * 7 + ["a"] * 3})

        label, share = majority_baseline(frame, "nutriscore_grade")

        assert label == "e"
        assert share == pytest.approx(0.7)

    def test_пустой_набор_не_делит_на_ноль(self):
        frame = pd.DataFrame({"nutriscore_grade": []})

        assert majority_baseline(frame, "nutriscore_grade") == ("", 0.0)


class TestКэш:
    def test_запись_и_чтение_совпадают(self, tmp_path):
        frame = _frame(20)
        path = save_dataset(frame, tmp_path / "dataset.parquet")

        restored = load_dataset(path)

        assert list(restored["code"]) == list(frame["code"])
        assert set(restored.columns) == set(frame.columns)

    def test_отсутствие_файла_объясняется_командой(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="analytics dataset"):
            load_dataset(tmp_path / "нет-такого.parquet")
