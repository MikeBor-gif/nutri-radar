"""Тесты внутренней модели источников.

`RawProduct` — контракт между двумя адаптерами (ADR-004), поэтому его правила
проверяются отдельно от чтения файлов: они относятся к предметной области,
а не к формату.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nutri_radar.ingest.models import MAIN_LANG_MARKER, RawProduct

LANGS = ["en", "ru", "de", "fr", "pl"]
MIN_LEN = 10


def make(**overrides) -> RawProduct:
    payload = {"code": "1000000000001", "source": "parquet"}
    payload.update(overrides)
    return RawProduct.model_validate(payload)


class TestBarcode:
    @pytest.mark.parametrize("bad", ["", "abc", None, "12", "123", "1234abc", "   "])
    def test_мусорный_штрихкод_отбивается(self, bad):
        with pytest.raises(ValidationError):
            make(code=bad)

    @pytest.mark.parametrize("good", ["1234", "3017620422003", "20012345"])
    def test_валидный_штрихкод_проходит(self, good):
        assert make(code=good).code == good

    def test_пробелы_обрезаются(self):
        assert make(code="  3017620422003  ").code == "3017620422003"


class TestMainLangMarker:
    """Служебная запись `main` дублирует главный язык и языком не является."""

    def test_main_исключается_из_состава(self):
        product = make(
            ingredients_text={
                MAIN_LANG_MARKER: "Sugar, palm oil, hazelnuts",
                "en": "Sugar, palm oil, hazelnuts",
                "fr": "Sucre, huile de palme, noisettes",
            }
        )

        assert MAIN_LANG_MARKER not in product.ingredients_text
        assert product.languages == ["en", "fr"]

    def test_main_исключается_и_из_названий(self):
        product = make(product_name={MAIN_LANG_MARKER: "Nutella", "en": "Nutella"})
        assert list(product.product_name) == ["en"]

    def test_продукт_только_с_main_остаётся_без_языков(self):
        product = make(ingredients_text={MAIN_LANG_MARKER: "Sugar, palm oil, hazelnuts"})
        assert product.languages == []
        assert product.has_usable_ingredients(LANGS, MIN_LEN) is False

    def test_пустые_тексты_выбрасываются(self):
        product = make(ingredients_text={"en": "   ", "ru": "Сахар, масло"})
        assert product.languages == ["ru"]


class TestNutriments:
    def test_отсутствующий_нутриент_не_становится_нулём(self):
        """Ноль осмыслен как значение, подмена уехала бы в обучение M4."""
        product = make(nutriments={"sugars": 12.5, "fiber": None, "salt": 0.0})

        assert "fiber" not in product.nutriments
        assert product.nutriments["salt"] == 0.0
        assert product.nutriments["sugars"] == 12.5

    def test_пустые_нутриенты_дают_отсутствие_данных(self):
        assert make(nutriments={}).has_nutrition_data is False

    def test_флаг_источника_перевешивает_наличие_значений(self):
        product = make(nutriments={"fat": 1.0}, no_nutrition_data=True)
        assert product.has_nutrition_data is False


class TestNutriscoreGrade:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("A", "a"), ("c", "c"), (" D ", "d"), ("e", "e")],
    )
    def test_буквы_нормализуются(self, raw, expected):
        assert make(nutriscore_grade=raw).nutriscore_grade == expected

    @pytest.mark.parametrize("service", ["unknown", "not-applicable", "", "  ", None])
    def test_служебные_значения_становятся_none(self, service):
        """Иначе они попали бы в обучение M4 как отдельный класс."""
        assert make(nutriscore_grade=service).nutriscore_grade is None


class TestUsableIngredients:
    def test_короткий_состав_не_годится(self):
        product = make(ingredients_text={"en": "Sugar"})
        assert product.has_usable_ingredients(LANGS, MIN_LEN) is False

    def test_язык_вне_списка_не_годится(self):
        product = make(ingredients_text={"ja": "砂糖、パーム油、ヘーゼルナッツ、ココア"})
        assert product.has_usable_ingredients(LANGS, MIN_LEN) is False

    def test_годные_языки_возвращаются_отсортированными(self):
        product = make(
            ingredients_text={
                "ru": "Сахар, пальмовое масло",
                "en": "Sugar",  # короче порога
                "fr": "Sucre, huile de palme",
            }
        )
        assert product.usable_ingredients_languages(LANGS, MIN_LEN) == ["fr", "ru"]


class TestPickText:
    def test_приоритет_у_главного_языка(self):
        product = make(
            lang="fr",
            ingredients_text={
                "en": "Sugar, palm oil, hazelnuts",
                "fr": "Sucre, huile de palme",
            },
        )
        picked = product.pick_ingredients_text(LANGS, MIN_LEN)
        assert picked is not None
        assert picked[0] == "fr"

    def test_если_главный_язык_не_годен_берётся_порядок_настроек(self):
        product = make(
            lang="en",
            ingredients_text={"en": "Sugar", "ru": "Сахар, пальмовое масло"},
        )
        picked = product.pick_ingredients_text(LANGS, MIN_LEN)
        assert picked is not None
        assert picked[0] == "ru"

    def test_без_годных_языков_возвращается_none(self):
        assert make(ingredients_text={"en": "Sugar"}).pick_ingredients_text(LANGS, MIN_LEN) is None


class TestQuality:
    def test_снятый_с_производства_не_проходит(self):
        assert make(obsolete=True).is_quality_ok is False

    def test_ошибки_качества_не_проходят(self):
        assert make(data_quality_errors=["en:nutrition-value-over-105"]).is_quality_ok is False

    def test_чистая_запись_проходит(self):
        assert make().is_quality_ok is True


class TestRawJson:
    def test_снимок_содержит_поля_для_products_raw(self):
        product = make(
            ingredients_text={"en": "Sugar, palm oil, hazelnuts"},
            unknown_ingredients_n=3,
            rev=42,
        )
        snapshot = product.to_raw_json()

        assert snapshot["code"] == "1000000000001"
        assert snapshot["ingredients_text"] == {"en": "Sugar, palm oil, hazelnuts"}
        assert snapshot["unknown_ingredients_n"] == 3
        assert snapshot["rev"] == 42
        assert snapshot["source"] == "parquet"

    def test_модель_неизменяема(self):
        """Данные источника не должны меняться по дороге в БД."""
        product = make()
        with pytest.raises(ValidationError):
            product.code = "9999999999999"
