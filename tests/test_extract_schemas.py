"""Тесты схемы извлечения.

Здесь проверяется **ключевая величина проекта** — число разных форм сахара.
Она считается свойством модели именно ради этих тестов: чтобы проверить её,
не нужна ни LLM, ни база (ARCHITECTURE.md, антипаттерн «логика в раннере»).
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from nutri_radar.extract.schemas import ExtractionResult, Ingredient, IngredientKind


def _ingredient(name: str, kind: IngredientKind, e_number: str | None = None) -> Ingredient:
    return Ingredient(canonical_name=name, kind=kind, e_number=e_number)


class TestDistinctSugarForms:
    def test_считает_разные_имена_а_не_вхождения(self):
        """Один сироп дважды — это одна форма, а не две."""
        result = ExtractionResult(
            ingredients=[
                _ingredient("glucose syrup", IngredientKind.SUGAR),
                _ingredient("glucose syrup", IngredientKind.SUGAR),
                _ingredient("sugar", IngredientKind.SUGAR),
            ]
        )

        assert result.distinct_sugar_forms == 2

    def test_реальный_случай_с_живой_проверки(self):
        """Состав «Sugar … Glucose-Fructose Syrup … Molasses» — три формы.

        Именно на нём наивный промпт вернул ноль форм сахара (см. план M2),
        поэтому случай зафиксирован тестом.
        """
        result = ExtractionResult(
            ingredients=[
                _ingredient("sugar", IngredientKind.SUGAR),
                _ingredient("glucose-fructose syrup", IngredientKind.SUGAR),
                _ingredient("molasses", IngredientKind.SUGAR),
                _ingredient("wheat flour", IngredientKind.BASE),
                _ingredient("palm oil", IngredientKind.FAT),
            ]
        )

        assert result.distinct_sugar_forms == 3

    def test_не_сахарные_виды_не_считаются(self):
        """Подсластитель — не форма сахара, это разные вещи."""
        result = ExtractionResult(
            ingredients=[
                _ingredient("sugar", IngredientKind.SUGAR),
                _ingredient("sorbitol", IngredientKind.SWEETENER),
                _ingredient("aspartame", IngredientKind.SWEETENER),
            ]
        )

        assert result.distinct_sugar_forms == 1

    def test_пустой_состав_даёт_ноль(self):
        assert ExtractionResult().distinct_sugar_forms == 0


class TestДругиеПроизводные:
    def test_e_добавки_считаются_по_наличию_номера(self):
        result = ExtractionResult(
            ingredients=[
                _ingredient("lecithin", IngredientKind.ADDITIVE, "E322"),
                _ingredient("citric acid", IngredientKind.ADDITIVE, "E330"),
                _ingredient("salt", IngredientKind.BASE),
            ]
        )

        assert result.e_additives_count == 2

    def test_пустая_строка_в_e_номере_это_отсутствие_номера(self):
        """Модель возвращает "" вместо null — встречалось на живых данных."""
        ingredient = Ingredient(canonical_name="salt", kind=IngredientKind.BASE, e_number="  ")

        assert ingredient.e_number is None

    def test_имя_приводится_к_нижнему_регистру(self):
        ingredient = Ingredient(canonical_name="  Glucose Syrup  ", kind=IngredientKind.SUGAR)

        assert ingredient.canonical_name == "glucose syrup"

    def test_пустые_аллергены_отбрасываются(self):
        result = ExtractionResult(allergens=["Milk", "", "  ", "Soy"])

        assert result.allergens == ["milk", "soy"]

    def test_нечитаемый_результат_не_годится_для_аналитики(self):
        readable = ExtractionResult(ingredients=[_ingredient("salt", IngredientKind.BASE)])
        unreadable = ExtractionResult(
            ingredients=[_ingredient("salt", IngredientKind.BASE)], unreadable=True
        )

        assert readable.is_usable
        assert not unreadable.is_usable
        # Пустой список ингредиентов тоже нечего анализировать.
        assert not ExtractionResult().is_usable

    def test_уверенность_вне_диапазона_отбивается(self):
        with pytest.raises(ValidationError):
            ExtractionResult(model_confidence=1.5)


class TestJsonSchemaForLlm:
    def test_ссылки_на_defs_развёрнуты(self):
        """Не всякий рантайм генерации по схеме понимает `$ref`."""
        schema = ExtractionResult.json_schema_for_llm()
        rendered = json.dumps(schema)

        assert "$defs" not in schema
        assert "$ref" not in rendered

    def test_перечисление_видов_попадает_в_схему(self):
        """Enum в схеме — то, что физически не даёт модели выдумать вид."""
        schema = ExtractionResult.json_schema_for_llm()
        rendered = json.dumps(schema)

        for kind in IngredientKind:
            assert kind.value in rendered

    def test_схема_описывает_ожидаемые_поля(self):
        schema = ExtractionResult.json_schema_for_llm()

        assert set(schema["properties"]) == {
            "ingredients",
            "allergens",
            "unreadable",
            "model_confidence",
        }
