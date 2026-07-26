"""Схема выхода извлечения — ядро проекта.

Этот класс играет две роли сразу: его JSON-схема уходит в параметр `format`
Ollama и ограничивает генерацию, и он же валидирует пришедший ответ.

Здесь живёт **число разных форм сахара** — величина, которой в Open Food Facts
нет вообще. OFF считает добавки (`additives_n`), но формы сахара не считает,
и это делает фичу уникальной, а не дублирующей существующее.

Производные величины — свойства модели, а не логика раннера: их тесты не должны
требовать поднятой LLM (см. ARCHITECTURE.md, антипаттерн «логика в раннере»).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class IngredientKind(StrEnum):
    """Тип ингредиента.

    Перечисление уходит в JSON-схему как `enum`, поэтому модель физически
    не может вернуть значение вне списка.
    """

    BASE = "base"
    SUGAR = "sugar"
    FAT = "fat"
    ADDITIVE = "additive"
    FLAVOURING = "flavouring"
    PRESERVATIVE = "preservative"
    SWEETENER = "sweetener"


class Ingredient(BaseModel):
    """Один ингредиент в каноническом виде."""

    # Английский и в нижнем регистре: имена сопоставляются с таксономией OFF,
    # где теги вида `en:sugar`. Проверено на живой модели — русский промпт
    # заставлял её переводить имена, и сопоставление ломалось.
    canonical_name: str = Field(description="Canonical ingredient name in English, lowercase")
    kind: IngredientKind
    e_number: str | None = Field(default=None, description="E-number if present, else null")

    @field_validator("canonical_name", mode="before")
    @classmethod
    def _normalize_name(cls, value: object) -> str:
        return str(value or "").strip().lower()

    @field_validator("e_number", mode="before")
    @classmethod
    def _empty_is_none(cls, value: object) -> object:
        """Модель возвращает пустую строку вместо null — приводим к None."""
        if isinstance(value, str) and not value.strip():
            return None
        return value


class ExtractionResult(BaseModel):
    """Результат разбора одного состава."""

    ingredients: list[Ingredient] = Field(default_factory=list)
    allergens: list[str] = Field(default_factory=list)
    # Флаг из раздела 8 брифа: такие записи не должны портить аналитику.
    unreadable: bool = False
    model_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("allergens", mode="before")
    @classmethod
    def _clean_allergens(cls, value: object) -> object:
        """Убрать пустые и привести к нижнему регистру.

        Модель иногда возвращает пустую строку в списке — на реальных данных
        такое встречалось.
        """
        if not isinstance(value, list):
            return value
        return [str(item).strip().lower() for item in value if str(item).strip()]

    @property
    def distinct_sugar_forms(self) -> int:
        """Сколько РАЗНЫХ названий сахара в составе.

        Ключевая производная фича проекта и самая наглядная часть демо.
        Считается по каноническим именам: «сироп глюкозы» дважды в одном
        составе — это одна форма, а сироп глюкозы и мальтодекстрин — две.
        """
        return len(self.sugar_names)

    @property
    def sugar_names(self) -> set[str]:
        return {i.canonical_name for i in self.ingredients if i.kind is IngredientKind.SUGAR}

    @property
    def e_additives_count(self) -> int:
        """Число ингредиентов с E-номером. Прямой аналог `additives_n` у OFF."""
        return sum(1 for i in self.ingredients if i.e_number)

    @property
    def is_usable(self) -> bool:
        """Годится ли результат для аналитики."""
        return not self.unreadable and bool(self.ingredients)

    @classmethod
    def json_schema_for_llm(cls) -> dict[str, Any]:
        """Схема для параметра `format` Ollama.

        Отдельный метод, а не `model_json_schema()` напрямую: pydantic
        генерирует `$defs` со ссылками `$ref`, и их нужно развернуть —
        не всякий рантайм генерации по схеме понимает ссылки.
        """
        schema = cls.model_json_schema()
        return _inline_refs(schema)


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Развернуть `$ref` на `$defs` прямо в схему."""
    defs = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                target = defs.get(ref.removeprefix("#/$defs/"), {})
                merged = {**resolve(target), **{k: v for k, v in node.items() if k != "$ref"}}
                return merged
            return {key: resolve(value) for key, value in node.items()}
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    return resolve(schema)
