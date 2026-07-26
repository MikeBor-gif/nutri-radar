"""Слайс извлечения структуры состава.

Ядро проекта: грязный многоязычный текст состава превращается в структуру
по JSON-схеме, включая число разных форм сахара — величину, которой в
Open Food Facts нет.
"""

from __future__ import annotations

from nutri_radar.extract.schemas import ExtractionResult, Ingredient, IngredientKind

__all__ = ["ExtractionResult", "Ingredient", "IngredientKind"]
