"""Репозитории: запросы к Postgres.

Наружу отдают Pydantic-модели, а не ORM-объекты: иначе ленивая загрузка
выстрелит `MissingGreenlet` при обращении к атрибуту вне активной сессии
(см. ARCHITECTURE.md, антипаттерны).
"""

from __future__ import annotations

from nutri_radar.db.repositories.extraction import ExtractionRepository, ExtractionRow
from nutri_radar.db.repositories.ingredient import AliasRow, IngredientAliasRepository
from nutri_radar.db.repositories.product import ProductRepository

__all__ = [
    "AliasRow",
    "ExtractionRepository",
    "ExtractionRow",
    "IngredientAliasRepository",
    "ProductRepository",
]
