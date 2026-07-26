"""Таблицы SQLAlchemy.

Каждая модель импортируется здесь, иначе автогенерация Alembic увидит пустую
схему: `Base.metadata` наполняется только при импорте модулей с моделями.

Таблица `product_embeddings` появится на M5, когда будет известна
размерность и стратегия индексации.
"""

from __future__ import annotations

from nutri_radar.db.models.extraction import IngredientAlias, ProductExtraction
from nutri_radar.db.models.product import Product, ProductRaw
from nutri_radar.db.models.run import Run, RunStage, RunStatus

__all__ = [
    "IngredientAlias",
    "Product",
    "ProductExtraction",
    "ProductRaw",
    "Run",
    "RunStage",
    "RunStatus",
]
