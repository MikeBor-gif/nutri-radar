"""Таблицы SQLAlchemy.

Каждая модель импортируется здесь, иначе автогенерация Alembic увидит пустую
схему: `Base.metadata` наполняется только при импорте модулей с моделями.

Таблицы `product_extraction`, `ingredients_dict` и `product_embeddings`
появятся на M2 и M5, когда будет известна форма их данных.
"""

from __future__ import annotations

from nutri_radar.db.models.product import Product, ProductRaw
from nutri_radar.db.models.run import Run, RunStage, RunStatus

__all__ = ["Product", "ProductRaw", "Run", "RunStage", "RunStatus"]
