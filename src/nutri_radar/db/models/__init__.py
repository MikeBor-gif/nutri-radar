"""Таблицы SQLAlchemy.

Каждая модель импортируется здесь, иначе автогенерация Alembic увидит пустую
схему: `Base.metadata` наполняется только при импорте модулей с моделями.

Таблица `product_embedding` добавлена на M5: `halfvec(1024)` под `bge-m3`,
индекс HNSW строится отдельным шагом после заливки, а не миграцией.
"""

from __future__ import annotations

from nutri_radar.db.models.embedding import EMBEDDING_DIM, ProductEmbedding
from nutri_radar.db.models.extraction import IngredientAlias, ProductExtraction
from nutri_radar.db.models.product import Product, ProductRaw
from nutri_radar.db.models.run import Run, RunStage, RunStatus

__all__ = [
    "EMBEDDING_DIM",
    "IngredientAlias",
    "Product",
    "ProductEmbedding",
    "ProductExtraction",
    "ProductRaw",
    "Run",
    "RunStage",
    "RunStatus",
]
