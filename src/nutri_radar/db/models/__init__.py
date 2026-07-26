"""Таблицы SQLAlchemy.

Каждая модель импортируется здесь, иначе автогенерация Alembic увидит пустую
схему: `Base.metadata` наполняется только при импорте модулей с моделями.

Таблицы продуктов (`products_raw`, `products`, `product_extraction`,
`ingredients_dict`, `product_embeddings`) проектируются на M1 и M2, когда
известна фактическая форма данных из дампа.
"""

from __future__ import annotations

from nutri_radar.db.models.run import Run, RunStage, RunStatus

__all__ = ["Run", "RunStage", "RunStatus"]
