"""Адаптеры источников данных.

Их два, и это принципиально (ADR-004):

* `parquet` — полный дамп, вложенные структуры `LIST<STRUCT(...)>`;
* `delta` — дельта-экспорты, JSONL в MongoDB-форме с плоскими ключами.

Оба возвращают `RawProduct`. Ниже этого слоя разница форматов не видна.
"""

from __future__ import annotations

from nutri_radar.ingest.sources.parquet import ParquetSource

__all__ = ["ParquetSource"]
