"""Слайс ingestion: дамп Open Food Facts → DuckDB-выборка → Postgres.

Массовые данные берутся только дампом. Живой API OFF в этом слайсе не
используется вообще — он нужен ровно в одном месте проекта, инструменте
`lookup_barcode` агента на M6.

Полный дамп и дельта-экспорты имеют разные схемы, поэтому в `sources/` живут
два адаптера, сводящие обе формы к одной внутренней модели (ADR-004).
"""

from __future__ import annotations

from nutri_radar.ingest.probe import ProbeReport, probe_schema

__all__ = ["ProbeReport", "probe_schema"]
