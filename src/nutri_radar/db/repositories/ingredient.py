"""Словарь алиасов ингредиентов в Postgres.

Seed-файл в `data/dictionaries/` — источник истины, который ведёт человек.
Таблица `ingredients_dict` — его копия в базе: она нужна, чтобы SQL-запросы
аналитики и агента могли канонизировать имена без чтения файлов, а `evals`
на M3 — сравнивать наши имена с таксономией Open Food Facts прямо в запросе.

Синхронизация односторонняя: файл → база. Обратного направления нет
намеренно, иначе правки разъехались бы между двумя местами.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from nutri_radar.db.models.extraction import IngredientAlias
from nutri_radar.logging import safe_extra

logger = logging.getLogger(__name__)

# Предел числа аргументов одного запроса в протоколе Postgres (int16).
_MAX_QUERY_ARGS = 32767


class AliasRow(BaseModel):
    """Одна строка `ingredients_dict` на границе с БД."""

    alias: str
    lang: str
    canonical_name: str
    # Строкой, а не перечислением: словарь наполняется данными, и жёсткая
    # связь с Python-перечислением здесь только мешала бы.
    kind: str | None = None


class IngredientAliasRepository:
    """Доступ к `ingredients_dict`."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert_batch(self, rows: list[AliasRow]) -> int:
        """Записать алиасы. Повторная загрузка seed обновляет, а не дублирует."""
        if not rows:
            return 0

        payload = [row.model_dump() for row in rows]
        columns = len(IngredientAlias.__table__.columns)
        chunk_size = max(_MAX_QUERY_ARGS // columns, 1)

        affected = 0
        for start in range(0, len(payload), chunk_size):
            chunk = payload[start : start + chunk_size]
            statement = insert(IngredientAlias).values(chunk)
            statement = statement.on_conflict_do_update(
                # Уникальность по (alias, lang): один алиас на языке ведёт
                # к одному каноническому имени.
                index_elements=["alias", "lang"],
                set_={
                    "canonical_name": statement.excluded.canonical_name,
                    "kind": statement.excluded.kind,
                },
            )
            result = await self._session.execute(statement)
            affected += int(getattr(result, "rowcount", 0) or 0)

        logger.info("Словарь записан в БД", extra=safe_extra(rows=len(rows), affected=affected))
        return affected

    async def all_aliases(self) -> list[AliasRow]:
        """Весь словарь. Он маленький (сотни строк) и читается целиком."""
        statement = select(
            IngredientAlias.alias,
            IngredientAlias.lang,
            IngredientAlias.canonical_name,
            IngredientAlias.kind,
        )
        rows = (await self._session.execute(statement)).all()
        return [
            AliasRow(alias=row[0], lang=row[1], canonical_name=row[2], kind=row[3]) for row in rows
        ]

    async def count(self) -> int:
        return int(
            (await self._session.scalar(select(func.count()).select_from(IngredientAlias))) or 0
        )
