"""Базовый класс моделей SQLAlchemy.

`naming_convention` задаётся здесь и до первой миграции. Без него Alembic
создаёт безымянные ограничения, а безымянное ограничение нельзя удалить
миграцией — придётся лезть в БД руками.
"""

from __future__ import annotations

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

# Шаблоны имён: ix — индекс, uq — уникальность, ck — check, fk — внешний ключ,
# pk — первичный ключ. Порядок и состав соответствуют рекомендации Alembic.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Корень декларативных моделей проекта."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)
