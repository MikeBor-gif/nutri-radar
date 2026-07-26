"""Асинхронный движок и сессии Postgres.

Движок создаётся один раз на процесс, а не на запрос: пул соединений теряет
смысл, если пересоздавать его каждый вызов.

Отказы БД заворачиваются в `DatabaseError` — наружу из этого модуля
`SQLAlchemyError` и `asyncpg`-исключения не протекают.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from nutri_radar.config import DatabaseSettings, get_settings
from nutri_radar.errors import DatabaseError

logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine(settings: DatabaseSettings | None = None) -> AsyncEngine:
    """Вернуть движок, создав его при первом обращении."""
    global _engine
    if _engine is None:
        db = settings or get_settings().db
        _engine = create_async_engine(
            db.dsn,
            pool_size=db.pool_size,
            pool_pre_ping=True,  # отсекает соединения, умершие после простоя
            echo=db.echo_sql,
            future=True,
        )
        logger.info(
            "Движок БД создан",
            extra={
                "dsn": db.safe_dsn,  # без пароля
                "pool_size": db.pool_size,
                "echo_sql": db.echo_sql,
            },
        )
    return _engine


def get_session_factory(
    settings: DatabaseSettings | None = None,
) -> async_sessionmaker[AsyncSession]:
    """Вернуть фабрику сессий, создав её при первом обращении."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(settings),
            # Обязательно False. С True после commit обращение к атрибуту
            # уйдёт в БД вне активной сессии и упадёт MissingGreenlet.
            expire_on_commit=False,
            autoflush=False,
        )
    return _session_factory


@asynccontextmanager
async def get_session(
    settings: DatabaseSettings | None = None,
) -> AsyncIterator[AsyncSession]:
    """Сессия с автоматическим commit при успехе и rollback при ошибке.

    Пример:
        async with get_session() as session:
            await session.execute(...)
    """
    factory = get_session_factory(settings)
    session = factory()
    logger.debug("Сессия БД открыта")
    try:
        yield session
        await session.commit()
        logger.debug("Сессия БД закоммичена")
    except SQLAlchemyError as exc:
        await session.rollback()
        logger.error("Откат сессии БД: %s", type(exc).__name__, exc_info=True)
        raise DatabaseError(f"Ошибка работы с БД: {exc}") from exc
    except Exception:
        # Ошибка прикладного кода внутри блока: транзакцию тоже откатываем,
        # но исключение не подменяем — оно не про БД.
        await session.rollback()
        logger.debug("Откат сессии БД из-за ошибки прикладного кода")
        raise
    finally:
        await session.close()
        logger.debug("Сессия БД закрыта")


async def dispose_engine() -> None:
    """Закрыть пул соединений. Вызывается при остановке процесса и в тестах."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        logger.info("Пул соединений БД закрыт")
    _engine = None
    _session_factory = None
