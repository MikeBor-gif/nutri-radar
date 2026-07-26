"""Окружение Alembic.

Отличия от сгенерированного шаблона:

* URL берётся из `nutri_radar.config`, а НЕ из `alembic.ini`. DSN содержит
  пароль, а `alembic.ini` лежит в репозитории — секрет туда попасть не должен.
* `target_metadata` указывает на `Base.metadata` и модели импортируются явно:
  без импорта автогенерация увидит пустую схему и предложит удалить все таблицы.
* `compare_type` и `compare_server_default` включены: без них Alembic
  пропускает смену типа колонки и значения по умолчанию.
"""

from __future__ import annotations

import asyncio
import logging
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# Импорт моделей наполняет Base.metadata. Без него автогенерация пуста.
from nutri_radar.config import get_settings
from nutri_radar.db.base import Base
from nutri_radar.db.models import *  # noqa: F403  — регистрация таблиц в metadata

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

logger = logging.getLogger("alembic.env")

target_metadata = Base.metadata

_settings = get_settings()

# DSN подставляется из конфига, но только если он не задан явно: интеграционные
# тесты передают свой URL через Config, чтобы гонять миграции на отдельной базе
# и не трогать рабочую. Значение в alembic.ini пустое, поэтому обычный запуск
# всегда берёт настройки проекта.
_explicit_url = config.get_main_option("sqlalchemy.url", default="")
if _explicit_url:
    _target_description = "URL передан явно"
else:
    config.set_main_option("sqlalchemy.url", _settings.db.dsn)
    _target_description = _settings.db.safe_dsn

# Куда именно применяются миграции — видно до их применения, чтобы случайно
# не миграть не ту базу. Пароль в safe_dsn отсутствует.
logger.info("Цель миграций: %s", _target_description)


def run_migrations_offline() -> None:
    """Режим генерации SQL без подключения к БД (`alembic upgrade head --sql`)."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        # NullPool: миграции — короткоживущий процесс, пул ему не нужен.
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
