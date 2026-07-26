"""Общий слой доступа к Postgres.

Публичный интерфейс модуля. Слайсы пайплайна обращаются только к нему;
сам модуль о слайсах ничего не знает (см. ARCHITECTURE.md, правила зависимостей).

Репозиториев пока нет — они появятся вместе с таблицами продуктов на M1.
"""

from __future__ import annotations

from nutri_radar.db.base import Base
from nutri_radar.db.session import (
    dispose_engine,
    get_engine,
    get_session,
    get_session_factory,
)

__all__ = [
    "Base",
    "dispose_engine",
    "get_engine",
    "get_session",
    "get_session_factory",
]
