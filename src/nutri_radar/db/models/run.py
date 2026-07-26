"""Журнал прогонов. Используется всеми стадиями пайплайна.

Зачем таблица, а не лог-файл: из неё берутся числа для отчётов и README —
время, токены, доля пропусков. Обязательное требование проекта: все числа
из кода, не из головы. Плюс она же обеспечивает возобновляемость: перезапуск
смотрит, что уже сделано, а не начинает заново.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import DateTime, Enum, Index, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from nutri_radar.db.base import Base


class RunStage(StrEnum):
    """Стадия пайплайна. Все значения заданы сразу, включая будущие.

    Если добавлять их по одному с каждым майлстоуном, придётся править
    ограничение миграцией каждый раз. Список стадий известен из брифа.
    """

    INGEST = "ingest"
    EXTRACT = "extract"
    EVALS = "evals"
    ANALYTICS = "analytics"
    RETRIEVAL = "retrieval"
    AGENT = "agent"


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# native_enum=False даёт VARCHAR вместо нативного типа Postgres: нативный enum
# требует ALTER TYPE для добавления значения, а CHECK меняется обычной миграцией.
#
# create_constraint=True обязателен. По умолчанию в SQLAlchemy он False, и тогда
# колонка становится обычным VARCHAR без всякой проверки — база молча примет
# любую строку, хотя тип объявлен перечислением.
_STAGE_ENUM = Enum(
    RunStage,
    native_enum=False,
    create_constraint=True,
    length=32,
    validate_strings=True,
    name="runstage",
)
_STATUS_ENUM = Enum(
    RunStatus,
    native_enum=False,
    create_constraint=True,
    length=16,
    validate_strings=True,
    name="runstatus",
)


class Run(Base):
    """Один прогон одной стадии."""

    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    stage: Mapped[RunStage] = mapped_column(_STAGE_ENUM, nullable=False)
    status: Mapped[RunStatus] = mapped_column(
        _STATUS_ENUM, nullable=False, default=RunStatus.RUNNING
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    items_processed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Доля пропусков — обязательная метрика проекта, поэтому считается всегда,
    # а не выясняется потом по логам.
    items_skipped: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Заполняются только стадиями с LLM. Без них сравнение версий промптов
    # и моделей в evals невозможно.
    model_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    error_message: Mapped[str | None] = mapped_column(String, nullable=True)

    # Параметры прогона целиком: языки, категории, размер батча, порог.
    # JSONB, а не колонки: набор параметров у стадий разный и будет меняться.
    params: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    __table_args__ = (
        # Типовой запрос: «последний прогон этой стадии». Без индекса он
        # превращается в полное сканирование, как только журнал разрастётся.
        Index("ix_runs_stage_started_at", "stage", started_at.desc()),
    )

    def __repr__(self) -> str:
        return (
            f"Run(id={self.id}, stage={self.stage}, status={self.status}, "
            f"processed={self.items_processed}, skipped={self.items_skipped})"
        )
