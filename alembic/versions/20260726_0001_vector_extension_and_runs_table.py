"""vector extension and runs table

Первая миграция проекта. Написана руками, а не автогенерацией: расширение
Postgres автогенератор не видит.

Расширение `vector` ставится сейчас, хотя используется только на M5
(эмбеддинги и HNSW-индекс). Причина: переставлять расширения посреди проекта
дороже, чем поставить сразу. Требует образ `pgvector/pgvector:pg16` —
на чистом `postgres:16` эта миграция упадёт.

Revision ID: 0001
Revises:
Create Date: 2026-07-26
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Значения перечислений дублируются здесь намеренно: миграция должна остаться
# воспроизводимой, даже если Python-перечисление в коде потом изменится.
_RUN_STAGES = ("ingest", "extract", "evals", "analytics", "retrieval", "agent")
_RUN_STATUSES = ("running", "completed", "failed")


def upgrade() -> None:
    """Расширение vector и журнал прогонов."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        # VARCHAR + CHECK вместо нативного enum Postgres: нативный требует
        # ALTER TYPE для добавления значения, CHECK меняется обычной миграцией.
        #
        # create_constraint=True обязателен: по умолчанию он False, и тогда
        # CHECK не создаётся вовсе — база примет любую строку.
        sa.Column(
            "stage",
            sa.Enum(
                *_RUN_STAGES,
                native_enum=False,
                create_constraint=True,
                length=32,
                name="runstage",
            ),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.Enum(
                *_RUN_STATUSES,
                native_enum=False,
                create_constraint=True,
                length=16,
                name="runstatus",
            ),
            nullable=False,
        ),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("items_processed", sa.Integer(), nullable=False),
        # Доля пропусков — обязательная метрика проекта, считается всегда.
        sa.Column("items_skipped", sa.Integer(), nullable=False),
        # Заполняются только стадиями с LLM.
        sa.Column("model_name", sa.String(length=128), nullable=True),
        sa.Column("prompt_version", sa.String(length=32), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("error_message", sa.String(), nullable=True),
        # JSONB, а не колонки: набор параметров у стадий разный и будет меняться.
        sa.Column("params", postgresql.JSONB(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_runs")),
    )

    # Типовой запрос «последний прогон этой стадии». Без индекса он станет
    # полным сканированием, как только журнал разрастётся.
    op.create_index(
        "ix_runs_stage_started_at",
        "runs",
        ["stage", sa.text("started_at DESC")],
        unique=False,
    )


def downgrade() -> None:
    """Полный откат. Пустой downgrade означал бы непроверенный цикл миграций."""
    op.drop_index("ix_runs_stage_started_at", table_name="runs")
    op.drop_table("runs")
    # Расширение снимаем последним: на нём могут висеть объекты из будущих
    # миграций, и они должны уйти раньше.
    op.execute("DROP EXTENSION IF EXISTS vector")
