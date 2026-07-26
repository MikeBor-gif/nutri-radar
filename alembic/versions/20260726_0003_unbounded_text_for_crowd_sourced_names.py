"""unbounded text for crowd-sourced names

Снимает ограничение длины с названий и брендов.

Причина найдена на реальном дампе: `generic_name` доходит до 1131 символа,
и 8 таких записей роняли заливку с `StringDataRightTruncationError`. Это
краудсорсинговые поля, которые заполняют люди, — угаданный предел даёт не
защиту, а отказ записи. В Postgres `text` и `varchar(n)` хранятся одинаково,
так что ограничение не экономило ничего.

Написано руками: автогенерация изменение `varchar(512)` → `text` не заметила,
несмотря на `compare_type=True`. Это её известное слепое пятно, и полагаться
на неё в таких правках нельзя.

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-26
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_COLUMNS = ("product_name", "generic_name", "brands")


def upgrade() -> None:
    """varchar(512) -> text."""
    for column in _COLUMNS:
        op.alter_column(
            "products",
            column,
            existing_type=sa.String(length=512),
            type_=sa.String(),
            existing_nullable=True,
        )


def downgrade() -> None:
    """text -> varchar(512).

    Откат обрезает значения длиннее 512 символов: иначе `ALTER TYPE` упадёт
    на тех же записях, из-за которых миграция и появилась. Потеря данных здесь
    неизбежна и потому сделана явной, а не спрятана в приведении типа.
    """
    for column in _COLUMNS:
        op.execute(
            f"UPDATE products SET {column} = left({column}, 512) "
            f"WHERE length({column}) > 512"
        )
        op.alter_column(
            "products",
            column,
            existing_type=sa.String(),
            type_=sa.String(length=512),
            existing_nullable=True,
        )
