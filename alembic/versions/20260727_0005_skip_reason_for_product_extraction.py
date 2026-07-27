"""skip_reason for product_extraction

Причина, по которой состав не превратился в разбор.

Без неё все пропуски выглядят одинаково: `unreadable=true` и пустой список
ингредиентов. Между тем состав, не влезший в контекст модели, состав, чей
ответ упёрся в лимит вывода, и состав, который модель сама объявила
нечитаемым, — три разные проблемы с тремя разными решениями. Доля пропусков
входит в обязательные метрики майлстоуна, и без разбивки по причинам это
число нечего обсуждать.

Строка, а не Postgres ENUM: список причин будет пополняться по мере
встреченных случаев, а миграция ради каждого нового значения — цена без
выгоды. Тот же довод, что и у `kind` в `ingredients_dict`.

Колонка nullable без значения по умолчанию: NULL означает «разбор состоялся»,
и это подавляющее большинство строк. Существующие строки остаются с NULL —
для успешных разборов это верно, а немногочисленные старые `unreadable`
переразбираются прогоном заново.

Revision ID: 0005
Revises: 0004
Create Date: 2026-07-27
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0005"
down_revision: str | Sequence[str] | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "product_extraction",
        sa.Column("skip_reason", sa.String(length=32), nullable=True),
    )
    # Индекс частичный: причина есть только у пропусков, а их доля мала.
    # Полный индекс по колонке, где почти везде NULL, — это оплаченные
    # страницы, которые никто не читает.
    op.create_index(
        "ix_product_extraction_skip_reason",
        "product_extraction",
        ["skip_reason"],
        unique=False,
        postgresql_where=sa.text("skip_reason IS NOT NULL"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_product_extraction_skip_reason", table_name="product_extraction")
    op.drop_column("product_extraction", "skip_reason")
