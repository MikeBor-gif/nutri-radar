"""product_embedding: векторы профилей для семантического поиска

Отдельная таблица, а не колонка в `products`. Вектор зависит от модели
эмбеддингов и от правила сборки профиля, и то и другое меняется независимо
от самого продукта. Колонка заставила бы переписывать строку продукта при
каждой перевекторизации и не дала бы держать рядом векторы двух моделей —
а сравнивать их придётся, как в M3 сравнивались версии промптов.

`halfvec(1024)`, а не `vector(1024)`: половинная точность занимает вдвое
меньше и в таблице, и в индексе HNSW, а потеря recall на этой размерности
неразличима. 146 тысяч векторов — 300 МБ против 600 МБ, и разница решает,
останется ли индекс в памяти.

**Индекс HNSW здесь не создаётся намеренно.** Он строится отдельной командой
после заливки: на заполненной таблице граф получается лучше, а сборка
быстрее. Создать его здесь значило бы построить на пустой таблице и потом
доращивать по одному вектору.

Хеш профиля хранится, чтобы прогон различал актуальные и устаревшие векторы
не сверяя тексты: без него изменение правила сборки прошло бы незамеченным,
и половина базы осталась бы векторизована по старому правилу.

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-05
"""

from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa

from alembic import op

revision: str = "0006"
down_revision: str | Sequence[str] | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_EMBEDDING_DIM = 1024


def upgrade() -> None:
    op.create_table(
        "product_embedding",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("code", sa.String(length=32), nullable=False),
        sa.Column("model_name", sa.String(length=128), nullable=False),
        sa.Column("profile_version", sa.String(length=32), nullable=False),
        sa.Column("profile_hash", sa.String(length=32), nullable=False),
        # NOT NULL намеренно: пустой вектор в косинусном расстоянии даёт
        # деление на ноль, а NULL молча выпадает из выдачи. Строка появляется
        # только вместе с вектором.
        sa.Column(
            "embedding",
            pgvector.sqlalchemy.HALFVEC(dim=_EMBEDDING_DIM),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["code"],
            ["products.code"],
            name=op.f("fk_product_embedding_code_products"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_product_embedding")),
        # Уникальность по продукту И модели: рядом должны уживаться векторы
        # двух моделей, иначе их не сравнить.
        sa.UniqueConstraint("code", "model_name", name="uq_product_embedding_code_model"),
    )
    # По этому индексу идёт возобновление прогона: «что уже посчитано этой
    # моделью». Без него оно превращается в полный перебор на каждом батче.
    op.create_index("ix_product_embedding_model", "product_embedding", ["model_name"])


def downgrade() -> None:
    op.drop_index("ix_product_embedding_model", table_name="product_embedding")
    op.drop_table("product_embedding")
