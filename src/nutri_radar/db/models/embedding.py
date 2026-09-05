"""Векторы профилей продуктов для семантического поиска.

**Почему отдельная таблица, а не колонка в `products`.** Вектор зависит
от модели эмбеддингов и от правила сборки профиля, и то и другое меняется
независимо от самого продукта. Колонка в `products` заставила бы переписывать
строку продукта при каждой перевекторизации и не дала бы держать рядом
векторы двух моделей — а сравнивать их придётся, как в M3 сравнивались
версии промптов.

**Почему `halfvec`, а не `vector`.** Половинная точность занимает вдвое
меньше и в таблице, и в индексе HNSW, а потеря recall на 1024 измерениях
неразличима. 146 тысяч векторов — это 300 МБ против 600 МБ, и разница
решает, останется ли индекс в памяти.

**Почему хранится хеш профиля.** Эмбеддинг — функция от текста, и без записи
о том, из какого текста он получен, изменение правила сборки прошло бы
незамеченным: половина базы оказалась бы векторизована по старому правилу,
а метрики посчитаны на смеси. Хеш позволяет прогону пропускать актуальные
векторы и переделывать устаревшие, не сверяя тексты целиком.

**Индекс HNSW здесь не объявляется.** Он строится отдельным шагом после
заливки: на заполненной таблице граф получается лучше, а сборка быстрее.
Объявить его в модели значило бы создавать его миграцией на пустой таблице.
"""

from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from nutri_radar.db.base import Base

_CODE_LEN = 32

# Размерность bge-m3. Значение дублирует `OllamaSettings.embedding_dim`
# намеренно: схема БД фиксируется миграцией и не может зависеть от
# переменной окружения — сменить размерность в `.env` и получить молча
# несовместимую колонку было бы худшим из возможных отказов.
EMBEDDING_DIM = 1024


class ProductEmbedding(Base):
    """Вектор профиля одного продукта, полученный одной моделью."""

    __tablename__ = "product_embedding"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(
        String(_CODE_LEN),
        ForeignKey("products.code", ondelete="CASCADE"),
        nullable=False,
    )

    # --- чем и из чего получен ---------------------------------------------
    #
    # Векторы разных моделей несравнимы, и вектор без имени модели — это
    # набор чисел неизвестно чего.
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    # Версия правила сборки профиля. Читается человеком; хеш читается кодом.
    profile_version: Mapped[str] = mapped_column(String(32), nullable=False)
    # sha256 текста профиля, укороченный. По нему прогон понимает, устарел
    # ли вектор, не перечитывая тексты.
    profile_hash: Mapped[str] = mapped_column(String(32), nullable=False)

    # NOT NULL намеренно: пустой вектор в косинусном расстоянии даёт деление
    # на ноль, а NULL молча выпадает из выдачи. Строка появляется только
    # вместе с вектором.
    embedding: Mapped[list[float]] = mapped_column(HALFVEC(EMBEDDING_DIM), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        # Уникальность по продукту и модели, а не по продукту: рядом должны
        # уживаться векторы двух моделей, иначе их не сравнить.
        UniqueConstraint("code", "model_name", name="uq_product_embedding_code_model"),
        # По этому индексу идёт возобновление прогона: «что уже посчитано
        # этой моделью». Без него оно превращается в полный перебор таблицы
        # на каждом батче.
        Index("ix_product_embedding_model", "model_name"),
    )

    def __repr__(self) -> str:
        return (
            f"ProductEmbedding(code={self.code!r}, model={self.model_name!r}, "
            f"profile={self.profile_version!r})"
        )
