"""Результаты извлечения и словарь ингредиентов.

Ключевое требование раздела 7 брифа: у продукта может быть **несколько
извлечений** разными версиями промпта и разными моделями. Без этого сравнение
версий в M3 невозможно — новая версия просто перетёрла бы старую, и сравнивать
стало бы не с чем.

Отсюда уникальность по `(code, model_name, prompt_version)`, а не по `code`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from nutri_radar.db.base import Base

_CODE_LEN = 32


class ProductExtraction(Base):
    """Один разбор состава конкретной моделью и версией промпта."""

    __tablename__ = "product_extraction"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(
        String(_CODE_LEN),
        ForeignKey("products.code", ondelete="CASCADE"),
        nullable=False,
    )

    # --- что именно разбирали ----------------------------------------------
    # Язык состава, отданного модели. Без него нельзя сравнить качество
    # извлечения по языкам, а многоязычность — главный аргумент проекта.
    source_lang: Mapped[str | None] = mapped_column(String(8), nullable=True)

    # --- результат ----------------------------------------------------------
    # Список ингредиентов как есть из схемы. JSONB, а не отдельная таблица:
    # он читается целиком и никогда не запрашивается по отдельному элементу.
    ingredients: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)

    # Ключевая производная фича проекта — отдельной колонкой, а не внутри JSONB:
    # по ней пойдут агрегаты и графики M4, а доставать её из JSONB на сотнях
    # тысяч строк дорого.
    distinct_sugar_forms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    e_additives_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ingredients_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    allergens: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False, default=list)

    # Записи с unreadable=true не идут в аналитику (раздел 8 брифа).
    unreadable: Mapped[bool] = mapped_column(nullable=False, default=False)
    model_confidence: Mapped[float | None] = mapped_column(Numeric(4, 3), nullable=True)

    # --- чем разбирали ------------------------------------------------------
    # Обе колонки обязательны: без них результат невозможно отнести к прогону,
    # и сравнение версий превращается в кашу.
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(32), nullable=False)

    # --- учёт стоимости -----------------------------------------------------
    # Время и стоимость полного прогона — обязательные метрики проекта,
    # и собираются они здесь, а не восстанавливаются по логам.
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    latency_s: Mapped[float | None] = mapped_column(Numeric(8, 3), nullable=True)

    extracted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        # Сердце версионирования: одна строка на связку продукт-модель-промпт.
        # Переизвлечение той же версией обновляет строку, другая версия
        # создаёт новую.
        UniqueConstraint(
            "code", "model_name", "prompt_version", name="uq_product_extraction_code_model_prompt"
        ),
        # Сравнение версий в M3 идёт срезами по промпту и модели.
        Index("ix_product_extraction_prompt_model", "prompt_version", "model_name"),
        # Отчёты M4 группируют по числу форм сахара.
        Index("ix_product_extraction_sugar_forms", "distinct_sugar_forms"),
    )

    def __repr__(self) -> str:
        return (
            f"ProductExtraction(code={self.code!r}, prompt={self.prompt_version!r}, "
            f"sugar_forms={self.distinct_sugar_forms})"
        )


class IngredientAlias(Base):
    """Словарь алиасов ингредиентов.

    Без него «сироп глюкозы», «glucose syrup» и «Glukosesirup» остаются тремя
    сущностями, и подсчёт форм сахара становится неверным.

    Наполняется человеком и seed-файлом, но **не моделью**: правило 6 брифа
    запрещает генерировать эталонные данные, и словарь — пограничный случай,
    поэтому решение зафиксировано явно.
    """

    __tablename__ = "ingredients_dict"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    canonical_name: Mapped[str] = mapped_column(String(128), nullable=False)
    alias: Mapped[str] = mapped_column(String(128), nullable=False)
    lang: Mapped[str] = mapped_column(String(8), nullable=False)
    # Тип из перечисления схемы извлечения. Хранится строкой: словарь
    # наполняется данными, и жёсткая связь с Python-перечислением здесь
    # только мешала бы.
    kind: Mapped[str | None] = mapped_column(String(32), nullable=True)

    __table_args__ = (
        # Один алиас на язык ведёт к одному каноническому имени.
        UniqueConstraint("alias", "lang", name="uq_ingredients_dict_alias_lang"),
        Index("ix_ingredients_dict_canonical", "canonical_name"),
    )

    def __repr__(self) -> str:
        return f"IngredientAlias({self.alias!r} [{self.lang}] -> {self.canonical_name!r})"
