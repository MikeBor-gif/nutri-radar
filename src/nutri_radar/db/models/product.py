"""Таблицы продуктов: сырое и нормализованное.

Разделение из раздела 7 брифа. `products_raw` хранит исходный набор отобранных
полей и **не перезаписывается деструктивно**: если завтра выяснится, что нужна
ещё одна колонка, её можно достать без повторного скачивания 7,7 ГБ.
`products` — нормализованный слой, по которому идут запросы и обучение.

Baseline-поля собственного парсера Open Food Facts (`ingredients_tags`,
`additives_n`, `unknown_ingredients_n` и прочие) лежат в `products` намеренно
(ADR-006): это то, с чем в M3 соревнуется наша LLM, и критерий отбора
LLM-корпуса на M2.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from nutri_radar.db.base import Base

# Штрихкод EAN-13 плюс запас на внутренние коды OFF и коды магазинов.
_CODE_LEN = 32

# Нутриенты на 100 г. Значения вроде энергии доходят до ~900 ккал, доли
# в граммах требуют дробной части — Numeric, а не Float: у Float накапливается
# ошибка, а эти числа уедут в обучение и в отчёты.
_NUTRIENT = Numeric(10, 3)


class ProductRaw(Base):
    """Исходные данные продукта в том виде, в каком пришли из источника."""

    __tablename__ = "products_raw"

    code: Mapped[str] = mapped_column(String(_CODE_LEN), primary_key=True)

    # Полный набор отобранных полей. JSONB, а не колонки: состав полей зависит
    # от версии схемы дампа (их пять: 999, 1001-1004), и жёсткая схема здесь
    # означала бы миграцию на каждое изменение апстрима.
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)

    # parquet или delta — из какого источника пришла запись. Нужно при разборе
    # расхождений: формы данных разные (ADR-004).
    source: Mapped[str] = mapped_column(String(16), nullable=False)

    # Версия дампа (размер и ETag). Требование раздела 7 брифа: знать, из какой
    # выгрузки взята запись.
    dump_version: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # Ревизия записи в OFF. По ней решается, обновлять ли: старый дамп не должен
    # откатывать данные назад.
    rev: Mapped[int | None] = mapped_column(Integer, nullable=True)

    imported_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    def __repr__(self) -> str:
        return f"ProductRaw(code={self.code!r}, source={self.source!r}, rev={self.rev})"


class Product(Base):
    """Нормализованный продукт. По этой таблице идут выборки и обучение."""

    __tablename__ = "products"

    code: Mapped[str] = mapped_column(
        String(_CODE_LEN),
        ForeignKey("products_raw.code", ondelete="CASCADE"),
        primary_key=True,
    )

    # --- тексты -------------------------------------------------------------
    #
    # Без ограничения длины. Это краудсорсинговые поля, которые заполняют люди:
    # угаданный предел даёт не защиту, а отказ записи. Проверено на дампе —
    # `generic_name` доходит до 1131 символа при «разумном» лимите 512.
    # В Postgres `text` и `varchar(n)` хранятся одинаково, так что ограничение
    # не экономит ничего, зато добавляет режим отказа.
    product_name: Mapped[str | None] = mapped_column(String, nullable=True)
    generic_name: Mapped[str | None] = mapped_column(String, nullable=True)
    brands: Mapped[str | None] = mapped_column(String, nullable=True)

    # Состав на одном языке — том, который реально пойдёт в LLM.
    # Полный словарь по языкам остаётся в products_raw.payload.
    ingredients_text: Mapped[str | None] = mapped_column(String, nullable=True)
    # Без этого поля непонятно, на каком языке текст отдали модели, и сравнивать
    # качество извлечения по языкам станет невозможно.
    ingredients_text_lang: Mapped[str | None] = mapped_column(String(8), nullable=True)

    lang: Mapped[str | None] = mapped_column(String(8), nullable=True)
    languages: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False, default=list)

    # --- классификация ------------------------------------------------------
    categories_tags: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
    food_groups_tags: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
    countries_tags: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
    labels_tags: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False, default=list)

    # --- готовые метки для M4 ----------------------------------------------
    # Только a-e: служебные unknown и not-applicable приводятся к NULL в
    # RawProduct, чтобы не попасть в обучение как отдельный класс.
    nutriscore_grade: Mapped[str | None] = mapped_column(String(1), nullable=True)
    nutriscore_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    nova_group: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # --- нутриенты на 100 г -------------------------------------------------
    # Отдельными колонками, а не JSONB: по ним пойдут агрегаты в отчётах и
    # обучение в M4, а доставать каждое значение из JSONB на 100 тыс. строк дорого.
    energy_kcal_100g: Mapped[float | None] = mapped_column(_NUTRIENT, nullable=True)
    fat_100g: Mapped[float | None] = mapped_column(_NUTRIENT, nullable=True)
    saturated_fat_100g: Mapped[float | None] = mapped_column(_NUTRIENT, nullable=True)
    carbohydrates_100g: Mapped[float | None] = mapped_column(_NUTRIENT, nullable=True)
    sugars_100g: Mapped[float | None] = mapped_column(_NUTRIENT, nullable=True)
    fiber_100g: Mapped[float | None] = mapped_column(_NUTRIENT, nullable=True)
    proteins_100g: Mapped[float | None] = mapped_column(_NUTRIENT, nullable=True)
    salt_100g: Mapped[float | None] = mapped_column(_NUTRIENT, nullable=True)
    sodium_100g: Mapped[float | None] = mapped_column(_NUTRIENT, nullable=True)
    # Входит в формулу Nutri-Score. Без него sanity-check M4 не сойдётся
    # (ADR-005). Разведка показала покрытие 83,7% — данных достаточно.
    fruits_vegetables_nuts_estimate_100g: Mapped[float | None] = mapped_column(
        _NUTRIENT, nullable=True
    )
    # Сырые баллы Nutri-Score — прямой вход формулы, а не производная от буквы.
    # Найдено разведкой, покрытие 73,7%: позволяет проверять sanity-check
    # регрессией на баллы, а не только классификацией на букву.
    nutrition_score_fr_100g: Mapped[float | None] = mapped_column(_NUTRIENT, nullable=True)

    nutrition_data_per: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # Явный флаг наличия данных — требование раздела 7 брифа. Считается по
    # факту, а не копируется из источника: флаг no_nutrition_data выставлен
    # не везде.
    has_nutrition_data: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # --- baseline парсера OFF (ADR-006) ------------------------------------
    ingredients_tags: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
    additives_tags: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
    allergens_tags: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
    traces_tags: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
    ingredients_analysis_tags: Mapped[list[str]] = mapped_column(
        ARRAY(String), nullable=False, default=list
    )
    ingredients_n: Mapped[int | None] = mapped_column(Integer, nullable=True)
    known_ingredients_n: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Ключевое поле: > 0 означает, что таксономия OFF не справилась. По нему
    # отбирается LLM-корпус на M2 (ADR-006).
    unknown_ingredients_n: Mapped[int | None] = mapped_column(Integer, nullable=True)
    additives_n: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # --- популярность и служебное ------------------------------------------
    unique_scans_n: Mapped[int | None] = mapped_column(Integer, nullable=True)
    popularity_key: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    completeness: Mapped[float | None] = mapped_column(Float, nullable=True)

    rev: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_modified_t: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        # Обучение и отчёты M4 фильтруют и группируют по целевой метке.
        Index("ix_products_nutriscore_grade", "nutriscore_grade"),
        Index("ix_products_nova_group", "nova_group"),
        # Фильтр корпуса и гибридный поиск M5 проверяют пересечение массивов —
        # это операция GIN, обычный btree здесь бесполезен.
        Index("ix_products_categories_tags", "categories_tags", postgresql_using="gin"),
        # Отбор LLM-корпуса на M2 идёт именно по этому полю (ADR-006).
        Index("ix_products_unknown_ingredients_n", "unknown_ingredients_n"),
        # Инкрементальное обновление ищет записи новее водяного знака.
        Index("ix_products_last_modified_t", "last_modified_t"),
    )

    def __repr__(self) -> str:
        return (
            f"Product(code={self.code!r}, grade={self.nutriscore_grade!r}, "
            f"lang={self.ingredients_text_lang!r})"
        )
