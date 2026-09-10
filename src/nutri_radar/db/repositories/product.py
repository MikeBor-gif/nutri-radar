"""Запись продуктов в Postgres.

Ключевое требование DoD майлстоуна: **повторный запуск не создаёт дублей**.
Отсюда upsert по `code` через `ON CONFLICT`, а не «сначала SELECT, потом
INSERT»: последнее дало бы на 100 тыс. строк 100 тыс. лишних запросов и гонку
при параллельных прогонах.

Второе требование, менее очевидное: **старый дамп не должен откатывать данные
назад**. Поэтому обновление происходит только если пришедшая ревизия свежее
сохранённой. Условие живёт в самом `ON CONFLICT ... WHERE`, а не в Python:
иначе между чтением и записью вклинилась бы гонка.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from nutri_radar.db.models.product import Product, ProductRaw
from nutri_radar.ingest.models import RawProduct
from nutri_radar.logging import safe_extra

logger = logging.getLogger(__name__)

# Предел числа аргументов одного запроса в протоколе Postgres: поле int16.
# asyncpg сообщает о нём как «the number of query arguments cannot exceed
# 32767». Ограничение на АРГУМЕНТЫ, а не на строки, поэтому допустимый размер
# куска зависит от числа колонок таблицы.
_MAX_QUERY_ARGS = 32767

# Соответствие «имя нутриента в источнике» → «колонка в products».
# Имена подтверждены разведкой дампа, а не взяты из документации по CSV.
NUTRIENT_COLUMNS = {
    "energy-kcal": "energy_kcal_100g",
    "fat": "fat_100g",
    "saturated-fat": "saturated_fat_100g",
    "carbohydrates": "carbohydrates_100g",
    "sugars": "sugars_100g",
    "fiber": "fiber_100g",
    "proteins": "proteins_100g",
    "salt": "salt_100g",
    "sodium": "sodium_100g",
    "fruits-vegetables-nuts-estimate-from-ingredients": "fruits_vegetables_nuts_estimate_100g",
    "nutrition-score-fr": "nutrition_score_fr_100g",
}


class ProductSummary(BaseModel):
    """Продукт на границе с БД: то, что нужно показать человеку.

    Отдельная модель, а не ORM-объект: `Product` за пределами `db/` дал бы
    `MissingGreenlet` на первом обращении к атрибуту вне сессии
    (см. ARCHITECTURE.md). И не весь `Product` целиком — сорок колонок
    нутриентов и служебных полей в карточке не нужны, а тащить их наружу
    значит превращать модель показа в копию схемы таблицы.
    """

    code: str
    product_name: str | None = None
    generic_name: str | None = None
    brands: str | None = None

    ingredients_text: str | None = None
    ingredients_text_lang: str | None = None
    lang: str | None = None

    nutriscore_grade: str | None = None
    nova_group: int | None = None

    categories_tags: list[str] = Field(default_factory=list)
    allergens_tags: list[str] = Field(default_factory=list)
    additives_tags: list[str] = Field(default_factory=list)

    energy_kcal_100g: float | None = None
    sugars_100g: float | None = None
    salt_100g: float | None = None


class ProductRepository:
    """Батчевая запись продуктов и чтение одного продукта по штрихкоду."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_code(self, code: str) -> ProductSummary | None:
        """Прочитать один продукт корпуса. Нет такого кода — `None`.

        Отсутствие продукта — нормальный ответ, а не ошибка: в корпусе
        147 тысяч продуктов из 4,63 млн строк дампа, и большинство реальных
        штрихкодов в него не входит по построению.
        """
        row = (
            await self._session.execute(select(Product).where(Product.code == code))
        ).scalar_one_or_none()
        if row is None:
            logger.debug("Продукта нет в корпусе", extra=safe_extra(code=code))
            return None

        logger.debug("Продукт прочитан из корпуса", extra=safe_extra(code=code))
        return ProductSummary.model_validate(row, from_attributes=True)

    async def upsert_batch(
        self,
        products: list[RawProduct],
        *,
        languages: list[str],
        min_ingredients_length: int,
        dump_version: str | None,
    ) -> tuple[int, int]:
        """Записать батч в `products_raw` и `products`.

        Returns:
            Пара (сколько строк затронуто в raw, сколько в products).
        """
        if not products:
            return 0, 0

        raw_rows = [self._raw_row(product, dump_version) for product in products]
        product_rows = [
            self._product_row(product, languages, min_ingredients_length) for product in products
        ]

        raw_count = await self._upsert(ProductRaw, raw_rows)
        # products пишется вторым: на нём внешний ключ на products_raw.
        product_count = await self._upsert(Product, product_rows)

        logger.debug(
            "Батч записан",
            extra={"raw": raw_count, "products": product_count, "batch": len(products)},
        )
        return raw_count, product_count

    async def _upsert(self, model: type[ProductRaw] | type[Product], rows: list[dict]) -> int:
        """Вставить или обновить строки по первичному ключу `code`.

        Батч режется на куски по числу **аргументов**, а не строк: asyncpg
        ограничивает их 32 767 (предел int16 в протоколе Postgres), и при
        42 колонках `products` батч из 1000 строк даёт 42 000 аргументов и
        падает с InterfaceError. Размер куска считается от числа колонок,
        поэтому добавление новой колонки не сломает заливку молча.
        """
        columns = len(model.__table__.columns)
        chunk_size = max(_MAX_QUERY_ARGS // columns, 1)

        affected = 0
        for start in range(0, len(rows), chunk_size):
            affected += await self._upsert_chunk(model, rows[start : start + chunk_size])
        return affected

    async def _upsert_chunk(self, model: type[ProductRaw] | type[Product], rows: list[dict]) -> int:
        statement = insert(model).values(rows)
        updatable = {
            column.name: statement.excluded[column.name]
            for column in model.__table__.columns
            if column.name != "code"
        }

        statement = statement.on_conflict_do_update(
            index_elements=["code"],
            set_=updatable,
            where=(
                # Сохранённая ревизия неизвестна — обновляем: отказывать не на
                # основании чего.
                model.rev.is_(None)
                # Обе известны — обновляем, только если пришедшая не старее.
                #
                # Случай «пришла запись БЕЗ ревизии, а сохранённая с ревизией»
                # намеренно не покрыт ни одним условием: сравнение с NULL даёт
                # NULL, условие ложно, обновления не будет. Именно это и нужно —
                # запись без ревизии не должна перетирать известную.
                | (statement.excluded.rev >= model.rev)
            ),
        )
        result = await self._session.execute(statement)
        # rowcount есть у CursorResult, но статически execute объявлен как
        # Result — берём через getattr, чтобы не врать типами.
        return int(getattr(result, "rowcount", 0) or 0)

    @staticmethod
    def _raw_row(product: RawProduct, dump_version: str | None) -> dict[str, Any]:
        return {
            "code": product.code,
            # Полный снимок: сырое не перезаписывается деструктивно, и новая
            # колонка достаётся отсюда без повторного скачивания дампа.
            "payload": product.to_raw_json(),
            "source": product.source,
            "dump_version": dump_version,
            "rev": product.rev,
        }

    @staticmethod
    def _product_row(
        product: RawProduct,
        languages: list[str],
        min_ingredients_length: int,
    ) -> dict[str, Any]:
        picked = product.pick_ingredients_text(languages, min_ingredients_length)
        ingredients_text, ingredients_lang = (picked[1], picked[0]) if picked else (None, None)

        row: dict[str, Any] = {
            "code": product.code,
            "product_name": product.pick_name(languages),
            "generic_name": next(iter(product.generic_name.values()), None),
            "brands": product.brands,
            "ingredients_text": ingredients_text,
            "ingredients_text_lang": ingredients_lang,
            "lang": product.lang,
            "languages": product.languages,
            "categories_tags": product.categories_tags,
            "food_groups_tags": product.food_groups_tags,
            "countries_tags": product.countries_tags,
            "labels_tags": product.labels_tags,
            "nutriscore_grade": product.nutriscore_grade,
            "nutriscore_score": product.nutriscore_score,
            "nova_group": product.nova_group,
            "nutrition_data_per": product.nutrition_data_per,
            "has_nutrition_data": product.has_nutrition_data,
            "ingredients_tags": product.ingredients_tags,
            "additives_tags": product.additives_tags,
            "allergens_tags": product.allergens_tags,
            "traces_tags": product.traces_tags,
            "ingredients_analysis_tags": product.ingredients_analysis_tags,
            "ingredients_n": product.ingredients_n,
            "known_ingredients_n": product.known_ingredients_n,
            "unknown_ingredients_n": product.unknown_ingredients_n,
            "additives_n": product.additives_n,
            "unique_scans_n": product.unique_scans_n,
            "popularity_key": product.popularity_key,
            "completeness": product.completeness,
            "rev": product.rev,
            "last_modified_t": product.last_modified_t,
        }

        # Отсутствующий нутриент остаётся NULL, а не нулём: ноль — осмысленное
        # значение, и подмена уехала бы в обучение M4 как настоящее измерение.
        for source_name, column in NUTRIENT_COLUMNS.items():
            row[column] = product.nutriments.get(source_name)

        return row
