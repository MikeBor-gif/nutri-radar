"""Адаптер полного дампа Open Food Facts (Parquet).

Три вещи, которые определяют весь код этого модуля и которые нельзя вывести
из документации OFF (`data-fields.txt` описывает CSV-экспорт, а не Parquet):

1. `ingredients_text`, `product_name`, `generic_name` — это
   `LIST<STRUCT(lang, text)>`, а не строки. Обращение как к колонке не работает.
2. Внутри этих списков есть служебная запись `lang = 'main'`, дублирующая текст
   главного языка. Разведка: 19 180 таких записей на 20 000 строк.
3. `nutriments` — список структур с полем `100g`, а не набор колонок.
   Плоского `sugars_100g` в Parquet не существует.

Вложенные структуры разворачиваются в SQL до списка пар, а в словарь
собираются уже в Python: `map_from_entries` в DuckDB падает на дублирующихся
ключах, а в краудсорсинговых данных дубли языков вполне возможны.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import duckdb
from pydantic import ValidationError

from nutri_radar.config import IngestSettings
from nutri_radar.errors import DataSourceError
from nutri_radar.ingest.models import MAIN_LANG_MARKER, RawProduct

logger = logging.getLogger(__name__)

# Нутриенты, которые переносим в нормализованную таблицу. Имена подтверждены
# разведкой (`ingest probe`), а не взяты из документации по CSV.
WANTED_NUTRIENTS = (
    "energy-kcal",
    "fat",
    "saturated-fat",
    "carbohydrates",
    "sugars",
    "fiber",
    "proteins",
    "salt",
    "sodium",
    "fruits-vegetables-nuts-estimate-from-ingredients",
    "nutrition-score-fr",
)

# Колонки дампа, которые вообще читаются. Parquet колоночный, поэтому короткий
# список — это прямая экономия чтения: из 111 колонок берём 30.
_MULTILANG_FIELDS = ("ingredients_text", "product_name", "generic_name")


def _multilang_expr(column: str) -> str:
    """Развернуть `LIST<STRUCT(lang,text)>` в список пар без служебного `main`."""
    return f"""
        list_transform(
            list_filter(
                coalesce({column}, []),
                x -> x.lang <> '{MAIN_LANG_MARKER}'
                     AND x.text IS NOT NULL
                     AND length(trim(x.text)) > 0
            ),
            x -> {{'lang': x.lang, 'text': trim(x.text)}}
        ) AS {column}
    """


def _nutriments_expr() -> str:
    """Оставить только нужные нутриенты со значением на 100 г.

    Записи без `100g` отбрасываются здесь, а не подменяются нулём: ноль —
    осмысленное значение, и подмена уехала бы в обучение M4 как измерение.
    """
    names = ", ".join(f"'{name}'" for name in WANTED_NUTRIENTS)
    return f"""
        list_transform(
            list_filter(
                coalesce(nutriments, []),
                x -> x.name IN ({names}) AND x['100g'] IS NOT NULL
            ),
            x -> {{'name': x.name, 'value': x['100g']}}
        ) AS nutriments
    """


SELECT_COLUMNS = f"""
    code,
    lang,
    {_multilang_expr("ingredients_text")},
    {_multilang_expr("product_name")},
    {_multilang_expr("generic_name")},
    brands,
    coalesce(categories_tags, []) AS categories_tags,
    coalesce(food_groups_tags, []) AS food_groups_tags,
    coalesce(countries_tags, []) AS countries_tags,
    coalesce(labels_tags, []) AS labels_tags,
    nutriscore_grade,
    nutriscore_score,
    nova_group,
    {_nutriments_expr()},
    nutrition_data_per,
    coalesce(no_nutrition_data, false) AS no_nutrition_data,
    coalesce(ingredients_tags, []) AS ingredients_tags,
    coalesce(ingredients_original_tags, []) AS ingredients_original_tags,
    coalesce(additives_tags, []) AS additives_tags,
    coalesce(allergens_tags, []) AS allergens_tags,
    coalesce(traces_tags, []) AS traces_tags,
    coalesce(ingredients_analysis_tags, []) AS ingredients_analysis_tags,
    ingredients_n,
    known_ingredients_n,
    unknown_ingredients_n,
    additives_n,
    ingredients AS ingredients_json,
    with_sweeteners,
    with_non_nutritive_sweeteners,
    coalesce(obsolete, false) AS obsolete,
    completeness,
    coalesce(data_quality_errors_tags, []) AS data_quality_errors,
    unique_scans_n,
    popularity_key,
    rev,
    last_modified_t,
    created_t,
    schema_version
"""


class ParquetSource:
    """Чтение продуктов из Parquet-дампа."""

    def __init__(self, settings: IngestSettings, *, path: Path | str | None = None) -> None:
        self._settings = settings
        self._path = str(path or settings.dump_path)

    def __repr__(self) -> str:
        return f"ParquetSource(path={self._path!r})"

    @property
    def path(self) -> str:
        return self._path

    def connect(self) -> duckdb.DuckDBPyConnection:
        """Соединение DuckDB с ограничением памяти.

        Без лимита DuckDB на файле 7,7 ГБ может съесть всю память и быть убитым
        операционной системой.
        """
        source = Path(self._path)
        if not source.exists() and "://" not in self._path:
            raise DataSourceError(
                f"Дамп не найден: {self._path}. Выполните `nutri-radar ingest dump`."
            )
        con = duckdb.connect()
        con.execute("INSTALL httpfs")
        con.execute("LOAD httpfs")
        con.execute(f"SET memory_limit = '{self._settings.duckdb_memory_limit}'")
        logger.debug(
            "Соединение DuckDB открыто",
            extra={"path": self._path, "memory_limit": self._settings.duckdb_memory_limit},
        )
        return con

    def iter_products(
        self,
        con: duckdb.DuckDBPyConnection,
        where: str,
        params: list[Any],
        *,
        limit: int | None = None,
    ) -> Iterator[list[RawProduct]]:
        """Отдавать продукты батчами.

        Итератор, а не список: в дампе 4,63 млн строк, и материализовать их
        целиком нельзя даже после фильтрации.

        Yields:
            Батчи `RawProduct` размером `IngestSettings.batch_size`.
        """
        sql = f"SELECT {SELECT_COLUMNS} FROM read_parquet(?) WHERE {where}"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"

        logger.debug("Запрос выборки", extra={"sql": " ".join(sql.split())[:400]})
        cursor = con.execute(sql, [self._path, *params])
        columns = [desc[0] for desc in cursor.description or []]

        skipped = 0
        while True:
            rows = cursor.fetchmany(self._settings.batch_size)
            if not rows:
                break

            batch: list[RawProduct] = []
            for row in rows:
                product = self._to_product(dict(zip(columns, row, strict=True)))
                if product is None:
                    skipped += 1
                    continue
                batch.append(product)

            if batch:
                yield batch

        if skipped:
            # Счётчик итоговый, а не по записи: на корпусе в сотни тысяч строк
            # лог на каждую отбраковку утопил бы вывод.
            logger.warning(
                "Записи отброшены при разборе",
                extra={"skipped": skipped, "reason": "не прошли валидацию RawProduct"},
            )

    @staticmethod
    def _to_product(row: dict[str, Any]) -> RawProduct | None:
        """Собрать `RawProduct` из строки DuckDB.

        Возвращает `None` для записей, которые не проходят валидацию: битый
        штрихкод не повод ронять прогон на 100 тыс. строк.
        """
        payload = dict(row)

        for field in _MULTILANG_FIELDS:
            entries = payload.get(field) or []
            # Словарь собирается здесь, а не в SQL: map_from_entries падает на
            # дублирующихся ключах, а дубли языков в краудсорсе возможны.
            payload[field] = {item["lang"]: item["text"] for item in entries}

        nutriments = payload.get("nutriments") or []
        payload["nutriments"] = {item["name"]: item["value"] for item in nutriments}

        payload["source"] = "parquet"

        try:
            return RawProduct.model_validate(payload)
        except ValidationError as exc:
            logger.debug(
                "Запись не прошла валидацию",
                extra={"code": row.get("code"), "error": exc.errors()[0]["msg"]},
            )
            return None
