"""Генератор синтетических фикстур источников.

Фикстуры создаются кодом, а не лежат бинарниками в репозитории: так видно,
что именно в них заложено, и их легко расширить новым граничным случаем.

**Критическое требование.** Схема фикстуры Parquet обязана совпадать с реальным
дампом по вложенным типам: `ingredients_text` — `LIST<STRUCT(lang, text)>`,
`nutriments` — `LIST<STRUCT(name, value, "100g", ...)>`. Упрощённая схема дала бы
зелёные тесты и падение на настоящих данных — худший вид зелёного набора.
Соответствие проверяется тестом `test_ingest_parquet.py`.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import duckdb

FIXTURES_DIR = Path(__file__).parent
PARQUET_FIXTURE = FIXTURES_DIR / "sample_dump.parquet"
DELTA_FIXTURE = FIXTURES_DIR / "sample_delta.jsonl.gz"

# Коды, которые ОБЯЗАНЫ пройти фильтр при языках en/ru/de/fr/pl,
# категориях снеков/напитков и пороге длины 10.
EXPECTED_MATCHING = frozenset(
    {
        "1000000000001",  # обычный снек, состав на en
        "1000000000002",  # состав на ru, есть unknown_ingredients_n
        "1000000000003",  # многоязычный состав, напиток
        "1000000000004",  # несколько уровней тегов категорий
        "1000000000005",  # без таблицы питательности, но состав есть
    }
)


def _product(
    code: str,
    *,
    ingredients: list[tuple[str, str]],
    names: list[tuple[str, str]] | None = None,
    categories: list[str],
    nutriments: list[tuple[str, float | None]] | None = None,
    lang: str = "en",
    obsolete: bool = False,
    quality_errors: list[str] | None = None,
    no_nutrition_data: bool = False,
    unknown_ingredients_n: int | None = 0,
    rev: int | None = 1,
    nutriscore_grade: str | None = "c",
) -> dict[str, Any]:
    """Одна строка будущего Parquet в виде словаря Python."""
    return {
        "code": code,
        "lang": lang,
        "ingredients_text": [{"lang": lg, "text": tx} for lg, tx in ingredients],
        # Сравнение с None, а не `or`: пустой список — осмысленное значение
        # («нутриентов нет»), и `or` подменил бы его значением по умолчанию.
        "product_name": [
            {"lang": lg, "text": tx}
            for lg, tx in (names if names is not None else [("en", f"Product {code}")])
        ],
        "generic_name": [],
        "brands": "Тестовый бренд",
        "categories_tags": categories,
        "food_groups_tags": [],
        "countries_tags": ["en:france"],
        "labels_tags": [],
        "nutriscore_grade": nutriscore_grade,
        "nutriscore_score": 5,
        "nova_group": 4,
        "nutriments": [
            {
                "name": name,
                "value": value,
                "100g": value,
                "serving": None,
                "unit": "g",
                "prepared_value": None,
                "prepared_100g": None,
                "prepared_serving": None,
                "prepared_unit": None,
            }
            for name, value in (
                nutriments if nutriments is not None else [("sugars", 12.5), ("fat", 3.0)]
            )
        ],
        "nutrition_data_per": "100g",
        "no_nutrition_data": no_nutrition_data,
        "ingredients_tags": ["en:sugar"],
        "ingredients_original_tags": ["en:sugar"],
        "additives_tags": ["en:e330"],
        "allergens_tags": [],
        "traces_tags": [],
        "ingredients_analysis_tags": ["en:palm-oil-free"],
        "ingredients_n": 3,
        "known_ingredients_n": 3,
        "unknown_ingredients_n": unknown_ingredients_n,
        "additives_n": 1,
        "ingredients": '{"parsed": true}',
        "with_sweeteners": 0,
        "with_non_nutritive_sweeteners": 0,
        "obsolete": obsolete,
        "completeness": 0.8,
        "data_quality_errors_tags": quality_errors or [],
        "unique_scans_n": 42,
        "popularity_key": 1000,
        "rev": rev,
        "last_modified_t": 1750000000,
        "created_t": 1700000000,
        "schema_version": 1004,
    }


def build_rows() -> list[dict[str, Any]]:
    """Строки фикстуры, покрывающие граничные случаи фильтра."""
    snack = ["en:snacks", "en:sweet-snacks"]
    drink = ["en:beverages"]

    return [
        # --- должны пройти фильтр -------------------------------------------
        _product(
            "1000000000001",
            ingredients=[("en", "Sugar, palm oil, hazelnuts, cocoa powder")],
            categories=snack,
        ),
        _product(
            "1000000000002",
            ingredients=[("ru", "Сахар, пальмовое масло, фундук, какао")],
            lang="ru",
            categories=snack,
            # Парсер OFF не справился — кандидат в LLM-корпус (ADR-006)
            unknown_ingredients_n=3,
        ),
        _product(
            "1000000000003",
            ingredients=[
                ("fr", "Eau, sucre, arome naturel de citron"),
                ("de", "Wasser, Zucker, naturliches Zitronenaroma"),
                # Служебная запись: дублирует главный язык и языком не является
                ("main", "Eau, sucre, arome naturel de citron"),
            ],
            lang="fr",
            categories=drink,
        ),
        _product(
            "1000000000004",
            ingredients=[("pl", "Cukier, olej palmowy, orzechy laskowe")],
            lang="pl",
            # Иерархия тегов: продукт несёт несколько уровней сразу
            categories=["en:snacks", "en:sweet-snacks", "en:chocolates", "en:spreads"],
        ),
        _product(
            "1000000000005",
            ingredients=[("en", "Water, sugar, citric acid, natural flavouring")],
            categories=drink,
            # Нет таблицы питательности — для корпуса это не помеха,
            # текст состава есть
            no_nutrition_data=True,
            nutriments=[],
            nutriscore_grade=None,
        ),
        # --- НЕ должны пройти фильтр ----------------------------------------
        _product(
            "2000000000001",
            # Состава нет вовсе
            ingredients=[],
            categories=snack,
        ),
        _product(
            "2000000000002",
            # Состав короче порога в 10 символов
            ingredients=[("en", "Sugar")],
            categories=snack,
        ),
        _product(
            "2000000000003",
            # Язык вне списка настроек
            ingredients=[("ja", "砂糖、パーム油、ヘーゼルナッツ、ココア")],
            lang="ja",
            categories=snack,
        ),
        _product(
            "2000000000004",
            ingredients=[("en", "Sugar, palm oil, hazelnuts")],
            categories=snack,
            obsolete=True,
        ),
        _product(
            "2000000000005",
            ingredients=[("en", "Sugar, palm oil, hazelnuts")],
            categories=snack,
            quality_errors=["en:nutrition-value-total-over-105"],
        ),
        _product(
            "2000000000006",
            # Категория вне списка настроек
            ingredients=[("en", "Water, salt, live cultures, rennet")],
            categories=["en:pet-food"],
        ),
        _product(
            "2000000000007",
            # Только служебная запись main — настоящих языков нет
            ingredients=[("main", "Sugar, palm oil, hazelnuts, cocoa")],
            categories=snack,
        ),
    ]


def write_parquet_fixture(path: Path = PARQUET_FIXTURE) -> Path:
    """Собрать Parquet с вложенными типами реального дампа."""
    rows = build_rows()
    con = duckdb.connect()
    # Строим через JSON: DuckDB выводит LIST<STRUCT> из вложенных объектов,
    # а не сплющивает их. Это и даёт совпадение с настоящей схемой.
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    tmp_json = path.with_suffix(".tmp.jsonl")
    tmp_json.write_text(payload, encoding="utf-8")
    try:
        con.execute(
            f"COPY (SELECT * FROM read_json_auto('{tmp_json.as_posix()}')) "
            f"TO '{path.as_posix()}' (FORMAT PARQUET)"
        )
    finally:
        tmp_json.unlink(missing_ok=True)
        con.close()
    return path


def build_delta_records() -> list[dict[str, Any]]:
    """Те же продукты, но в MongoDB-форме дельта-экспортов.

    Ключевое отличие формата (ADR-004): состав лежит ПЛОСКИМИ ключами
    `ingredients_text_en`, а нутриенты — плоским объектом. Первый продукт
    намеренно совпадает с продуктом из Parquet-фикстуры: на нём проверяется
    инвариант «оба источника дают идентичный RawProduct».
    """
    return [
        {
            "code": "1000000000001",
            "lang": "en",
            "ingredients_text_en": "Sugar, palm oil, hazelnuts, cocoa powder",
            "product_name_en": "Product 1000000000001",
            "brands": "Тестовый бренд",
            "categories_tags": ["en:snacks", "en:sweet-snacks"],
            "countries_tags": ["en:france"],
            "nutriscore_grade": "c",
            "nutriscore_score": 5,
            "nova_group": 4,
            "nutriments": {"sugars_100g": 12.5, "fat_100g": 3.0},
            "nutrition_data_per": "100g",
            "ingredients_tags": ["en:sugar"],
            "ingredients_original_tags": ["en:sugar"],
            "additives_tags": ["en:e330"],
            "ingredients_analysis_tags": ["en:palm-oil-free"],
            "ingredients_n": 3,
            "known_ingredients_n": 3,
            "unknown_ingredients_n": 0,
            "additives_n": 1,
            "completeness": 0.8,
            "unique_scans_n": 42,
            "popularity_key": 1000,
            "rev": 1,
            "last_modified_t": 1750000000,
            "created_t": 1700000000,
        },
        {
            # Более свежая ревизия того же продукта: проверяет, что дельта
            # обновляет запись, а не создаёт дубль
            "code": "1000000000002",
            "lang": "ru",
            "ingredients_text_ru": "Сахар, пальмовое масло, фундук, какао, лецитин",
            "product_name_ru": "Обновлённое название",
            "categories_tags": ["en:snacks", "en:sweet-snacks"],
            "nutriscore_grade": "d",
            "nutriments": {"sugars_100g": 40.0},
            "unknown_ingredients_n": 3,
            "unique_scans_n": 17,
            "rev": 99,
            "last_modified_t": 1760000000,
        },
        {
            # Категория вне корпуса: фильтр обязан отсеять и в дельте тоже
            "code": "2000000000006",
            "lang": "en",
            "ingredients_text_en": "Water, salt, live cultures, rennet",
            "categories_tags": ["en:pet-food"],
            "unique_scans_n": 5,
            "rev": 5,
        },
        {
            # Категория подходит, но продукт никто ни разу не сканировал —
            # заброшенная запись. Отсекается критерием require_scanned.
            "code": "2000000000008",
            "lang": "en",
            "ingredients_text_en": "Sugar, palm oil, hazelnuts, cocoa powder",
            "categories_tags": ["en:sweet-snacks"],
            "unique_scans_n": 0,
            "rev": 3,
        },
    ]


def write_delta_fixture(path: Path = DELTA_FIXTURE) -> Path:
    """Собрать gzip-JSONL дельта-экспорта."""
    lines = "\n".join(
        json.dumps(record, ensure_ascii=False) for record in build_delta_records()
    )
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(lines + "\n")
    return path


def write_all() -> tuple[Path, Path]:
    return write_parquet_fixture(), write_delta_fixture()


if __name__ == "__main__":
    parquet_path, delta_path = write_all()
    print(f"Parquet: {parquet_path}")
    print(f"Дельта:  {delta_path}")
