"""Тесты адаптера дельта-экспортов.

Главный тест здесь — `TestFormatsAgree`: один и тот же продукт, записанный
в Parquet-форме и в MongoDB-форме, обязан дать идентичный `RawProduct`.
Ради этого инварианта и существует слой нормализации (ADR-004); если он
нарушится, весь смысл двух адаптеров пропадает.
"""

from __future__ import annotations

import pytest

from nutri_radar.config import IngestSettings
from nutri_radar.ingest.select import iter_selected, matches_corpus
from nutri_radar.ingest.sources.delta import (
    DELTA_RETENTION_DAYS,
    check_retention_gap,
    iter_records,
    parse_index,
    select_files,
    to_raw_product,
)
from nutri_radar.ingest.sources.parquet import ParquetSource

from .data.make_fixtures import (
    DELTA_FIXTURE,
    PARQUET_FIXTURE,
    build_delta_records,
    write_delta_fixture,
    write_parquet_fixture,
)


@pytest.fixture(scope="module", autouse=True)
def _fixture_files():
    write_parquet_fixture()
    write_delta_fixture()
    yield


@pytest.fixture
def ingest_settings() -> IngestSettings:
    return IngestSettings(
        languages=["en", "ru", "de", "fr", "pl"],
        category_tags=["en:snacks", "en:sweet-snacks", "en:beverages"],
        min_ingredients_length=10,
        batch_size=2,
    )


class TestIndex:
    def test_файлы_упорядочены_по_времени(self):
        files = parse_index("1690086400.json.gz\n1689913600.json.gz\n1690000000.json.gz\n")

        assert [f.timestamp for f in files] == [1689913600, 1690000000, 1690086400]

    def test_мусорные_строки_игнорируются(self):
        files = parse_index("1690000000.json.gz\n\nсовсем не файл\nreadme.txt\n")

        assert len(files) == 1

    def test_отбираются_только_файлы_новее_водяного_знака(self):
        files = parse_index("1689913600.json.gz\n1690000000.json.gz\n1690086400.json.gz\n")

        selected = select_files(files, watermark=1690000000)

        assert [f.timestamp for f in selected] == [1690086400]

    def test_без_водяного_знака_берутся_все(self):
        files = parse_index("1689913600.json.gz\n1690000000.json.gz\n")
        assert len(select_files(files, watermark=None)) == 2

    def test_url_строится_от_адреса_индекса(self):
        files = parse_index("1690000000.json.gz\n")
        url = files[0].url("https://example.test/data/delta/index.txt")

        assert url == "https://example.test/data/delta/1690000000.json.gz"


class TestRetentionGap:
    def test_разрыв_обнаруживается(self):
        """Хранение дельт ограничено, и часть истории могла исчезнуть."""
        files = parse_index("1690000000.json.gz\n")

        assert check_retention_gap(files, watermark=1600000000) is True

    def test_непрерывная_история_не_считается_разрывом(self):
        files = parse_index("1690000000.json.gz\n1690086400.json.gz\n")

        assert check_retention_gap(files, watermark=1690000000) is False

    def test_без_водяного_знака_разрыва_нет(self):
        files = parse_index("1690000000.json.gz\n")
        assert check_retention_gap(files, watermark=None) is False

    def test_срок_хранения_зафиксирован(self):
        assert DELTA_RETENTION_DAYS == 14


class TestMongoForm:
    """В дельтах язык — часть имени ключа, а нутриенты приходят плоско."""

    def test_плоские_языковые_ключи_собираются_в_словарь(self):
        product = to_raw_product(
            {
                "code": "1000000000001",
                "ingredients_text_en": "Sugar, palm oil",
                "ingredients_text_ru": "Сахар, пальмовое масло",
            }
        )

        assert product is not None
        assert product.ingredients_text == {
            "en": "Sugar, palm oil",
            "ru": "Сахар, пальмовое масло",
        }

    def test_служебные_суффиксы_не_принимаются_за_язык(self):
        product = to_raw_product(
            {
                "code": "1000000000001",
                "ingredients_text_en": "Sugar, palm oil",
                "ingredients_text_debug_tags": "мусор",
                "ingredients_text_with_allergens": "тоже не язык",
            }
        )

        assert product is not None
        assert list(product.ingredients_text) == ["en"]

    def test_плоские_нутриенты_разворачиваются(self):
        product = to_raw_product(
            {
                "code": "1000000000001",
                "nutriments": {"sugars_100g": 12.5, "fat_100g": 3.0, "salt_serving": 1.0},
            }
        )

        assert product is not None
        # `salt_serving` не на 100 г — не наш формат, отбрасывается
        assert product.nutriments == {"sugars": 12.5, "fat": 3.0}

    def test_нечисловые_нутриенты_пропускаются(self):
        product = to_raw_product(
            {"code": "1000000000001", "nutriments": {"sugars_100g": "не число"}}
        )

        assert product is not None
        assert product.nutriments == {}

    def test_источник_помечен_как_delta(self):
        product = to_raw_product({"code": "1000000000001"})

        assert product is not None
        assert product.source == "delta"

    def test_битый_штрихкод_даёт_none_а_не_исключение(self):
        """Одна плохая запись не должна ронять применение всего файла."""
        assert to_raw_product({"code": "мусор"}) is None
        assert to_raw_product({}) is None


class TestReadFile:
    def test_записи_читаются_из_gzip(self):
        records = list(iter_records(DELTA_FIXTURE))

        assert len(records) == len(build_delta_records())
        assert records[0]["code"] == "1000000000001"


class TestFilterOnDelta:
    def test_фильтр_корпуса_применяется_и_к_дельтам(self, ingest_settings):
        """Дельта может принести продукт из чужой категории."""
        accepted = []
        for record in iter_records(DELTA_FIXTURE):
            product = to_raw_product(record)
            if product is not None and matches_corpus(product, ingest_settings):
                accepted.append(product.code)

        assert accepted == ["1000000000001", "1000000000002"]
        assert "2000000000006" not in accepted


class TestFormatsAgree:
    """Инвариант ADR-004: оба источника дают идентичный RawProduct."""

    COMPARED_FIELDS = (
        "code",
        "lang",
        "ingredients_text",
        "product_name",
        "brands",
        "categories_tags",
        "countries_tags",
        "nutriscore_grade",
        "nutriscore_score",
        "nova_group",
        "nutriments",
        "nutrition_data_per",
        "ingredients_tags",
        "ingredients_original_tags",
        "additives_tags",
        "ingredients_analysis_tags",
        "ingredients_n",
        "known_ingredients_n",
        "unknown_ingredients_n",
        "additives_n",
        "completeness",
        "unique_scans_n",
        "popularity_key",
        "rev",
        "last_modified_t",
        "created_t",
    )

    def test_один_продукт_в_двух_форматах_совпадает(self, ingest_settings):
        source = ParquetSource(ingest_settings, path=str(PARQUET_FIXTURE))
        con = source.connect()
        try:
            from_parquet = {
                p.code: p for batch in iter_selected(source, ingest_settings, con) for p in batch
            }["1000000000001"]
        finally:
            con.close()

        record = next(r for r in build_delta_records() if r["code"] == "1000000000001")
        from_delta = to_raw_product(record)
        assert from_delta is not None

        differences = {
            field: (getattr(from_parquet, field), getattr(from_delta, field))
            for field in self.COMPARED_FIELDS
            if getattr(from_parquet, field) != getattr(from_delta, field)
        }

        assert not differences, f"Форматы разъехались: {differences}"

    def test_поле_source_различается_намеренно(self, ingest_settings):
        """Единственное поле, которое обязано отличаться."""
        record = next(r for r in build_delta_records() if r["code"] == "1000000000001")
        from_delta = to_raw_product(record)

        assert from_delta is not None
        assert from_delta.source == "delta"
