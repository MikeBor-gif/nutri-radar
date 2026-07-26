"""Тесты Parquet-адаптера и фильтра корпуса.

DoD майлстоуна прямо требует тесты фильтрации на маленьком синтетическом
Parquet. Проверяется не количество отобранных строк, а **конкретный набор
кодов**: совпадение по количеству может оказаться случайным.
"""

from __future__ import annotations

import pytest

from nutri_radar.config import IngestSettings
from nutri_radar.errors import DataSourceError
from nutri_radar.ingest.select import collect_stats, iter_selected, matches_corpus
from nutri_radar.ingest.sources.parquet import ParquetSource

from .data.make_fixtures import EXPECTED_MATCHING, PARQUET_FIXTURE, write_parquet_fixture


@pytest.fixture(scope="module", autouse=True)
def _fixture_file():
    write_parquet_fixture()
    yield PARQUET_FIXTURE


@pytest.fixture
def ingest_settings() -> IngestSettings:
    """Настройки корпуса под фикстуру: те же языки и категории, что в проекте."""
    return IngestSettings(
        languages=["en", "ru", "de", "fr", "pl"],
        category_tags=["en:snacks", "en:sweet-snacks", "en:beverages"],
        min_ingredients_length=10,
        batch_size=2,
    )


@pytest.fixture
def source(ingest_settings: IngestSettings) -> ParquetSource:
    return ParquetSource(ingest_settings, path=str(PARQUET_FIXTURE))


@pytest.fixture
def connection(source: ParquetSource):
    con = source.connect()
    yield con
    con.close()


def collect_products(source, settings, con) -> dict:
    return {p.code: p for batch in iter_selected(source, settings, con) for p in batch}


class TestSchemaFidelity:
    """Схема фикстуры обязана совпадать с реальным дампом по вложенным типам.

    Упрощённая схема дала бы зелёные тесты и падение на настоящих данных —
    это худший вид зелёного набора.
    """

    def test_многоязычные_поля_это_список_структур(self, connection):
        rows = connection.execute(
            "DESCRIBE SELECT ingredients_text, product_name "
            f"FROM read_parquet('{PARQUET_FIXTURE.as_posix()}')"
        ).fetchall()
        types = {name: type_ for name, type_, *_ in rows}

        for field in ("ingredients_text", "product_name"):
            assert "STRUCT" in types[field], f"{field} должен быть списком структур"
            assert "lang" in types[field]
            assert types[field].endswith("[]"), f"{field} должен быть списком"

    def test_нутриенты_это_список_структур_с_ключом_100g(self, connection):
        row = connection.execute(
            f"DESCRIBE SELECT nutriments FROM read_parquet('{PARQUET_FIXTURE.as_posix()}')"
        ).fetchone()
        assert row is not None
        type_ = row[1]

        assert "STRUCT" in type_ and type_.endswith("[]")
        assert '"100g"' in type_ or "100g" in type_
        assert "name" in type_


class TestFilter:
    def test_отбираются_ровно_ожидаемые_коды(self, source, ingest_settings, connection):
        got = set(collect_products(source, ingest_settings, connection))

        assert got == set(EXPECTED_MATCHING)

    @pytest.mark.parametrize(
        ("code", "reason"),
        [
            ("2000000000001", "состава нет вовсе"),
            ("2000000000002", "состав короче порога"),
            ("2000000000003", "язык вне списка настроек"),
            ("2000000000004", "снят с производства"),
            ("2000000000005", "ошибки качества данных"),
            ("2000000000006", "категория вне списка"),
            ("2000000000007", "только служебная запись main"),
        ],
    )
    def test_граничные_случаи_отсеиваются(self, source, ingest_settings, connection, code, reason):
        got = collect_products(source, ingest_settings, connection)
        assert code not in got, f"должен отсеяться: {reason}"

    def test_сужение_категорий_сужает_выборку(self, source, connection):
        narrow = IngestSettings(
            languages=["en", "ru", "de", "fr", "pl"],
            category_tags=["en:beverages"],
            min_ingredients_length=10,
            batch_size=2,
        )
        got = set(collect_products(source, narrow, connection))

        assert got == {"1000000000003", "1000000000005"}

    def test_повышение_порога_длины_сужает_выборку(self, source, connection):
        strict = IngestSettings(
            languages=["en", "ru", "de", "fr", "pl"],
            category_tags=["en:snacks", "en:sweet-snacks", "en:beverages"],
            min_ingredients_length=200,
            batch_size=2,
        )
        assert collect_products(source, strict, connection) == {}


class TestParsing:
    def test_многоязычный_состав_разворачивается_в_словарь(
        self, source, ingest_settings, connection
    ):
        product = collect_products(source, ingest_settings, connection)["1000000000003"]

        assert product.ingredients_text["fr"].startswith("Eau, sucre")
        assert product.ingredients_text["de"].startswith("Wasser, Zucker")

    def test_служебная_запись_main_не_попадает_в_словарь(self, source, ingest_settings, connection):
        product = collect_products(source, ingest_settings, connection)["1000000000003"]

        assert "main" not in product.ingredients_text
        assert product.languages == ["de", "fr"]

    def test_нутриенты_собираются_в_плоский_словарь(self, source, ingest_settings, connection):
        product = collect_products(source, ingest_settings, connection)["1000000000001"]

        assert product.nutriments == {"sugars": 12.5, "fat": 3.0}

    def test_отсутствующий_нутриент_даёт_none_а_не_ноль(self, source, ingest_settings, connection):
        """Ключевая проверка: нули уехали бы в обучение M4 как измерения."""
        product = collect_products(source, ingest_settings, connection)["1000000000001"]

        assert "fiber" not in product.nutriments
        assert product.nutriments.get("fiber") is None

    def test_продукт_без_питательности_имеет_пустые_нутриенты(
        self, source, ingest_settings, connection
    ):
        product = collect_products(source, ingest_settings, connection)["1000000000005"]

        assert product.nutriments == {}
        assert product.has_nutrition_data is False

    def test_baseline_парсера_off_переносится(self, source, ingest_settings, connection):
        """Поля OFF нужны для сравнения в M3 и отбора корпуса в M2 (ADR-006)."""
        product = collect_products(source, ingest_settings, connection)["1000000000002"]

        assert product.unknown_ingredients_n == 3
        assert product.additives_tags == ["en:e330"]
        assert product.ingredients_analysis_tags == ["en:palm-oil-free"]

    def test_источник_помечен_как_parquet(self, source, ingest_settings, connection):
        product = collect_products(source, ingest_settings, connection)["1000000000001"]
        assert product.source == "parquet"


class TestBatching:
    def test_итератор_отдаёт_батчи_заданного_размера(self, source, ingest_settings, connection):
        batches = list(iter_selected(source, ingest_settings, connection))

        assert all(len(batch) <= ingest_settings.batch_size for batch in batches)
        assert sum(len(batch) for batch in batches) == len(EXPECTED_MATCHING)

    def test_limit_ограничивает_выборку(self, source, ingest_settings, connection):
        total = sum(
            len(batch) for batch in iter_selected(source, ingest_settings, connection, limit=2)
        )
        assert total == 2


class TestStats:
    def test_статистика_совпадает_с_выборкой(self, source, ingest_settings, connection):
        stats = collect_stats(source, ingest_settings, connection)

        assert stats.matched_rows == len(EXPECTED_MATCHING)
        assert stats.total_rows > stats.matched_rows
        assert set(stats.by_language) <= set(ingest_settings.languages)

    def test_считаются_кандидаты_в_llm_корпус(self, source, ingest_settings, connection):
        """Отбор LLM-корпуса на M2 идёт по unknown_ingredients_n (ADR-006)."""
        stats = collect_stats(source, ingest_settings, connection)
        assert stats.unknown_ingredients == 1


class TestErrors:
    def test_отсутствующий_дамп_даёт_понятную_ошибку(self, ingest_settings):
        missing = ParquetSource(ingest_settings, path="data/нет-такого-файла.parquet")

        with pytest.raises(DataSourceError, match="ingest dump"):
            missing.connect()


class TestSqlAndPythonFiltersAgree:
    """SQL-фильтр и питоновский предикат обязаны давать одинаковый результат.

    Критерии живут в двух формах: SQL для Parquet и Python для дельт. Разъезд
    между ними — самая вероятная и самая незаметная поломка в этом слайсе.
    """

    def test_обе_формы_отбирают_одинаковые_коды(self, source, ingest_settings, connection):
        from_sql = set(collect_products(source, ingest_settings, connection))

        # Читаем ВСЕ строки фикстуры без фильтра и прогоняем питоновским предикатом
        all_products = {
            p.code: p for batch in source.iter_products(connection, "TRUE", []) for p in batch
        }
        from_python = {
            code
            for code, product in all_products.items()
            if matches_corpus(product, ingest_settings)
        }

        assert from_python == from_sql
