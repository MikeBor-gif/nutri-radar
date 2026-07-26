"""Интеграционные тесты заливки. Требуют поднятого Postgres.

Здесь проверяется главный пункт DoD майлстоуна: **повторный запуск не создаёт
дублей**. Это невозможно проверить моками — нужны настоящие `ON CONFLICT`
и настоящее ограничение уникальности.

Запуск: `uv run pytest -m integration`
"""

from __future__ import annotations

import time

import pytest
from sqlalchemy import text

from nutri_radar.config import IngestSettings, Settings
from nutri_radar.db.models.run import RunStage, RunStatus
from nutri_radar.db.repositories.product import ProductRepository
from nutri_radar.db.session import dispose_engine, get_session
from nutri_radar.ingest.load import load_corpus
from nutri_radar.ingest.models import RawProduct
from nutri_radar.ingest.sources.parquet import ParquetSource

from .data.make_fixtures import EXPECTED_MATCHING, PARQUET_FIXTURE, write_parquet_fixture

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def _fixture_file():
    write_parquet_fixture()
    yield


@pytest.fixture
async def _clean_engine():
    """Движок кэшируется в модуле — сбрасываем между тестами."""
    await dispose_engine()
    yield
    await dispose_engine()


@pytest.fixture
def load_settings(integration_settings: Settings) -> Settings:
    """Настройки интеграционной БД плюс параметры корпуса под фикстуру."""
    ingest = IngestSettings(
        **{
            **integration_settings.ingest.model_dump(),
            "languages": ["en", "ru", "de", "fr", "pl"],
            "category_tags": ["en:snacks", "en:sweet-snacks", "en:beverages"],
            "min_ingredients_length": 10,
            "batch_size": 2,
        }
    )
    return integration_settings.model_copy(update={"ingest": ingest})


@pytest.fixture
def patched_source(monkeypatch):
    """Подменить путь дампа на фикстуру."""
    original = ParquetSource
    monkeypatch.setattr(
        "nutri_radar.ingest.load.ParquetSource",
        lambda settings, **kwargs: original(settings, path=str(PARQUET_FIXTURE)),
    )


async def count_rows(settings: Settings, table: str) -> int:
    async with get_session(settings.db) as session:
        return (await session.execute(text(f"SELECT count(*) FROM {table}"))).scalar_one()


async def fetch_product(settings: Settings, code: str) -> dict | None:
    async with get_session(settings.db) as session:
        row = (
            await session.execute(
                text(
                    "SELECT product_name, rev, nutriscore_grade, sugars_100g, fiber_100g, "
                    "ingredients_text_lang, has_nutrition_data "
                    "FROM products WHERE code = :code"
                ),
                {"code": code},
            )
        ).one_or_none()
    if row is None:
        return None
    return dict(row._mapping)


def make_product(code: str, *, rev: int | None, name: str) -> RawProduct:
    return RawProduct(
        code=code,
        source="parquet",
        lang="en",
        rev=rev,
        ingredients_text={"en": "Sugar, palm oil, hazelnuts, cocoa powder"},
        product_name={"en": name},
        categories_tags=["en:snacks"],
    )


async def upsert(settings: Settings, product: RawProduct) -> None:
    async with get_session(settings.db) as session:
        await ProductRepository(session).upsert_batch(
            [product],
            languages=settings.ingest.languages,
            min_ingredients_length=settings.ingest.min_ingredients_length,
            dump_version="тестовая-версия",
        )


class TestIdempotency:
    """Главный пункт DoD: повторный запуск не создаёт дублей."""

    async def test_первая_заливка_даёт_ожидаемое_число_строк(
        self, load_settings, migrated_database, patched_source, _clean_engine
    ):
        result = await load_corpus(load_settings)

        assert result.processed == len(EXPECTED_MATCHING)
        assert await count_rows(load_settings, "products") == len(EXPECTED_MATCHING)
        assert await count_rows(load_settings, "products_raw") == len(EXPECTED_MATCHING)

    async def test_повторная_заливка_не_меняет_число_строк(
        self, load_settings, migrated_database, patched_source, _clean_engine
    ):
        await load_corpus(load_settings)
        before = await count_rows(load_settings, "products")

        await load_corpus(load_settings)
        after = await count_rows(load_settings, "products")

        assert after == before == len(EXPECTED_MATCHING)

    async def test_три_прогона_подряд_не_плодят_строк(
        self, load_settings, migrated_database, patched_source, _clean_engine
    ):
        for _ in range(3):
            await load_corpus(load_settings)

        assert await count_rows(load_settings, "products") == len(EXPECTED_MATCHING)


class TestRevisionGuard:
    """Старый дамп не должен откатывать данные назад."""

    CODE = "1000000000001"

    async def test_свежая_ревизия_обновляет(self, load_settings, migrated_database, _clean_engine):
        await upsert(load_settings, make_product(self.CODE, rev=10, name="старое"))
        await upsert(load_settings, make_product(self.CODE, rev=50, name="новое"))

        product = await fetch_product(load_settings, self.CODE)

        assert product is not None
        assert product["product_name"] == "новое"
        assert product["rev"] == 50

    async def test_старая_ревизия_не_откатывает(
        self, load_settings, migrated_database, _clean_engine
    ):
        await upsert(load_settings, make_product(self.CODE, rev=50, name="новое"))
        await upsert(load_settings, make_product(self.CODE, rev=10, name="СТАРОЕ"))

        product = await fetch_product(load_settings, self.CODE)

        assert product is not None
        assert product["product_name"] == "новое", "запись откатилась назад"
        assert product["rev"] == 50

    async def test_запись_без_ревизии_не_перетирает_известную(
        self, load_settings, migrated_database, _clean_engine
    ):
        await upsert(load_settings, make_product(self.CODE, rev=50, name="новое"))
        await upsert(load_settings, make_product(self.CODE, rev=None, name="БЕЗ РЕВИЗИИ"))

        product = await fetch_product(load_settings, self.CODE)

        assert product is not None
        assert product["product_name"] == "новое"
        assert product["rev"] == 50

    async def test_равная_ревизия_обновляет(self, load_settings, migrated_database, _clean_engine):
        """Повторный прогон того же дампа должен проходить, а не отвергаться."""
        await upsert(load_settings, make_product(self.CODE, rev=50, name="первое"))
        await upsert(load_settings, make_product(self.CODE, rev=50, name="второе"))

        product = await fetch_product(load_settings, self.CODE)

        assert product is not None
        assert product["product_name"] == "второе"

    async def test_известная_ревизия_поверх_неизвестной_проходит(
        self, load_settings, migrated_database, _clean_engine
    ):
        await upsert(load_settings, make_product(self.CODE, rev=None, name="без ревизии"))
        await upsert(load_settings, make_product(self.CODE, rev=7, name="с ревизией"))

        product = await fetch_product(load_settings, self.CODE)

        assert product is not None
        assert product["product_name"] == "с ревизией"
        assert product["rev"] == 7


class TestStoredValues:
    async def test_отсутствующий_нутриент_остаётся_null(
        self, load_settings, migrated_database, patched_source, _clean_engine
    ):
        """Сквозная проверка: ноль вместо NULL уехал бы в обучение M4."""
        await load_corpus(load_settings)

        product = await fetch_product(load_settings, "1000000000001")

        assert product is not None
        assert product["sugars_100g"] is not None
        assert product["fiber_100g"] is None, "пропуск подменён нулём"

    async def test_язык_состава_сохраняется(
        self, load_settings, migrated_database, patched_source, _clean_engine
    ):
        await load_corpus(load_settings)

        assert (await fetch_product(load_settings, "1000000000002"))[
            "ingredients_text_lang"
        ] == "ru"
        assert (await fetch_product(load_settings, "1000000000003"))[
            "ingredients_text_lang"
        ] == "fr"

    async def test_продукт_без_питательности_помечен(
        self, load_settings, migrated_database, patched_source, _clean_engine
    ):
        await load_corpus(load_settings)

        product = await fetch_product(load_settings, "1000000000005")

        assert product is not None
        assert product["has_nutrition_data"] is False
        assert product["nutriscore_grade"] is None

    async def test_сырой_снимок_сохраняет_исходную_форму(
        self, load_settings, migrated_database, patched_source, _clean_engine
    ):
        """`products_raw` не перезаписывается деструктивно (раздел 7 брифа)."""
        await load_corpus(load_settings)

        async with get_session(load_settings.db) as session:
            row = (
                await session.execute(
                    text(
                        "SELECT payload, source, dump_version FROM products_raw "
                        "WHERE code = '1000000000003'"
                    )
                )
            ).one()

        payload = row.payload
        # Полный словарь по языкам остаётся в сыром слое, хотя в products
        # уехал только один язык.
        assert set(payload["ingredients_text"]) == {"de", "fr"}
        assert row.source == "parquet"


class TestRunJournal:
    async def test_прогон_записывается_в_журнал(
        self, load_settings, migrated_database, patched_source, _clean_engine
    ):
        """Доля пропусков — обязательная метрика, она должна читаться из БД."""
        result = await load_corpus(load_settings)

        async with get_session(load_settings.db) as session:
            row = (
                await session.execute(
                    text(
                        "SELECT stage, status, items_processed, items_skipped, params, "
                        "finished_at FROM runs WHERE id = :id"
                    ),
                    {"id": result.run_id},
                )
            ).one()

        assert row.stage == RunStage.INGEST.value
        assert row.status == RunStatus.COMPLETED.value
        assert row.items_processed == len(EXPECTED_MATCHING)
        assert row.items_skipped == 0
        assert row.finished_at is not None
        assert row.params["mode"] == "select"

    async def test_dry_run_не_пишет_ни_в_бд_ни_в_журнал(
        self, load_settings, migrated_database, patched_source, _clean_engine
    ):
        result = await load_corpus(load_settings, dry_run=True)

        assert result.processed == len(EXPECTED_MATCHING)
        assert result.run_id is None
        assert await count_rows(load_settings, "products") == 0
        assert await count_rows(load_settings, "runs") == 0


class TestLargeBatch:
    """Батч боевого размера. На маленьких этот класс ошибок не проявляется.

    asyncpg ограничивает число аргументов запроса 32 767 (int16 в протоколе
    Postgres). У `products` больше сорока колонок, поэтому батч из 1000 строк
    даёт свыше 42 000 аргументов и падает. Тесты с батчем по 2 и по 200 строк
    этого не ловили — ошибка вылезла только на настоящем прогоне.
    """

    async def test_батч_боевого_размера_записывается(
        self, load_settings, migrated_database, _clean_engine
    ):
        products = [
            make_product(f"8{index:012d}", rev=1, name=f"Товар {index}") for index in range(1000)
        ]

        async with get_session(load_settings.db) as session:
            await ProductRepository(session).upsert_batch(
                products,
                languages=load_settings.ingest.languages,
                min_ingredients_length=load_settings.ingest.min_ingredients_length,
                dump_version="тест",
            )

        assert await count_rows(load_settings, "products") == 1000

    async def test_размер_куска_считается_от_числа_колонок(self):
        """Константа-подбор сломалась бы при добавлении колонки."""
        from nutri_radar.db.models.product import Product
        from nutri_radar.db.repositories.product import _MAX_QUERY_ARGS

        columns = len(Product.__table__.columns)
        chunk = _MAX_QUERY_ARGS // columns

        assert chunk * columns <= _MAX_QUERY_ARGS
        assert chunk >= 1


class TestBatchPerformance:
    """Ловит переход на построчные запросы вместо батчевых.

    Не бенчмарк: порог с большим запасом. Смысл в том, что регрессия
    производительности иначе всплыла бы только на сотне тысяч строк.
    """

    async def test_заливка_тысячи_строк_укладывается_в_порог(
        self, load_settings, migrated_database, _clean_engine
    ):
        products = [
            make_product(f"9{index:012d}", rev=1, name=f"Товар {index}") for index in range(1000)
        ]

        started = time.perf_counter()
        async with get_session(load_settings.db) as session:
            repository = ProductRepository(session)
            for start in range(0, len(products), 200):
                await repository.upsert_batch(
                    products[start : start + 200],
                    languages=load_settings.ingest.languages,
                    min_ingredients_length=load_settings.ingest.min_ingredients_length,
                    dump_version="тест",
                )
        elapsed = time.perf_counter() - started

        assert await count_rows(load_settings, "products") == 1000
        # Батчами это доли секунды; построчно 1000 строк заняли бы десятки секунд.
        assert elapsed < 15.0, f"заливка 1000 строк заняла {elapsed:.1f} с"
