"""Интеграционные тесты хранения извлечений. Требуют поднятого Postgres.

Здесь проверяется пункт DoD майлстоуна, который моками проверить нельзя:
**повторный запуск пропускает уже обработанное и не создаёт дублей**. Нужны
настоящий `ON CONFLICT` и настоящее ограничение уникальности по
`(code, model_name, prompt_version)`.

Модель по-прежнему подделывается `FakeLLM` — в сеть не ходим и здесь.

Запуск: `uv run pytest -m integration`
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from nutri_radar.config import ExtractSettings, Settings
from nutri_radar.db.models.product import Product, ProductRaw
from nutri_radar.db.repositories.extraction import ExtractionRepository, ExtractionRow
from nutri_radar.db.session import dispose_engine, get_session
from nutri_radar.extract.corpus import CorpusItem
from nutri_radar.extract.runner import run_extraction
from nutri_radar.llm.adapters.fake import FakeLLM

pytestmark = pytest.mark.integration

MODEL = "тестовая-модель:3b"
RESPONSE = {
    "ingredients": [
        {"canonical_name": "sugar", "kind": "sugar", "e_number": None},
        {"canonical_name": "glucose syrup", "kind": "sugar", "e_number": None},
        {"canonical_name": "lecithin", "kind": "additive", "e_number": "E322"},
    ],
    "allergens": ["milk"],
    "unreadable": False,
    "model_confidence": 0.9,
}


@pytest.fixture
async def _clean_engine():
    """Движок кэшируется в модуле — сбрасываем между тестами."""
    await dispose_engine()
    yield
    await dispose_engine()


@pytest.fixture
def extract_settings(integration_settings: Settings) -> Settings:
    return integration_settings.model_copy(
        update={"extract": ExtractSettings(prompt_version="v1", batch_size=2, retry_backoff_s=0.0)}
    )


def _items(count: int) -> list[CorpusItem]:
    return [
        CorpusItem(
            code=f"{index:013d}",
            ingredients_text="Sugar, Milk, Glucose Syrup",
            lang="en",
            nutriscore_grade="d",
            unknown_ingredients_n=2,
            stratum="unknown",
        )
        for index in range(count)
    ]


async def _seed_products(settings: Settings, items: list[CorpusItem]) -> None:
    """Продукты нужны из-за внешнего ключа `product_extraction.code`."""
    async with get_session(settings.db) as session:
        for item in items:
            session.add(ProductRaw(code=item.code, payload={}, source="test"))
        await session.flush()
        for item in items:
            session.add(
                Product(
                    code=item.code,
                    ingredients_text=item.ingredients_text,
                    ingredients_text_lang=item.lang,
                    nutriscore_grade=item.nutriscore_grade,
                    unknown_ingredients_n=item.unknown_ingredients_n,
                )
            )


async def _count(settings: Settings, table: str) -> int:
    async with get_session(settings.db) as session:
        return (await session.execute(text(f"SELECT count(*) FROM {table}"))).scalar_one()


def _row(code: str, *, prompt_version: str = "v1", sugar_forms: int = 2) -> ExtractionRow:
    return ExtractionRow(
        code=code,
        source_lang="en",
        ingredients=RESPONSE["ingredients"],
        distinct_sugar_forms=sugar_forms,
        e_additives_count=1,
        ingredients_count=3,
        allergens=["milk"],
        model_name=MODEL,
        prompt_version=prompt_version,
        input_tokens=100,
        output_tokens=50,
        latency_s=1.5,
    )


class TestРепозиторийИзвлечений:
    async def test_повторная_запись_не_создаёт_дублей(
        self, migrated_database, extract_settings: Settings, _clean_engine
    ):
        items = _items(3)
        await _seed_products(extract_settings, items)

        rows = [_row(item.code) for item in items]
        async with get_session(extract_settings.db) as session:
            await ExtractionRepository(session).upsert_batch(rows)
        async with get_session(extract_settings.db) as session:
            await ExtractionRepository(session).upsert_batch(rows)

        assert await _count(extract_settings, "product_extraction") == 3

    async def test_повторная_запись_обновляет_значения(
        self, migrated_database, extract_settings: Settings, _clean_engine
    ):
        items = _items(1)
        await _seed_products(extract_settings, items)
        code = items[0].code

        async with get_session(extract_settings.db) as session:
            await ExtractionRepository(session).upsert_batch([_row(code, sugar_forms=2)])
        async with get_session(extract_settings.db) as session:
            await ExtractionRepository(session).upsert_batch([_row(code, sugar_forms=5)])

        async with get_session(extract_settings.db) as session:
            value = (
                await session.execute(
                    text("SELECT distinct_sugar_forms FROM product_extraction WHERE code = :code"),
                    {"code": code},
                )
            ).scalar_one()

        assert value == 5

    async def test_другая_версия_промпта_даёт_отдельную_строку(
        self, migrated_database, extract_settings: Settings, _clean_engine
    ):
        """Сердце версионирования: без этого сравнение версий в M3 невозможно."""
        items = _items(2)
        await _seed_products(extract_settings, items)

        async with get_session(extract_settings.db) as session:
            repository = ExtractionRepository(session)
            await repository.upsert_batch([_row(item.code, prompt_version="v1") for item in items])
            await repository.upsert_batch([_row(item.code, prompt_version="v3") for item in items])

        assert await _count(extract_settings, "product_extraction") == 4

    async def test_extracted_codes_различает_версии(
        self, migrated_database, extract_settings: Settings, _clean_engine
    ):
        items = _items(3)
        await _seed_products(extract_settings, items)
        codes = [item.code for item in items]

        async with get_session(extract_settings.db) as session:
            await ExtractionRepository(session).upsert_batch([_row(codes[0]), _row(codes[1])])

        async with get_session(extract_settings.db) as session:
            repository = ExtractionRepository(session)
            done_v1 = await repository.extracted_codes(codes, model_name=MODEL, prompt_version="v1")
            done_v3 = await repository.extracted_codes(codes, model_name=MODEL, prompt_version="v3")
            done_other_model = await repository.extracted_codes(
                codes, model_name="другая-модель", prompt_version="v1"
            )

        assert done_v1 == {codes[0], codes[1]}
        assert done_v3 == set()
        assert done_other_model == set()


class TestВозобновляемостьПрогона:
    async def test_повторный_запуск_пропускает_обработанное(
        self, migrated_database, extract_settings: Settings, _clean_engine
    ):
        """Пункт DoD целиком: пропуск обработанного и отсутствие дублей."""
        items = _items(5)
        await _seed_products(extract_settings, items)

        first_llm = FakeLLM(default_response=RESPONSE, model_name=MODEL)
        first = await run_extraction(first_llm, items, extract_settings)

        assert first.processed == 5
        assert first.already_done == 0
        assert first_llm.call_count == 5
        assert await _count(extract_settings, "product_extraction") == 5

        second_llm = FakeLLM(default_response=RESPONSE, model_name=MODEL)
        second = await run_extraction(second_llm, items, extract_settings)

        assert second.already_done == 5
        assert second.processed == 0
        # Главное: модель не вызывалась ни разу — часы GPU не потрачены заново.
        assert second_llm.call_count == 0
        assert await _count(extract_settings, "product_extraction") == 5

    async def test_прерванный_прогон_дообрабатывает_остаток(
        self, migrated_database, extract_settings: Settings, _clean_engine
    ):
        items = _items(6)
        await _seed_products(extract_settings, items)

        # Первый прогон обработал только часть корпуса — как после Ctrl+C.
        llm = FakeLLM(default_response=RESPONSE, model_name=MODEL)
        await run_extraction(llm, items[:2], extract_settings)

        resumed_llm = FakeLLM(default_response=RESPONSE, model_name=MODEL)
        resumed = await run_extraction(resumed_llm, items, extract_settings)

        assert resumed.already_done == 2
        assert resumed.processed == 4
        assert resumed_llm.call_count == 4
        assert await _count(extract_settings, "product_extraction") == 6

    async def test_другая_версия_промпта_обрабатывает_корпус_заново(
        self, migrated_database, extract_settings: Settings, _clean_engine
    ):
        items = _items(3)
        await _seed_products(extract_settings, items)

        await run_extraction(
            FakeLLM(default_response=RESPONSE, model_name=MODEL),
            items,
            extract_settings,
            prompt_version="v1",
        )
        second = await run_extraction(
            FakeLLM(default_response=RESPONSE, model_name=MODEL),
            items,
            extract_settings,
            prompt_version="v3",
        )

        assert second.already_done == 0
        assert second.processed == 3
        assert await _count(extract_settings, "product_extraction") == 6

    async def test_прогон_попадает_в_журнал(
        self, migrated_database, extract_settings: Settings, _clean_engine
    ):
        """Время и токены берутся из БД, а не восстанавливаются по логам."""
        items = _items(2)
        await _seed_products(extract_settings, items)

        await run_extraction(
            FakeLLM(default_response=RESPONSE, model_name=MODEL), items, extract_settings
        )

        async with get_session(extract_settings.db) as session:
            row = (
                await session.execute(
                    text(
                        "SELECT status, items_processed, model_name, prompt_version, "
                        "input_tokens, output_tokens FROM runs WHERE stage = 'extract'"
                    )
                )
            ).one()

        status, processed, model_name, prompt_version, input_tokens, output_tokens = row
        assert status == "completed"
        assert processed == 2
        assert model_name == MODEL
        assert prompt_version == "v1"
        assert input_tokens > 0
        assert output_tokens > 0
