"""Тесты раннера извлечения. Сети нет — модель подделывается `FakeLLM`.

Проверяется поведение, ради которого раннер существует: битая запись не роняет
прогон, параллелизм не превышает настройку (на 6 ГБ VRAM это не абстракция),
ретрай идёт только на недоступности модели.

Все прогоны здесь идут с `dry_run=True`: так раннер не обращается к БД, и тест
остаётся герметичным. Запись в базу и отсутствие дублей проверяются отдельно,
на настоящем Postgres — в `test_extract_repository.py`.
"""

from __future__ import annotations

import asyncio

import pytest

from nutri_radar.config import ExtractSettings, OllamaSettings, Settings
from nutri_radar.errors import LLMUnavailableError
from nutri_radar.extract.corpus import CorpusItem
from nutri_radar.extract.preprocess import PreprocessStats
from nutri_radar.extract.prompts import load_prompt
from nutri_radar.extract.runner import _extract_one, run_extraction
from nutri_radar.extract.schemas import ExtractionResult
from nutri_radar.llm.adapters.fake import FakeLLM
from nutri_radar.tracing import NoOpTracer

GOOD_RESPONSE = {
    "ingredients": [
        {"canonical_name": "sugar", "kind": "sugar", "e_number": None},
        {"canonical_name": "glucose-fructose syrup", "kind": "sugar", "e_number": None},
        {"canonical_name": "lecithin", "kind": "additive", "e_number": "E322"},
    ],
    "allergens": ["milk"],
    "unreadable": False,
    "model_confidence": 0.8,
}

# Модель ответила, но не по схеме: нет обязательного kind.
BROKEN_RESPONSE = {"ingredients": [{"canonical_name": "sugar"}]}

# Состав, на котором ответ не помещается в лимит вывода: длинные списки орехов
# и сухофруктов дают десятки ингредиентов.
LONG_TEXT = "Almonds, Banana, Blueberries, Cashews, Cranberries, Dates, Figs"


def _items(count: int, *, text: str = "Sugar, Milk, Glucose-Fructose Syrup") -> list[CorpusItem]:
    return [
        CorpusItem(
            code=f"{index:013d}",
            ingredients_text=text,
            lang="en",
            nutriscore_grade="d",
            unknown_ingredients_n=1,
            stratum="unknown",
        )
        for index in range(count)
    ]


@pytest.fixture
def extract_settings(settings: Settings) -> Settings:
    """Настройки прогона: без пауз между ретраями, чтобы тест не спал."""
    return settings.model_copy(
        update={
            "extract": ExtractSettings(
                prompt_version="v1",
                batch_size=3,
                max_retries=3,
                retry_backoff_s=0.0,
                max_consecutive_failures=4,
            )
        }
    )


def _with_concurrency(settings: Settings, value: int) -> Settings:
    ollama = OllamaSettings(**{**settings.ollama.model_dump(), "max_concurrency": value})
    return settings.model_copy(update={"ollama": ollama})


class TestУспешныйПрогон:
    async def test_все_продукты_обработаны(self, extract_settings: Settings):
        llm = FakeLLM(default_response=GOOD_RESPONSE)

        result = await run_extraction(llm, _items(7), extract_settings, dry_run=True)

        assert result.total == 7
        assert result.processed == 7
        assert result.invalid == 0
        assert llm.call_count == 7

    async def test_считаются_токены_и_формы_сахара(self, extract_settings: Settings):
        llm = FakeLLM(default_response=GOOD_RESPONSE)

        result = await run_extraction(llm, _items(3), extract_settings, dry_run=True)

        assert result.usage.output_tokens > 0
        assert result.usage.input_tokens > 0
        # Две разные формы сахара в каждом ответе.
        assert result.mean_sugar_forms == 2

    async def test_версия_промпта_и_модель_попадают_в_итог(self, extract_settings: Settings):
        llm = FakeLLM(default_response=GOOD_RESPONSE, model_name="проверочная-модель")

        result = await run_extraction(
            llm, _items(1), extract_settings, prompt_version="v2", dry_run=True
        )

        assert result.prompt_version == "v2"
        assert result.model_name == "проверочная-модель"

    async def test_батчи_режут_корпус_но_не_теряют_продукты(self, extract_settings: Settings):
        llm = FakeLLM(default_response=GOOD_RESPONSE)

        result = await run_extraction(llm, _items(7), extract_settings, dry_run=True)

        # batch_size=3 → 3 + 3 + 1
        assert result.batches == 3
        assert result.processed == 7


class TestБитаяЗаписьНеРоняетПрогон:
    async def test_невалидный_ответ_считается_и_пропускается(self, extract_settings: Settings):
        llm = FakeLLM(default_response=BROKEN_RESPONSE)

        result = await run_extraction(llm, _items(4), extract_settings, dry_run=True)

        assert result.invalid == 4
        assert result.processed == 0
        assert result.invalid_share == 1.0

    async def test_плохие_и_хорошие_ответы_в_одном_прогоне(self, extract_settings: Settings):
        # Ответ выбирается по подстроке промпта: у плохого продукта свой состав.
        llm = FakeLLM(
            responses={"ПЛОХОЙ": BROKEN_RESPONSE},
            default_response=GOOD_RESPONSE,
        )
        items = _items(3) + _items(1, text="ПЛОХОЙ СОСТАВ")

        result = await run_extraction(llm, items, extract_settings, dry_run=True)

        assert result.processed == 3
        assert result.invalid == 1

    async def test_пустой_состав_помечается_нечитаемым_и_не_идёт_в_модель(
        self, extract_settings: Settings
    ):
        llm = FakeLLM(default_response=GOOD_RESPONSE)
        items = _items(2) + _items(1, text="   ")

        result = await run_extraction(llm, items, extract_settings, dry_run=True)

        # Продукт учтён как обработанный (иначе он вечно оставался бы
        # необработанным и перезапуск упирался бы в него снова).
        assert result.processed == 3
        assert result.unreadable == 1
        # Но в модель он не отправлялся.
        assert llm.call_count == 2

    async def test_слишком_длинный_состав_в_модель_не_идёт(self, extract_settings: Settings):
        llm = FakeLLM(default_response=GOOD_RESPONSE)
        huge = "sugar, " * extract_settings.ollama.num_ctx
        items = _items(1, text=huge)

        result = await run_extraction(llm, items, extract_settings, dry_run=True)

        assert result.unreadable == 1
        assert llm.call_count == 0


class TestРетраи:
    async def test_недоступность_модели_ретраится(self, extract_settings: Settings):
        """Два отказа подряд, третья попытка успешна — продукт обработан."""
        llm = FakeLLM(default_response=GOOD_RESPONSE, fail_times=2)

        result = await run_extraction(llm, _items(1), extract_settings, dry_run=True)

        assert result.processed == 1
        assert result.unavailable == 0
        assert llm.call_count == 3

    async def test_невалидный_ответ_не_ретраится(self, extract_settings: Settings):
        """При temperature=0 повтор дал бы тот же ответ — это не отказ сети."""
        llm = FakeLLM(default_response=BROKEN_RESPONSE)

        await run_extraction(llm, _items(1), extract_settings, dry_run=True)

        assert llm.call_count == 1

    async def test_попытки_исчерпаны_продукт_помечен_отказом(self, extract_settings: Settings):
        llm = FakeLLM(default_response=GOOD_RESPONSE, fail_times=99)

        result = await run_extraction(llm, _items(1), extract_settings, dry_run=True)

        assert result.unavailable == 1
        assert result.processed == 0
        assert llm.call_count == extract_settings.extract.max_retries

    async def test_отказы_подряд_останавливают_прогон(self, extract_settings: Settings):
        """Если Ollama упала, молотить оставшиеся тысячи продуктов бессмысленно."""
        llm = FakeLLM(default_response=GOOD_RESPONSE, fail_times=999)

        with pytest.raises(LLMUnavailableError, match="подряд"):
            await run_extraction(llm, _items(30), extract_settings, dry_run=True)

        # Остановились на границе батча, а не прошли весь корпус.
        assert llm.call_count < 30 * extract_settings.extract.max_retries


class TestОборванныйОтвет:
    """Длинный состав упирается в лимит вывода, и JSON обрывается.

    Сценарий не выдуман: на нём встал полный прогон M2 (`reports/run_v3.err.log`,
    продукт 0718604977580). Обрыв детерминирован — повтор тратит ещё минуты
    генерации и приходит к той же обрезке.
    """

    async def test_не_ретраится(self, extract_settings: Settings):
        llm = FakeLLM(default_response=GOOD_RESPONSE, truncate_marker=LONG_TEXT)

        await run_extraction(llm, _items(1, text=LONG_TEXT), extract_settings, dry_run=True)

        assert llm.call_count == 1

    async def test_считается_невалидным_а_не_отказом(self, extract_settings: Settings):
        """Модель работает — она просто не уместила ответ. Это не недоступность."""
        llm = FakeLLM(default_response=GOOD_RESPONSE, truncate_marker=LONG_TEXT)

        result = await run_extraction(
            llm, _items(1, text=LONG_TEXT), extract_settings, dry_run=True
        )

        assert result.invalid == 1
        assert result.unavailable == 0
        assert result.processed == 0

    async def test_пачка_обрывов_подряд_не_роняет_прогон(self, extract_settings: Settings):
        """Регрессия: обрыв инкрементил счётчик отказов и останавливал прогон.

        Длинные составы идут в корпусе кучно, и порога
        `max_consecutive_failures` хватало, чтобы уронить шестичасовой прогон
        на данных, которые всего лишь не помещаются в ответ.
        """
        count = extract_settings.extract.max_consecutive_failures * 3
        llm = FakeLLM(default_response=GOOD_RESPONSE, truncate_marker=LONG_TEXT)

        result = await run_extraction(
            llm, _items(count, text=LONG_TEXT), extract_settings, dry_run=True
        )

        assert result.invalid == count
        assert llm.call_count == count

    async def test_обрыв_не_мешает_соседям_по_батчу(self, extract_settings: Settings):
        llm = FakeLLM(
            responses={LONG_TEXT: GOOD_RESPONSE},
            default_response=GOOD_RESPONSE,
            truncate_marker=LONG_TEXT,
        )
        items = _items(2) + _items(1, text=LONG_TEXT)

        result = await run_extraction(llm, items, extract_settings, dry_run=True)

        assert result.processed == 2
        assert result.invalid == 1

    async def test_потраченные_токены_учтены(self, extract_settings: Settings):
        """Продукт в выборку не попал, но генерация до лимита реально оплачена."""
        llm = FakeLLM(default_response=GOOD_RESPONSE, truncate_marker=LONG_TEXT)

        result = await run_extraction(
            llm, _items(1, text=LONG_TEXT), extract_settings, dry_run=True
        )

        assert result.usage.output_tokens > 0

    async def test_пишется_строка_чтобы_перезапуск_не_упирался_снова(
        self, extract_settings: Settings
    ):
        """Иначе продукт вечно «необработан», и каждый рестарт жжёт на нём минуты."""
        llm = FakeLLM(default_response=GOOD_RESPONSE, truncate_marker=LONG_TEXT)

        outcome = await _extract_one(
            llm,
            _items(1, text=LONG_TEXT)[0],
            prompt=load_prompt("v1"),
            schema=ExtractionResult.model_json_schema(),
            settings=extract_settings,
            semaphore=asyncio.Semaphore(1),
            tracer=NoOpTracer(),
            stats=PreprocessStats(),
        )

        assert outcome.status == "invalid"
        assert outcome.row is not None
        assert outcome.row.unreadable is True


class TestПараллелизм:
    async def test_не_превышает_настройку(self, extract_settings: Settings):
        """На 6 ГБ VRAM неограниченный веер запросов кладёт GPU."""
        settings = _with_concurrency(extract_settings, 3)
        llm = FakeLLM(default_response=GOOD_RESPONSE, latency_s=0.01)

        await run_extraction(llm, _items(12), settings, dry_run=True)

        assert llm.max_in_flight <= 3

    async def test_единица_означает_строго_последовательно(self, extract_settings: Settings):
        settings = _with_concurrency(extract_settings, 1)
        llm = FakeLLM(default_response=GOOD_RESPONSE, latency_s=0.01)

        await run_extraction(llm, _items(6), settings, dry_run=True)

        assert llm.max_in_flight == 1

    async def test_параллелизм_действительно_используется(self, extract_settings: Settings):
        """Иначе тест выше проходил бы и при полностью последовательном коде."""
        settings = _with_concurrency(extract_settings, 3)
        # Батч должен вмещать все продукты, иначе граница батча ограничит веер.
        settings = settings.model_copy(
            update={
                "extract": ExtractSettings(
                    **{**settings.extract.model_dump(), "batch_size": 6},
                )
            }
        )
        llm = FakeLLM(default_response=GOOD_RESPONSE, latency_s=0.02)

        await run_extraction(llm, _items(6), settings, dry_run=True)

        assert llm.max_in_flight > 1


class TestПустойВход:
    async def test_пустой_корпус_не_роняет_и_ничего_не_вызывает(self, extract_settings: Settings):
        llm = FakeLLM(default_response=GOOD_RESPONSE)

        result = await run_extraction(llm, [], extract_settings, dry_run=True)

        assert result.total == 0
        assert result.processed == 0
        assert llm.call_count == 0
