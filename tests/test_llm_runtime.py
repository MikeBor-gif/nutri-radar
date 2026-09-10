"""Тесты очереди к моделям.

Проверяемое свойство одно и оно про железо: **две модели не оказываются
загружены одновременно**. На 6 ГБ VRAM `qwen2.5:3b` и `bge-m3` не помещаются
вместе, и в сервисе с параллельными запросами это перестаёт быть вопросом
дисциплины — порядок держится кодом или не держится вовсе.

Проверяется наблюдаемым протоколом, а не внутренним состоянием: если
в записи вызовов между входом в одну модель и выходом из неё встретился
вход в другую, инвариант нарушен. Спрашивать очередь, всё ли у неё хорошо,
значило бы мерить её самооценку.

Сети здесь нет вовсе: очередь не знает про HTTP, она знает про имена моделей.
"""

from __future__ import annotations

import asyncio

import pytest

from nutri_radar.config import Settings
from nutri_radar.llm.runtime import ModelRuntime, get_runtime


@pytest.fixture
def runtime(settings: Settings) -> ModelRuntime:
    return ModelRuntime(settings.ollama)


def _interleaved(log: list[str]) -> bool:
    """Нашлось ли в протоколе перекрытие двух разных моделей."""
    active: set[str] = set()
    for event in log:
        kind, model = event.split(":", 1)
        if kind == "in":
            if active and model not in active:
                return True
            active.add(model)
        else:
            active.discard(model)
    return False


class TestВзаимноеИсключение:
    async def test_две_модели_не_работают_одновременно(self, runtime: ModelRuntime) -> None:
        log: list[str] = []

        async def call(model: str) -> None:
            async with runtime.hold(model):
                log.append(f"in:{model}")
                # Уступаем управление внутри критической секции: без этого
                # тест прошёл бы даже на сломанной очереди, потому что
                # корутина без await не даёт другим шанса вклиниться.
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                log.append(f"out:{model}")

        await asyncio.gather(*(call(model) for model in ("A", "B", "A", "B")))

        assert not _interleaved(log), f"Модели перекрылись: {log}"
        assert len(log) == 8

    async def test_вызовы_одной_модели_идут_в_пределах_параллелизма(
        self, settings: Settings
    ) -> None:
        runtime = ModelRuntime(settings.ollama.model_copy(update={"max_concurrency": 2}))
        peak = 0

        async def call() -> None:
            nonlocal peak
            async with runtime.hold("A"):
                peak = max(peak, runtime.active_calls)
                await asyncio.sleep(0)

        await asyncio.gather(*(call() for _ in range(4)))

        # Два слота настроены — два и должны использоваться. Если бы очередь
        # всё сериализовала, параллелизм из конфига был бы декорацией.
        assert peak == 2

    async def test_смена_модели_ждёт_окончания_всех_вызовов(self, settings: Settings) -> None:
        runtime = ModelRuntime(settings.ollama.model_copy(update={"max_concurrency": 2}))
        log: list[str] = []
        started = asyncio.Event()

        async def long_call() -> None:
            async with runtime.hold("A"):
                log.append("in:A")
                started.set()
                await asyncio.sleep(0.05)
                log.append("out:A")

        async def other_model() -> None:
            await started.wait()
            async with runtime.hold("B"):
                log.append("in:B")
                log.append("out:B")

        await asyncio.gather(long_call(), other_model())

        assert log == ["in:A", "out:A", "in:B", "out:B"]


class TestВыгрузка:
    async def test_предыдущая_модель_выгружается_при_переключении(
        self, runtime: ModelRuntime
    ) -> None:
        unloaded: list[str] = []

        async def unload_a() -> None:
            unloaded.append("A")

        async with runtime.hold("A", unload=unload_a):
            pass
        assert unloaded == [], (
            "Выгрузка на выходе не нужна: следующий запрос придёт к той же модели"
        )

        async with runtime.hold("B"):
            pass
        assert unloaded == ["A"], "Переключение обязано освободить VRAM"

    async def test_модель_без_выгрузки_не_ломает_переключение(self, runtime: ModelRuntime) -> None:
        # У модели генерации метода выгрузки нет — она уходит по keep_alive.
        # Это штатный случай, а не повод падать.
        async with runtime.hold("generator"):
            pass
        async with runtime.hold("embedder"):
            pass
        assert runtime.current_model == "embedder"

    async def test_выгрузка_вызывается_один_раз_на_переключение(
        self, runtime: ModelRuntime
    ) -> None:
        calls: list[str] = []

        async def unload() -> None:
            calls.append("unload")

        for _ in range(3):
            async with runtime.hold("A", unload=unload):
                pass
        assert calls == [], "Повторные обращения к той же модели не выгружают её"

        async with runtime.hold("B"):
            pass
        assert calls == ["unload"]


class TestЭкземплярНаЦикл:
    def test_разные_циклы_получают_разные_очереди(self, settings: Settings) -> None:
        """Примитивы asyncio привязываются к циклу, и общий экземпляр упал бы.

        CLI вызывает `asyncio.run` на каждую команду, pytest — на каждый тест.
        Очередь, пережившая цикл, во втором вызове дала бы `RuntimeError`
        вместо работы.
        """

        async def take() -> ModelRuntime:
            return get_runtime(settings)

        # Сравниваются сами объекты, а не их `id()`. Времена жизни двух
        # очередей не пересекались бы, а CPython переиспользует адреса
        # освобождённых объектов — тест на `id()` проходил бы или падал
        # в зависимости от истории аллокаций в прогоне. Ссылки на оба
        # объекта живы до конца проверки, поэтому совпасть они не могут.
        first = asyncio.run(take())
        second = asyncio.run(take())
        assert first is not second

    async def test_внутри_одного_цикла_очередь_одна(self, settings: Settings) -> None:
        assert get_runtime(settings) is get_runtime(settings)

    def test_вне_цикла_очередь_не_выдаётся(self, settings: Settings) -> None:
        with pytest.raises(RuntimeError):
            get_runtime(settings)


class TestДиагностика:
    async def test_счётчик_обращений_обнуляется(self, runtime: ModelRuntime) -> None:
        async with runtime.hold("A"):
            assert runtime.active_calls == 1
        assert runtime.active_calls == 0
        assert runtime.current_model == "A"

    async def test_ошибка_внутри_секции_освобождает_очередь(self, runtime: ModelRuntime) -> None:
        """Иначе первая же ошибка модели заклинила бы сервис навсегда."""
        with pytest.raises(ValueError):
            async with runtime.hold("A"):
                raise ValueError("модель ответила ерундой")

        assert runtime.active_calls == 0
        async with runtime.hold("B"):
            pass
