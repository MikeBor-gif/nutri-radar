"""Очередь к локальным моделям: одна на процесс.

**Зачем это нужно.** На RTX 3060 Laptop с 6 ГБ VRAM `qwen2.5:3b` и `bge-m3`
одновременно не помещаются. В CLI это решалось руками: прогон эмбеддингов
заканчивался явным `await model.unload()`, и только потом создавалась модель
генерации. Команда однопоточная, порядок виден в коде — этого хватало.

В долгоживущем сервисе руками так сделать нельзя. Два параллельных запроса
к API или два сообщения боту начнут выгружать модели друг у друга: первый
запрос загрузил `bge-m3`, второй тут же требует `qwen2.5:3b`, Ollama
вытесняет первую модель, первый запрос получает свой вектор и просит
генерацию — и цикл повторяется. Каждый шаг стоит секунды загрузки весов
с диска, и латентность растёт на порядок при нулевой полезной нагрузке.
Дефолты Ollama от этого не спасают: она честно выполнит всё, что попросили.

**Поэтому доступ к моделям сериализован структурно.** Не «мы же не будем
слать два запроса», а инвариант: в каждый момент времени загружена ровно
одна модель, и переключение на другую происходит только тогда, когда
обращений к текущей не осталось. Переключение сопровождается явной
выгрузкой предыдущей — иначе следующая загрузка будет ждать истечения
`keep_alive`.

Параллелизм внутри одной модели берётся из `OLLAMA__MAX_CONCURRENCY`
(по умолчанию 1 — по измерению из ADR-017, а не из осторожности).

**Почему на каждый событийный цикл своя очередь.** Примитивы `asyncio`
привязываются к циклу при первом ожидании, и один экземпляр, переживший
`asyncio.run()`, во втором вызове упал бы с `RuntimeError`. CLI запускает
`asyncio.run()` на каждую команду, тесты — на каждый тест. Очередь живёт
ровно столько, сколько живёт цикл, и это ограничение реализации, а не
свойство железа: у процесса всё равно одна видеокарта, а два событийных
цикла в одном процессе одновременно не работают.
"""

from __future__ import annotations

import asyncio
import logging
import time
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from nutri_radar.config import OllamaSettings, Settings, get_settings
from nutri_radar.logging import safe_extra

logger = logging.getLogger(__name__)

# Корутина, выгружающая модель из памяти. Знание о том, *как* выгружать,
# остаётся в адаптере: очередь только решает, *когда* это делать.
Unloader = Callable[[], Awaitable[None]]


class ModelRuntime:
    """Очередь к моделям Ollama на одном событийном цикле.

    Инвариант: одновременно занята одна модель. Обращения к ней идут
    параллельно в пределах `max_concurrency`, обращения к другой ждут,
    пока текущая освободится и будет выгружена.

    При `max_concurrency > 1` непрерывный поток обращений к одной модели
    теоретически может задержать переключение на другую. На значении по
    умолчанию (1) этого не происходит: счётчик обращается в ноль после
    каждого вызова, и `asyncio.Condition` будит ожидающих по очереди.
    """

    def __init__(self, settings: OllamaSettings) -> None:
        self._settings = settings
        self._condition = asyncio.Condition()
        self._current: str | None = None
        self._active = 0
        self._unloaders: dict[str, Unloader] = {}
        # Счётчики переключений. Нужны не для диагностики, а для замера:
        # цена смены модели на 6 ГБ VRAM — это число, которое идёт в README
        # как ограничение железа, и посчитать его можно только здесь.
        self._loads = 0
        self._switches = 0
        logger.debug(
            "Очередь к моделям создана",
            extra=safe_extra(
                max_concurrency=settings.max_concurrency,
                warn_after_s=settings.lock_warn_after_s,
            ),
        )

    @property
    def current_model(self) -> str | None:
        """Какая модель занята прямо сейчас. Для диагностики и тестов."""
        return self._current

    @property
    def active_calls(self) -> int:
        """Сколько обращений выполняется прямо сейчас."""
        return self._active

    @property
    def loads(self) -> int:
        """Сколько раз очередь загружала модель, считая самую первую."""
        return self._loads

    @property
    def switches(self) -> int:
        """Сколько раз одна модель сменила другую.

        Первая загрузка сюда не входит: платить за неё придётся в любом
        случае, а замер спрашивает про **цену чередования**. Смешать их
        значило бы завысить цену ровно на одну загрузку — и тем сильнее,
        чем короче прогон.
        """
        return self._switches

    @asynccontextmanager
    async def hold(self, model: str, *, unload: Unloader | None = None) -> AsyncIterator[None]:
        """Занять очередь под обращение к `model`.

        Args:
            model: имя модели Ollama. Разные имена — разные веса в VRAM.
            unload: как выгрузить эту модель, когда очередь переключится
                на другую. Не передан — модель останется в памяти до
                истечения `keep_alive`. Для моделей генерации это штатно:
                отдельного метода выгрузки у адаптера нет.
        """
        await self._acquire(model, unload)
        try:
            yield
        finally:
            await self._release()

    async def _acquire(self, model: str, unload: Unloader | None) -> None:
        started = time.perf_counter()
        async with self._condition:
            # Пустить можно в двух случаях: это та же модель и есть
            # свободный слот, либо обращений нет вовсе и модель можно
            # переключить. Всё остальное ждёт.
            await self._condition.wait_for(
                lambda: (
                    (self._current == model and self._active < self._settings.max_concurrency)
                    or self._active == 0
                )
            )
            waited = time.perf_counter() - started

            if unload is not None:
                self._unloaders[model] = unload

            if self._current != model:
                await self._switch_to(model)

            self._active += 1

        if waited >= self._settings.lock_warn_after_s:
            # Не отказ: длинная генерация законно держит GPU десятки секунд.
            # Но минутные ожидания — признак, что запросов больше, чем
            # переваривает железо, и узнать об этом надо здесь, а не из
            # жалобы «сервис тормозит».
            logger.warning(
                "Долгое ожидание очереди к модели",
                extra=safe_extra(model=model, waited_s=round(waited, 1)),
            )
        else:
            logger.debug(
                "Очередь занята",
                extra=safe_extra(model=model, waited_s=round(waited, 3), active=self._active),
            )

    async def _switch_to(self, model: str) -> None:
        """Сменить загруженную модель. Вызывается под удержанным условием."""
        previous = self._current
        if previous is not None:
            await self._unload(previous)
            self._switches += 1
        self._loads += 1
        self._current = model
        logger.info(
            "Активная модель переключена",
            extra=safe_extra(
                model=model,
                previous=previous or "—",
                switches=self._switches,
                loads=self._loads,
            ),
        )

    async def _unload(self, model: str) -> None:
        """Выгрузить модель, если адаптер объяснил, как это делается.

        **Контракт: выгрузка не бросает исключений.** Не удалось выгрузить —
        адаптер логирует это сам и возвращает управление, потому что отказ
        не критичен: модель уйдёт по `keep_alive`. Ловить здесь `Exception`
        значило бы прятать настоящую ошибку в адаптере под видом заботы
        о пользователе.
        """
        unload = self._unloaders.get(model)
        if unload is None:
            logger.debug(
                "Выгрузка не задана, модель уйдёт по keep_alive",
                extra=safe_extra(model=model),
            )
            return
        await unload()
        logger.info("Модель выгружена", extra=safe_extra(model=model))

    async def _release(self) -> None:
        async with self._condition:
            self._active -= 1
            logger.debug(
                "Очередь освобождена",
                extra=safe_extra(model=self._current or "—", active=self._active),
            )
            # Модель НЕ выгружается на выходе: следующий запрос, скорее
            # всего, придёт к ней же, и выгрузка ради чистоты означала бы
            # перезагрузку весов на каждый запрос. Выгрузка происходит
            # ровно при переключении.
            self._condition.notify_all()


# Экземпляр на событийный цикл. Слабые ключи: завершившийся цикл не
# удерживает очередь в памяти, а вместе с ней и ссылки на адаптеры.
_runtimes: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, ModelRuntime] = (
    weakref.WeakKeyDictionary()
)


def get_runtime(settings: Settings | None = None) -> ModelRuntime:
    """Получить очередь текущего событийного цикла.

    Raises:
        RuntimeError: вызвано вне корутины. Очередь имеет смысл только
            внутри работающего цикла.
    """
    settings = settings or get_settings()
    loop = asyncio.get_running_loop()
    runtime = _runtimes.get(loop)
    if runtime is None:
        runtime = ModelRuntime(settings.ollama)
        _runtimes[loop] = runtime
    return runtime
