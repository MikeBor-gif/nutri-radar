"""Сборка конвейера поиска: одна на весь проект.

Цепочка «текст → вектор → поиск → ответ» одинакова для CLI, HTTP-API,
бота и MCP. До M7 она жила в `retrieval/cli.py`, и три новые точки входа
получили бы по своей копии — четыре места, где надо не забыть выгрузить
модель эмбеддингов перед генерацией и не перепутать порядок.

Поэтому сборка вынесена сюда, а `cli.py` переведён на неё. Точки входа
остаются тонкими, как требует `ARCHITECTURE.md`: разобрать вход, вызвать
эту функцию, отформатировать ответ.

Обращения к моделям идут через `llm.runtime`: на 6 ГБ VRAM `bge-m3`
и `qwen2.5:3b` не помещаются одновременно, и в сервисе с параллельными
запросами порядок загрузки нельзя держать в голове — он держится очередью.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass

import httpx

from nutri_radar.config import Settings, get_settings
from nutri_radar.llm.adapters import OllamaEmbeddings, OllamaLLM
from nutri_radar.llm.runtime import get_runtime
from nutri_radar.logging import safe_extra
from nutri_radar.retrieval.rag import RagAnswer
from nutri_radar.retrieval.rag import answer as rag_answer
from nutri_radar.retrieval.search import SearchFilters, SearchResult
from nutri_radar.retrieval.search import search as search_vectors

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AskResult:
    """Ответ вместе с выдачей, на которой он построен.

    Возвращается одним объектом, потому что ответ без выдачи непроверяем:
    по нему нельзя понять, действительно ли модель сослалась на найденное
    или сочинила штрихкод. Это же свойство меряет метрика подтверждённости.
    """

    search: SearchResult
    answer: RagAnswer

    # Векторизация плюс поиск. Отделено от времени генерации: когда ответ
    # приходит за десять секунд, надо сразу видеть, ушли они в поиск
    # или в модель.
    retrieval_latency_s: float = 0.0


async def embed_texts(
    texts: Sequence[str],
    *,
    client: httpx.AsyncClient,
    settings: Settings | None = None,
) -> list[list[float]]:
    """Векторизовать тексты моделью эмбеддингов.

    Проходит через очередь моделей: если сейчас занята модель генерации,
    вызов подождёт её и выгрузит, а не полезет в занятую видеопамять.
    """
    settings = settings or get_settings()
    if not texts:
        return []

    model = OllamaEmbeddings(client, settings.ollama)
    runtime = get_runtime(settings)
    started = time.perf_counter()
    async with runtime.hold(model.model_name, unload=model.unload):
        vectors = await model.embed(texts)
    logger.debug(
        "Тексты векторизованы",
        extra=safe_extra(
            texts=len(texts),
            model=model.model_name,
            latency_s=round(time.perf_counter() - started, 3),
        ),
    )
    return vectors


async def search_by_text(
    query: str,
    *,
    client: httpx.AsyncClient,
    settings: Settings | None = None,
    limit: int | None = None,
    filters: SearchFilters | None = None,
) -> SearchResult:
    """Найти продукты по смыслу запроса: векторизация плюс поиск."""
    settings = settings or get_settings()
    started = time.perf_counter()

    vector = (await embed_texts([query], client=client, settings=settings))[0]
    embed_latency = time.perf_counter() - started

    result = await search_vectors(
        vector, query=query, limit=limit, filters=filters, settings=settings
    )
    logger.info(
        "Поиск по тексту выполнен",
        extra=safe_extra(
            query_chars=len(query),
            filters=(filters or SearchFilters()).describe(),
            found=len(result.hits),
            embed_s=round(embed_latency, 3),
            search_s=round(result.latency_s, 3),
        ),
    )
    return result


async def ask(
    question: str,
    *,
    client: httpx.AsyncClient,
    settings: Settings | None = None,
    limit: int | None = None,
    filters: SearchFilters | None = None,
) -> AskResult:
    """Ответить на вопрос строго по найденным продуктам.

    Отказ «не знаю» формируется кодом при пустой или нерелевантной выдаче —
    модель об этом не спрашивается (см. `rag.answer`). Здесь этот случай
    отдельно не обрабатывается намеренно: единственное место, где решается
    судьба отказа, должно остаться одно.
    """
    settings = settings or get_settings()
    started = time.perf_counter()

    result = await search_by_text(
        question, client=client, settings=settings, limit=limit, filters=filters
    )
    embed_and_search = time.perf_counter() - started

    llm = OllamaLLM(client, settings.ollama)
    runtime = get_runtime(settings)
    async with runtime.hold(llm.model_name):
        answer = await rag_answer(question, result, llm, settings)

    logger.info(
        "Ответ собран",
        extra=safe_extra(
            question_chars=len(question),
            refused=answer.refused,
            sources=len(answer.sources),
            cited=len(answer.cited),
            invented=len(answer.cited_outside_sources),
            tokens=answer.input_tokens + answer.output_tokens,
            retrieval_s=round(embed_and_search, 3),
            generation_s=round(answer.latency_s, 3),
        ),
    )
    return AskResult(search=result, answer=answer, retrieval_latency_s=embed_and_search)
