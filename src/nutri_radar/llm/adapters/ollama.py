"""Адаптер локальной модели через Ollama.

Три вещи, проверенные на живой модели `qwen2.5:3b-instruct-q4_K_M`:

1. Параметр `format` принимает JSON-схему, и она соблюдается — валидный JSON
   во всех попытках, включая схемы с `enum` и nullable-полями.
2. `num_ctx` задаётся **явно**. Дефолт Ollama — 4096, и он подбирается
   динамически по доступной VRAM, то есть молча обрезает вход.
3. Скорость 2,5–10,5 с на продукт при входе около 220 токенов.

Параллелизм ограничен семафором: на 6 ГБ VRAM неограниченный веер запросов
положит GPU.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Sequence
from typing import Any

import httpx

from nutri_radar.config import OllamaSettings
from nutri_radar.errors import ExtractionError, LLMUnavailableError
from nutri_radar.llm.models import LLMResponse, TokenUsage
from nutri_radar.logging import safe_extra

logger = logging.getLogger(__name__)

# Доля от лимита вывода, начиная с которой ответ считается подозрительно
# близким к обрезанию. Обрезанный JSON может пройти схему, но список
# ингредиентов окажется неполным.
_TRUNCATION_RATIO = 0.98


class OllamaLLM:
    """Локальная модель. Реализация порта `StructuredLLM`."""

    def __init__(self, client: httpx.AsyncClient, settings: OllamaSettings) -> None:
        self._client = client
        self._settings = settings
        # Семафор, а не голый gather: 6 ГБ VRAM не переживут неограниченный
        # веер запросов, а Ollama начнёт выгружать и грузить модель заново.
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)
        logger.info(
            "Адаптер Ollama создан",
            extra={
                "model": settings.model,
                "num_ctx": settings.num_ctx,
                "keep_alive": settings.keep_alive,
                "max_concurrency": settings.max_concurrency,
                "base_url": settings.base_url,
            },
        )

    @property
    def model_name(self) -> str:
        return self._settings.model

    async def generate(
        self,
        prompt: str,
        *,
        json_schema: dict[str, Any],
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        limit = max_output_tokens or self._settings.max_output_tokens
        payload = {
            "model": self._settings.model,
            "messages": [{"role": "user", "content": prompt}],
            # Ограничение генерации схемой, а не просьба в промпте.
            "format": json_schema,
            "stream": False,
            "options": {
                # Явно: дефолт Ollama мал и молча режет вход.
                "num_ctx": self._settings.num_ctx,
                "num_predict": limit,
                "temperature": self._settings.temperature,
            },
            # Модели грузятся по очереди: рядом живёт bge-m3 на 1,2 ГБ.
            "keep_alive": self._settings.keep_alive,
        }

        async with self._semaphore:
            started = time.perf_counter()
            try:
                response = await self._client.post(
                    "/api/chat", json=payload, timeout=self._settings.timeout_s
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                logger.error(
                    "Ollama недоступна",
                    extra=safe_extra(
                        base_url=self._settings.base_url,
                        model=self._settings.model,
                        error=type(exc).__name__,
                    ),
                )
                raise LLMUnavailableError(
                    f"Ollama недоступна ({self._settings.base_url}): {exc}"
                ) from exc
            latency = time.perf_counter() - started

        return self._to_response(response.json(), latency, limit)

    def _to_response(self, payload: dict[str, Any], latency: float, limit: int) -> LLMResponse:
        content = payload.get("message", {}).get("content", "")
        usage = TokenUsage(
            input_tokens=int(payload.get("prompt_eval_count") or 0),
            output_tokens=int(payload.get("eval_count") or 0),
        )
        truncated = usage.output_tokens >= int(limit * _TRUNCATION_RATIO)

        # Ollama отдаёт содержимое строкой даже при генерации по схеме.
        try:
            raw_json = json.loads(content) if isinstance(content, str) else dict(content)
        except (json.JSONDecodeError, TypeError) as exc:
            # Две разные причины с противоположной реакцией на ретрай.
            #
            # Длинный состав упёрся в `num_predict`, и JSON оборвался посреди
            # массива. Это проблема данных: при temperature=0 повтор даст ту же
            # обрезку, потратив ещё столько же времени на генерацию. Такой
            # продукт пропускается как невалидный, а не ретраится.
            if truncated:
                logger.warning(
                    "Ответ обрезан лимитом вывода и не разобрался — продукт невалиден",
                    extra=safe_extra(
                        model=self._settings.model,
                        output_tokens=usage.output_tokens,
                        limit=limit,
                        head=str(content)[:200],
                    ),
                )
                raise ExtractionError(
                    f"Ollama оборвала JSON на лимите вывода ({usage.output_tokens} "
                    f"из {limit} токенов): состав длиннее, чем помещается в ответ",
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    latency_s=latency,
                ) from exc
            # Модель ответила мусором, не упёршись в лимит: похоже на сбой
            # загрузки модели. Вот это ретрай лечит.
            logger.error(
                "Ollama вернула не-JSON при заданной схеме",
                extra=safe_extra(model=self._settings.model, head=str(content)[:200]),
            )
            raise LLMUnavailableError(f"Ollama вернула не-JSON: {exc}") from exc

        if truncated:
            logger.warning(
                "Ответ упёрся в лимит вывода — список ингредиентов может быть неполным",
                extra=safe_extra(output_tokens=usage.output_tokens, limit=limit),
            )

        logger.debug(
            "Ответ Ollama получен",
            extra=safe_extra(
                latency_s=round(latency, 2),
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
            ),
        )
        return LLMResponse(
            raw_json=raw_json,
            usage=usage,
            latency_s=round(latency, 3),
            model_name=self._settings.model,
            truncated=truncated,
        )


class OllamaEmbeddings:
    """Модель эмбеддингов. Реализация порта `EmbeddingModel`.

    Отдельный класс, а не метод в `OllamaLLM`: у моделей разный жизненный
    цикл. На 6 ГБ VRAM `qwen2.5:3b` и `bge-m3` одновременно не помещаются,
    и держать их в одном объекте значило бы притворяться, что помещаются.

    `keep_alive` управляется явно по той же причине: после прогона
    эмбеддингов модель надо выгрузить, иначе следующая загрузка модели
    генерации упрётся в занятую память.
    """

    def __init__(self, client: httpx.AsyncClient, settings: OllamaSettings) -> None:
        self._client = client
        self._settings = settings
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)
        logger.info(
            "Адаптер эмбеддингов Ollama создан",
            extra={
                "model": settings.embedding_model,
                "dimensions": settings.embedding_dim,
                "keep_alive": settings.keep_alive,
                "base_url": settings.base_url,
            },
        )

    @property
    def model_name(self) -> str:
        return self._settings.embedding_model

    @property
    def dimensions(self) -> int:
        return self._settings.embedding_dim

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Векторизовать батч. Порядок ответа совпадает с порядком входа."""
        if not texts:
            return []

        payload = {
            "model": self._settings.embedding_model,
            "input": list(texts),
            "keep_alive": self._settings.keep_alive,
            "options": {"num_ctx": self._settings.num_ctx},
        }

        async with self._semaphore:
            started = time.perf_counter()
            try:
                response = await self._client.post(
                    "/api/embed", json=payload, timeout=self._settings.timeout_s
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                logger.error(
                    "Ollama недоступна при векторизации",
                    extra=safe_extra(
                        base_url=self._settings.base_url,
                        model=self._settings.embedding_model,
                        error=type(exc).__name__,
                    ),
                )
                raise LLMUnavailableError(
                    f"Ollama недоступна ({self._settings.base_url}): {exc}"
                ) from exc
            latency = time.perf_counter() - started

        vectors = response.json().get("embeddings") or []

        # Проверяем длину батча и размерность, а не доверяем. Расхождение
        # размерности иначе всплывёт вставкой в `vector(1024)` через слайс
        # и два модуля от места, где оно возникло.
        if len(vectors) != len(texts):
            raise ExtractionError(f"Ollama вернула {len(vectors)} векторов на {len(texts)} текстов")
        for vector in vectors:
            if len(vector) != self._settings.embedding_dim:
                raise ExtractionError(
                    f"Размерность вектора {len(vector)} не совпала с ожидаемой "
                    f"{self._settings.embedding_dim}: модель "
                    f"{self._settings.embedding_model} вернула не то, что объявлено "
                    "в настройках"
                )

        logger.debug(
            "Батч векторизован",
            extra=safe_extra(
                texts=len(texts),
                latency_s=round(latency, 2),
                per_text_s=round(latency / len(texts), 4),
            ),
        )
        return [[float(x) for x in vector] for vector in vectors]

    async def unload(self) -> None:
        """Выгрузить модель из памяти.

        Нужно между стадиями: 6 ГБ VRAM не держат две модели, и без явной
        выгрузки следующая загрузка либо ждёт истечения `keep_alive`, либо
        уезжает в своп. Отказ здесь не критичен — модель выгрузится сама
        по таймауту, поэтому ошибка только логируется.
        """
        try:
            await self._client.post(
                "/api/embed",
                json={"model": self._settings.embedding_model, "input": [], "keep_alive": 0},
                timeout=self._settings.timeout_s,
            )
            logger.info(
                "Модель эмбеддингов выгружена",
                extra=safe_extra(model=self._settings.embedding_model),
            )
        except httpx.HTTPError as exc:
            logger.warning(
                "Выгрузить модель не удалось — освободится по keep_alive",
                extra=safe_extra(error=type(exc).__name__),
            )
