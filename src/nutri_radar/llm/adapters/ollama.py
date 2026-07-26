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
from typing import Any

import httpx

from nutri_radar.config import OllamaSettings
from nutri_radar.errors import LLMUnavailableError
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

        # Ollama отдаёт содержимое строкой даже при генерации по схеме.
        try:
            raw_json = json.loads(content) if isinstance(content, str) else dict(content)
        except (json.JSONDecodeError, TypeError) as exc:
            # При заданном format такого быть не должно, но если случилось —
            # это отказ модели, а не проблема данных.
            logger.error(
                "Ollama вернула не-JSON при заданной схеме",
                extra=safe_extra(model=self._settings.model, head=str(content)[:200]),
            )
            raise LLMUnavailableError(f"Ollama вернула не-JSON: {exc}") from exc

        truncated = usage.output_tokens >= int(limit * _TRUNCATION_RATIO)
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
