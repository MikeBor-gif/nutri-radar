"""Подделка модели для тестов.

Правило 4 брифа: тест не ходит в сеть. Вся LLM-часть проверяется через этот
адаптер — он детерминирован и умеет воспроизводить нужные сценарии отказа.

Отдельно считает **максимальный одновременный** параллелизм: без этого нельзя
проверить, что раннер соблюдает `max_concurrency`, а на 6 ГБ VRAM нарушение
этого лимита кладёт GPU.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from nutri_radar.errors import LLMUnavailableError
from nutri_radar.llm.models import LLMResponse, TokenUsage


class FakeLLM:
    """Детерминированная реализация порта `StructuredLLM`."""

    def __init__(
        self,
        responses: dict[str, dict[str, Any]] | None = None,
        *,
        default_response: dict[str, Any] | None = None,
        model_name: str = "fake-model",
        fail_times: int = 0,
        latency_s: float = 0.0,
        response_factory: Callable[[str], dict[str, Any]] | None = None,
    ) -> None:
        """
        Args:
            responses: ответ по подстроке промпта. Первая подходящая — побеждает.
            default_response: что отдавать, если ничего не совпало.
            fail_times: сколько первых вызовов должны упасть с
                `LLMUnavailableError` — для проверки ретраев.
            latency_s: искусственная задержка, чтобы измерять параллелизм.
            response_factory: полностью своя логика ответа по промпту.
        """
        self._responses = responses or {}
        self._default = default_response or {
            "ingredients": [],
            "allergens": [],
            "unreadable": False,
            "model_confidence": 1.0,
        }
        self._model_name = model_name
        self._fail_times = fail_times
        self._latency_s = latency_s
        self._factory = response_factory

        self.calls: list[str] = []
        self._in_flight = 0
        self.max_in_flight = 0

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def call_count(self) -> int:
        return len(self.calls)

    async def generate(
        self,
        prompt: str,
        *,
        json_schema: dict[str, Any],
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls.append(prompt)

        if self._fail_times > 0:
            self._fail_times -= 1
            raise LLMUnavailableError("подделка: модель недоступна")

        self._in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            if self._latency_s:
                await asyncio.sleep(self._latency_s)
            payload = self._resolve(prompt)
        finally:
            self._in_flight -= 1

        return LLMResponse(
            raw_json=payload,
            usage=TokenUsage(input_tokens=len(prompt) // 4, output_tokens=42),
            latency_s=self._latency_s,
            model_name=self._model_name,
        )

    def _resolve(self, prompt: str) -> dict[str, Any]:
        if self._factory is not None:
            return self._factory(prompt)
        for needle, payload in self._responses.items():
            if needle in prompt:
                return payload
        return self._default
