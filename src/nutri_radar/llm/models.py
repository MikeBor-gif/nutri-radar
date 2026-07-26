"""Общие типы для работы с языковыми моделями.

Живут отдельно от портов, чтобы адаптеры и потребители не тянули друг друга.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    """Расход токенов на один вызов.

    Считается всегда: время и стоимость полного прогона — обязательные метрики
    проекта, и собрать их постфактум по логам нельзя.
    """

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class LLMResponse(BaseModel):
    """Ответ модели, ограниченный JSON-схемой."""

    raw_json: dict[str, Any] = Field(default_factory=dict)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_s: float = 0.0
    model_name: str = ""
    # Ответ упёрся в лимит вывода. Такой JSON схема может пропустить, но список
    # ингредиентов окажется обрезанным — результат выглядит валидным, а он неполон.
    truncated: bool = False
