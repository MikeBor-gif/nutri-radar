"""Порт языковой модели.

Единственное, что знают о моделях слайсы пайплайна. Конкретный адаптер
подставляет composition root по значению `LLM__PROVIDER` — ни `extract`,
ни `evals`, ни `agent` не импортируют `ollama` или `anthropic` напрямую
(см. ARCHITECTURE.md, антипаттерн «конкретный адаптер внутри слайса»).

Порт сознательно узкий: один метод. Расширять по факту потребности, а не
проектировать впрок.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from nutri_radar.llm.models import LLMResponse


@runtime_checkable
class StructuredLLM(Protocol):
    """Генерация, ограниченная JSON-схемой.

    Ограничение схемой — не просьба в промпте, а параметр генерации: свободный
    JSON от модели 3B ненадёжен (раздел 3a брифа).
    """

    @property
    def model_name(self) -> str:
        """Имя модели. Уезжает в `product_extraction` рядом с результатом."""
        ...

    async def generate(
        self,
        prompt: str,
        *,
        json_schema: dict[str, Any],
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        """Получить ответ, соответствующий схеме.

        Raises:
            LLMUnavailableError: модель недоступна или вернула отказ.
                Такую ошибку имеет смысл ретраить.
        """
        ...
