"""Инструменты агента и их реестр.

**Описание инструмента живёт рядом с его кодом**, а не в промпте. Промпт
собирается из реестра, поэтому добавление инструмента не требует правки
текста руками — а значит, описание не может разъехаться с поведением.
Разъехавшееся описание хуже отсутствующего: модель вызывает инструмент
по обещанию, которого он не выполняет.

**Схема аргументов — часть контракта.** Она уходит в промпт и в JSON-схему
ответа модели, так что вызов с лишним или пропущенным аргументом
невозможен по построению, а не отлавливается проверками.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from nutri_radar.logging import safe_extra

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolResult:
    """Что инструмент вернул агенту.

    Ошибка — не исключение, а результат с `ok=False`. Модель должна
    получить текст ошибки и попробовать иначе: агент, падающий от неверного
    аргумента, бесполезен ровно в тех случаях, ради которых он и нужен.
    """

    ok: bool
    content: str
    # Что показать человеку в трассировке помимо текста: число строк,
    # найденные штрихкоды, потраченное время.
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def failure(cls, message: str, **meta: Any) -> ToolResult:
        return cls(ok=False, content=message, meta=meta)


@dataclass(frozen=True)
class Tool:
    """Один инструмент агента."""

    name: str
    # Описание читает модель. Пишется как инструкция для того, кто видит
    # инструмент впервые: когда применять и когда НЕ применять. Второе
    # важнее — слабая модель тянется к первому подходящему инструменту.
    description: str
    # JSON-схема аргументов. Уходит и в промпт, и в схему ответа модели.
    parameters: dict[str, Any]
    run: Callable[..., Awaitable[ToolResult]]

    def describe(self) -> str:
        """Как инструмент выглядит в промпте."""
        args = ", ".join(
            f"{name}: {spec.get('type', 'string')}"
            for name, spec in self.parameters.get("properties", {}).items()
        )
        return f"- {self.name}({args})\n  {self.description}"


class ToolRegistry:
    """Реестр инструментов, из которого собирается промпт и схема действия."""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.add(tool)

    def add(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Инструмент {tool.name!r} уже зарегистрирован")
        self._tools[tool.name] = tool
        logger.debug("Инструмент зарегистрирован", extra=safe_extra(tool=tool.name))

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def describe(self) -> str:
        """Блок инструментов для промпта."""
        return "\n".join(self._tools[name].describe() for name in self.names)

    def action_schema(self) -> dict[str, Any]:
        """JSON-схема ответа модели: либо вызов инструмента, либо ответ.

        `enum` по именам инструментов делает вызов несуществующего
        инструмента физически невозможным — самая частая ошибка слабой
        модели отсекается схемой, а не проверкой после.
        """
        return {
            "type": "object",
            "properties": {
                # Рассуждение первым полем: модель, которой дали место
                # подумать до выбора действия, выбирает заметно лучше.
                "thought": {"type": "string"},
                "action": {"type": "string", "enum": [*self.names, "final_answer"]},
                "arguments": {"type": "object"},
                "answer": {"type": "string"},
            },
            "required": ["thought", "action"],
        }

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Вызвать инструмент по имени.

        Неизвестное имя и неверные аргументы возвращаются результатом,
        а не исключением: это сообщение модели о том, что надо исправить.
        """
        tool = self.get(name)
        if tool is None:
            return ToolResult.failure(
                f"Инструмента {name!r} нет. Доступны: {', '.join(self.names)}."
            )
        try:
            return await tool.run(**arguments)
        except TypeError as exc:
            # Модель передала не те аргументы. Это её ошибка, и она должна
            # о ней узнать текстом, а не уронить прогон.
            expected = ", ".join(tool.parameters.get("properties", {}))
            return ToolResult.failure(
                f"Неверные аргументы для {name!r}: {exc}. Ожидаются: {expected}."
            )
        except Exception as exc:
            # Любой отказ инструмента — сообщение модели, а не падение
            # цикла: агент, умирающий от одной неудачной попытки,
            # бесполезен ровно там, где нужен.
            logger.warning(
                "Инструмент отказал",
                extra=safe_extra(tool=name, error=type(exc).__name__),
            )
            return ToolResult.failure(f"Инструмент {name!r} отказал: {exc}")


__all__ = ["Tool", "ToolRegistry", "ToolResult"]
