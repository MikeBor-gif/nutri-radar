"""Тесты MCP-сервера.

Главное проверяемое свойство — **одно описание инструмента на проект**.
Описание это не документация, а часть промпта: по нему модель выбирает,
что вызвать. Разойдясь, копия для агента и копия для MCP дали бы разное
поведение при одинаковом на вид наборе инструментов, и разбирать такое
пришлось бы долго.

Второе — **ошибка инструмента не роняет сессию**. У клиента на другом
конце stdio есть право узнать, что не получилось, и попробовать иначе.

Сети и БД здесь нет: сам вызов инструмента подставной (правило 4 брифа).
"""

from __future__ import annotations

import pytest

from nutri_radar.agent.registry import build_registry
from nutri_radar.agent.tools import Tool, ToolRegistry, ToolResult
from nutri_radar.config import Settings
from nutri_radar.mcp_server import server as mcp_server
from nutri_radar.wording import evaluative_labels_in


@pytest.fixture
def registry(settings: Settings) -> ToolRegistry:
    return build_registry(settings)


class TestНаборИнструментов:
    async def test_состав_совпадает_с_реестром_агента(
        self, settings: Settings, registry: ToolRegistry
    ) -> None:
        server = mcp_server.create_server(settings)
        exposed = sorted(tool.name for tool in await server.list_tools())

        assert exposed == registry.names

    async def test_описания_берутся_из_реестра(
        self, settings: Settings, registry: ToolRegistry
    ) -> None:
        """Расхождение здесь — признак второй копии описаний."""
        server = mcp_server.create_server(settings)

        for tool in await server.list_tools():
            source = registry.get(tool.name)
            assert source is not None
            assert tool.description == source.description

    async def test_обязательные_аргументы_совпадают_с_реестром(
        self, settings: Settings, registry: ToolRegistry
    ) -> None:
        """Схема выводится из аннотаций обёртки, и разъехаться она может.

        Обязательные аргументы — та часть, где расхождение видно сразу:
        клиент не пришлёт то, чего в схеме нет, и инструмент упадёт.
        """
        server = mcp_server.create_server(settings)

        for tool in await server.list_tools():
            source = registry.get(tool.name)
            assert source is not None
            expected = set(source.parameters.get("required", []))
            actual = set(tool.input_schema.get("required", []))
            assert actual == expected, f"Аргументы {tool.name} разошлись с реестром"

    async def test_забытая_обёртка_ломает_сборку(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Инструмент, добавленный агенту, не должен тихо исчезнуть из MCP."""

        def _registry_with_extra(_settings: Settings | None = None) -> ToolRegistry:
            registry = build_registry(settings)
            registry.add(
                Tool(
                    name="новый_инструмент",
                    description="появился у агента, но не обёрнут для MCP",
                    parameters={"type": "object", "properties": {}},
                    run=lambda: ToolResult(ok=True, content=""),
                )
            )
            return registry

        monkeypatch.setattr(mcp_server, "build_registry", _registry_with_extra)

        with pytest.raises(KeyError, match="новый_инструмент"):
            mcp_server.create_server(settings)


class TestВызов:
    async def test_вызов_доходит_до_реестра(self, registry: ToolRegistry) -> None:
        seen: dict[str, object] = {}

        async def _call(name: str, arguments: dict[str, object]) -> ToolResult:
            seen["name"] = name
            seen["arguments"] = arguments
            return ToolResult(ok=True, content="нашёл")

        registry.call = _call  # type: ignore[method-assign]
        text = await mcp_server._call(registry, "vector_search", query="шоколад", limit=3)

        assert text == "нашёл"
        assert seen["name"] == "vector_search"
        assert seen["arguments"] == {"query": "шоколад", "limit": 3}

    async def test_непереданные_аргументы_не_доезжают(self, registry: ToolRegistry) -> None:
        """У инструмента свои значения по умолчанию из настроек.

        Прислать ему `lang=None` значило бы навязать пустой фильтр вместо
        настроенного поведения.
        """
        seen: dict[str, object] = {}

        async def _call(name: str, arguments: dict[str, object]) -> ToolResult:
            seen["arguments"] = arguments
            return ToolResult(ok=True, content="")

        registry.call = _call  # type: ignore[method-assign]
        await mcp_server._call(registry, "vector_search", query="шоколад", lang=None, limit=None)

        assert seen["arguments"] == {"query": "шоколад"}

    async def test_ошибка_инструмента_возвращается_текстом(self, registry: ToolRegistry) -> None:
        """Уронить сессию из-за неверного аргумента — заставить клиента
        переподключаться вместо того, чтобы он исправил запрос."""

        async def _call(name: str, arguments: dict[str, object]) -> ToolResult:
            return ToolResult.failure("«шоколад» не похож на штрихкод")

        registry.call = _call  # type: ignore[method-assign]
        text = await mcp_server._call(registry, "lookup_barcode", barcode="шоколад")

        assert "не похож на штрихкод" in text


class TestОписаниеСервера:
    def test_инструкции_несут_атрибуцию_и_дисклеймер(self) -> None:
        # Клиент MCP — это чужой агент, и он тоже обязан узнать, чьи данные
        # получает и какого они качества.
        assert "ODbL" in mcp_server.INSTRUCTIONS
        assert "краудсорсинговые" in mcp_server.INSTRUCTIONS

    def test_в_инструкциях_нет_оценочных_ярлыков(self) -> None:
        assert evaluative_labels_in(mcp_server.INSTRUCTIONS) == []
