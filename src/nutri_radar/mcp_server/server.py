"""MCP-сервер: инструменты агента наружу.

**Что это даёт.** Три инструмента, которые M6 написал для своего агента,
становятся доступны любому MCP-клиенту — Claude Desktop, IDE, чужому
агенту. Проект перестаёт быть закрытым демо и превращается в источник
данных, которым можно пользоваться извне.

**Обёртка тонкая.** Имена и описания инструментов берутся из того же
реестра, что видит агент (`agent/registry.py`). Второй копии описаний
не появляется — а описание инструмента это то, по чему модель выбирает,
что вызвать: разойдясь, две копии дали бы разное поведение агента и
внешнего клиента при одинаковом на вид наборе инструментов.

Сигнатуры обёрток при этом написаны руками: MCP выводит схему аргументов
из аннотаций, а реестр хранит её готовым JSON. Это единственное, что
здесь продублировано, и продублировано осознанно — аргументов один-два
на инструмент, они типизированы, и попытка собрать сигнатуру динамически
дала бы хрупкий код ради экономии шести строк.

**Логи идут в stderr.** stdout занят протоколом: строка лога, попавшая
в него, ломает сессию клиента.
"""

from __future__ import annotations

import logging

from mcp.server import MCPServer

from nutri_radar import __version__
from nutri_radar.agent.registry import build_registry
from nutri_radar.agent.tools import ToolRegistry
from nutri_radar.config import Settings, get_settings
from nutri_radar.logging import safe_extra
from nutri_radar.wording import ATTRIBUTION, DISCLAIMER

logger = logging.getLogger(__name__)

INSTRUCTIONS = f"""Разбор состава пищевых продуктов на данных Open Food Facts.

Инструменты работают по локальному корпусу в 147 тысяч продуктов и,
для lookup_barcode, по живому API Open Food Facts.

{ATTRIBUTION}

{DISCLAIMER}"""


async def _call(registry: ToolRegistry, name: str, **arguments: object) -> str:
    """Вызвать инструмент реестра и вернуть текст клиенту.

    Ошибка инструмента возвращается текстом, а не исключением: у клиента
    на другом конце stdio есть право узнать, что именно не получилось,
    и попробовать иначе. Уронить сессию из-за неверного аргумента значило
    бы заставить его переподключаться.
    """
    # Необязательные аргументы, которые клиент не прислал, до инструмента
    # не доезжают: у него свои значения по умолчанию из настроек.
    payload = {key: value for key, value in arguments.items() if value is not None}
    result = await registry.call(name, payload)

    logger.info(
        "Инструмент вызван через MCP",
        extra=safe_extra(tool=name, ok=result.ok, chars=len(result.content)),
    )
    return result.content


def create_server(settings: Settings | None = None) -> MCPServer:
    """Собрать MCP-сервер поверх реестра инструментов агента."""
    settings = settings or get_settings()
    registry = build_registry(settings)

    server = MCPServer(
        name="nutri-radar",
        version=__version__,
        instructions=INSTRUCTIONS,
    )

    async def vector_search(query: str, lang: str | None = None, limit: int | None = None) -> str:
        return await _call(registry, "vector_search", query=query, lang=lang, limit=limit)

    async def sql_query(query: str) -> str:
        return await _call(registry, "sql_query", query=query)

    async def lookup_barcode(barcode: str) -> str:
        return await _call(registry, "lookup_barcode", barcode=barcode)

    wrappers = {
        "vector_search": vector_search,
        "sql_query": sql_query,
        "lookup_barcode": lookup_barcode,
    }

    # Обход по реестру, а не по словарю обёрток: инструмент, добавленный
    # агенту и забытый здесь, обязан сломать сборку сервера, а не тихо
    # исчезнуть из выдачи MCP.
    for name in registry.names:
        tool = registry.get(name)
        wrapper = wrappers.get(name)
        if tool is None or wrapper is None:
            raise KeyError(
                f"Инструмент {name!r} есть в реестре агента, но не обёрнут для MCP. "
                "Добавьте обёртку в mcp_server/server.py."
            )
        server.add_tool(wrapper, name=tool.name, description=tool.description)

    logger.info(
        "MCP-сервер собран",
        extra=safe_extra(version=__version__, tools=registry.names),
    )
    return server


def main(settings: Settings | None = None) -> None:
    """Запустить сервер на stdio. Точка входа CLI."""
    server = create_server(settings)
    logger.info("MCP-сервер слушает stdio")
    server.run(transport="stdio")
