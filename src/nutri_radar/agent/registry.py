"""Сборка реестра инструментов агента.

Отдельный модуль, а не функция в `cli.py`: на M7 у реестра появились три
потребителя — CLI, HTTP-API и MCP-сервер. Держать сборку в CLI значило бы,
что API импортирует точку входа, а точки входа друг о друге знать не должны
(`ARCHITECTURE.md`).
"""

from __future__ import annotations

import logging

from nutri_radar.agent.tools import ToolRegistry
from nutri_radar.agent.tools.lookup_barcode import lookup_barcode_tool
from nutri_radar.agent.tools.sql_query import sql_query_tool
from nutri_radar.agent.tools.vector_search import vector_search_tool
from nutri_radar.config import Settings, get_settings
from nutri_radar.logging import safe_extra

logger = logging.getLogger(__name__)


def build_registry(settings: Settings | None = None) -> ToolRegistry:
    """Собрать реестр инструментов.

    Порядок регистрации не важен — реестр сортирует по имени, чтобы промпт
    был одинаковым между запусками. Разный порядок инструментов в промпте
    менял бы поведение модели, и сравнивать прогоны стало бы нельзя.
    """
    settings = settings or get_settings()
    registry = ToolRegistry(
        [
            lookup_barcode_tool(settings),
            sql_query_tool(settings),
            vector_search_tool(settings),
        ]
    )
    logger.debug("Реестр инструментов собран", extra=safe_extra(tools=registry.names))
    return registry
