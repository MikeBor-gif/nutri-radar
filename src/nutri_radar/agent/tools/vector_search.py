"""Инструмент `vector_search`: поиск по смыслу состава.

Самый безопасный из трёх: параметры типизированы, произвольного кода
модель не сочиняет. Обёртка над поиском M5 — дублировать поиск внутри
агента значило бы получить две расходящиеся реализации, и через месяц
никто не сказал бы, какая из них считала метрики.

Это единственный разрешённый импорт слайса в слайс (`ARCHITECTURE.md`:
`agent` → `retrieval`).

**Описание инструмента честно говорит о его слабости.** M5 измерил:
отрицание («без пальмового масла») векторный поиск не ловит — 1 попадание
из 5. Умолчать об этом в описании значило бы отправлять модель в заведомо
плохой инструмент; вместо этого описание переадресует такие запросы
в `sql_query`.
"""

from __future__ import annotations

import logging

import httpx

from nutri_radar.agent.tools import Tool, ToolResult
from nutri_radar.config import Settings, get_settings
from nutri_radar.llm.factory import build_embeddings
from nutri_radar.logging import safe_extra
from nutri_radar.retrieval.search import SearchFilters, SearchHit, search

logger = logging.getLogger(__name__)

# Сколько символов состава показывать модели. Полный состав бывает
# в две тысячи символов, и пять таких съедают контекст, не добавляя
# понимания: для выбора продукта хватает начала списка, где стоят
# ингредиенты с наибольшей долей.
_INGREDIENTS_PREVIEW = 300


def format_hits(hits: list[SearchHit]) -> str:
    """Выдача в виде, который модель разберёт и процитирует.

    Штрихкод в квадратных скобках — в той же форме, которую модель обязана
    воспроизвести в ответе. Чем ближе форма в наблюдении к требуемой,
    тем реже она изобретает свою.
    """
    if not hits:
        return "Ничего не найдено."

    blocks = []
    for hit in hits:
        parts = [f"[{hit.code}] {hit.product_name or 'без названия'}"]
        if hit.nutriscore_grade:
            parts.append(f"оценка {hit.nutriscore_grade}")
        if hit.ingredients_text:
            parts.append(f"состав: {hit.ingredients_text[:_INGREDIENTS_PREVIEW]}")
        blocks.append("; ".join(parts))
    return "\n".join(blocks)


async def run_vector_search(
    query: str,
    lang: str = "",
    limit: int = 0,
    settings: Settings | None = None,
) -> ToolResult:
    """Найти продукты по смыслу запроса."""
    settings = settings or get_settings()
    top_k = limit or settings.retrieval.top_k

    async with httpx.AsyncClient(base_url=settings.ollama.base_url) as client:
        model = build_embeddings(settings, client)
        vector = (await model.embed([query]))[0]

    result = await search(
        vector,
        query=query,
        limit=top_k,
        filters=SearchFilters(lang=lang or None),
        settings=settings,
    )
    logger.info(
        "vector_search отработал",
        extra=safe_extra(query=query[:120], found=len(result.hits), lang=lang or "любой"),
    )
    return ToolResult(
        ok=True,
        content=format_hits(result.hits),
        meta={"found": len(result.hits), "codes": result.codes},
    )


def vector_search_tool(settings: Settings | None = None) -> Tool:
    """Собрать инструмент. Настройки замыкаются, чтобы реестр не знал о них."""
    settings = settings or get_settings()

    return Tool(
        name="vector_search",
        description=(
            "Найти продукты по смыслу состава: «шоколад с пальмовым маслом», "
            "«снеки с глутаматом». Применяй, когда ищешь продукты по свойству "
            "состава. НЕ применяй, когда известен штрихкод — для этого есть "
            "lookup_barcode. Отрицание («без пальмового масла») этот инструмент "
            "ловит плохо: для него бери sql_query с NOT ILIKE."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Что искать, словами"},
                "lang": {
                    "type": "string",
                    "description": "Язык состава: ru, en, de, fr, pl",
                },
                "limit": {"type": "integer", "description": "Сколько вернуть"},
            },
            "required": ["query"],
        },
        run=lambda query, lang="", limit=0: run_vector_search(query, lang, limit, settings),
    )
