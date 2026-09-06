"""Инструмент `sql_query`: выполнение SQL, сочинённого моделью.

Самый опасный инструмент агента, и единственный, где защита обязана быть
**структурной**. Модель генерирует текст; текст уходит в базу. Просить её
в промпте «не пиши DELETE» защитой не является: промпт — это пожелание,
а инъекция — это выполнение.

Что защищает на самом деле:

1. **Read-only транзакция.** Postgres сам отвергнет любую запись, что бы
   ни пришло. Это последняя линия, и она не зависит ни от разбора текста,
   ни от белых списков.
2. **Ровно один оператор.** Точка с запятой запрещена, поэтому
   `SELECT 1; DROP TABLE products` не разложится на две команды.
3. **Только `SELECT`.** Запрос, начинающийся с чего угодно другого,
   не выполняется.
4. **Белый список таблиц.** Модель не должна ходить в `alembic_version`
   или в служебные таблицы, даже читая.
5. **Обязательный `LIMIT`.** Не защита от вреда, а защита от того, что
   модель получит десять тысяч строк и утонет в них вместе с контекстом.
6. **Таймаут выражения.** Запрос без индекса по 146 тысячам строк с
   `ILIKE '%...%'` выполним, но остановит прогон.

Каждая проверка покрыта тестом. Отказ возвращается модели текстом ошибки:
она должна суметь исправиться, а не уронить цикл.
"""

from __future__ import annotations

import logging
import re

from sqlalchemy import text

from nutri_radar.agent.tools import Tool, ToolResult
from nutri_radar.config import Settings, get_settings
from nutri_radar.db.session import get_session
from nutri_radar.logging import safe_extra

logger = logging.getLogger(__name__)

# Куда модели можно смотреть. Не «всё, кроме служебного», а перечень:
# белый список не расширяется сам при появлении новой таблицы, чёрный —
# расширяется, и однажды забудут дописать.
ALLOWED_TABLES = frozenset(
    {"products", "product_extraction", "product_embedding", "ingredients_dict"}
)

_SELECT_START = re.compile(r"^\s*select\b", re.IGNORECASE)
_LIMIT = re.compile(r"\blimit\s+\d+", re.IGNORECASE)
# Имена после FROM и JOIN. Схема не разбирается всерьёз — для белого списка
# достаточно найти всё, что похоже на источник строк, и проверить каждое.
_SOURCES = re.compile(r"\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_.]*)", re.IGNORECASE)


def validate_sql(query: str, max_rows: int) -> str | None:
    """Проверить запрос до выполнения. Возвращает текст ошибки или `None`.

    Проверки идут до базы, чтобы модель получала внятное объяснение,
    а не `SyntaxError` из драйвера. Но они **не единственная** защита:
    read-only транзакция ниже держит оборону, даже если разбор текста
    что-то пропустил.
    """
    stripped = query.strip().rstrip(";").strip()

    if not stripped:
        return "Пустой запрос."
    if ";" in stripped:
        return (
            "Точка с запятой запрещена: разрешён ровно один оператор. Уберите её и всё, что после."
        )
    if not _SELECT_START.match(stripped):
        return "Разрешён только SELECT. Запись и изменение схемы недоступны."

    sources = {name.lower().split(".")[-1] for name in _SOURCES.findall(stripped)}
    forbidden = sources - ALLOWED_TABLES
    if forbidden:
        return (
            f"Таблицы {', '.join(sorted(forbidden))} недоступны. "
            f"Доступны: {', '.join(sorted(ALLOWED_TABLES))}."
        )

    if not _LIMIT.search(stripped):
        return f"Добавьте LIMIT (не больше {max_rows}): без него ответ не поместится."

    return None


def _format_rows(columns: list[str], rows: list[tuple]) -> str:
    """Результат в виде, который модель разберёт.

    Не JSON: слабая модель на JSON внутри наблюдения начинает отвечать
    JSON-ом вместо вызова следующего инструмента. Простые строки безопаснее.
    """
    if not rows:
        return "Строк не найдено."
    header = " | ".join(columns)
    body = "\n".join(
        " | ".join("" if value is None else str(value) for value in row) for row in rows
    )
    return f"{header}\n{body}\n({len(rows)} строк)"


async def run_sql_query(query: str, settings: Settings | None = None) -> ToolResult:
    """Выполнить SELECT и вернуть строки модели."""
    settings = settings or get_settings()
    max_rows = settings.agent.sql_max_rows

    error = validate_sql(query, max_rows)
    if error is not None:
        logger.info("SQL отклонён", extra=safe_extra(reason=error, query=query[:200]))
        return ToolResult.failure(error, query=query[:200])

    stripped = query.strip().rstrip(";").strip()
    timeout_ms = int(settings.agent.tool_timeout_s * 1000)

    async with get_session(settings.db) as session:
        # Обе строки — часть защиты, а не оптимизация. Транзакция read-only
        # отвергает запись на уровне Postgres, таймаут не даёт запросу
        # без индекса остановить весь прогон.
        await session.execute(text("SET TRANSACTION READ ONLY"))
        await session.execute(text(f"SET LOCAL statement_timeout = {timeout_ms}"))
        result = await session.execute(text(stripped))
        columns = list(result.keys())
        rows = result.fetchmany(max_rows)

    logger.info(
        "SQL выполнен",
        extra=safe_extra(rows=len(rows), columns=len(columns), query=query[:200]),
    )
    return ToolResult(
        ok=True,
        content=_format_rows(columns, [tuple(row) for row in rows]),
        meta={"rows": len(rows)},
    )


def sql_query_tool(settings: Settings | None = None) -> Tool:
    """Собрать инструмент. Настройки замыкаются, чтобы реестр не знал о них."""
    settings = settings or get_settings()
    tables = ", ".join(sorted(ALLOWED_TABLES))

    return Tool(
        name="sql_query",
        description=(
            f"Выполнить SELECT по базе и получить строки. Таблицы: {tables}. "
            "Обязателен LIMIT. Применяй, когда нужны точные числа, подсчёты "
            "или фильтр по колонке (оценка, язык, число ингредиентов). "
            "НЕ применяй для поиска по смыслу состава — для этого есть "
            "vector_search."
        ),
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "SELECT-запрос с LIMIT"}},
            "required": ["query"],
        },
        run=lambda query: run_sql_query(query, settings),
    )
