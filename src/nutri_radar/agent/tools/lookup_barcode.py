"""Инструмент `lookup_barcode`: свежие данные по одному штрихкоду.

**Сам поход в живой API живёт не здесь, а в `nutri_radar.openfoodfacts`.**
На M7 у него появился второй потребитель — карточка продукта, — и общая
граница переехала в общий слой (см. докстринг того модуля). Здесь остаётся
то, что относится к агенту: описание инструмента для модели, проверка
аргумента и превращение продукта в текст, который модель сможет
процитировать.

**«Продукт не найден» — нормальный ответ, а не ошибка.** Штрихкода может
не быть в OFF, и агент должен уметь об этом сказать. Пометить такой вызов
неудачным значило бы толкнуть модель повторять его в надежде, что во второй
раз получится.
"""

from __future__ import annotations

import logging

import httpx

from nutri_radar.agent.tools import Tool, ToolResult
from nutri_radar.config import Settings, get_settings
from nutri_radar.errors import DataSourceError
from nutri_radar.logging import safe_extra
from nutri_radar.openfoodfacts import FIELDS, USER_AGENT, fetch_product, normalize_barcode

logger = logging.getLogger(__name__)

# Реэкспорт: перечень полей и User-Agent — часть контракта с OFF, и проверять
# его удобнее там же, где проверяется инструмент.
__all__ = [
    "FIELDS",
    "USER_AGENT",
    "format_product",
    "lookup_barcode_tool",
    "run_lookup_barcode",
]


def format_product(code: str, product: dict) -> str:
    """Продукт в виде, который модель разберёт и процитирует."""
    parts = [f"[{code}] {product.get('product_name') or 'без названия'}"]
    if product.get("brands"):
        parts.append(f"бренд: {product['brands']}")
    if product.get("nutriscore_grade"):
        parts.append(f"оценка: {product['nutriscore_grade']}")
    if product.get("nova_group"):
        parts.append(f"NOVA: {product['nova_group']}")
    if product.get("ingredients_text"):
        parts.append(f"состав: {product['ingredients_text']}")
    return "\n".join(parts)


async def run_lookup_barcode(
    barcode: str,
    settings: Settings | None = None,
    *,
    client: httpx.AsyncClient | None = None,
) -> ToolResult:
    """Спросить у Open Food Facts один продукт по штрихкоду.

    Args:
        barcode: код, 6-14 цифр.
        settings: настройки.
        client: клиент HTTP. Передаётся тестом, чтобы вызов не уходил
            в сеть (правило 4 брифа); в бою создаётся внутри клиента OFF.
    """
    settings = settings or get_settings()
    code = normalize_barcode(barcode)

    if code is None:
        # Проверка до сети: модель охотно подставляет название вместо кода,
        # и ходить с ним в API значит ждать таймаут ради заведомого отказа.
        return ToolResult.failure(
            f"«{str(barcode).strip()}» не похож на штрихкод: ожидаются 6-14 цифр без пробелов."
        )

    try:
        product = await fetch_product(code, timeout_s=settings.agent.tool_timeout_s, client=client)
    except DataSourceError as exc:
        # Отказ сети возвращается модели текстом, а не исключением: она
        # должна уметь попробовать другой инструмент, а не уронить прогон.
        logger.warning("Инструмент не смог обратиться к OFF", extra=safe_extra(code=code))
        return ToolResult.failure(str(exc))

    if product is None:
        return ToolResult(
            ok=True,
            content=f"Продукт [{code}] в Open Food Facts не найден.",
            meta={"found": False},
        )

    return ToolResult(
        ok=True,
        content=format_product(code, product.model_dump(exclude_none=True)),
        meta={"found": True},
    )


def lookup_barcode_tool(
    settings: Settings | None = None, *, client: httpx.AsyncClient | None = None
) -> Tool:
    """Собрать инструмент."""
    settings = settings or get_settings()

    return Tool(
        name="lookup_barcode",
        description=(
            "Получить продукт по штрихкоду из Open Food Facts. Применяй, "
            "когда штрихкод известен и нужны свежие данные. НЕ применяй для "
            "поиска — только для одного конкретного кода."
        ),
        parameters={
            "type": "object",
            "properties": {"barcode": {"type": "string", "description": "Штрихкод, 6-14 цифр"}},
            "required": ["barcode"],
        },
        run=lambda barcode: run_lookup_barcode(barcode, settings, client=client),
    )
