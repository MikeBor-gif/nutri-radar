"""Инструмент `lookup_barcode`: единственный поход в живой API OFF.

**Это ровно то место, о котором говорит бриф.** Разработчики Open Food
Facts прямо просят не выкачивать базу через API — массовые данные проект
берёт дампом. Здесь один штрихкод за вызов, с таймаутом и User-Agent,
по которому OFF может опознать проект: это их требование, а не вежливость.

**«Продукт не найден» — нормальный ответ, а не ошибка.** Штрихкода может
не быть в OFF, и агент должен уметь сказать об этом. Пометить такой вызов
неудачным значило бы толкнуть модель повторять его в надежде, что во второй
раз получится.

**Поля запрашиваются перечнем.** Ответ OFF по одному продукту без фильтра —
десятки килобайт, из которых агенту нужны шесть полей. Остальное съело бы
контекст и замедлило бы каждый шаг.
"""

from __future__ import annotations

import logging
import re

import httpx

from nutri_radar.agent.tools import Tool, ToolResult
from nutri_radar.config import Settings, get_settings
from nutri_radar.logging import safe_extra

logger = logging.getLogger(__name__)

API_URL = "https://world.openfoodfacts.org/api/v2/product/{code}.json"

# Требование OFF: User-Agent с именем приложения, версией и контактом.
# Без него запросы имеют право быть отклонены.
USER_AGENT = "NutriRadar/0.1 (portfolio project; https://github.com/MikeBor-gif/nutri-radar)"

FIELDS = "code,product_name,brands,ingredients_text,nutriscore_grade,nova_group"

_BARCODE = re.compile(r"^\d{6,14}$")


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
            в сеть (правило 4 брифа); в бою создаётся здесь.
    """
    settings = settings or get_settings()
    code = str(barcode).strip()

    if not _BARCODE.match(code):
        # Проверка до сети: модель охотно подставляет название вместо кода,
        # и ходить с ним в API значит ждать таймаут ради заведомого отказа.
        return ToolResult.failure(
            f"«{code}» не похож на штрихкод: ожидаются 6-14 цифр без пробелов."
        )

    owns_client = client is None
    http = client or httpx.AsyncClient()
    try:
        response = await http.get(
            API_URL.format(code=code),
            params={"fields": FIELDS},
            headers={"User-Agent": USER_AGENT},
            timeout=settings.agent.tool_timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError as exc:
        logger.warning(
            "API OFF недоступен",
            extra=safe_extra(code=code, error=type(exc).__name__),
        )
        return ToolResult.failure(f"API Open Food Facts недоступен: {exc}")
    finally:
        if owns_client:
            await http.aclose()

    # `status` 0 означает «нет такого продукта». Это ответ, а не сбой:
    # пометить его ошибкой значило бы толкнуть модель на повтор.
    if not payload.get("status"):
        logger.info("Продукт не найден в OFF", extra=safe_extra(code=code))
        return ToolResult(
            ok=True,
            content=f"Продукт [{code}] в Open Food Facts не найден.",
            meta={"found": False},
        )

    logger.info("Продукт получен из OFF", extra=safe_extra(code=code))
    return ToolResult(
        ok=True,
        content=format_product(code, dict(payload.get("product") or {})),
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
