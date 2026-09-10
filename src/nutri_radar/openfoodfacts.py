"""Живой API Open Food Facts: единственный модуль проекта, который туда ходит.

**Граница, а не удобство.** Разработчики OFF прямо просят не выкачивать базу
через API — массовые данные проект берёт дампом (раздел «Данные» брифа).
Здесь один штрихкод за вызов, с таймаутом и с User-Agent, по которому OFF
может опознать проект: это их требование, а не вежливость.

До M7 этот код жил внутри `agent/tools/lookup_barcode.py` и был единственным
местом обращения к API. На M7 у него появился второй потребитель — карточка
продукта по штрихкоду, которого нет в корпусе (147 тысяч продуктов из
4,63 млн строк дампа, то есть большинство реальных сканов). Тянуть
инструмент агента из слайса `retrieval` нельзя: это запрещённая зависимость
между слайсами (`ARCHITECTURE.md`), а копия HTTP-вызова превратила бы одну
границу в две.

Поэтому вызов переехал в общий слой, к остальным внешним системам. Граница
осталась одна — просто теперь это модуль, а не функция: любой поход в живой
API проходит здесь, и проверить это можно поиском по импортам.

**«Продукт не найден» — ответ, а не ошибка.** Штрихкода может не быть в OFF,
и потребитель обязан уметь об этом сказать. Отказ сети — другое дело:
он заворачивается в доменное исключение на границе слоя, наружу
`httpx.HTTPError` не протекает.
"""

from __future__ import annotations

import logging
import re

import httpx
from pydantic import BaseModel

from nutri_radar.errors import DataSourceError
from nutri_radar.logging import safe_extra

logger = logging.getLogger(__name__)

API_URL = "https://world.openfoodfacts.org/api/v2/product/{code}.json"

# Требование OFF: User-Agent с именем приложения, версией и контактом.
# Без него запросы имеют право быть отклонены.
USER_AGENT = "NutriRadar/0.1 (portfolio project; https://github.com/MikeBor-gif/nutri-radar)"

# Поля запрашиваются перечнем: ответ OFF по одному продукту без фильтра —
# десятки килобайт, из которых нужны шесть полей. Остальное съело бы
# контекст модели и замедлило бы каждый шаг агента.
FIELDS = "code,product_name,brands,ingredients_text,nutriscore_grade,nova_group"

# Штрихкод: 6-14 цифр. EAN-8, EAN-13, UPC и внутренние коды OFF.
_BARCODE = re.compile(r"^\d{6,14}$")


class OffProduct(BaseModel):
    """Продукт из живого API. Имена полей совпадают с именами в ответе OFF.

    Совпадение неслучайно и удобно: разбор ответа сводится к валидации,
    а обратное преобразование в словарь не требует таблицы соответствий.
    """

    code: str
    product_name: str | None = None
    brands: str | None = None
    ingredients_text: str | None = None
    nutriscore_grade: str | None = None
    nova_group: int | None = None


def normalize_barcode(text: str) -> str | None:
    """Привести ввод к штрихкоду. Не похоже на штрихкод — `None`.

    Проверка до сети: и модель, и человек охотно подставляют вместо кода
    название продукта, а ходить с ним в API значит ждать таймаут ради
    заведомого отказа.
    """
    code = str(text).strip()
    return code if _BARCODE.match(code) else None


async def fetch_product(
    code: str,
    *,
    timeout_s: float,
    client: httpx.AsyncClient | None = None,
) -> OffProduct | None:
    """Спросить у Open Food Facts один продукт по штрихкоду.

    Args:
        code: штрихкод, уже прошедший `normalize_barcode`.
        timeout_s: таймаут запроса.
        client: клиент HTTP. Передаётся тестом, чтобы вызов не уходил
            в сеть (правило 4 брифа); в бою создаётся здесь.

    Returns:
        Продукт или `None`, если такого кода в OFF нет.

    Raises:
        DataSourceError: API недоступен или ответил ошибкой. Отличается
            от `None` намеренно: «нет такого продукта» и «не смогли
            спросить» — разные факты, и лечатся они разным.
    """
    owns_client = client is None
    http = client or httpx.AsyncClient()
    try:
        response = await http.get(
            API_URL.format(code=code),
            params={"fields": FIELDS},
            headers={"User-Agent": USER_AGENT},
            timeout=timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError as exc:
        logger.warning(
            "API OFF недоступен",
            extra=safe_extra(code=code, error=type(exc).__name__),
        )
        raise DataSourceError(f"API Open Food Facts недоступен: {exc}") from exc
    finally:
        if owns_client:
            await http.aclose()

    # `status` 0 означает «нет такого продукта».
    if not payload.get("status"):
        logger.info("Продукт не найден в OFF", extra=safe_extra(code=code))
        return None

    product = dict(payload.get("product") or {})
    # Код берём свой, а не из ответа: OFF иногда возвращает его в другой
    # нормализации, и карточка перестала бы совпадать с тем, что спросили.
    product["code"] = code
    logger.info("Продукт получен из OFF", extra=safe_extra(code=code))
    return OffProduct.model_validate(product)
