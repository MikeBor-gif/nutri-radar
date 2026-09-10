"""Корень, готовность среды и карточка продукта по штрихкоду."""

from __future__ import annotations

import logging
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException, Path, Response

from nutri_radar import __version__
from nutri_radar.api.app import get_app_settings, get_http_client
from nutri_radar.api.schemas import (
    HealthCheck,
    HealthResponse,
    ProductCardResponse,
    ServiceInfo,
)
from nutri_radar.config import Settings
from nutri_radar.health import check_health
from nutri_radar.logging import safe_extra
from nutri_radar.openfoodfacts import normalize_barcode
from nutri_radar.retrieval.product_card import load_card

logger = logging.getLogger(__name__)

router = APIRouter(tags=["продукты"])


@router.get("/", response_model=ServiceInfo, summary="Что это за сервис")
async def root(settings: Annotated[Settings, Depends(get_app_settings)]) -> ServiceInfo:
    """Описание сервиса вместе с атрибуцией и дисклеймером.

    Атрибуция лежит здесь, а не только в подвале документации: лицензия
    ODbL требует указывать источник, а корень — первое, что открывают.
    """
    return ServiceInfo(
        name="Nutri Radar",
        version=__version__,
        description=(
            "Разбор состава пищевых продуктов: поиск по смыслу, ответы "
            "со ссылками на штрихкоды и число разных форм сахара в составе."
        ),
    )


@router.get("/healthz", response_model=HealthResponse, summary="Готовность среды")
async def healthz(
    response: Response,
    settings: Annotated[Settings, Depends(get_app_settings)],
    client: Annotated[httpx.AsyncClient, Depends(get_http_client)],
) -> HealthResponse:
    """Проверить БД, расширение vector, миграции, Ollama и ключ Anthropic.

    Код 503 при любом FAIL. WARN не роняет проверку: отсутствие ключа
    Anthropic — штатное состояние проекта, а не поломка среды.
    """
    report = await check_health(settings, http_client=client)
    if not report.is_healthy:
        # Оркестратору нужен код, а не разбор тела ответа: 200 с «healthy:
        # false» внутри означал бы, что сервис считается живым.
        response.status_code = 503
        logger.warning(
            "Среда не готова",
            extra=safe_extra(failures=[c.name for c in report.failures]),
        )

    return HealthResponse(
        healthy=report.is_healthy,
        checks=[
            HealthCheck(name=check.name, status=check.status.value, detail=check.detail)
            for check in report.checks
        ],
    )


@router.get(
    "/products/{barcode}",
    response_model=ProductCardResponse,
    summary="Карточка продукта по штрихкоду",
)
async def product_by_barcode(
    settings: Annotated[Settings, Depends(get_app_settings)],
    client: Annotated[httpx.AsyncClient, Depends(get_http_client)],
    barcode: Annotated[str, Path(min_length=6, max_length=14)],
) -> ProductCardResponse:
    """Отдать карточку: сначала корпус проекта, затем живой API OFF.

    Число разных форм сахара есть только у продуктов, прошедших через
    извлечение. Для остальных поле остаётся пустым — это честнее нуля,
    который читался бы как «сахара не нашли».
    """
    if normalize_barcode(barcode) is None:
        # 400, а не 404: «это не штрихкод» и «такого штрихкода нет» —
        # разные ответы, и лечатся они по-разному.
        raise HTTPException(status_code=400, detail="Штрихкод — это 6-14 цифр без пробелов.")

    # Живой API OFF дёргается одним запросом на один штрихкод: массовые
    # данные проект берёт дампом, и это ограничение брифа.
    card = await load_card(barcode, settings=settings, client=client)
    if card is None:
        raise HTTPException(
            status_code=404,
            detail=f"Продукт [{barcode}] не найден ни в корпусе проекта, ни в Open Food Facts.",
        )

    logger.info(
        "Карточка отдана",
        extra=safe_extra(
            code=card.code,
            source=card.source.value,
            has_extraction=card.has_extraction,
        ),
    )
    return ProductCardResponse.from_card(card)
