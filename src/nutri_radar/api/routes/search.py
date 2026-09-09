"""Семантический поиск по составам продуктов."""

from __future__ import annotations

import logging
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends

from nutri_radar.api.app import get_app_settings, get_http_client
from nutri_radar.api.schemas import SearchRequest, SearchResponse
from nutri_radar.config import Settings
from nutri_radar.logging import safe_extra
from nutri_radar.retrieval.pipeline import search_by_text
from nutri_radar.retrieval.search import SearchFilters

logger = logging.getLogger(__name__)

router = APIRouter(tags=["поиск"])


@router.post("/search", response_model=SearchResponse, summary="Найти продукты по смыслу")
async def search(
    request: SearchRequest,
    settings: Annotated[Settings, Depends(get_app_settings)],
    client: Annotated[httpx.AsyncClient, Depends(get_http_client)],
) -> SearchResponse:
    """Найти продукты, похожие на запрос по смыслу состава.

    Фильтры живут в SQL, а не в тексте запроса: по оценке и группе NOVA
    ищут числом, а не смыслом, и подмешивать их в эмбеддинг значило бы
    получить выдачу по совпадению оценки.
    """
    filters = SearchFilters(
        lang=request.lang,
        category=request.category,
        grade_in=tuple(request.grade_in),
        nova_in=tuple(request.nova_in),
    )

    result = await search_by_text(
        request.query,
        client=client,
        settings=settings,
        limit=request.limit,
        filters=filters,
    )

    logger.info(
        "Поиск отдан",
        extra=safe_extra(
            found=len(result.hits),
            filters=filters.describe(),
            latency_s=round(result.latency_s, 3),
        ),
    )
    return SearchResponse.from_result(result)
