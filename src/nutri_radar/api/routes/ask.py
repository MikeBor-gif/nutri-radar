"""Ответ на вопрос строго по найденным продуктам (RAG)."""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException

from nutri_radar.api.app import get_app_settings, get_http_client
from nutri_radar.api.schemas import AskRequest, AskResponse
from nutri_radar.config import Settings
from nutri_radar.logging import safe_extra
from nutri_radar.retrieval.pipeline import ask as pipeline_ask
from nutri_radar.retrieval.search import SearchFilters

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ответы"])


@router.post("/ask", response_model=AskResponse, summary="Ответ со ссылками на штрихкоды")
async def ask(
    request: AskRequest,
    settings: Annotated[Settings, Depends(get_app_settings)],
    client: Annotated[httpx.AsyncClient, Depends(get_http_client)],
) -> AskResponse:
    """Ответить на вопрос по продуктам, найденным в базе.

    Если релевантного не нашлось, ответ — честное «не знаю». Отказ
    формируется кодом, а не просьбой в промпте: модель об этом даже
    не спрашивается.
    """
    filters = SearchFilters(lang=request.lang)

    try:
        outcome = await asyncio.wait_for(
            pipeline_ask(
                request.question,
                client=client,
                settings=settings,
                limit=request.limit,
                filters=filters,
            ),
            timeout=settings.api.request_timeout_s,
        )
    except TimeoutError as exc:
        # Висящий запрос хуже отказа: клиент не знает, ждать ему или нет.
        # 504, а не 503: сервис жив, не уложился конкретный запрос.
        logger.warning(
            "Ответ не уложился в таймаут",
            extra=safe_extra(timeout_s=settings.api.request_timeout_s),
        )
        raise HTTPException(
            status_code=504,
            detail=(
                f"Ответ не уложился в {settings.api.request_timeout_s:.0f} с. "
                "Обычно это значит, что Ollama занята другой моделью."
            ),
        ) from exc

    answer = outcome.answer
    if answer.cited_outside_sources:
        # Не прячем и не исправляем: выдуманный штрихкод — факт о работе
        # системы, и клиент должен его увидеть.
        logger.warning(
            "Модель сослалась на штрихкоды вне выдачи",
            extra=safe_extra(invented=answer.cited_outside_sources),
        )

    logger.info(
        "Ответ отдан",
        extra=safe_extra(
            refused=answer.refused,
            sources=len(answer.sources),
            tokens=answer.input_tokens + answer.output_tokens,
        ),
    )
    return AskResponse.from_answer(answer, retrieval_latency_s=outcome.retrieval_latency_s)
