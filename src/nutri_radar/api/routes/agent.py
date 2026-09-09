"""Прогон агента с инструментами по одному вопросу.

**Отдельный таймаут.** На `qwen2.5:3b` цикл агента доходил до восьми шагов
и минут работы, а до ответа добирался в 20% случаев (M6, ADR-030). Держать
для него общий таймаут запроса значило бы задрать общий до агентского —
и тогда он перестал бы ловить зависший поиск, ради чего и существует.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException

from nutri_radar.agent.loop import AgentRun, run_agent
from nutri_radar.agent.registry import build_registry
from nutri_radar.api.app import get_app_settings, get_http_client
from nutri_radar.api.schemas import AgentRequest, AgentResponse, AgentStep
from nutri_radar.config import Settings
from nutri_radar.llm.adapters import OllamaLLM
from nutri_radar.llm.runtime import get_runtime
from nutri_radar.logging import safe_extra
from nutri_radar.tracing import get_tracer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["агент"])


def _to_response(run: AgentRun) -> AgentResponse:
    return AgentResponse(
        question=run.question,
        answer=run.answer or None,
        stop_reason=run.stop_reason,
        steps=[
            AgentStep(
                number=step.number,
                action=step.action,
                arguments=dict(step.arguments),
                ok=step.result.ok if step.result else None,
                is_final=step.is_final,
            )
            for step in run.steps
        ],
        tool_calls=run.tool_calls,
        failed_calls=run.failed_calls,
        repeated_calls=run.repeated_calls,
        total_tokens=run.total_tokens,
        latency_s=round(run.latency_s, 2),
        model_name=run.model_name,
    )


@router.post("/ask", response_model=AgentResponse, summary="Спросить агента с инструментами")
async def agent_ask(
    request: AgentRequest,
    settings: Annotated[Settings, Depends(get_app_settings)],
    client: Annotated[httpx.AsyncClient, Depends(get_http_client)],
) -> AgentResponse:
    """Прогнать агента и вернуть весь протокол, а не только ответ.

    Пустой ответ при `stop_reason != "ответ"` — нормальный исход: агент
    не обязан справляться, и скрывать это было бы враньём о качестве.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Вопрос не может быть пустым.")

    tools = build_registry(settings)
    llm = OllamaLLM(client, settings.ollama)
    tracer = get_tracer(settings)

    async def go() -> AgentRun:
        # Очередь моделей: агент занимает GPU надолго, и запрос к поиску,
        # пришедший параллельно, не должен вытеснить его модель.
        async with get_runtime(settings).hold(llm.model_name):
            return await run_agent(question, llm, tools, settings, tracer=tracer)

    try:
        run = await asyncio.wait_for(go(), timeout=settings.api.agent_timeout_s)
    except TimeoutError as exc:
        logger.warning(
            "Агент не уложился в таймаут",
            extra=safe_extra(timeout_s=settings.api.agent_timeout_s),
        )
        raise HTTPException(
            status_code=504,
            detail=(
                f"Агент не уложился в {settings.api.agent_timeout_s:.0f} с. "
                "Это штатный исход на локальной модели, а не поломка."
            ),
        ) from exc

    logger.info(
        "Агент отработал",
        extra=safe_extra(
            stop_reason=run.stop_reason,
            steps=len(run.steps),
            tool_calls=run.tool_calls,
            repeated=run.repeated_calls,
            tokens=run.total_tokens,
        ),
    )
    return _to_response(run)
