"""Свободный вопрос: ответ по базе со ссылками на штрихкоды.

Ловит всё, что не оказалось командой, штрихкодом или фото, — поэтому
роутер подключается последним.
"""

from __future__ import annotations

import asyncio
import logging

import httpx
from aiogram import F, Router
from aiogram.types import Message

from nutri_radar.bot.errors import log_failure, user_message
from nutri_radar.bot.texts import NOT_A_BARCODE, THINKING
from nutri_radar.config import Settings
from nutri_radar.errors import NutriRadarError
from nutri_radar.logging import safe_extra
from nutri_radar.retrieval.pipeline import ask as pipeline_ask

logger = logging.getLogger(__name__)

router = Router(name="ask")

# Короче этого вопрос не бывает. Отсекается до модели: гнать «??» через
# эмбеддинг и генерацию — это секунды работы GPU ради заведомого мусора.
MIN_QUESTION_CHARS = 4


@router.message(F.text)
async def ask(message: Message, settings: Settings, http_client: httpx.AsyncClient) -> None:
    """Ответить на вопрос по продуктам из базы."""
    question = (message.text or "").strip()
    if len(question) < MIN_QUESTION_CHARS:
        await message.answer(NOT_A_BARCODE)
        return

    placeholder = await message.answer(THINKING)
    try:
        outcome = await asyncio.wait_for(
            pipeline_ask(question, client=http_client, settings=settings),
            timeout=settings.bot.request_timeout_s,
        )
    except (NutriRadarError, TimeoutError) as exc:
        log_failure(exc, where="ask")
        await placeholder.edit_text(user_message(exc))
        return

    answer = outcome.answer
    if answer.cited_outside_sources:
        # Не прячем и не правим: выдуманный штрихкод — факт о работе
        # системы. Пользователю он виден как ссылка, которая никуда
        # не ведёт, и это лучше, чем тихо вычищенный текст.
        logger.warning(
            "Модель сослалась на штрихкоды вне выдачи",
            extra=safe_extra(invented=answer.cited_outside_sources),
        )

    text = answer.text
    # Условие по `cited`, а не по `sources`: выдача может быть непустой,
    # а модель — не сослаться ни на один штрихкод. Приписка «пришлите любой
    # штрихкод из ответа» под таким ответом отправляет человека искать то,
    # чего в тексте нет. Поймано на живом прогоне бота.
    if not answer.refused and answer.cited:
        text += "\n\nПришлите любой штрихкод из ответа, чтобы увидеть карточку продукта."

    logger.info(
        "Ответ отправлен",
        extra=safe_extra(
            refused=answer.refused,
            sources=len(answer.sources),
            cited=len(answer.cited),
            tokens=answer.input_tokens + answer.output_tokens,
        ),
    )
    await placeholder.edit_text(text)
