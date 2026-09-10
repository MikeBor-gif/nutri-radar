"""Фото штрихкода: прочитать код с картинки и показать карточку."""

from __future__ import annotations

import asyncio
import io
import logging

import httpx
from aiogram import Bot, F, Router
from aiogram.types import Message

from nutri_radar.bot import keyboards
from nutri_radar.bot.barcode_image import decode_barcode
from nutri_radar.bot.errors import log_failure, user_message
from nutri_radar.bot.texts import (
    BARCODE_NOT_RECOGNISED,
    NOT_FOUND,
    PHOTO_TOO_LARGE,
    READING_PHOTO,
)
from nutri_radar.config import Settings
from nutri_radar.errors import NutriRadarError
from nutri_radar.logging import safe_extra
from nutri_radar.retrieval.product_card import format_card, load_card

logger = logging.getLogger(__name__)

router = Router(name="photo")

_MEGABYTE = 1024 * 1024


@router.message(F.photo)
async def by_photo(
    message: Message, bot: Bot, settings: Settings, http_client: httpx.AsyncClient
) -> None:
    """Прочитать штрихкод с присланного фото.

    Telegram отдаёт несколько размеров одного снимка. Берём самый большой:
    штрихкод — мелкая деталь, и на превью его полосы сливаются. Предел
    размера при этом остаётся: он ограничивает память, а не качество.
    """
    if not message.photo:  # pragma: no cover — фильтр уже проверил
        return

    largest = message.photo[-1]
    if (largest.file_size or 0) > settings.bot.max_photo_bytes:
        await message.answer(
            PHOTO_TOO_LARGE.format(
                size_mb=(largest.file_size or 0) / _MEGABYTE,
                limit_mb=settings.bot.max_photo_bytes / _MEGABYTE,
            )
        )
        return

    placeholder = await message.answer(READING_PHOTO)

    # Файл живёт только в памяти: на диск он не попадает ни во временный
    # каталог, ни в кэш. Фотография этикетки — это в том числе место
    # и время покупки (бриф: данные не хранятся дольше ответа).
    buffer = io.BytesIO()
    await bot.download(largest, destination=buffer)
    code = decode_barcode(buffer.getvalue())
    buffer.close()

    if code is None:
        await placeholder.edit_text(BARCODE_NOT_RECOGNISED)
        return

    try:
        card = await asyncio.wait_for(
            load_card(code, settings=settings, client=http_client),
            timeout=settings.bot.request_timeout_s,
        )
    except (NutriRadarError, TimeoutError) as exc:
        log_failure(exc, where="by_photo")
        await placeholder.edit_text(user_message(exc))
        return

    if card is None:
        await placeholder.edit_text(NOT_FOUND.format(code=code))
        return

    logger.info(
        "Карточка по фото отправлена",
        extra=safe_extra(source=card.source.value, has_extraction=card.has_extraction),
    )
    await placeholder.edit_text(
        format_card(card),
        reply_markup=keyboards.card_keyboard(card.code, has_extraction=card.has_extraction),
    )
