"""Команды `/start` и `/help`.

`/start` несёт атрибуцию ODbL и дисклеймер о краудсорсинговых данных.
Это прямое требование брифа: **в первом сообщении, а не в подвале.**
Проверяется тестом — требование обязано ломать сборку, а не ждать вычитки.
"""

from __future__ import annotations

import logging

from aiogram import Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from nutri_radar.bot.texts import HELP, START

logger = logging.getLogger(__name__)

router = Router(name="commands")


@router.message(CommandStart())
async def start(message: Message) -> None:
    """Первое сообщение: что это, что умеет, откуда данные."""
    logger.debug("Команда start")
    await message.answer(START)


@router.message(Command("help"))
async def help_command(message: Message) -> None:
    """Справка."""
    logger.debug("Команда help")
    await message.answer(HELP)
