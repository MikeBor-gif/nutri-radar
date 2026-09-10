"""Приватность и наблюдаемость апдейтов Telegram.

**Требование брифа: пользовательские данные не хранятся дольше, чем нужно
для ответа.** Лог — это хранение. Строка «пользователь 123456789 спросил
про диабет» живёт в файле месяцами и переживает всё, что бот честно забыл
сразу после ответа.

Поэтому в лог не попадает ни текст сообщения, ни идентификатор чата
в открытом виде. Вместо идентификатора пишется короткий хэш: он позволяет
связать несколько строк одного диалога при разборе инцидента и не позволяет
узнать, кто это был. Соли нет намеренно — она потребовала бы хранить саму
соль, то есть ещё одну вещь, которую надо защищать; для несвязываемости
между перезапусками этого достаточно, а от подбора по списку известных
id хэш всё равно не защищает. Это осознанный размен, а не недосмотр.

Из содержания сообщения в лог уходит только длина и тип: этого хватает,
чтобы понять, что бот получил и сколько времени потратил.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware
from aiogram.types import CallbackQuery, Message, TelegramObject, Update

from nutri_radar.logging import safe_extra

logger = logging.getLogger(__name__)

# Четыре байта: достаточно, чтобы различить диалоги в одном логе, и мало,
# чтобы значение считалось идентификатором личности.
_DIGEST_SIZE = 4


def chat_fingerprint(chat_id: int | None) -> str:
    """Короткий отпечаток чата для логов. Обратно не разворачивается."""
    if chat_id is None:
        return "—"
    return hashlib.blake2s(str(chat_id).encode(), digest_size=_DIGEST_SIZE).hexdigest()


def _describe(event: TelegramObject) -> dict[str, Any]:
    """Что можно сказать об апдейте, не раскрывая его содержания.

    Middleware висит на уровне `update`, чтобы охватить и сообщения,
    и нажатия кнопок, — поэтому сюда приходит `Update`, а не `Message`.
    Разворачивать его надо здесь: без этого описание любого апдейта
    выродилось бы в бесполезное «kind: Update».
    """
    if isinstance(event, Update):
        event = event.message or event.callback_query or event

    if isinstance(event, CallbackQuery):
        # Данные кнопки — это действие и штрихкод, то есть наш собственный
        # формат, а не ввод пользователя. Действие писать можно, код — нет.
        action, _, _code = (event.data or "").partition(":")
        return {"kind": f"callback:{action or 'без данных'}"}

    if not isinstance(event, Message):
        return {"kind": type(event).__name__}

    if event.photo:
        kind = "photo"
    elif event.text and event.text.startswith("/"):
        # Команда — не пользовательские данные: множество команд известно
        # заранее и ничего о человеке не сообщает.
        kind = f"command:{event.text.split()[0]}"
    elif event.text:
        kind = "text"
    else:
        kind = "other"

    return {
        "kind": kind,
        # Длина, а не текст. Она объясняет латентность и объём работы,
        # но не содержит вопроса пользователя.
        "chars": len(event.text or ""),
    }


class PrivacyLoggingMiddleware(BaseMiddleware):
    """Пишет в лог факт обработки апдейта и ничего сверх того."""

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        chat = data.get("event_chat")
        fingerprint = chat_fingerprint(getattr(chat, "id", None))
        described = _describe(event)

        logger.debug("Апдейт принят", extra=safe_extra(chat=fingerprint, **described))
        started = time.perf_counter()
        try:
            return await handler(event, data)
        except Exception as exc:
            # Тип ошибки и тип апдейта — да, содержание — нет. Иначе
            # разбор падения превращается в чтение переписки.
            logger.error(
                "Хендлер упал",
                extra=safe_extra(
                    chat=fingerprint,
                    error=type(exc).__name__,
                    **described,
                ),
                exc_info=True,
            )
            raise
        finally:
            logger.info(
                "Апдейт обработан",
                extra=safe_extra(
                    chat=fingerprint,
                    latency_s=round(time.perf_counter() - started, 3),
                    **described,
                ),
            )
