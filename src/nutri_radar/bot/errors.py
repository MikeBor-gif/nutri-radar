"""Доменные ошибки в сообщения пользователю.

То же, что `api/errors.py`, но для чата: пользователю нужен не код ответа,
а понятная фраза и понимание, стоит ли повторить. «Источник недоступен» и
«ничего не нашлось» — разные вещи, и смешивать их нельзя: во втором случае
человек сделает вывод о продукте, которого делать нельзя.
"""

from __future__ import annotations

import logging

from nutri_radar.bot.texts import MODEL_UNAVAILABLE, SOURCE_UNAVAILABLE, TIMEOUT
from nutri_radar.errors import (
    DatabaseError,
    DataSourceError,
    ExtractionError,
    LLMUnavailableError,
    NutriRadarError,
)
from nutri_radar.logging import safe_extra

logger = logging.getLogger(__name__)


def user_message(error: BaseException) -> str:
    """Что сказать пользователю про этот отказ."""
    match error:
        case TimeoutError():
            return TIMEOUT
        case LLMUnavailableError() | ExtractionError():
            return MODEL_UNAVAILABLE
        case DataSourceError() | DatabaseError():
            return SOURCE_UNAVAILABLE
        case NutriRadarError():
            return SOURCE_UNAVAILABLE
        case _:
            # Наружу это не выходит: неизвестные ошибки не глотаются
            # хендлером, а поднимаются дальше и попадают в лог целиком.
            raise error


def log_failure(error: BaseException, *, where: str) -> None:
    """Записать отказ без содержания сообщения пользователя."""
    logger.warning(
        "Ответ пользователю заменён на сообщение об отказе",
        extra=safe_extra(where=where, error=type(error).__name__),
    )
