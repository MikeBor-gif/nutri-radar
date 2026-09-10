"""Доменные ошибки в коды HTTP.

**Зачем отдельный модуль.** Без него любой отказ ниже по стеку — упавшая
Ollama, недоступный Postgres, молчащий Open Food Facts — приходит клиенту
как 500 с трассировкой в логе. Клиент не может отличить «сломался сервис»
от «сломался источник», а по 500 нельзя решить, имеет ли смысл повторить
запрос.

Соответствие построено по одному правилу: **чей отказ**. Наш — 500,
внешней системы, от которой мы зависим, — 503 (попробуйте позже),
внешнего источника данных — 502 (мы спросили, нам ответили плохо).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from nutri_radar.errors import (
    ConfigurationError,
    DatabaseError,
    DataSourceError,
    ExtractionError,
    LLMUnavailableError,
    NutriRadarError,
)
from nutri_radar.logging import safe_extra

logger = logging.getLogger(__name__)

# `NutriRadarError` — предок всех остальных, и он работает как запасной
# вариант для доменных ошибок, которым не назначили свой код. Порядок
# записей здесь роли не играет: Starlette ищет обработчик по MRO самого
# исключения и берёт самый специфичный из зарегистрированных.
ERROR_STATUS: tuple[tuple[type[NutriRadarError], int, str], ...] = (
    (
        DatabaseError,
        503,
        "База данных недоступна. Проверьте, поднят ли Postgres.",
    ),
    (
        LLMUnavailableError,
        503,
        "Языковая модель недоступна. Проверьте, запущена ли Ollama.",
    ),
    (
        ExtractionError,
        502,
        "Модель ответила, но ответ непригоден. Повтор не поможет.",
    ),
    (
        DataSourceError,
        502,
        "Внешний источник данных недоступен или ответил ошибкой.",
    ),
    (
        ConfigurationError,
        500,
        "Сервис сконфигурирован неверно. Смотрите .env.example.",
    ),
    (
        NutriRadarError,
        500,
        "Внутренняя ошибка сервиса.",
    ),
)


def _handler(status: int, hint: str) -> Callable[[Request, Exception], Awaitable[JSONResponse]]:
    async def handle(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "—")
        # Уровень зависит от того, чей отказ. 500 — наш код или наша
        # конфигурация, это ERROR и повод чинить. 502 и 503 — внешняя
        # система, это WARNING и повод смотреть на неё, а не на нас.
        # Одинаковый уровень для обоих случаев превратил бы лог в шум,
        # в котором настоящая поломка не видна.
        log = logger.error if status == 500 else logger.warning
        log(
            "Запрос завершился доменной ошибкой",
            extra=safe_extra(
                request_id=request_id,
                status=status,
                error=type(exc).__name__,
                path=request.url.path,
            ),
        )
        return JSONResponse(
            status_code=status,
            content={
                "error": type(exc).__name__,
                # Сообщение доменной ошибки написано для человека и секретов
                # не содержит: они вычищаются на уровне настроек, а сюда
                # приходит уже безопасный текст.
                "detail": str(exc),
                "hint": hint,
                "request_id": request_id,
            },
        )

    return handle


def register_error_handlers(app: FastAPI) -> None:
    """Повесить обработчики доменных ошибок на приложение."""
    for error_type, status, hint in ERROR_STATUS:
        app.add_exception_handler(error_type, _handler(status, hint))
    logger.debug(
        "Обработчики ошибок зарегистрированы",
        extra=safe_extra(count=len(ERROR_STATUS)),
    )
