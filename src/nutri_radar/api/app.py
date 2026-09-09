"""HTTP-API: сборка приложения.

Точка входа по `ARCHITECTURE.md`: разобрать запрос, вызвать слайс,
отформатировать ответ. Логики здесь нет и быть не должно — если в роуте
появилось ветвление по домену, оно уехало не туда.

**Общий клиент на процесс.** `httpx.AsyncClient` создаётся один раз
в `lifespan`, а не на запрос: клиент держит пул соединений, и создание его
на каждый запрос означало бы новое TCP-соединение к Ollama каждый раз.
Пул закрывается на остановке вместе с пулом Postgres.

**Атрибуция ODbL** попадает в описание OpenAPI и в корневой ответ. Лицензия
требует указывать источник, и в HTTP-API видное место — это описание схемы,
которое читает всякий, кто открыл `/docs`.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from nutri_radar import __version__
from nutri_radar.api.errors import register_error_handlers
from nutri_radar.config import Settings, get_settings
from nutri_radar.db.session import dispose_engine
from nutri_radar.logging import safe_extra
from nutri_radar.wording import ATTRIBUTION, DISCLAIMER

logger = logging.getLogger(__name__)

# Заголовок с идентификатором запроса. Без него разобрать параллельные
# запросы в логе нельзя: строки перемешаны, и понять, какая относится
# к жалобе пользователя, невозможно.
REQUEST_ID_HEADER = "X-Request-ID"

DESCRIPTION = f"""
Разбор состава пищевых продуктов: семантический поиск по составам,
ответы со ссылками на штрихкоды и карточка продукта с числом разных форм
сахара — величиной, которой в Open Food Facts нет.

{ATTRIBUTION}

{DISCLAIMER}
"""


def _client_holder(app: FastAPI) -> httpx.AsyncClient:
    """Достать общий HTTP-клиент. Отсутствие клиента — ошибка сборки."""
    client: httpx.AsyncClient | None = getattr(app.state, "http_client", None)
    if client is None:  # pragma: no cover — не достигается при живом lifespan
        raise RuntimeError("HTTP-клиент не создан: приложение собрано без lifespan")
    return client


def get_http_client(request: Request) -> httpx.AsyncClient:
    """Зависимость FastAPI: общий клиент к Ollama."""
    return _client_holder(request.app)


def get_app_settings(request: Request) -> Settings:
    """Зависимость FastAPI: настройки приложения.

    Берутся из состояния приложения, а не через `get_settings()` в каждом
    роуте: так тест может собрать приложение с другими настройками, не
    трогая переменные окружения.
    """
    return request.app.state.settings  # type: ignore[no-any-return]


def create_app(settings: Settings | None = None) -> FastAPI:
    """Собрать приложение. Composition root точки входа."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.http_client = httpx.AsyncClient(base_url=settings.ollama.base_url)
        logger.info(
            "API запущен",
            extra=safe_extra(
                version=__version__,
                environment=settings.app.environment,
                bind=f"{settings.api.host}:{settings.api.port}",
                ollama=settings.ollama.base_url,
            ),
        )
        try:
            yield
        finally:
            await app.state.http_client.aclose()
            # Пул Postgres закрывается здесь же: иначе при остановке
            # контейнера соединения повиснут до таймаута сервера.
            await dispose_engine()
            logger.info("API остановлен")

    app = FastAPI(
        title="Nutri Radar",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    app.state.settings = settings

    if settings.api.cors_origins:
        # Включается только когда происхождения заданы явно. Разрешать «*»
        # по умолчанию — дыра без потребителя: веб-фронтенда у проекта нет.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.api.cors_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )
        logger.info(
            "CORS включён",
            extra=safe_extra(origins=len(settings.api.cors_origins)),
        )

    @app.middleware("http")
    async def access_log(request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
        """Идентификатор запроса и запись о каждом обращении.

        Тело запроса в лог не пишется намеренно: в нём вопрос пользователя,
        а пользовательские данные проект не хранит дольше, чем нужно для
        ответа (раздел «Границы продукта» брифа).
        """
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:12]
        request.state.request_id = request_id

        started = time.perf_counter()
        logger.debug(
            "Запрос принят",
            extra=safe_extra(request_id=request_id, method=request.method, path=request.url.path),
        )
        response = await call_next(request)
        elapsed = time.perf_counter() - started

        response.headers[REQUEST_ID_HEADER] = request_id
        logger.info(
            "Запрос обработан",
            extra=safe_extra(
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                latency_s=round(elapsed, 3),
            ),
        )
        return response

    register_error_handlers(app)

    # Импорт роутеров здесь, а не на уровне модуля: они зависят от
    # зависимостей, объявленных выше, и циклический импорт иначе неизбежен.
    from nutri_radar.api.routes import agent, ask, products, search

    app.include_router(products.router)
    app.include_router(search.router)
    app.include_router(ask.router)
    app.include_router(agent.router)

    logger.debug("Приложение собрано", extra=safe_extra(routes=len(app.routes)))
    return app
