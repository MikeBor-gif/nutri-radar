"""Telegram-бот: сборка и запуск.

Точка входа по `ARCHITECTURE.md`: разобрать сообщение, вызвать слайс,
отформатировать ответ. Логики здесь нет.

**Long polling, а не вебхуки.** Вебхук требует публичного HTTPS-адреса,
то есть домена и сертификата, а бот должен подниматься на ноутбуке одной
командой. Переход на вебхуки — вопрос деплоя, а не архитектуры бота.

**Отсутствие токена — не падение стека.** В CI и на чужой машине токена
нет. Бот сообщает об этом понятной ошибкой конфигурации и не запускается;
всё остальное — API, база, CLI — работает.
"""

from __future__ import annotations

import asyncio
import logging

import httpx
from aiogram import Bot, Dispatcher

from nutri_radar import __version__
from nutri_radar.bot.middlewares import PrivacyLoggingMiddleware
from nutri_radar.config import Settings, get_settings
from nutri_radar.db.session import dispose_engine
from nutri_radar.errors import ConfigurationError
from nutri_radar.logging import safe_extra

logger = logging.getLogger(__name__)


def create_dispatcher(settings: Settings, client: httpx.AsyncClient) -> Dispatcher:
    """Собрать диспетчер с роутерами и общими зависимостями.

    Настройки и HTTP-клиент кладутся в данные диспетчера: aiogram передаст
    их хендлерам аргументами. Так хендлер остаётся чистой функцией от входа
    и зависимостей, а не лезет за ними в глобальное состояние.
    """
    # Импорт здесь, а не наверху: хендлеры импортируют этот модуль за
    # текстами и клавиатурами, и на уровне модуля вышел бы цикл.
    from nutri_radar.bot.handlers import ask, barcode, commands, photo

    dispatcher = Dispatcher(settings=settings, http_client=client)
    dispatcher.update.outer_middleware(PrivacyLoggingMiddleware())

    # Порядок важен: роутеры разбирают сообщение первым подошедшим фильтром.
    # Команды и штрихкоды узнаются точно, поэтому идут раньше; свободный
    # вопрос ловит всё остальное и обязан быть последним.
    dispatcher.include_router(commands.router)
    dispatcher.include_router(barcode.router)
    dispatcher.include_router(photo.router)
    dispatcher.include_router(ask.router)

    logger.debug("Диспетчер собран", extra=safe_extra(routers=4))
    return dispatcher


async def run_bot(settings: Settings | None = None) -> None:
    """Запустить бота на long polling. Возвращается только при остановке.

    Raises:
        ConfigurationError: токен не задан. Отдельная ошибка, а не падение
            по месту использования: человеку нужно знать, какой ключ
            положить в `.env`, а не читать трассировку aiogram.
    """
    settings = settings or get_settings()

    if not settings.bot.is_configured:
        raise ConfigurationError(
            "BOT__TOKEN не задан — бот не может запуститься. "
            "Получите токен у @BotFather и положите его в .env "
            "(эталонный список ключей — в .env.example)."
        )

    bot = Bot(token=settings.bot.token.get_secret_value())
    client = httpx.AsyncClient(base_url=settings.ollama.base_url)
    dispatcher = create_dispatcher(settings, client)

    logger.info(
        "Бот запускается",
        extra=safe_extra(
            version=__version__,
            environment=settings.app.environment,
            ollama=settings.ollama.base_url,
        ),
    )
    try:
        # Накопленные за время простоя апдейты отбрасываются: отвечать
        # на вопрос, заданный сутки назад, — это не забота, а неожиданность.
        await bot.delete_webhook(drop_pending_updates=True)
        await dispatcher.start_polling(bot)
    finally:
        await client.aclose()
        await bot.session.close()
        await dispose_engine()
        logger.info("Бот остановлен")


def main(settings: Settings | None = None) -> None:
    """Синхронная обёртка для точки входа CLI."""
    try:
        asyncio.run(run_bot(settings))
    except KeyboardInterrupt:  # pragma: no cover — ручная остановка
        logger.info("Остановка по Ctrl+C")
