"""Команды запуска витрины: `serve api`, `serve bot`, `serve mcp`.

Отдельный модуль, а не три команды в `cli.py`: каждая точка входа
собирается по-своему, и держать их сборку в корневом CLI значило бы,
что корень знает про FastAPI, aiogram и MCP сразу.

Команды тонкие: разобрать флаги, собрать точку входа, передать ей
управление. Логики нет.
"""

from __future__ import annotations

import logging

import typer
import uvicorn

from nutri_radar import __version__
from nutri_radar.config import get_settings
from nutri_radar.errors import ConfigurationError
from nutri_radar.logging import safe_extra

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="serve",
    help="Запустить точку входа: HTTP-API, Telegram-бот или MCP-сервер.",
    no_args_is_help=True,
)


@app.command()
def api(
    host: str = typer.Option("", "--host", help="Переопределить API__HOST."),
    port: int = typer.Option(0, "--port", help="Переопределить API__PORT."),
    reload: bool = typer.Option(False, "--reload", help="Перезапуск при правке кода."),
) -> None:
    """Запустить HTTP-API на uvicorn."""
    settings = get_settings()
    bind_host = host or settings.api.host
    bind_port = port or settings.api.port

    logger.info(
        "Запуск HTTP-API",
        extra=safe_extra(version=__version__, host=bind_host, port=bind_port, reload=reload),
    )
    typer.echo(f"Nutri Radar API -> http://{bind_host}:{bind_port}/docs")

    # Строка импорта, а не объект: без неё uvicorn не умеет --reload,
    # потому что перезагрузка требует повторного импорта модуля.
    uvicorn.run(
        "nutri_radar.api.factory:app",
        host=bind_host,
        port=bind_port,
        reload=reload,
        # Логи настраивает проект, а не uvicorn: иначе рядом окажутся два
        # формата и два уровня, и `LOG_LEVEL` перестанет что-либо значить.
        log_config=None,
    )


@app.command()
def bot() -> None:
    """Запустить Telegram-бота на long polling."""
    from nutri_radar.bot import main as run_bot

    settings = get_settings()
    if not settings.bot.is_configured:
        # Понятная ошибка вместо трассировки aiogram: человеку нужно знать,
        # какой ключ положить в .env.
        typer.echo(
            "BOT__TOKEN не задан. Получите токен у @BotFather и положите "
            "его в .env — эталонный список ключей в .env.example.",
            err=True,
        )
        raise typer.Exit(code=2)

    typer.echo("Telegram-бот запущен. Остановка — Ctrl+C.")
    try:
        run_bot(settings)
    except ConfigurationError as exc:  # pragma: no cover — проверено выше
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc


@app.command()
def mcp() -> None:
    """Запустить MCP-сервер на stdio.

    Вывод для человека здесь запрещён: stdout занят протоколом, и любая
    посторонняя строка ломает сессию клиента. Диагностика идёт в stderr
    через обычное логирование.
    """
    from nutri_radar.mcp_server import main as run_mcp

    logger.info("Запуск MCP-сервера на stdio")
    run_mcp(get_settings())
