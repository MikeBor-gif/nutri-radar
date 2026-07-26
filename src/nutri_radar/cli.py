"""Корень CLI. Точка входа `nutri-radar`.

Слайсы пайплайна регистрируют здесь свои под-приложения по мере появления:
`ingest` — M1, `extract` — M2, `evals` — M3, `analytics` — M4,
`retrieval` — M5, `agent` — M6.

CLI — тонкая точка входа: разбирает аргументы, вызывает слой ниже,
форматирует вывод. Доменной логики здесь нет (см. ARCHITECTURE.md).
"""

from __future__ import annotations

import logging

import typer

from nutri_radar import __version__
from nutri_radar.logging import setup_logging

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="nutri-radar",
    help="Разбор состава пищевых продуктов на данных Open Food Facts.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def main(
    log_level: str = typer.Option(
        "",
        "--log-level",
        help="Переопределить LOG_LEVEL из конфига (DEBUG, INFO, WARNING, ERROR).",
    ),
    json_logs: bool = typer.Option(
        False,
        "--json-logs",
        help="Логи одной JSON-строкой на запись — для контейнеров и CI.",
    ),
) -> None:
    """Общая настройка для всех команд."""
    # Конфиг читается здесь, а не на уровне модуля: иначе `--help` требовал бы
    # заполненного .env, и CLI нельзя было бы даже посмотреть без настройки.
    from nutri_radar.config import get_settings

    settings = get_settings()
    setup_logging(
        level=log_level or settings.app.log_level,
        json_output=json_logs,
        secrets=settings.secret_values(),
    )
    logger.debug(
        "CLI запущен",
        extra={"version": __version__, "environment": settings.app.environment},
    )


@app.command()
def version() -> None:
    """Показать версию."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
