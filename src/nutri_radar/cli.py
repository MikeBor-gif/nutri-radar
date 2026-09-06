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
from nutri_radar.agent import cli as agent_cli
from nutri_radar.analytics import cli as analytics_cli
from nutri_radar.evals import cli as evals_cli
from nutri_radar.extract import cli as extract_cli
from nutri_radar.health import CheckStatus, run_health_check
from nutri_radar.ingest import cli as ingest_cli
from nutri_radar.logging import setup_logging
from nutri_radar.retrieval import cli as retrieval_cli

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="nutri-radar",
    help="Разбор состава пищевых продуктов на данных Open Food Facts.",
    no_args_is_help=True,
    add_completion=False,
)

# Слайсы пайплайна подключаются здесь по мере появления.
app.add_typer(ingest_cli.app, name="ingest")
app.add_typer(extract_cli.app, name="extract")
app.add_typer(evals_cli.app, name="evals")
app.add_typer(analytics_cli.app, name="analytics")
app.add_typer(retrieval_cli.app, name="retrieval")
app.add_typer(agent_cli.app, name="agent")


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
        # Флаг только включает JSON, но не выключает: в контейнере режим задаёт
        # переменная JSON_LOGS, и флага в команде там нет.
        json_output=json_logs or settings.app.json_logs,
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


# Символы статусов вынесены из кода вывода: менять оформление в одном месте.
_STATUS_MARK = {
    CheckStatus.OK: "OK  ",
    CheckStatus.WARN: "WARN",
    CheckStatus.FAIL: "FAIL",
}


@app.command()
def health(
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Машиночитаемый вывод вместо таблицы.",
    ),
) -> None:
    """Проверить готовность среды: БД, расширение vector, миграции, Ollama, ключ.

    Код возврата 0, если нет ни одного FAIL (WARN не роняет), иначе 1.
    """
    report = run_health_check()

    if as_json:
        typer.echo(report.model_dump_json(indent=2))
    else:
        # Вывод для человека идёт в stdout через Typer, логи — в stderr.
        # print() в библиотечном коде запрещён, здесь это точка входа.
        width = max(len(check.name) for check in report.checks)
        for check in report.checks:
            mark = _STATUS_MARK[check.status]
            typer.echo(f"[{mark}] {check.name.ljust(width)}  {check.detail}")

        summary = (
            f"OK: {sum(1 for c in report.checks if c.status is CheckStatus.OK)}, "
            f"WARN: {len(report.warnings)}, FAIL: {len(report.failures)}"
        )
        typer.echo(f"\n{summary}")
        if not report.is_healthy:
            typer.echo("Среда не готова: устраните FAIL выше.", err=True)

    raise typer.Exit(code=report.exit_code)


if __name__ == "__main__":
    app()
