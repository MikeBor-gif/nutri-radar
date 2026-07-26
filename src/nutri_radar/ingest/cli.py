"""CLI слайса ingestion.

Точка входа тонкая: разбирает аргументы, вызывает слой ниже, форматирует вывод.
Доменной логики здесь нет (см. ARCHITECTURE.md).
"""

from __future__ import annotations

import logging

import typer

from nutri_radar.config import get_settings
from nutri_radar.ingest.download import download_dump
from nutri_radar.ingest.probe import format_report, probe_schema

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="ingest",
    help="Загрузка данных Open Food Facts: дамп, выборка корпуса, дельты.",
    no_args_is_help=True,
)


@app.command()
def dump(
    force: bool = typer.Option(
        False,
        "--force",
        help="Перекачать заново, игнорируя существующий файл.",
    ),
) -> None:
    """Скачать Parquet-дамп Open Food Facts (7,7 ГБ) идемпотентно.

    Повторный запуск не качает заново, если файл уже на месте и размер совпадает.
    Оборванная загрузка продолжается с места обрыва.
    """
    result = download_dump(get_settings().ingest, force=force)

    if result.skipped:
        typer.echo(f"Дамп уже на месте: {result.path} ({result.size / 1024**3:.2f} ГБ)")
    else:
        typer.echo(
            f"Скачано: {result.path} ({result.size / 1024**3:.2f} ГБ) "
            f"за {result.elapsed_s / 60:.1f} мин"
        )
    typer.echo(f"Версия дампа: {result.dump_version}")


@app.command()
def probe(
    local: bool = typer.Option(
        False,
        "--local",
        help="Читать скачанный дамп вместо URL. Быстрее в разы — после `ingest dump`.",
    ),
    rows: int | None = typer.Option(
        None,
        "--rows",
        help="Размер выборки. По умолчанию INGEST__PROBE_SAMPLE_ROWS.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Машиночитаемый вывод."),
) -> None:
    """Разведать схему дампа: языки состава, имена нутриентов, версия схемы.

    Отвечает на вопросы, которые нельзя решить по документации: `data-fields.txt`
    описывает CSV-экспорт, а у Parquet другая схема с вложенными структурами.

    Без --local читает дамп по сети: работает до скачивания 7,7 ГБ, но каждый
    запрос занимает минуты.
    """
    settings = get_settings().ingest
    report = probe_schema(settings, local=local, rows=rows)

    if as_json:
        typer.echo(report.model_dump_json(indent=2))
    else:
        typer.echo(format_report(report))

    # Ненайденный нужный нутриент — повод для ненулевого кода возврата:
    # дальше проектировать схему нельзя, имена не подтверждены.
    if report.missing_required_nutrients:
        raise typer.Exit(code=1)
