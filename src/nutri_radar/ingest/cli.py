"""CLI слайса ingestion.

Точка входа тонкая: разбирает аргументы, вызывает слой ниже, форматирует вывод.
Доменной логики здесь нет (см. ARCHITECTURE.md).
"""

from __future__ import annotations

import logging

import typer

from nutri_radar.config import get_settings
from nutri_radar.ingest.download import download_dump
from nutri_radar.ingest.load import run_load_corpus
from nutri_radar.ingest.probe import format_report, probe_schema
from nutri_radar.ingest.select import collect_stats, format_stats
from nutri_radar.ingest.sources.parquet import ParquetSource

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


@app.command(name="select")
def select_corpus(
    limit: int | None = typer.Option(None, "--limit", help="Ограничить число строк (отладка)."),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Посчитать и показать статистику, ничего не записывая в БД.",
    ),
    stats: bool = typer.Option(
        False,
        "--stats",
        help="Подробная статистика выборки: языки, оценки, пригодность для M2 и M4.",
    ),
) -> None:
    """Отобрать корпус из дампа и залить в Postgres.

    Фильтрация идёт в DuckDB до Postgres: в базу уезжает подмножество, а не
    все 4,63 млн строк. Повторный запуск не создаёт дублей.
    """
    settings = get_settings()

    if stats:
        source = ParquetSource(settings.ingest)
        con = source.connect()
        try:
            collected = collect_stats(source, settings.ingest, con)
        finally:
            con.close()
        typer.echo(format_stats(collected))
        if not dry_run:
            typer.echo("")

    result = run_load_corpus(settings, limit=limit, dry_run=dry_run)

    action = "Посчитано" if dry_run else "Залито"
    typer.echo(
        f"{action}: {result.processed} строк за {result.elapsed_s} с "
        f"({result.rows_per_second:.0f} строк/с)"
    )
    if result.skipped:
        typer.echo(f"Пропущено: {result.skipped} ({result.skip_share:.2%})")
    if result.run_id is not None:
        typer.echo(f"Прогон в runs: id={result.run_id}")


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
