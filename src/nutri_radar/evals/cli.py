"""CLI слайса `evals`: выборка, разметка, статус.

Composition root слайса. Метрики и гейт добавятся здесь же по мере готовности.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from collections.abc import Awaitable, Callable

import typer

from nutri_radar.config import get_settings
from nutri_radar.db.session import dispose_engine
from nutri_radar.evals.annotate import annotate_session
from nutri_radar.evals.sample import select_sample
from nutri_radar.evals.schemas import (
    GOLD_FILE,
    SAMPLE_FILE,
    GoldRecord,
    SampleItem,
    read_jsonl,
    write_jsonl,
)

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="evals",
    help="Оценка качества: эталон, метрики, сравнение систем.",
    no_args_is_help=True,
)


def _run[T](coro_factory: Callable[[], Awaitable[T]]) -> T:
    """Выполнить корутину и закрыть пул соединений."""

    async def _main() -> T:
        try:
            return await coro_factory()
        finally:
            await dispose_engine()

    return asyncio.run(_main())


@app.command()
def sample(
    size: int = typer.Option(0, "--size", help="Продуктов в выборке; 0 — EVALS__GOLD_SIZE."),
    force: bool = typer.Option(
        False, "--force", help="Перезаписать выборку, даже если разметка уже начата."
    ),
) -> None:
    """Отобрать продукты для ручной разметки.

    Выборка детерминирована и коммитится в репозиторий: её состав — часть
    эксперимента, а не деталь запуска.
    """
    settings = get_settings()
    already = read_jsonl(GOLD_FILE, GoldRecord)
    if already and not force:
        # Пересобрать выборку под начатой разметкой значит обесценить её:
        # размеченные продукты могут в новую выборку не попасть.
        typer.echo(
            f"Разметка уже начата ({len(already)} записей в {GOLD_FILE}).\n"
            "Пересбор выборки сделает её несопоставимой с размеченным.\n"
            "Если это осознанно — повторите с --force."
        )
        raise typer.Exit(code=1)

    items = _run(lambda: select_sample(settings, size=size or None))
    write_jsonl(SAMPLE_FILE, items)

    by_lang = Counter(item.lang for item in items)
    typer.echo(f"Отобрано {len(items)} продуктов → {SAMPLE_FILE}")
    for lang, count in sorted(by_lang.items()):
        typer.echo(f"  {lang}: {count}")


@app.command()
def annotate(
    annotator: str = typer.Option(..., "--annotator", help="Кто размечает. Пишется в запись."),
    limit: int = typer.Option(0, "--limit", help="Сколько продуктов за сессию; 0 — все."),
    assisted: bool = typer.Option(
        False,
        "--assisted",
        help="Показывать предсказания модели. Смещает разметку — пишется в запись.",
    ),
) -> None:
    """Разметить эталон вручную.

    Метки ставит человек (правило 6 брифа). Прогресс сохраняется после каждого
    продукта, так что сессию можно прервать в любой момент.
    """
    added = annotate_session(
        annotator=annotator,
        ask=lambda prompt: typer.prompt(prompt, default="", show_default=False),
        show=typer.echo,
        assisted=assisted,
        limit=limit or None,
    )
    typer.echo(
        f"\nРазмечено за сессию: {added}. Всего в эталоне: {len(read_jsonl(GOLD_FILE, GoldRecord))}"
    )


@app.command()
def status() -> None:
    """Показать, сколько размечено и что осталось."""
    sample_items = read_jsonl(SAMPLE_FILE, SampleItem)
    gold = read_jsonl(GOLD_FILE, GoldRecord)

    if not sample_items:
        typer.echo("Выборка не собрана. Начните с `nutri-radar evals sample`.")
        raise typer.Exit(code=1)

    annotated = {record.code for record in gold}
    by_lang_total = Counter(item.lang for item in sample_items)
    by_lang_done = Counter(item.lang for item in sample_items if item.code in annotated)

    typer.echo(f"Размечено: {len(annotated)} из {len(sample_items)}")
    for lang in sorted(by_lang_total):
        typer.echo(f"  {lang}: {by_lang_done[lang]} из {by_lang_total[lang]}")

    if assisted_count := sum(1 for record in gold if record.assisted):
        # Отдельной строкой: разметка с подсказкой слабее слепой, и это
        # должно быть видно, а не выясняться при разборе результатов.
        typer.echo(f"\nИз них с подсказкой модели: {assisted_count} — учитывать отдельно.")
