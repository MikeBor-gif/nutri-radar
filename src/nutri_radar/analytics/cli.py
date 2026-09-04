"""CLI слайса `analytics`.

Composition root слайса: собирает зависимости и форматирует вывод, логики
не содержит (`ARCHITECTURE.md`).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path

import typer

from nutri_radar.analytics.dataset import (
    TARGETS,
    fetch_dataset,
    load_dataset,
    majority_baseline,
    prepare,
    save_dataset,
    split,
)
from nutri_radar.config import get_settings
from nutri_radar.db.session import dispose_engine

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="analytics",
    help="Предсказание оценки качества по тексту состава: TF-IDF, эмбеддинги, LLM.",
    no_args_is_help=True,
)


def _run[T](coro_factory: Callable[[], Awaitable[T]]) -> T:
    """Выполнить корутину и закрыть движок.

    Движок закрывается здесь, а не в слое ниже: пул принадлежит процессу,
    и слайс не должен решать за него, когда он кончился.
    """

    async def main() -> T:
        try:
            return await coro_factory()
        finally:
            await dispose_engine()

    return asyncio.run(main())


@app.command()
def dataset(
    out: str = typer.Option("", "--out", help="Куда сохранить; пусто — data/analytics/."),
) -> None:
    """Выгрузить обучающий набор из базы в parquet.

    Кэш, а не артефакт репозитория: выгрузка занимает минуты, повторять её
    на каждый эксперимент незачем. Воспроизводимость держится на seed,
    а не на файле.
    """
    settings = get_settings()
    frame = _run(lambda: fetch_dataset(settings))
    path = save_dataset(frame, Path(out) if out else None)

    typer.echo(f"Выгружено строк: {len(frame)} -> {path}")
    for target in TARGETS:
        subset = prepare(frame, target)
        label, share = majority_baseline(subset, target)
        typer.echo(f"  {target}: {len(subset)} строк, самый частый класс «{label}» — {share:.1%}")


@app.command()
def describe(
    target: str = typer.Option(
        "nutriscore_grade", "--target", help=f"Что предсказываем: {', '.join(TARGETS)}."
    ),
) -> None:
    """Показать состав набора и сплита без обучения моделей.

    Отдельная команда, потому что смотреть на распределение классов нужно
    до того, как появятся числа точности, а не после.
    """
    settings = get_settings()
    frame = prepare(load_dataset(), target)
    train, test = split(frame, target, settings)

    label, share = majority_baseline(frame, target)
    typer.echo(f"Задача: {target}")
    typer.echo(f"  строк всего: {len(frame)}")
    seed = settings.analytics.random_seed
    typer.echo(f"  train / test: {len(train)} / {len(test)}  (seed {seed})")
    typer.echo(f"  baseline большинства класса: «{label}» — {share:.1%}")

    typer.echo("\n  Классы:")
    for value, count in frame[target].value_counts().sort_index().items():
        typer.echo(f"    {value}: {count} ({count / len(frame):.1%})")

    typer.echo("\n  Языки:")
    for value, count in frame["lang"].value_counts().head(10).items():
        typer.echo(f"    {value}: {count} ({count / len(frame):.1%})")
