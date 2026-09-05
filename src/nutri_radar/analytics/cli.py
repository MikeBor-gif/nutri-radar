"""CLI слайса `analytics`.

Composition root слайса: собирает зависимости и форматирует вывод, логики
не содержит (`ARCHITECTURE.md`).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx
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
from nutri_radar.analytics.embeddings import benchmark as benchmark_embeddings
from nutri_radar.analytics.features import SET_COMMON, SET_FULL, load_scores
from nutri_radar.analytics.report import (
    build_report,
    plot_accuracy_vs_cost,
    plot_confusion,
    write_report,
)
from nutri_radar.analytics.tasks.grade_from_text import (
    format_score,
    run_embeddings,
    run_tfidf,
    run_tfidf_small,
    run_zero_shot,
)
from nutri_radar.analytics.tasks.sanity_check import format_sanity, run_sanity_check
from nutri_radar.config import get_settings
from nutri_radar.db.session import dispose_engine
from nutri_radar.llm.adapters import OllamaEmbeddings, OllamaLLM

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


@app.command()
def sanity(
    target: str = typer.Option(
        "nutriscore_grade", "--target", help=f"Что предсказываем: {', '.join(TARGETS)}."
    ),
    sample: int = typer.Option(20000, "--sample", help="Сколько строк брать на проверку."),
) -> None:
    """Проверить, что задача решаема и модель учит состав, а не язык.

    Запускается ДО сравнения подходов: если внутри языков модель не обгоняет
    базлайн, вся таблица M4 измеряет смещение корпуса по странам, и узнать
    это надо раньше, чем она появится.
    """
    settings = get_settings()
    frame = prepare(load_dataset(), target)
    result = run_sanity_check(frame, target, settings, sample_size=sample)
    typer.echo(format_sanity(result))


@app.command()
def tfidf(
    target: str = typer.Option(
        "nutriscore_grade", "--target", help=f"Что предсказываем: {', '.join(TARGETS)}."
    ),
) -> None:
    """Обучить TF-IDF + логрегрессию и посчитать метрики.

    Считает сразу два результата: на полном тесте и на общей подвыборке,
    той же, на которой потом померяются эмбеддинги и LLM. Без второго
    числа сравнение подходов было бы сравнением разных задач.
    """
    settings = get_settings()
    frame = prepare(load_dataset(), target)
    full, common = run_tfidf(frame, target, settings)

    typer.echo(format_score(full, "полный тест"))
    typer.echo("")
    typer.echo(format_score(common, "общая подвыборка"))


@app.command("embed-benchmark")
def embed_benchmark(
    target: str = typer.Option(
        "nutriscore_grade", "--target", help=f"Что предсказываем: {', '.join(TARGETS)}."
    ),
) -> None:
    """Замерить скорость векторизации и экстраполировать на корпус.

    Запускается ДО полного прогона: решение о размере корпуса принимается
    по числу, а не по плану. Тот же порядок, что в M2.
    """
    settings = get_settings()
    frame = prepare(load_dataset(), target)
    sample = frame.sample(
        n=min(settings.analytics.benchmark_size, len(frame)),
        random_state=settings.analytics.random_seed,
    )

    async def run() -> None:
        async with httpx.AsyncClient(base_url=settings.ollama.base_url) as client:
            model = OllamaEmbeddings(client, settings.ollama)
            result = await benchmark_embeddings(model, sample["ingredients_text"].tolist())
            typer.echo(result.format(len(frame)))
            await model.unload()

    asyncio.run(run())


@app.command("tfidf-small")
def tfidf_small(
    target: str = typer.Option(
        "nutriscore_grade", "--target", help=f"Что предсказываем: {', '.join(TARGETS)}."
    ),
) -> None:
    """TF-IDF на том же урезанном train, что и эмбеддинги.

    Контрольный прогон. Без него разницу между TF-IDF и эмбеддингами
    невозможно отличить от разницы в размере обучающей выборки: полный
    корпус эмбеддингам не по карману (219 минут GPU по замеру).
    """
    settings = get_settings()
    frame = prepare(load_dataset(), target)
    full, common = run_tfidf_small(frame, target, settings)

    typer.echo(format_score(full, "полный тест"))
    typer.echo("")
    typer.echo(format_score(common, "общая подвыборка"))


@app.command()
def embed(
    target: str = typer.Option(
        "nutriscore_grade", "--target", help=f"Что предсказываем: {', '.join(TARGETS)}."
    ),
) -> None:
    """Векторизовать корпус bge-m3 и обучить логрегрессию на векторах.

    Прогон возобновляемый: векторы кэшируются пачками, перезапуск
    продолжает с невекторизованных.
    """
    settings = get_settings()
    frame = prepare(load_dataset(), target)

    async def run() -> tuple[object, object]:
        async with httpx.AsyncClient(base_url=settings.ollama.base_url) as client:
            model = OllamaEmbeddings(client, settings.ollama)
            try:
                return await run_embeddings(frame, target, model, settings)
            finally:
                # Выгружаем всегда: следующая стадия — LLM zero-shot, и она
                # не влезет в 6 ГБ рядом с bge-m3.
                await model.unload()

    full, common = asyncio.run(run())
    typer.echo(format_score(full, "полный тест"))
    typer.echo("")
    typer.echo(format_score(common, "общая подвыборка"))


@app.command("zero-shot")
def zero_shot(
    target: str = typer.Option(
        "nutriscore_grade", "--target", help=f"Что предсказываем: {', '.join(TARGETS)}."
    ),
) -> None:
    """Прогнать LLM zero-shot на общей подвыборке.

    Только на подвыборке: бриф запрещает гонять через модель больше
    нескольких тысяч продуктов. Прогон возобновляемый — прогресс пишется
    после каждого продукта.
    """
    settings = get_settings()
    frame = prepare(load_dataset(), target)

    async def run() -> object:
        async with httpx.AsyncClient(base_url=settings.ollama.base_url) as client:
            llm = OllamaLLM(client, settings.ollama)
            return await run_zero_shot(frame, target, llm, settings)

    score = asyncio.run(run())
    typer.echo(format_score(score, "общая подвыборка"))


@app.command()
def report(
    target: str = typer.Option(
        "nutriscore_grade", "--target", help=f"Что предсказываем: {', '.join(TARGETS)}."
    ),
    plots: bool = typer.Option(True, "--plots/--no-plots", help="Рисовать графики."),
) -> None:
    """Собрать отчёт сравнения подходов из сохранённых результатов.

    Читает то, что посчитали отдельные команды. Ничего не пересчитывает:
    подходы считаются часами, и пересборка отчёта не должна их трогать.
    """
    text = build_report(target)
    typer.echo(text)
    path = write_report(text, target)
    typer.echo(f"\nОтчёт записан -> {path}")

    if not plots:
        return

    common = load_scores(target, SET_COMMON)
    if common:
        typer.echo(f"График -> {plot_accuracy_vs_cost(common, target)}")
    for score in load_scores(target, SET_FULL) + common:
        plot_confusion(score, target)
