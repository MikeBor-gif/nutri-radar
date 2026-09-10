"""CLI слайса извлечения. Точка входа `nutri-radar extract`.

Здесь же **composition root**: единственное место, где по `LLM__PROVIDER`
выбирается конкретный адаптер модели. Раннер, замер и отбор корпуса знают
только порт `StructuredLLM` — иначе провайдера нельзя было бы сменить через
`.env`, а тесты полезли бы в сеть (ARCHITECTURE.md, антипаттерн «конкретный
адаптер внутри слайса»).

Логики здесь нет: команды разбирают аргументы, вызывают слой ниже и печатают
результат. Вывод для человека идёт в stdout, логи — в stderr.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

import httpx
import typer

from nutri_radar.config import Settings, get_settings
from nutri_radar.db.repositories.extraction import ExtractionRepository
from nutri_radar.db.session import dispose_engine, get_session
from nutri_radar.extract import benchmark as benchmark_module
from nutri_radar.extract.corpus import (
    CorpusItem,
    collect_stats,
    format_stats,
    sample_for_benchmark,
    select_llm_corpus,
)
from nutri_radar.extract.normalize import (
    NormalizationStats,
    format_unknown_report,
    load_db_index,
    normalize_ingredients,
    sync_seed_to_db,
)
from nutri_radar.extract.prompts import available_versions
from nutri_radar.extract.runner import ExtractionRunResult, format_result, run_extraction
from nutri_radar.extract.schemas import Ingredient
from nutri_radar.llm.factory import build_llm

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="extract",
    help="Извлечение структуры состава: корпус, замер, прогон, словарь.",
    no_args_is_help=True,
)
dict_app = typer.Typer(
    name="dict",
    help="Словарь алиасов ингредиентов. Наполняется человеком, не моделью.",
    no_args_is_help=True,
)
app.add_typer(dict_app, name="dict")


def _run[T](coro_factory: Callable[[], Awaitable[T]]) -> T:
    """Выполнить корутину и закрыть пул соединений.

    Без `dispose_engine` движок, созданный внутри одного цикла событий,
    переживёт его и на следующем `asyncio.run` отдаст соединения от
    закрытого цикла.
    """

    async def _main() -> T:
        try:
            return await coro_factory()
        finally:
            await dispose_engine()

    return asyncio.run(_main())


def _http_client(settings: Settings) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=settings.ollama.base_url, timeout=settings.ollama.timeout_s)


async def _corpus(settings: Settings, size: int | None) -> list[CorpusItem]:
    items = await select_llm_corpus(settings, size=size)
    if not items:
        raise typer.Exit(code=1)
    return items


@app.command()
def corpus(
    size: int = typer.Option(
        0, "--size", help="Размер корпуса; 0 — значение EXTRACT__CORPUS_SIZE."
    ),
) -> None:
    """Отобрать LLM-корпус и показать его состав.

    Ничего не записывает: выборка детерминирована seed, поэтому прогон
    воспроизведёт ровно этот же набор.
    """
    settings = get_settings()
    items = _run(lambda: _corpus(settings, size or None))
    typer.echo(format_stats(collect_stats(items)))


@app.command()
def benchmark(
    prompt: str = typer.Option(
        "",
        "--prompt",
        "-p",
        help="Версии промпта через запятую; по умолчанию все доступные.",
    ),
    size: int = typer.Option(
        0, "--size", help="Продуктов в замере; 0 — значение EXTRACT__BENCHMARK_SIZE."
    ),
    report: bool = typer.Option(True, "--report/--no-report", help="Писать отчёт в reports/."),
) -> None:
    """Замер перед полным прогоном: секунды, токены, экстраполяция.

    Обязателен до полного прогона (раздел 3a брифа). Все версии меряются
    на ОДНИХ И ТЕХ ЖЕ продуктах — иначе сравнение версий ничего не значит.
    Выборка представительная по языкам: первые N по коду дали бы одну страну.
    """
    settings = get_settings()
    versions = [part.strip() for part in prompt.split(",") if part.strip()] or available_versions()
    sample_size = size or settings.extract.benchmark_size

    async def _work() -> list[benchmark_module.BenchmarkResult]:
        items = sample_for_benchmark(await _corpus(settings, None), sample_size)
        results = []
        async with _http_client(settings) as client:
            llm = build_llm(settings, client)
            for version in versions:
                result = await benchmark_module.run_benchmark(
                    llm, items, settings, prompt_version=version
                )
                if report:
                    benchmark_module.write_report(result, settings)
                results.append(result)
        return results

    for result in _run(_work):
        typer.echo("")
        typer.echo(benchmark_module.format_result(result, settings))


@app.command()
def run(
    prompt: str = typer.Option(
        "", "--prompt", "-p", help="Версия промпта; по умолчанию EXTRACT__PROMPT_VERSION."
    ),
    size: int = typer.Option(0, "--size", help="Размер корпуса; 0 — значение из настроек."),
    limit: int = typer.Option(0, "--limit", help="Обработать не больше N продуктов корпуса."),
    resume: bool = typer.Option(
        True, "--resume/--no-resume", help="Пропускать уже разобранное этой версией."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Прогнать модель, но ничего не писать в БД."
    ),
) -> None:
    """Полный прогон извлечения по корпусу.

    Прогон возобновляемый: Ctrl+C стоит не больше одного батча, повторный
    запуск продолжает с необработанных продуктов.
    """
    settings = get_settings()

    async def _work() -> ExtractionRunResult:
        items = await _corpus(settings, size or None)
        if limit:
            items = items[:limit]
        async with _http_client(settings) as client:
            llm = build_llm(settings, client)
            return await run_extraction(
                llm,
                items,
                settings,
                prompt_version=prompt or None,
                resume=resume,
                dry_run=dry_run,
            )

    typer.echo(format_result(_run(_work)))


@dict_app.command("sync")
def dict_sync() -> None:
    """Залить seed-словарь из `data/dictionaries/` в таблицу `ingredients_dict`.

    Направление одностороннее: файл — источник истины, база его копия.
    """
    affected = _run(lambda: sync_seed_to_db())
    typer.echo(f"Записано строк словаря: {affected}")


@dict_app.command("unknown")
def dict_unknown(
    top: int = typer.Option(0, "--top", help="Сколько имён показать; 0 — значение из настроек."),
    prompt: str = typer.Option("", "--prompt", "-p", help="Ограничить версией промпта."),
    model: str = typer.Option("", "--model", help="Ограничить моделью."),
) -> None:
    """Показать самые частые имена, которых нет в словаре.

    Это ответ на вопрос «куда пополнять словарь» — по данным, а не на глаз.
    Доля неизвестных имён здесь же: наш прямой аналог `unknown_ingredients_n`
    у Open Food Facts.
    """
    settings = get_settings()
    limit = top or settings.extract.unknown_report_top

    async def _work() -> NormalizationStats:
        index = await load_db_index(settings)
        async with get_session(settings.db) as session:
            rows = await ExtractionRepository(session).iter_ingredients(
                model_name=model or None, prompt_version=prompt or None
            )

        stats = NormalizationStats()
        for ingredients, lang in rows:
            parsed = [Ingredient.model_validate(item) for item in ingredients]
            stats.add(normalize_ingredients(parsed, index, lang=lang))
        stats.log_summary()
        return stats

    stats = _run(_work)
    if not stats.total:
        typer.echo("Извлечений в базе нет — сначала выполните `extract run`.", err=True)
        raise typer.Exit(code=1)
    typer.echo(format_unknown_report(stats, limit))
