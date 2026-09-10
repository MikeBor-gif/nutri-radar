"""CLI слайса `retrieval`.

Composition root слайса: собирает зависимости и форматирует вывод, логики
не содержит (`ARCHITECTURE.md`).
"""

from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import Awaitable, Callable

import httpx
import typer

from nutri_radar.config import get_settings
from nutri_radar.db.session import dispose_engine
from nutri_radar.llm.adapters import OllamaEmbeddings
from nutri_radar.llm.factory import build_llm
from nutri_radar.llm.runtime import get_runtime
from nutri_radar.retrieval.embed import (
    build_index,
    count_candidates,
    embed_corpus,
    iter_profiles,
)
from nutri_radar.retrieval.metrics import (
    GoldQuery,
    RetrievalReport,
    append_query,
    format_report,
    matching_codes,
    read_queries,
    score_grounding,
    score_property,
    score_recall,
    write_report,
)
from nutri_radar.retrieval.pipeline import ask as pipeline_ask
from nutri_radar.retrieval.pipeline import embed_texts, search_by_text
from nutri_radar.retrieval.rag import answer as rag_answer
from nutri_radar.retrieval.search import SearchFilters
from nutri_radar.retrieval.search import search as search_products

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="retrieval",
    help="Семантический поиск по составам и ответы со ссылками на штрихкоды.",
    no_args_is_help=True,
)


def _printable(text: str) -> str:
    """Сделать текст выводимым в текущую консоль.

    Профили собраны из краудсорсинговых полей на пяти языках, а консоль
    Windows по умолчанию однобайтная: французское `é` роняет вывод
    с `UnicodeEncodeError`. Диагностическая команда не должна падать
    из-за кодировки терминала — она для того и нужна, чтобы посмотреть
    на данные до трёхчасового прогона.
    """
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def _run[T](coro_factory: Callable[[], Awaitable[T]]) -> T:
    """Выполнить корутину и закрыть движок."""

    async def main() -> T:
        try:
            return await coro_factory()
        finally:
            await dispose_engine()

    return asyncio.run(main())


@app.command()
def benchmark() -> None:
    """Замерить скорость векторизации и экстраполировать на корпус.

    Запускается ДО полного прогона: решение о размере корпуса принимается
    по числу, а не по плану. Тот же порядок, что в M2 и M4.
    """
    settings = get_settings()
    size = settings.retrieval.benchmark_size

    async def run() -> None:
        total = await count_candidates(settings)
        async with httpx.AsyncClient(base_url=settings.ollama.base_url) as client:
            model = OllamaEmbeddings(client, settings.ollama)
            progress = await embed_corpus(model, settings, limit=size)
            await model.unload()

        typer.echo(progress.format())
        if progress.embedded:
            minutes = progress.per_product * total / 60
            typer.echo(
                f"\nКорпус целиком: {total} продуктов, "
                f"экстраполяция {minutes:.0f} минут "
                f"({minutes / 60:.1f} часа)."
            )
        else:
            typer.echo("\nВсё уже векторизовано — замерять нечего.")

    _run(run)


@app.command()
def embed(
    limit: int = typer.Option(0, "--limit", help="Сколько продуктов просмотреть; 0 — все."),
) -> None:
    """Векторизовать профили корпуса и записать в pgvector.

    Прогон возобновляемый: продукт пропускается, если его вектор посчитан
    той же моделью И хеш профиля совпал. Смена правила сборки профиля
    делает старые векторы устаревшими автоматически.
    """
    settings = get_settings()

    async def run() -> None:
        async with httpx.AsyncClient(base_url=settings.ollama.base_url) as client:
            model = OllamaEmbeddings(client, settings.ollama)
            try:
                progress = await embed_corpus(model, settings, limit=limit or None)
            finally:
                # Выгружаем всегда: рядом живёт модель генерации, и на 6 ГБ
                # VRAM они не помещаются вдвоём.
                await model.unload()
        typer.echo(progress.format())

    _run(run)


@app.command()
def profiles(
    limit: int = typer.Option(3, "--limit", help="Сколько профилей показать."),
) -> None:
    """Показать собранные профили, не векторизуя.

    Отдельная команда, потому что смотреть на то, что уходит в эмбеддинг,
    нужно ДО трёхчасового прогона, а не после.
    """
    settings = get_settings()

    async def run() -> None:
        shown = 0
        async for profile in iter_profiles(settings, limit=limit):
            typer.echo(f"--- {profile.code} (хеш {profile.hash}, {profile.version}) ---")
            typer.echo(_printable(profile.text))
            typer.echo("")
            shown += 1
        typer.echo(f"Показано профилей: {shown}")

    _run(run)


@app.command("build-index")
def build_index_command() -> None:
    """Построить индекс HNSW поверх залитых векторов.

    Запускается ПОСЛЕ заливки: на заполненной таблице граф получается
    лучше, а сборка быстрее. Миграция создала бы индекс на пустой таблице.
    """
    typer.echo(_run(lambda: build_index(get_settings())))


@app.command()
def search(
    query: str = typer.Argument(..., help="Что искать."),
    limit: int = typer.Option(0, "--limit", help="Сколько вернуть; 0 — из настроек."),
    lang: str = typer.Option("", "--lang", help="Ограничить языком состава."),
    grade: str = typer.Option("", "--grade", help="Оценки через запятую: a,b."),
) -> None:
    """Найти продукты по смыслу запроса."""
    settings = get_settings()
    filters = SearchFilters(
        lang=lang or None,
        grade_in=tuple(g.strip() for g in grade.split(",") if g.strip()),
    )

    async def run() -> None:
        async with httpx.AsyncClient(base_url=settings.ollama.base_url) as client:
            result = await search_by_text(
                query,
                client=client,
                settings=settings,
                limit=limit or None,
                filters=filters,
            )
        typer.echo(
            f"Найдено {len(result.hits)} за {result.latency_s * 1000:.0f} мс "
            f"({result.filters.describe()})"
        )
        for hit in result.hits:
            name = _printable(hit.product_name or "без названия")
            typer.echo(
                f"  {hit.similarity:.3f}  {hit.code}  {name[:60]}  [{hit.nutriscore_grade or '-'}]"
            )

    _run(run)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Вопрос о продуктах."),
    limit: int = typer.Option(0, "--limit", help="Сколько продуктов дать модели."),
) -> None:
    """Ответить на вопрос строго по найденным продуктам.

    Если релевантного не нашлось, отказ формируется кодом и модель
    не вызывается вовсе: просить её не выдумывать и надеяться —
    не проверяемое свойство системы.
    """
    settings = get_settings()

    async def run() -> None:
        async with httpx.AsyncClient(base_url=settings.ollama.base_url) as client:
            answer = (
                await pipeline_ask(question, client=client, settings=settings, limit=limit or None)
            ).answer

        typer.echo(_printable(answer.text))
        typer.echo("")
        if answer.refused:
            typer.echo("Отказ: релевантного в базе не нашлось.")
            return
        typer.echo(
            f"Источников в выдаче: {len(answer.sources)}, ссылок в ответе: {len(answer.cited)}"
        )
        if answer.cited_outside_sources:
            typer.echo(f"ВЫДУМАННЫЕ ссылки: {', '.join(answer.cited_outside_sources)}")

    _run(run)


@app.command("add-query")
def add_query(
    query: str = typer.Option(..., "--query", help="Формулировка запроса."),
    expected: str = typer.Option(..., "--expected", help="Штрихкоды через запятую."),
    kind: str = typer.Option("", "--kind", help="Тип запроса для разбивки в отчёте."),
    lang: str = typer.Option("", "--lang", help="Язык запроса."),
    author: str = typer.Option(..., "--author", help="Кто составил."),
) -> None:
    """Записать эталонный запрос.

    Запросы составляет ЧЕЛОВЕК (правило 6 брифа) и делает это ДО первого
    прогона поиска: эталон, написанный после того, как автор увидел выдачу,
    измеряет согласие системы с самой собой.
    """
    codes = [code.strip() for code in expected.split(",") if code.strip()]
    gold = GoldQuery(query=query, expected=codes, kind=kind, lang=lang, author=author)
    if not gold.is_usable:
        typer.echo("Запрос без формулировки или без эталонных штрихкодов не годится.")
        raise typer.Exit(code=2)

    path = append_query(gold)
    typer.echo(f"Записано: «{query}» -> {len(codes)} штрихкодов ({path})")


@app.command()
def evaluate(
    limit: int = typer.Option(0, "--limit", help="Сколько возвращать; 0 — из настроек."),
    with_rag: bool = typer.Option(True, "--rag/--no-rag", help="Считать и подтверждённость."),
) -> None:
    """Посчитать recall@k и подтверждённость на эталонных запросах."""
    settings = get_settings()
    queries = read_queries()
    top_k = limit or settings.retrieval.top_k

    async def run() -> None:
        report = RetrievalReport(k=top_k)
        async with httpx.AsyncClient(base_url=settings.ollama.base_url) as client:
            vectors = await embed_texts(
                [item.query for item in queries], client=client, settings=settings
            )

            results = []
            for gold, vector in zip(queries, vectors, strict=True):
                result = await search_products(
                    vector, query=gold.query, limit=top_k, settings=settings
                )
                if gold.predicate:
                    # Свойство проверяется тем же условием, которым эталон
                    # и определялся: один запрос в базу на выдачу.
                    matching = await matching_codes(gold.predicate, result.codes, settings)
                    report.properties.append(score_property(gold, result, matching))
                if gold.expected:
                    report.recalls.append(score_recall(gold, result))
                results.append((gold, result))

            if with_rag:
                llm = build_llm(settings, client)
                # Одно удержание очереди на весь цикл: между вопросами
                # модель не выгружается, иначе прогон по эталону превратился
                # бы в двадцать перезагрузок весов.
                async with get_runtime(settings).hold(llm.model_name):
                    for gold, result in results:
                        answer = await rag_answer(gold.query, result, llm, settings)
                        report.groundings.append(score_grounding(answer))

        text = format_report(report)
        typer.echo(_printable(text))
        typer.echo(f"\nОтчёт записан -> {write_report(text)}")

    _run(run)
