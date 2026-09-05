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
from nutri_radar.retrieval.embed import count_candidates, embed_corpus, iter_profiles

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
