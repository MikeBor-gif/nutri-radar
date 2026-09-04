"""CLI слайса `evals`: выборка, разметка, предсказания, метрики, гейт, отчёт.

Composition root слайса: собирает зависимости и форматирует вывод, логики
не содержит.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from collections.abc import Awaitable, Callable
from functools import partial
from pathlib import Path

import anthropic
import typer

from nutri_radar.config import get_settings
from nutri_radar.db.session import dispose_engine
from nutri_radar.evals.annotate import annotate_session
from nutri_radar.evals.compare import (
    OFF_SYSTEM,
    collect_cloud_predictions,
    collect_local_predictions,
    collect_off_predictions,
)
from nutri_radar.evals.gate import (
    collect_current,
    format_gate,
    metrics_snapshot,
    run_gate,
    write_baseline,
)
from nutri_radar.evals.report import build_report, read_all_predictions, write_report
from nutri_radar.evals.sample import select_sample
from nutri_radar.evals.schemas import (
    GOLD_FILE,
    PREDICTIONS_DIR,
    SAMPLE_FILE,
    GoldRecord,
    PredictionRecord,
    SampleItem,
    predictions_path,
    read_jsonl,
    write_jsonl,
)
from nutri_radar.extract.normalize import load_seed_index
from nutri_radar.extract.prompts import load_prompt
from nutri_radar.llm.adapters.anthropic import AnthropicLLM

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


def _save_predictions(system: str, predictions: list[PredictionRecord]) -> int:
    """Записать предсказания одной системы. Возвращает число строк."""
    return write_jsonl(predictions_path(system), predictions)


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
    # Стрелка только ASCII: символа U+2192 нет в cp1251, и на консоли Windows
    # вывод падает с UnicodeEncodeError — кириллица и тире там есть, а он нет.
    typer.echo(f"Отобрано {len(items)} продуктов -> {SAMPLE_FILE}")
    for lang, count in sorted(by_lang.items()):
        typer.echo(f"  {lang}: {count}")


@app.command()
def annotate(
    annotator: str = typer.Option(..., "--annotator", help="Кто размечает. Пишется в запись."),
    limit: int = typer.Option(0, "--limit", help="Сколько продуктов за сессию; 0 — все."),
    lang: str = typer.Option(
        "",
        "--lang",
        help="Размечать только один язык: de, en, fr, pl, ru. Пусто — все подряд.",
    ),
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
    try:
        added = annotate_session(
            annotator=annotator,
            ask=lambda prompt: typer.prompt(prompt, default="", show_default=False),
            show=typer.echo,
            assisted=assisted,
            limit=limit or None,
            lang=lang or None,
        )
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=2) from exc
    typer.echo(
        f"\nРазмечено за сессию: {added}. Всего в эталоне: {len(read_jsonl(GOLD_FILE, GoldRecord))}"
    )


@app.command()
def predict(
    system: str = typer.Option(
        "off,local",
        "--system",
        help="Какие системы прогнать: off, local, cloud (через запятую).",
    ),
    dump: str = typer.Option(
        "", "--dump", help="Дамп OFF для baseline парсера; пусто — INGEST__DATA_DIR."
    ),
) -> None:
    """Собрать предсказания систем по выборке.

    Каждая система пишется отдельным файлом в `data/evals/predictions/`.
    Файлы коммитятся: на них работает гейт в CI, которому нельзя ни GPU,
    ни базы, ни сети.
    """
    settings = get_settings()
    sample_items = read_jsonl(SAMPLE_FILE, SampleItem)
    if not sample_items:
        typer.echo("Выборка не собрана. Начните с `nutri-radar evals sample`.")
        raise typer.Exit(code=1)

    wanted = {part.strip().lower() for part in system.split(",") if part.strip()}
    written: list[tuple[str, int]] = []

    if "off" in wanted:
        dump_path = Path(dump) if dump else settings.ingest.dump_path
        predictions = collect_off_predictions(sample_items, dump_path)
        written.append((OFF_SYSTEM, _save_predictions(OFF_SYSTEM, predictions)))

    if "local" in wanted:
        predictions = _run(partial(collect_local_predictions, sample_items, settings))
        written.append(
            (settings.ollama.model, _save_predictions(settings.ollama.model, predictions))
        )

    if "cloud" in wanted:
        if settings.anthropic.api_key is None:
            # Отказ явный и до прогона: облачные вызовы стоят денег, и молча
            # пропустить систему значит получить сравнение трёх систем там,
            # где отчёт обещает четыре.
            typer.echo(
                "ANTHROPIC__API_KEY не задан — облачные системы прогнать нельзя.\n"
                "Укажите ключ в .env или уберите `cloud` из --system."
            )
            raise typer.Exit(code=1)
        prompt = load_prompt(settings.extract.prompt_version)
        for model in (settings.anthropic.model, settings.anthropic.cheap_model):
            llm = AnthropicLLM(
                anthropic.AsyncAnthropic(
                    api_key=settings.anthropic.api_key.get_secret_value(),
                    timeout=settings.anthropic.timeout_s,
                ),
                settings.anthropic.model_copy(update={"model": model}),
            )
            predictions = _run(
                partial(collect_cloud_predictions, sample_items, llm, prompt, settings)
            )
            written.append((model, _save_predictions(model, predictions)))

    for name, count in written:
        typer.echo(f"{name}: {count} предсказаний -> {predictions_path(name)}")


@app.command()
def gate() -> None:
    """Проверить, не просели ли метрики против базлайна.

    Работает без GPU, без БД и без сети — только файлы из репозитория.
    Выход с кодом 1 при просадке: это точка входа для CI.
    """
    settings = get_settings()
    result = run_gate(settings)
    typer.echo(format_gate(result, settings.evals.max_f1_drop))
    if not result.passed:
        raise typer.Exit(code=1)


@app.command()
def report(
    out: str = typer.Option("", "--out", help="Куда писать отчёт; пусто — reports/ с датой."),
    save: bool = typer.Option(True, "--save/--no-save", help="Писать файл или только в консоль."),
) -> None:
    """Собрать отчёт приёмки: таблица сравнения и материал для вопросов M2.

    Работает по тем же файлам, что и гейт: эталон, предсказания, словарь.
    Ни GPU, ни базы, ни сети — отчёт пересобирается одной командой из того,
    что лежит в репозитории.
    """
    settings = get_settings()
    gold = read_jsonl(GOLD_FILE, GoldRecord)
    predictions_by_system = read_all_predictions(PREDICTIONS_DIR)

    text = build_report(gold, predictions_by_system, load_seed_index(), settings)
    typer.echo(text)

    if save:
        path = write_report(text, Path(out) if out else None)
        typer.echo(f"\nОтчёт записан -> {path}")


@app.command()
def baseline() -> None:
    """Зафиксировать текущие метрики как базлайн для гейта.

    Отдельная команда, а не автообновление: базлайн, который переписывается
    сам, не сторожит ничего — любая просадка молча стала бы новой нормой.
    """
    gold = read_jsonl(GOLD_FILE, GoldRecord)
    if not gold:
        typer.echo("Эталон пуст — фиксировать нечего. Сначала разметьте продукты.")
        raise typer.Exit(code=1)

    results = collect_current(gold, PREDICTIONS_DIR)
    if not results:
        typer.echo("Нет предсказаний. Сначала соберите их: `nutri-radar evals predict`.")
        raise typer.Exit(code=1)

    snapshot = {system: metrics_snapshot(result) for system, result in results.items()}
    path = write_baseline(snapshot)
    typer.echo(f"Базлайн зафиксирован для {len(snapshot)} систем -> {path}")
    for system, metrics in sorted(snapshot.items()):
        values = ", ".join(f"{k} {v:.3f}" for k, v in sorted(metrics.items()))
        typer.echo(f"  {system}: {values}")


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
