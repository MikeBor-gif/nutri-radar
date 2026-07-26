"""Заливка отобранного корпуса в Postgres.

DoD майлстоуна: корпус в базе, повторный запуск не создаёт дублей.

Прогресс пишется в таблицу `runs` — не для красоты: доля продуктов с
пропущенными данными входит в обязательные метрики проекта, и она должна
считаться из БД, а не выясняться чтением логов задним числом.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx
from sqlalchemy import select

from nutri_radar.config import IngestSettings, Settings, get_settings
from nutri_radar.db.models.run import Run, RunStage, RunStatus
from nutri_radar.db.repositories.product import ProductRepository
from nutri_radar.db.session import dispose_engine, get_session
from nutri_radar.errors import DatabaseError
from nutri_radar.ingest.download import read_dump_version
from nutri_radar.ingest.models import RawProduct
from nutri_radar.ingest.select import iter_selected, matches_corpus
from nutri_radar.ingest.sources.delta import (
    check_retention_gap,
    download_delta,
    fetch_index,
    iter_records,
    select_files,
    to_raw_product,
)
from nutri_radar.ingest.sources.parquet import ParquetSource

logger = logging.getLogger(__name__)


@dataclass
class LoadResult:
    """Итог заливки. Те же числа уезжают в `runs`."""

    run_id: int | None = None
    processed: int = 0
    skipped: int = 0
    batches: int = 0
    elapsed_s: float = 0.0
    failed_batches: list[int] = field(default_factory=list)

    @property
    def rows_per_second(self) -> float:
        return self.processed / self.elapsed_s if self.elapsed_s else 0.0

    @property
    def skip_share(self) -> float:
        total = self.processed + self.skipped
        return self.skipped / total if total else 0.0


async def _open_run(settings: Settings, params: dict[str, object]) -> int:
    """Открыть запись прогона и вернуть её id."""
    async with get_session(settings.db) as session:
        # Незавершённый прогон — признак предыдущего обрыва. Не блокируем:
        # upsert идемпотентен, поэтому повтор безопасен. Но знать об этом надо.
        stale = (
            (
                await session.execute(
                    select(Run).where(Run.stage == RunStage.INGEST, Run.status == RunStatus.RUNNING)
                )
            )
            .scalars()
            .all()
        )
        if stale:
            logger.warning(
                "Есть незавершённые прогоны ingestion — предыдущий запуск оборвался. "
                "Повтор безопасен: запись идемпотентна",
                extra={"stale_run_ids": [run.id for run in stale]},
            )

        run = Run(stage=RunStage.INGEST, status=RunStatus.RUNNING, params=params)
        session.add(run)
        await session.flush()
        return run.id


async def _close_run(
    settings: Settings,
    run_id: int,
    result: LoadResult,
    *,
    error: str | None = None,
    params: dict[str, object] | None = None,
) -> None:
    async with get_session(settings.db) as session:
        run = await session.get(Run, run_id)
        if run is None:
            logger.error("Запись прогона исчезла", extra={"run_id": run_id})
            return
        run.status = RunStatus.FAILED if error else RunStatus.COMPLETED
        run.items_processed = result.processed
        run.items_skipped = result.skipped
        run.error_message = error
        run.finished_at = datetime.now(UTC)
        if params is not None:
            # Водяной знак дельт хранится в params прогона: следующий запуск
            # берёт его оттуда и продолжает с нужного места.
            run.params = params


async def load_corpus(
    settings: Settings | None = None,
    *,
    limit: int | None = None,
    dry_run: bool = False,
) -> LoadResult:
    """Отобрать корпус из дампа и залить в Postgres.

    Args:
        settings: настройки; по умолчанию из `get_settings()`.
        limit: ограничить число строк — для отладки на подмножестве.
        dry_run: посчитать, но не писать в БД.
    """
    settings = settings or get_settings()
    ingest: IngestSettings = settings.ingest

    source = ParquetSource(ingest)
    dump_version = read_dump_version(ingest)
    run_params = {
        "mode": "select",
        "languages": ingest.languages,
        "category_tags_count": len(ingest.category_tags),
        "min_ingredients_length": ingest.min_ingredients_length,
        "batch_size": ingest.batch_size,
        "limit": limit,
        "dump_version": dump_version,
    }

    result = LoadResult()
    run_id = None if dry_run else await _open_run(settings, run_params)
    result.run_id = run_id

    logger.info(
        "Заливка корпуса начата",
        extra={
            "run_id": run_id,
            "dry_run": dry_run,
            "batch_size": ingest.batch_size,
            "limit": limit,
            "dump_version": dump_version,
        },
    )

    started = time.perf_counter()
    con = source.connect()
    error: str | None = None
    try:
        for index, batch in enumerate(iter_selected(source, ingest, con, limit=limit), start=1):
            result.batches = index
            if dry_run:
                result.processed += len(batch)
            else:
                try:
                    await _write_batch(settings, batch, ingest, dump_version)
                    result.processed += len(batch)
                except DatabaseError:
                    # Битый батч не роняет прогон целиком: помечаем, считаем,
                    # идём дальше. Иначе одна плохая строка обнуляет часы работы.
                    result.skipped += len(batch)
                    result.failed_batches.append(index)
                    logger.error(
                        "Батч не записан, продолжаем",
                        extra={"batch": index, "size": len(batch)},
                        exc_info=True,
                    )

            if index % 10 == 0:
                elapsed = time.perf_counter() - started
                logger.info(
                    "Прогресс заливки",
                    extra={
                        "batches": index,
                        "processed": result.processed,
                        "skipped": result.skipped,
                        "rows_per_s": round(result.processed / max(elapsed, 1e-6), 1),
                    },
                )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.error("Заливка прервана", exc_info=True)
        raise
    finally:
        con.close()
        result.elapsed_s = round(time.perf_counter() - started, 2)
        if run_id is not None:
            await _close_run(settings, run_id, result, error=error)

    _log_result(result, ingest, dry_run=dry_run)
    return result


async def _last_delta_watermark(settings: Settings) -> int | None:
    """Таймстамп последнего успешно применённого файла дельт."""
    async with get_session(settings.db) as session:
        row = (
            await session.execute(
                select(Run)
                .where(
                    Run.stage == RunStage.INGEST,
                    Run.status == RunStatus.COMPLETED,
                    Run.params["mode"].astext == "delta",
                )
                .order_by(Run.started_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()

    if row is None:
        return None
    watermark = row.params.get("watermark")
    return int(watermark) if watermark is not None else None


async def load_deltas(
    settings: Settings | None = None,
    *,
    days: int | None = None,
    dry_run: bool = False,
    client: httpx.Client | None = None,
) -> LoadResult:
    """Применить дельта-экспорты поверх корпуса.

    Фильтр корпуса применяется и здесь: дельта может принести продукт из
    ненужной категории или на ненужном языке.

    Args:
        settings: настройки; по умолчанию из `get_settings()`.
        days: ограничить окно вручную; по умолчанию — от последнего прогона.
        dry_run: посчитать, ничего не записывая.
        client: HTTP-клиент; подменяется в тестах.
    """
    settings = settings or get_settings()
    ingest = settings.ingest

    owns_client = client is None
    client = client or httpx.Client(timeout=ingest.download_timeout_s, follow_redirects=True)

    result = LoadResult()
    started = time.perf_counter()
    watermark = await _last_delta_watermark(settings)

    if days is not None:
        manual = int(time.time()) - days * 86400
        watermark = max(watermark or 0, manual)
        logger.info("Окно задано вручную", extra={"days": days, "watermark": watermark})

    try:
        index = fetch_index(client, ingest)
        check_retention_gap(index, watermark)
        selected = select_files(index, watermark)

        if not selected:
            logger.info("Новых дельт нет — корпус актуален")
            result.elapsed_s = round(time.perf_counter() - started, 2)
            return result

        run_params: dict[str, object] = {
            "mode": "delta",
            "files": len(selected),
            "watermark_before": watermark,
        }
        run_id = None if dry_run else await _open_run(settings, run_params)
        result.run_id = run_id

        applied_watermark = watermark
        for file in selected:
            try:
                path = download_delta(client, file, ingest)
            except httpx.HTTPError as exc:
                # Один недоступный файл не отменяет остальные, но водяной знак
                # дальше него двигать нельзя: между ними образуется дыра.
                logger.warning(
                    "Файл дельты не скачан, дальше не идём",
                    extra={"name": file.name, "error": type(exc).__name__},
                )
                break

            matched, skipped = await _apply_delta_file(settings, path, dry_run=dry_run)
            result.processed += matched
            result.skipped += skipped
            result.batches += 1
            applied_watermark = file.timestamp
            logger.info(
                "Дельта применена",
                extra={
                    "file": file.name,
                    "matched": matched,
                    "skipped": skipped,
                    "watermark": applied_watermark,
                },
            )

        result.elapsed_s = round(time.perf_counter() - started, 2)
        if run_id is not None:
            run_params["watermark"] = applied_watermark
            await _close_run(settings, run_id, result, params=run_params)
    finally:
        if owns_client:
            client.close()

    logger.info(
        "Применение дельт завершено",
        extra={
            "files": result.batches,
            "processed": result.processed,
            "skipped": result.skipped,
            "elapsed_s": result.elapsed_s,
        },
    )
    return result


async def _apply_delta_file(
    settings: Settings,
    path: Path,
    *,
    dry_run: bool,
) -> tuple[int, int]:
    """Применить один файл дельты. Возвращает (принято, отброшено)."""
    ingest = settings.ingest
    batch: list[RawProduct] = []
    matched = 0
    skipped = 0

    for record in iter_records(path):
        product = to_raw_product(record)
        if product is None or not matches_corpus(product, ingest):
            skipped += 1
            continue

        batch.append(product)
        matched += 1
        if len(batch) >= ingest.batch_size:
            if not dry_run:
                await _write_batch(settings, batch, ingest, dump_version=f"delta:{path.name}")
            batch = []

    if batch and not dry_run:
        await _write_batch(settings, batch, ingest, dump_version=f"delta:{path.name}")

    return matched, skipped


def run_load_deltas(
    settings: Settings | None = None,
    *,
    days: int | None = None,
    dry_run: bool = False,
) -> LoadResult:
    """Синхронная обёртка для CLI с закрытием пула соединений."""

    async def _run() -> LoadResult:
        try:
            return await load_deltas(settings, days=days, dry_run=dry_run)
        finally:
            await dispose_engine()

    return asyncio.run(_run())


def run_load_corpus(
    settings: Settings | None = None,
    *,
    limit: int | None = None,
    dry_run: bool = False,
) -> LoadResult:
    """Синхронная обёртка для CLI.

    Закрывает пул соединений после прогона. Без этого движок, созданный внутри
    одного цикла событий, переживёт его и на следующем `asyncio.run` отдаст
    соединения от закрытого цикла.
    """

    async def _run() -> LoadResult:
        try:
            return await load_corpus(settings, limit=limit, dry_run=dry_run)
        finally:
            await dispose_engine()

    return asyncio.run(_run())


async def _write_batch(
    settings: Settings,
    batch: list,
    ingest: IngestSettings,
    dump_version: str | None,
) -> None:
    """Один батч в одной транзакции: обрыв не оставит половину батча."""
    async with get_session(settings.db) as session:
        repository = ProductRepository(session)
        await repository.upsert_batch(
            batch,
            languages=ingest.languages,
            min_ingredients_length=ingest.min_ingredients_length,
            dump_version=dump_version,
        )


def _log_result(result: LoadResult, ingest: IngestSettings, *, dry_run: bool) -> None:
    logger.info(
        "Заливка завершена",
        extra={
            "run_id": result.run_id,
            "dry_run": dry_run,
            "processed": result.processed,
            "skipped": result.skipped,
            "batches": result.batches,
            "elapsed_s": result.elapsed_s,
            "rows_per_s": round(result.rows_per_second, 1),
        },
    )

    if result.failed_batches:
        logger.warning(
            "Часть батчей не записана",
            extra={"failed_batches": result.failed_batches},
        )

    if result.skip_share > ingest.max_skip_share:
        logger.warning(
            "Доля пропусков выше порога — вероятно, сломался фильтр или адаптер",
            extra={
                "skip_share": f"{result.skip_share:.2%}",
                "threshold": f"{ingest.max_skip_share:.2%}",
            },
        )
