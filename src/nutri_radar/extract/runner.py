"""Полный прогон извлечения по LLM-корпусу.

Раннер отвечает только за оркестрацию: батчи, параллелизм, ретраи, учёт
токенов и времени. Что считается формой сахара и как классифицируется
ингредиент — в `schemas.py` и `normalize.py`, потому что это должно
тестироваться без поднятой LLM (ARCHITECTURE.md, антипаттерн «логика
в раннере»).

Четыре свойства, ради которых он существует:

1. **Возобновляемость.** Перед прогоном спрашиваем БД, что уже разобрано
   этой моделью и этой версией промпта, и пропускаем. Состояние живёт
   в таблице, а не в процессе, поэтому Ctrl+C стоит максимум одного батча.
2. **Ограниченный параллелизм.** Семафор из настроек. На 6 ГБ VRAM
   неограниченный веер запросов кладёт GPU, а Ollama начинает выгружать
   и грузить модель заново — прогон становится медленнее последовательного.
3. **Битая запись не роняет прогон.** Невалидный ответ считается и
   пропускается: доля пропусков входит в обязательные метрики проекта,
   поэтому она считается, а не проглатывается.
4. **Ретрай только на недоступности модели.** `LLMUnavailableError` —
   инфраструктурная проблема, её лечит пауза. `ValidationError`
   и `ExtractionError` — проблема данных или промпта: при `temperature=0`
   повтор даст тот же ответ, и прогон встанет на месте. Отдельно важен
   обрыв ответа на лимите вывода: генерация до лимита стоит минуты, поэтому
   ретрай такого продукта — самый дорогой способ ничего не добиться.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import ValidationError
from sqlalchemy import select

from nutri_radar.config import Settings, get_settings
from nutri_radar.db.models.run import Run, RunStage, RunStatus
from nutri_radar.db.repositories.extraction import ExtractionRepository, ExtractionRow
from nutri_radar.db.session import get_session
from nutri_radar.errors import ExtractionError, LLMUnavailableError
from nutri_radar.extract.corpus import CorpusItem
from nutri_radar.extract.preprocess import PreprocessStats, prepare_text
from nutri_radar.extract.prompts import Prompt, load_prompt
from nutri_radar.extract.schemas import ExtractionResult, SkipReason
from nutri_radar.llm.models import LLMResponse, TokenUsage
from nutri_radar.llm.ports import StructuredLLM
from nutri_radar.logging import safe_extra
from nutri_radar.tracing import NoOpTracer, Tracer

logger = logging.getLogger(__name__)

# Исход обработки одного продукта. Все четыре считаются отдельно: «сколько
# пропущено» без причины пропуска — бесполезное число.
ItemStatus = Literal["ok", "unreadable", "invalid", "unavailable"]


@dataclass(frozen=True)
class _Outcome:
    """Что случилось с одним продуктом."""

    status: ItemStatus
    row: ExtractionRow | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass
class ExtractionRunResult:
    """Итог прогона. Те же числа уезжают в `runs` и в отчёт приёмки."""

    run_id: int | None = None
    model_name: str = ""
    prompt_version: str = ""

    total: int = 0
    # Пропущено на входе, потому что уже разобрано этой версией. Не ошибка,
    # а признак того, что возобновляемость работает.
    already_done: int = 0

    processed: int = 0
    unreadable: int = 0
    invalid: int = 0
    unavailable: int = 0

    batches: int = 0
    elapsed_s: float = 0.0
    usage: TokenUsage = field(default_factory=TokenUsage)
    sugar_forms_total: int = 0

    @property
    def attempted(self) -> int:
        """Сколько продуктов реально ушло в работу на этом запуске."""
        return self.processed + self.invalid + self.unavailable

    @property
    def invalid_share(self) -> float:
        return self.invalid / self.attempted if self.attempted else 0.0

    @property
    def seconds_per_item(self) -> float:
        """Фактическая скорость — для сверки с экстраполяцией замера."""
        return self.elapsed_s / self.attempted if self.attempted else 0.0

    @property
    def mean_sugar_forms(self) -> float:
        return self.sugar_forms_total / self.processed if self.processed else 0.0


async def _open_run(settings: Settings, params: dict[str, object]) -> int:
    """Открыть запись прогона и вернуть её id."""
    async with get_session(settings.db) as session:
        # Незавершённый прогон — след предыдущего обрыва. Не блокируем:
        # запись идемпотентна по (code, model, prompt), повтор безопасен.
        stale = (
            (
                await session.execute(
                    select(Run).where(
                        Run.stage == RunStage.EXTRACT, Run.status == RunStatus.RUNNING
                    )
                )
            )
            .scalars()
            .all()
        )
        if stale:
            logger.warning(
                "Есть незавершённые прогоны извлечения — предыдущий запуск оборвался. "
                "Повтор безопасен: уже разобранное будет пропущено",
                extra=safe_extra(stale_run_ids=[run.id for run in stale]),
            )

        run = Run(stage=RunStage.EXTRACT, status=RunStatus.RUNNING, params=params)
        session.add(run)
        await session.flush()
        return run.id


async def _close_run(
    settings: Settings,
    run_id: int,
    result: ExtractionRunResult,
    *,
    error: str | None = None,
) -> None:
    async with get_session(settings.db) as session:
        run = await session.get(Run, run_id)
        if run is None:
            logger.error("Запись прогона исчезла", extra=safe_extra(run_id=run_id))
            return
        run.status = RunStatus.FAILED if error else RunStatus.COMPLETED
        run.items_processed = result.processed
        # Всё, что не легло в таблицу: невалидные ответы, отказы модели
        # и нечитаемые составы. Доля пропусков — обязательная метрика проекта.
        run.items_skipped = result.invalid + result.unavailable
        run.model_name = result.model_name
        run.prompt_version = result.prompt_version
        run.input_tokens = result.usage.input_tokens
        run.output_tokens = result.usage.output_tokens
        run.error_message = error
        run.finished_at = datetime.now(UTC)


async def _filter_done(
    settings: Settings,
    items: Sequence[CorpusItem],
    *,
    model_name: str,
    prompt_version: str,
) -> list[CorpusItem]:
    """Убрать продукты, уже разобранные этой моделью и версией промпта."""
    codes = [item.code for item in items]
    async with get_session(settings.db) as session:
        done = await ExtractionRepository(session).extracted_codes(
            codes, model_name=model_name, prompt_version=prompt_version
        )
    return [item for item in items if item.code not in done]


def _to_row(
    item: CorpusItem,
    extraction: ExtractionResult,
    *,
    model_name: str,
    prompt_version: str,
    usage: TokenUsage,
    latency_s: float,
    truncated: bool = False,
) -> ExtractionRow:
    """Собрать строку БД из результата извлечения.

    Производные величины берутся из свойств модели, а не пересчитываются
    здесь: иначе появилось бы два места, где считается число форм сахара.

    Разбор состоялся, но `unreadable` мог возникнуть двумя путями: ответ
    упёрся в лимит вывода и список неполон, либо модель сама объявила состав
    нечитаемым. Причины разные, и различать их нужно уже здесь.
    """
    skip_reason: SkipReason | None = None
    if truncated:
        skip_reason = SkipReason.OUTPUT_LIMIT
    elif extraction.unreadable:
        skip_reason = SkipReason.MODEL_UNREADABLE

    return ExtractionRow(
        code=item.code,
        source_lang=item.lang,
        ingredients=[ingredient.model_dump() for ingredient in extraction.ingredients],
        distinct_sugar_forms=extraction.distinct_sugar_forms,
        e_additives_count=extraction.e_additives_count,
        ingredients_count=len(extraction.ingredients),
        allergens=extraction.allergens,
        unreadable=extraction.unreadable,
        skip_reason=skip_reason,
        model_confidence=extraction.model_confidence,
        model_name=model_name,
        prompt_version=prompt_version,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        latency_s=round(latency_s, 3),
    )


def _unreadable_row(
    item: CorpusItem,
    *,
    model_name: str,
    prompt_version: str,
    skip_reason: SkipReason,
    usage: TokenUsage | None = None,
    latency_s: float | None = None,
) -> ExtractionRow:
    """Строка для состава, который разобрать не удалось.

    Два случая. Текст пустой или длиннее контекста — в модель он не уходит,
    токенов нет. Либо ответ оборвался на лимите вывода — вызов состоялся,
    и тогда `usage` с `latency_s` заполнены: строка обязана нести свою
    стоимость, иначе пересчёт цены прогона по таблице занизит её ровно
    на самых дорогих продуктах.

    `skip_reason` обязателен: без него оба случая в таблице неразличимы,
    а лечатся они разным — один размером контекста, другой лимитом вывода.

    Записывается намеренно: иначе такие продукты остались бы «необработанными»
    навсегда и каждый перезапуск снова упирался бы в них. Аналитика записи
    с `unreadable` не берёт (раздел 8 брифа), так что метрики не портятся.
    """
    usage = usage or TokenUsage()
    return ExtractionRow(
        code=item.code,
        source_lang=item.lang,
        unreadable=True,
        skip_reason=skip_reason,
        model_name=model_name,
        prompt_version=prompt_version,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        latency_s=round(latency_s, 3) if latency_s is not None else None,
    )


async def _generate_with_retry(
    llm: StructuredLLM,
    prompt_text: str,
    *,
    schema: dict[str, Any],
    settings: Settings,
    tracer: Tracer,
    code: str,
    prompt_version: str,
) -> LLMResponse | None:
    """Вызвать модель с ретраями. `None` — попытки исчерпаны.

    Ретраится ТОЛЬКО недоступность модели: пауза лечит перезагрузку модели
    в Ollama, но не лечит промпт, который эта модель не понимает.
    """
    extract = settings.extract
    for attempt in range(1, extract.max_retries + 1):
        try:
            with tracer.span("extract_one", code=code, prompt_version=prompt_version):
                return await llm.generate(prompt_text, json_schema=schema)
        except LLMUnavailableError as exc:
            if attempt >= extract.max_retries:
                logger.error(
                    "Модель недоступна, попытки исчерпаны",
                    extra=safe_extra(code=code, attempts=attempt, error=str(exc)),
                )
                return None
            # Экспоненциальная задержка: если Ollama перезагружает модель,
            # частые повторы только мешают ей это доделать.
            delay = extract.retry_backoff_s * 2 ** (attempt - 1)
            logger.warning(
                "Модель недоступна, повтор после паузы",
                extra=safe_extra(code=code, attempt=attempt, delay_s=delay),
            )
            await asyncio.sleep(delay)
    return None


async def _extract_one(
    llm: StructuredLLM,
    item: CorpusItem,
    *,
    prompt: Prompt,
    schema: dict[str, Any],
    settings: Settings,
    semaphore: asyncio.Semaphore,
    tracer: Tracer,
    stats: PreprocessStats,
) -> _Outcome:
    """Разобрать один состав. Исключения наружу не выпускает."""
    model_name = llm.model_name
    prepared = prepare_text(item.ingredients_text, num_ctx=settings.ollama.num_ctx)
    stats.add(prepared)

    if not prepared.is_usable:
        logger.debug(
            "Состав в модель не отправляется",
            extra=safe_extra(
                code=item.code,
                length=prepared.cleaned_length,
                too_long=prepared.too_long,
            ),
        )
        return _Outcome(
            status="unreadable",
            row=_unreadable_row(
                item,
                model_name=model_name,
                prompt_version=prompt.version,
                skip_reason=SkipReason.TOO_LONG if prepared.too_long else SkipReason.EMPTY,
            ),
        )

    rendered = prompt.render(prepared.cleaned, lang=item.lang)

    # Семафор держится на всё время попыток: пауза перед повтором тоже занимает
    # слот. Иначе при недоступной Ollama в неё одновременно ломились бы все
    # задачи батча сразу, как только освободился бы слот.
    async with semaphore:
        try:
            response = await _generate_with_retry(
                llm,
                rendered,
                schema=schema,
                settings=settings,
                tracer=tracer,
                code=item.code,
                prompt_version=prompt.version,
            )
        except ExtractionError as exc:
            # Ответ непригоден по вине данных, а не модели: ретрая не было
            # и не будет. Считаем невалидным — счётчик подряд идущих отказов
            # не трогаем, иначе десяток длинных составов уронит весь прогон.
            #
            # Строка всё равно пишется, и `unreadable` в ней честный: состав
            # не помещается в лимит вывода этой модели. Без записи продукт
            # остался бы «необработанным» навсегда, и каждый перезапуск снова
            # тратил бы на него полную генерацию до лимита.
            logger.warning(
                "Ответ непригоден — продукт помечен нечитаемым",
                extra=safe_extra(code=item.code, error=str(exc)),
            )
            usage = TokenUsage(input_tokens=exc.input_tokens, output_tokens=exc.output_tokens)
            return _Outcome(
                status="invalid",
                row=_unreadable_row(
                    item,
                    model_name=model_name,
                    prompt_version=prompt.version,
                    skip_reason=SkipReason.OUTPUT_LIMIT,
                    usage=usage,
                    latency_s=exc.latency_s,
                ),
                usage=usage,
            )

    if response is None:
        return _Outcome(status="unavailable")

    try:
        extraction = ExtractionResult.model_validate(response.raw_json)
    except ValidationError as exc:
        # Не ретраим: схема задана параметром генерации, и если ответ ей
        # не соответствует — повтор при temperature=0 даст то же самое.
        logger.warning(
            "Невалидный ответ модели — продукт пропущен",
            extra=safe_extra(code=item.code, error=exc.errors()[0]["msg"]),
        )
        return _Outcome(status="invalid", usage=response.usage)

    if response.truncated:
        # Обрезанный ответ проходит схему, но список ингредиентов неполон.
        # Помечаем как нечитаемый, чтобы он не уехал в аналитику как полный.
        extraction.unreadable = True
        logger.warning(
            "Ответ обрезан лимитом вывода — помечен как нечитаемый",
            extra=safe_extra(code=item.code, output_tokens=response.usage.output_tokens),
        )

    row = _to_row(
        item,
        extraction,
        model_name=model_name,
        prompt_version=prompt.version,
        usage=response.usage,
        latency_s=response.latency_s,
        truncated=response.truncated,
    )
    return _Outcome(
        status="unreadable" if extraction.unreadable else "ok",
        row=row,
        usage=response.usage,
    )


async def _write_batch(settings: Settings, rows: list[ExtractionRow]) -> None:
    """Один батч в одной транзакции: обрыв не оставит половину батча."""
    if not rows:
        return
    async with get_session(settings.db) as session:
        await ExtractionRepository(session).upsert_batch(rows)


async def run_extraction(
    llm: StructuredLLM,
    items: Sequence[CorpusItem],
    settings: Settings | None = None,
    *,
    prompt_version: str | None = None,
    tracer: Tracer | None = None,
    resume: bool = True,
    dry_run: bool = False,
) -> ExtractionRunResult:
    """Прогнать корпус через модель и записать результаты.

    Args:
        llm: реализация порта `StructuredLLM`. Конкретный адаптер выбирает
            composition root по `LLM__PROVIDER` — раннер провайдера не знает.
        items: корпус из `select_llm_corpus`. Порядок детерминирован,
            поэтому прогоны разных версий промпта идут по одним продуктам.
        settings: настройки; по умолчанию из `get_settings()`.
        prompt_version: версия промпта; по умолчанию `EXTRACT__PROMPT_VERSION`.
        tracer: трассировка; по умолчанию no-op (Langfuse появляется на M6).
        resume: пропускать уже разобранное этой моделью и версией.
        dry_run: прогнать модель, но ничего не писать в БД и не открывать
            запись прогона. Для отладки промпта на нескольких продуктах.

    Raises:
        LLMUnavailableError: модель отказала `EXTRACT__MAX_CONSECUTIVE_FAILURES`
            раз подряд. Записанное до этого момента сохранено, перезапуск
            продолжит с того же места.
    """
    settings = settings or get_settings()
    tracer = tracer or NoOpTracer()
    version = prompt_version or settings.extract.prompt_version
    prompt = load_prompt(version)
    schema = ExtractionResult.json_schema_for_llm()
    model_name = llm.model_name

    result = ExtractionRunResult(model_name=model_name, prompt_version=version, total=len(items))

    pending = list(items)
    if resume and not dry_run:
        pending = await _filter_done(settings, items, model_name=model_name, prompt_version=version)
        result.already_done = len(items) - len(pending)

    run_params: dict[str, object] = {
        "corpus_size": len(items),
        "pending": len(pending),
        "already_done": result.already_done,
        "prompt_version": version,
        "model": model_name,
        "num_ctx": settings.ollama.num_ctx,
        "max_concurrency": settings.ollama.max_concurrency,
        "batch_size": settings.extract.batch_size,
    }

    logger.info("Прогон извлечения начат", extra=safe_extra(dry_run=dry_run, **run_params))

    if not pending:
        logger.info("Все продукты корпуса уже разобраны этой версией промпта — работы нет")
        return result

    run_id = None if dry_run else await _open_run(settings, run_params)
    result.run_id = run_id

    # Семафор на весь прогон, а не на батч: иначе конец батча простаивал бы,
    # ожидая самый долгий состав, вместо того чтобы брать следующий.
    semaphore = asyncio.Semaphore(settings.ollama.max_concurrency)
    stats = PreprocessStats()
    batch_size = settings.extract.batch_size
    consecutive_failures = 0

    started = time.perf_counter()
    error: str | None = None
    try:
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            result.batches += 1

            outcomes = await asyncio.gather(
                *(
                    _extract_one(
                        llm,
                        item,
                        prompt=prompt,
                        schema=schema,
                        settings=settings,
                        semaphore=semaphore,
                        tracer=tracer,
                        stats=stats,
                    )
                    for item in batch
                )
            )

            rows = [outcome.row for outcome in outcomes if outcome.row is not None]
            if not dry_run:
                await _write_batch(settings, rows)

            for outcome in outcomes:
                result.usage = result.usage + outcome.usage
                match outcome.status:
                    case "ok":
                        result.processed += 1
                        consecutive_failures = 0
                        if outcome.row is not None:
                            result.sugar_forms_total += outcome.row.distinct_sugar_forms
                    case "unreadable":
                        result.processed += 1
                        result.unreadable += 1
                        consecutive_failures = 0
                    case "invalid":
                        result.invalid += 1
                        # Невалидный ответ — это работающая модель, а не отказ:
                        # счётчик подряд идущих отказов не трогаем.
                    case "unavailable":
                        result.unavailable += 1
                        consecutive_failures += 1

            elapsed = time.perf_counter() - started
            logger.info(
                "Прогресс извлечения",
                extra=safe_extra(
                    batch=result.batches,
                    done=result.attempted,
                    pending=len(pending) - result.attempted,
                    invalid=result.invalid,
                    unavailable=result.unavailable,
                    s_per_item=round(elapsed / max(result.attempted, 1), 2),
                    tokens=result.usage.total,
                ),
            )

            if consecutive_failures >= settings.extract.max_consecutive_failures:
                raise LLMUnavailableError(
                    f"Модель отказала {consecutive_failures} раз подряд — прогон остановлен. "
                    "Записанное сохранено, перезапуск продолжит с этого места."
                )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.error("Прогон извлечения прерван", exc_info=True)
        raise
    finally:
        result.elapsed_s = round(time.perf_counter() - started, 2)
        # Прогон закрывается и при отказе: иначе упавший запуск навсегда
        # остаётся в статусе running и предупреждение о незавершённых
        # прогонах начнёт срабатывать на каждом следующем запуске.
        if run_id is not None:
            await _close_run(settings, run_id, result, error=error)
        stats.log_summary()

    _log_result(result, settings, dry_run=dry_run)
    return result


def _log_result(result: ExtractionRunResult, settings: Settings, *, dry_run: bool) -> None:
    logger.info(
        "Прогон извлечения завершён",
        extra=safe_extra(
            run_id=result.run_id,
            dry_run=dry_run,
            model=result.model_name,
            prompt_version=result.prompt_version,
            processed=result.processed,
            already_done=result.already_done,
            unreadable=result.unreadable,
            invalid=result.invalid,
            unavailable=result.unavailable,
            elapsed_s=result.elapsed_s,
            s_per_item=round(result.seconds_per_item, 2),
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            mean_sugar_forms=round(result.mean_sugar_forms, 2),
        ),
    )

    if result.invalid_share > settings.extract.max_invalid_share:
        logger.warning(
            "Доля невалидных ответов выше порога — результаты прогона ненадёжны",
            extra=safe_extra(
                invalid_share=f"{result.invalid_share:.1%}",
                threshold=f"{settings.extract.max_invalid_share:.1%}",
            ),
        )


def format_result(result: ExtractionRunResult) -> str:
    """Человекочитаемая сводка для CLI и для отчёта приёмки."""
    return "\n".join(
        [
            f"Модель:            {result.model_name}",
            f"Промпт:            {result.prompt_version}",
            f"Корпус:            {result.total}",
            f"Уже было разобрано: {result.already_done}",
            "",
            f"Обработано:        {result.processed}",
            f"  из них unreadable: {result.unreadable}",
            f"Невалидных:        {result.invalid} ({result.invalid_share:.1%})",
            f"Отказов модели:    {result.unavailable}",
            "",
            f"Время:             {result.elapsed_s:.0f} с "
            f"({result.seconds_per_item:.2f} с на продукт)",
            f"Токены:            вход {result.usage.input_tokens}, "
            f"выход {result.usage.output_tokens}",
            f"Форм сахара:       {result.mean_sugar_forms:.2f} в среднем",
        ]
    )
