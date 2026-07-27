"""Замер перед полным прогоном.

Прямое требование раздела 3a брифа: перед полным прогоном — замер на 20
продуктах, секунды на продукт, токены, оценка общего времени, результат
в отчёт.

Это не удобство, а страховка. Полный прогон идёт часами, и узнать его
стоимость нужно **до**, а не после. Отдельно замер отвечает на вопрос, можно
ли вообще запускать эту версию промпта: если доля невалидных ответов высока,
сутки работы GPU дадут мусор.

Медиана и p95, а не среднее: время на продукт зависит от длины состава,
и среднее по длинному хвосту даёт оптимистичную оценку.
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from nutri_radar.config import Settings, get_settings
from nutri_radar.errors import ExtractionError, LLMUnavailableError
from nutri_radar.extract.corpus import CorpusItem
from nutri_radar.extract.preprocess import prepare_text
from nutri_radar.extract.prompts import load_prompt
from nutri_radar.extract.schemas import ExtractionResult
from nutri_radar.llm.ports import StructuredLLM
from nutri_radar.logging import safe_extra

logger = logging.getLogger(__name__)

REPORTS_DIR = Path("reports")


@dataclass
class BenchmarkResult:
    """Итог замера одной версии промпта."""

    prompt_version: str
    model_name: str
    sample_size: int = 0

    latencies: list[float] = field(default_factory=list)
    input_tokens: list[int] = field(default_factory=list)
    output_tokens: list[int] = field(default_factory=list)
    sugar_forms: list[int] = field(default_factory=list)
    ingredients_counts: list[int] = field(default_factory=list)

    invalid: int = 0
    unavailable: int = 0
    unreadable: int = 0
    truncated: int = 0

    @property
    def succeeded(self) -> int:
        return len(self.latencies)

    @property
    def invalid_share(self) -> float:
        total = self.succeeded + self.invalid
        return self.invalid / total if total else 0.0

    @property
    def median_latency(self) -> float:
        return statistics.median(self.latencies) if self.latencies else 0.0

    @property
    def p95_latency(self) -> float:
        """95-й перцентиль. На маленькой выборке это просто верхняя граница."""
        if not self.latencies:
            return 0.0
        ordered = sorted(self.latencies)
        index = min(int(len(ordered) * 0.95), len(ordered) - 1)
        return ordered[index]

    @property
    def mean_input_tokens(self) -> float:
        return statistics.mean(self.input_tokens) if self.input_tokens else 0.0

    @property
    def mean_output_tokens(self) -> float:
        return statistics.mean(self.output_tokens) if self.output_tokens else 0.0

    @property
    def mean_sugar_forms(self) -> float:
        return statistics.mean(self.sugar_forms) if self.sugar_forms else 0.0

    @property
    def zero_sugar_share(self) -> float:
        """Доля составов, где не найдено ни одной формы сахара.

        Ключевой индикатор для этого проекта: корпус — снеки, кондитерка
        и сладкие напитки, поэтому поголовные нули означают, что промпт
        не работает, а не что сахара нет.
        """
        if not self.sugar_forms:
            return 0.0
        return sum(1 for value in self.sugar_forms if value == 0) / len(self.sugar_forms)

    def extrapolate(self, corpus_size: int, concurrency: int = 1) -> tuple[float, int]:
        """Оценка полного прогона: (часы, суммарные токены).

        Считается по медиане **без деления на параллелизм**. Раньше здесь стояло
        деление — предполагалось, что два одновременных запроса дают двукратное
        ускорение. Измерение это опровергло: на одних и тех же 40 продуктах
        прогон с параллелизмом 2 дал 20,96 с на продукт против медианы 14,20 с
        последовательного замера. На 6 ГБ VRAM модель занимает GPU целиком,
        и второй запрос не выполняется параллельно, а ждёт своей очереди,
        добавляя накладные расходы.

        Аргумент `concurrency` оставлен ради совместимости вызовов и потому,
        что на другом железе ускорение может появиться. Но по умолчанию оценка
        консервативная: лучше переоценить время прогона, чем недооценить.
        """
        if not self.latencies:
            return 0.0, 0
        seconds = self.median_latency * corpus_size
        tokens = int((self.mean_input_tokens + self.mean_output_tokens) * corpus_size)
        return seconds / 3600, tokens


async def run_benchmark(
    llm: StructuredLLM,
    items: list[CorpusItem],
    settings: Settings | None = None,
    *,
    prompt_version: str | None = None,
) -> BenchmarkResult:
    """Прогнать выборку через модель и собрать статистику.

    Выборка берётся с начала уже отобранного корпуса — она детерминирована
    тем же seed, поэтому замеры разных версий промпта идут по **одним и тем же**
    продуктам и сравнимы между собой.
    """
    settings = settings or get_settings()
    version = prompt_version or settings.extract.prompt_version
    prompt = load_prompt(version)
    schema = ExtractionResult.json_schema_for_llm()

    result = BenchmarkResult(
        prompt_version=version, model_name=llm.model_name, sample_size=len(items)
    )

    logger.info(
        "Замер запущен",
        extra=safe_extra(
            prompt_version=version,
            model=llm.model_name,
            sample_size=len(items),
            num_ctx=settings.ollama.num_ctx,
        ),
    )

    for index, item in enumerate(items, start=1):
        prepared = prepare_text(item.ingredients_text, num_ctx=settings.ollama.num_ctx)
        if not prepared.is_usable:
            result.invalid += 1
            continue

        try:
            response = await llm.generate(
                prompt.render(prepared.cleaned, lang=item.lang), json_schema=schema
            )
        except LLMUnavailableError:
            result.unavailable += 1
            logger.error("Модель недоступна на замере", extra=safe_extra(code=item.code))
            continue
        except ExtractionError as exc:
            # Обрыв на лимите вывода — это отказ данных, а не модели. В замере
            # он должен попадать в долю невалидных, иначе экстраполяция
            # обещает полный прогон там, где часть продуктов не разбирается.
            result.invalid += 1
            logger.warning(
                "Непригодный ответ на замере",
                extra=safe_extra(code=item.code, error=str(exc)),
            )
            continue

        try:
            extraction = ExtractionResult.model_validate(response.raw_json)
        except ValidationError as exc:
            result.invalid += 1
            logger.warning(
                "Невалидный ответ на замере",
                extra=safe_extra(code=item.code, error=exc.errors()[0]["msg"]),
            )
            continue

        result.latencies.append(response.latency_s)
        result.input_tokens.append(response.usage.input_tokens)
        result.output_tokens.append(response.usage.output_tokens)
        result.sugar_forms.append(extraction.distinct_sugar_forms)
        result.ingredients_counts.append(len(extraction.ingredients))
        if extraction.unreadable:
            result.unreadable += 1
        if response.truncated:
            result.truncated += 1

        # На 20 продуктах лог по каждому уместен и полезен: видно разброс.
        logger.info(
            "Замер: продукт обработан",
            extra=safe_extra(
                n=index,
                code=item.code,
                lang=item.lang,
                latency_s=round(response.latency_s, 2),
                sugar_forms=extraction.distinct_sugar_forms,
                ingredients=len(extraction.ingredients),
            ),
        )

    _log_result(result, settings)
    return result


def _log_result(result: BenchmarkResult, settings: Settings) -> None:
    hours, tokens = result.extrapolate(settings.extract.corpus_size)
    logger.info(
        "Замер завершён",
        extra=safe_extra(
            prompt_version=result.prompt_version,
            succeeded=result.succeeded,
            invalid=result.invalid,
            median_s=round(result.median_latency, 2),
            p95_s=round(result.p95_latency, 2),
            projected_hours=round(hours, 1),
            projected_tokens=tokens,
        ),
    )

    if result.invalid_share > settings.extract.max_invalid_share:
        logger.warning(
            "Доля невалидных ответов выше порога — полный прогон запускать нельзя",
            extra=safe_extra(
                invalid_share=f"{result.invalid_share:.1%}",
                threshold=f"{settings.extract.max_invalid_share:.1%}",
            ),
        )
    if hours > settings.extract.max_run_hours:
        logger.warning(
            "Экстраполяция превышает порог — стоит сузить корпус, а не ждать",
            extra=safe_extra(
                projected_hours=round(hours, 1), threshold=settings.extract.max_run_hours
            ),
        )
    if result.zero_sugar_share > 0.5:
        logger.warning(
            "Больше половины составов без единой формы сахара — на корпусе из "
            "снеков и сладких напитков это признак неработающего промпта",
            extra=safe_extra(zero_sugar_share=f"{result.zero_sugar_share:.0%}"),
        )


def format_result(result: BenchmarkResult, settings: Settings) -> str:
    """Человекочитаемый отчёт."""
    hours, tokens = result.extrapolate(settings.extract.corpus_size)
    return "\n".join(
        [
            f"Промпт:            {result.prompt_version}",
            f"Модель:            {result.model_name}",
            f"Выборка:           {result.sample_size} продуктов",
            "",
            f"Успешно:           {result.succeeded}",
            f"Невалидных:        {result.invalid} ({result.invalid_share:.1%})",
            f"Модель недоступна: {result.unavailable}",
            f"unreadable:        {result.unreadable}",
            f"Обрезанных:        {result.truncated}",
            "",
            f"Секунд на продукт: медиана {result.median_latency:.2f}, p95 {result.p95_latency:.2f}",
            f"Токенов вход:      {result.mean_input_tokens:.0f} в среднем",
            f"Токенов выход:     {result.mean_output_tokens:.0f} в среднем",
            "",
            f"Форм сахара:       {result.mean_sugar_forms:.2f} в среднем",
            f"Составов без сахара: {result.zero_sugar_share:.0%}",
            f"Ингредиентов:      "
            f"{statistics.mean(result.ingredients_counts) if result.ingredients_counts else 0:.1f}"
            " в среднем",
            "",
            f"ЭКСТРАПОЛЯЦИЯ на {settings.extract.corpus_size} продуктов:",
            f"  время:  {hours:.1f} часов (по медиане, без скидки на параллелизм —",
            "           измерение показало, что на 6 ГБ VRAM он не ускоряет)",
            f"  токены: {tokens}",
        ]
    )


def write_report(result: BenchmarkResult, settings: Settings) -> Path:
    """Сохранить отчёт в `reports/`. Требование DoD: замер в отчёт."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_model = result.model_name.replace(":", "_").replace("/", "_")
    path = REPORTS_DIR / f"benchmark_{safe_model}_{result.prompt_version}.md"

    body = "\n".join(
        [
            f"# Замер извлечения: {result.prompt_version}",
            "",
            f"Дата: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "```",
            format_result(result, settings),
            "```",
            "",
            "Замер обязателен до полного прогона (раздел 3a брифа).",
            "Медиана и p95, а не среднее: время зависит от длины состава,",
            "и среднее по длинному хвосту даёт оптимистичную оценку.",
        ]
    )
    path.write_text(body, encoding="utf-8")
    logger.info("Отчёт замера сохранён", extra=safe_extra(path=str(path)))
    return path


def measure_wall_clock(started: float) -> float:
    """Фактическое время прогона — для сверки с экстраполяцией."""
    return time.perf_counter() - started
