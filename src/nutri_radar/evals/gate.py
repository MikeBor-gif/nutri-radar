"""Гейт качества для CI: пересчёт метрик из закоммиченных файлов.

Требование раздела «Ограничения» спецификации: гейт работает **без GPU,
без БД и без сети**. Отсюда вся его конструкция — чистая функция от трёх
файлов в репозитории: эталон, предсказания систем, зафиксированный базлайн.
Словарь алиасов тоже читается с диска, а не из базы.

Из этого следует главное свойство: **гейт детерминирован**. Он не вызывает
модель и не пересчитывает предсказания, поэтому не может позеленеть или
покраснеть от разброса между прогонами, задокументированного в ADR-018.
Красный гейт означает, что изменился код, а не погода.

Пустой эталон — не провал, а пропуск. Разметку ведёт человек (правило 6),
и держать сборку красной, пока она не закончена, значит приучить всех
не смотреть на неё.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from nutri_radar.config import Settings, get_settings
from nutri_radar.evals.metrics import ComparisonResult, score_system
from nutri_radar.evals.schemas import (
    GOLD_FILE,
    PREDICTIONS_DIR,
    GoldRecord,
    PredictionRecord,
    read_jsonl,
)
from nutri_radar.extract.normalize import load_seed_index
from nutri_radar.logging import safe_extra

logger = logging.getLogger(__name__)

BASELINE_FILE = Path("data/evals/baseline.json")


@dataclass(frozen=True)
class Regression:
    """Просадка одной метрики у одной системы."""

    system: str
    metric: str
    baseline: float
    current: float

    @property
    def drop_points(self) -> float:
        """Просадка в пунктах. Метрики хранятся долями, порог задан в пунктах."""
        return (self.baseline - self.current) * 100

    def __str__(self) -> str:
        return (
            f"{self.system} / {self.metric}: было {self.baseline:.3f}, "
            f"стало {self.current:.3f} (−{self.drop_points:.1f} пункта)"
        )


@dataclass
class GateResult:
    """Итог проверки."""

    passed: bool
    skipped: bool = False
    reason: str = ""
    regressions: list[Regression] = field(default_factory=list)
    current: dict[str, dict[str, float]] = field(default_factory=dict)


def metrics_snapshot(result: ComparisonResult) -> dict[str, float]:
    """Что именно сторожит гейт.

    Только те величины, чья просадка означает ухудшение продукта. Токены
    и латентность сюда не входят: они меняются от смены модели и железа,
    и падать из-за них сборке незачем.
    """
    score = result.overall
    return {
        "f1_ingredients": round(score.ingredients.f1, 4),
        "f1_typed": round(score.typed.f1, 4),
        "sugar_accuracy": round(score.sugar.accuracy, 4),
    }


def load_baseline(path: Path | None = None) -> dict[str, dict[str, float]]:
    """Прочитать зафиксированный базлайн. Нет файла — пустой словарь."""
    file = path or BASELINE_FILE
    if not file.exists():
        logger.info("Базлайн не найден", extra=safe_extra(path=str(file)))
        return {}
    data = json.loads(file.read_text(encoding="utf-8"))
    return {system: dict(metrics) for system, metrics in data.items()}


def write_baseline(snapshot: dict[str, dict[str, float]], path: Path | None = None) -> Path:
    """Зафиксировать базлайн.

    Отдельная команда, а не автоматическое обновление при каждом прогоне:
    базлайн, который переписывается сам, не сторожит ничего — любая просадка
    молча становится новой нормой.
    """
    file = path or BASELINE_FILE
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("Базлайн записан", extra=safe_extra(path=str(file), systems=len(snapshot)))
    return file


def collect_current(
    gold: list[GoldRecord],
    predictions_dir: Path,
) -> dict[str, ComparisonResult]:
    """Посчитать метрики всех систем, чьи предсказания лежат в каталоге."""
    index = load_seed_index()
    results: dict[str, ComparisonResult] = {}

    for file in sorted(predictions_dir.glob("*.jsonl")):
        predictions = read_jsonl(file, PredictionRecord)
        if not predictions:
            logger.warning("Пустой файл предсказаний", extra=safe_extra(path=str(file)))
            continue
        result = score_system(gold, predictions, index)
        results[result.system] = result

    return results


def run_gate(
    settings: Settings | None = None,
    *,
    gold_path: Path | None = None,
    predictions_dir: Path | None = None,
    baseline_path: Path | None = None,
) -> GateResult:
    """Проверить, не просели ли метрики против базлайна.

    Returns:
        Итог с перечнем просадок. `skipped=True`, если сравнивать не с чем —
        это не провал: эталон размечает человек, и до конца разметки держать
        сборку красной бессмысленно.
    """
    settings = settings or get_settings()
    max_drop = settings.evals.max_f1_drop

    gold = read_jsonl(gold_path or GOLD_FILE, GoldRecord)
    if not gold:
        reason = "Эталон пуст — разметка ещё не сделана, проверять нечего."
        logger.info("Гейт пропущен", extra=safe_extra(reason=reason))
        return GateResult(passed=True, skipped=True, reason=reason)

    results = collect_current(gold, predictions_dir or PREDICTIONS_DIR)
    if not results:
        reason = "Нет файлов предсказаний — сравнивать нечего."
        logger.info("Гейт пропущен", extra=safe_extra(reason=reason))
        return GateResult(passed=True, skipped=True, reason=reason)

    current = {system: metrics_snapshot(result) for system, result in results.items()}
    baseline = load_baseline(baseline_path)
    if not baseline:
        reason = (
            "Базлайн не зафиксирован — первый прогон. Зафиксируйте его командой `evals baseline`."
        )
        logger.info("Гейт пропущен", extra=safe_extra(reason=reason))
        return GateResult(passed=True, skipped=True, reason=reason, current=current)

    regressions: list[Regression] = []
    for system, metrics in baseline.items():
        if system not in current:
            # Система пропала из предсказаний. Это тоже регрессия: сравнение
            # обещало четыре системы, а показывает три.
            regressions.append(
                Regression(system=system, metric="нет предсказаний", baseline=1.0, current=0.0)
            )
            continue
        for metric, was in metrics.items():
            now = current[system].get(metric, 0.0)
            if (was - now) * 100 > max_drop:
                regressions.append(
                    Regression(system=system, metric=metric, baseline=was, current=now)
                )

    passed = not regressions
    logger.info(
        "Гейт отработал",
        extra=safe_extra(
            passed=passed,
            systems=len(current),
            regressions=len(regressions),
            max_drop=max_drop,
        ),
    )
    return GateResult(passed=passed, regressions=regressions, current=current)


def format_gate(result: GateResult, max_drop: float) -> str:
    """Человекочитаемый итог для лога CI."""
    if result.skipped:
        return f"ГЕЙТ ПРОПУЩЕН: {result.reason}"

    lines = [f"Порог просадки: {max_drop:.1f} пункта F1"]
    for system, metrics in sorted(result.current.items()):
        values = ", ".join(f"{name} {value:.3f}" for name, value in sorted(metrics.items()))
        lines.append(f"  {system}: {values}")

    if result.passed:
        lines.append("\nГЕЙТ ПРОЙДЕН: просадок нет.")
    else:
        lines.append("\nГЕЙТ НЕ ПРОЙДЕН:")
        lines.extend(f"  {regression}" for regression in result.regressions)
    return "\n".join(lines)
