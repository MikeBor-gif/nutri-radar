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

Базлайн хранит **отпечаток эталона**, на котором он снят. Разметка идёт
заходами по языкам, и без отпечатка добавление новых языков сдвинуло бы
метрики без единой правки кода: гейт объявил бы регрессией смену линейки,
а не ухудшение продукта. Хуже того, он мог бы и промолчать — если новые
продукты окажутся легче, реальная просадка утонула бы в среднем.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
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


@dataclass(frozen=True)
class GoldFingerprint:
    """Отпечаток эталона, на котором снят базлайн.

    Числа хранятся ради человека, читающего `baseline.json`: «40 продуктов,
    de 20, ru 20» сразу говорит, на чём мерили. `digest` — ради машины:
    состав можно поменять, не меняя счётчиков, и тогда сравнение молча
    поедет.
    """

    products: int
    by_lang: dict[str, int]
    digest: str

    def describe(self) -> str:
        langs = ", ".join(f"{lang}: {count}" for lang, count in sorted(self.by_lang.items()))
        return f"{self.products} продуктов ({langs})"

    def to_json(self) -> dict[str, object]:
        return {
            "products": self.products,
            "by_lang": dict(sorted(self.by_lang.items())),
            "digest": self.digest,
        }

    @classmethod
    def from_json(cls, data: dict[str, object]) -> GoldFingerprint:
        by_lang = data.get("by_lang") or {}
        return cls(
            products=int(data.get("products", 0)),
            by_lang={str(k): int(v) for k, v in dict(by_lang).items()},
            digest=str(data.get("digest", "")),
        )


@dataclass(frozen=True)
class Baseline:
    """Зафиксированные метрики вместе с эталоном, на котором они сняты."""

    systems: dict[str, dict[str, float]] = field(default_factory=dict)
    gold: GoldFingerprint | None = None

    def __bool__(self) -> bool:
        return bool(self.systems)


@dataclass
class GateResult:
    """Итог проверки."""

    passed: bool
    skipped: bool = False
    reason: str = ""
    regressions: list[Regression] = field(default_factory=list)
    current: dict[str, dict[str, float]] = field(default_factory=dict)
    # Базлайн снят на другом эталоне. Это не просадка качества, и путать
    # одно с другим нельзя: чинится оно не кодом, а перефиксацией базлайна.
    stale: bool = False


def gold_fingerprint(gold: list[GoldRecord]) -> GoldFingerprint:
    """Посчитать отпечаток эталона.

    Хеш берётся от отсортированных кодов: порядок строк в JSONL зависит от
    того, в каком порядке человек размечал, и различать прогоны по нему
    значило бы краснеть от перестановки.
    """
    codes = sorted(record.code for record in gold)
    digest = hashlib.sha256("\n".join(codes).encode("utf-8")).hexdigest()[:12]
    return GoldFingerprint(
        products=len(gold),
        by_lang=dict(Counter(record.lang for record in gold)),
        digest=digest,
    )


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


def load_baseline(path: Path | None = None) -> Baseline:
    """Прочитать зафиксированный базлайн. Нет файла — пустой базлайн."""
    file = path or BASELINE_FILE
    if not file.exists():
        logger.info("Базлайн не найден", extra=safe_extra(path=str(file)))
        return Baseline()

    data = json.loads(file.read_text(encoding="utf-8"))
    systems = {system: dict(metrics) for system, metrics in dict(data.get("systems", {})).items()}
    gold_data = data.get("gold")
    gold = GoldFingerprint.from_json(dict(gold_data)) if gold_data else None
    logger.debug(
        "Базлайн прочитан",
        extra=safe_extra(
            path=str(file),
            systems=len(systems),
            gold=gold.describe() if gold else "нет отпечатка",
        ),
    )
    return Baseline(systems=systems, gold=gold)


def write_baseline(
    snapshot: dict[str, dict[str, float]],
    path: Path | None = None,
    *,
    gold: GoldFingerprint | None = None,
) -> Path:
    """Зафиксировать базлайн.

    Отдельная команда, а не автоматическое обновление при каждом прогоне:
    базлайн, который переписывается сам, не сторожит ничего — любая просадка
    молча становится новой нормой.

    Args:
        snapshot: метрики по системам.
        path: куда писать; по умолчанию `data/evals/baseline.json`.
        gold: отпечаток эталона, на котором сняты метрики. Без него гейт
            не отличит просадку качества от смены линейки.
    """
    file = path or BASELINE_FILE
    file.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"systems": snapshot}
    if gold is not None:
        payload["gold"] = gold.to_json()
    file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Базлайн записан",
        extra=safe_extra(
            path=str(file),
            systems=len(snapshot),
            gold=gold.describe() if gold else "без отпечатка",
        ),
    )
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

    fingerprint = gold_fingerprint(gold)
    if baseline.gold is not None and baseline.gold != fingerprint:
        # Красный, а не пропуск: базлайн, снятый на другом эталоне, не сторожит
        # ничего, и зелёная сборка означала бы «проверено», хотя не проверено.
        # Но и не просадка: причина не в коде, и чинится она перефиксацией.
        reason = (
            f"Эталон изменился с момента фиксации базлайна: было "
            f"{baseline.gold.describe()}, стало {fingerprint.describe()}. "
            "Метрики на разных наборах не сравнимы — пересчитайте базлайн "
            "командой `evals baseline` и перепроверьте отчёт."
        )
        logger.warning(
            "Базлайн снят на другом эталоне",
            extra=safe_extra(
                baseline_gold=baseline.gold.describe(),
                current_gold=fingerprint.describe(),
                baseline_digest=baseline.gold.digest,
                current_digest=fingerprint.digest,
            ),
        )
        return GateResult(passed=False, stale=True, reason=reason, current=current)

    regressions: list[Regression] = []
    for system, metrics in baseline.systems.items():
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

    if result.stale:
        lines = [f"ГЕЙТ НЕ ПРОЙДЕН: {result.reason}", "", "Текущие метрики:"]
        for system, metrics in sorted(result.current.items()):
            values = ", ".join(f"{name} {value:.3f}" for name, value in sorted(metrics.items()))
            lines.append(f"  {system}: {values}")
        return "\n".join(lines)

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
