"""Метрики сравнения систем.

От правила сопоставления зависят **все** числа майлстоуна, поэтому оно
зафиксировано здесь явно и до прогона, а не подогнано под результат.

**Имена сопоставляются через словарь алиасов.** «Сироп глюкозы», `glucose
syrup` и `Glukosesirup` — одна сущность; считать их разными значит наказывать
систему за язык исходного состава, а не за ошибку.

**Типы сравниваются свои у каждой стороны.** Это неочевидная и важная деталь.
`normalize_ingredient` даёт типу из словаря побеждать тип модели — так и надо
в продуктовом коде, но в метрике это подлог: прогони обе стороны через словарь,
и у всех известных ингредиентов типы совпадут по построению, а F1 по типам
уедет к единице, измеряя словарь вместо систем. Поэтому для сопоставления
берётся каноническое имя из словаря, а тип — тот, что система реально выдала.

**Микро-усреднение, а не макро.** Складываются TP/FP/FN по всем продуктам,
и уже из суммы считается F1. На выборке в 100 продуктов, где у одного состава
два ингредиента, а у другого двадцать, макро-усреднение дало бы короткому
составу тот же вес, что длинному.
"""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass, field

from nutri_radar.evals.schemas import GoldRecord, PredictionRecord
from nutri_radar.extract.normalize import AliasIndex, normalize_key
from nutri_radar.extract.schemas import ExtractionResult, IngredientKind
from nutri_radar.logging import safe_extra

logger = logging.getLogger(__name__)


def match_key(name: str, index: AliasIndex, *, lang: str | None = None) -> str:
    """Ключ, по которому ингредиент эталона сопоставляется с предсказанным.

    Сначала словарь алиасов, затем — нормализованное имя как есть. Второй шаг
    обязателен: словарь неполон по построению (его ведёт человек), и без
    отката незнакомые имена не совпадали бы даже сами с собой.
    """
    entry = index.lookup(name, lang=lang)
    if entry is not None:
        return normalize_key(entry.canonical_name)
    return normalize_key(name)


def name_set(
    extraction: ExtractionResult, index: AliasIndex, *, lang: str | None = None
) -> set[str]:
    """Множество канонических имён."""
    keys = {match_key(item.canonical_name, index, lang=lang) for item in extraction.ingredients}
    return {key for key in keys if key}


def typed_set(
    extraction: ExtractionResult, index: AliasIndex, *, lang: str | None = None
) -> set[tuple[str, str]]:
    """Множество пар «имя, тип».

    Тип берётся **из самой системы**, а не из словаря: иначе метрика
    сравнивала бы словарь с самим собой.
    """
    pairs = {
        (match_key(item.canonical_name, index, lang=lang), item.kind.value)
        for item in extraction.ingredients
    }
    return {(name, kind) for name, kind in pairs if name}


def sugar_forms(extraction: ExtractionResult, index: AliasIndex, *, lang: str | None = None) -> int:
    """Число разных форм сахара после канонизации имён.

    Считается по именам из словаря, но по типу самой системы — по той же
    причине, что и `typed_set`. Именно эта величина — главная фича проекта,
    и мерить её надо честно.
    """
    names = {
        match_key(item.canonical_name, index, lang=lang)
        for item in extraction.ingredients
        if item.kind is IngredientKind.SUGAR
    }
    return len({name for name in names if name})


@dataclass
class PrfScore:
    """Precision / recall / F1, накопленные по продуктам."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, gold: set, predicted: set) -> None:
        self.tp += len(gold & predicted)
        self.fp += len(predicted - gold)
        self.fn += len(gold - predicted)

    @property
    def precision(self) -> float:
        total = self.tp + self.fp
        return self.tp / total if total else 0.0

    @property
    def recall(self) -> float:
        total = self.tp + self.fn
        return self.tp / total if total else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass
class SugarScore:
    """Насколько верно система считает формы сахара.

    Две метрики, потому что они отвечают на разные вопросы. Доля точных
    попаданий — «как часто число верно». Средняя ошибка — «насколько сильно
    промахивается, когда промахивается». Отдельно считается, сколько раз
    система вернула ноль там, где человек нашёл сахар: это прямой ответ
    на вопрос M2 о 44% составов без единой формы.
    """

    exact: int = 0
    total: int = 0
    abs_errors: list[int] = field(default_factory=list)
    missed_all: int = 0
    gold_zero: int = 0

    def add(self, gold_count: int, predicted_count: int) -> None:
        self.total += 1
        self.abs_errors.append(abs(gold_count - predicted_count))
        if gold_count == predicted_count:
            self.exact += 1
        if gold_count == 0:
            self.gold_zero += 1
        elif predicted_count == 0:
            self.missed_all += 1

    @property
    def accuracy(self) -> float:
        return self.exact / self.total if self.total else 0.0

    @property
    def mae(self) -> float:
        return statistics.mean(self.abs_errors) if self.abs_errors else 0.0


@dataclass
class SystemScore:
    """Итог по одной системе."""

    system: str
    ingredients: PrfScore = field(default_factory=PrfScore)
    typed: PrfScore = field(default_factory=PrfScore)
    sugar: SugarScore = field(default_factory=SugarScore)

    products: int = 0
    # Продукты, по которым система не дала ответа вовсе. Считаются отдельно
    # для отчёта, но в метрику входят как пустой ответ — со всеми эталонными
    # ингредиентами в FN. Иначе система, промолчавшая на трети выборки,
    # получила бы recall по одной трети, где справилась, и выглядела бы
    # лучше той, что ответила на всех.
    missing: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latencies: list[float] = field(default_factory=list)

    @property
    def median_latency(self) -> float:
        return statistics.median(self.latencies) if self.latencies else 0.0


@dataclass
class ComparisonResult:
    """Сравнение одной системы: итог целиком и разбивка по языкам."""

    overall: SystemScore
    by_lang: dict[str, SystemScore] = field(default_factory=dict)

    @property
    def system(self) -> str:
        return self.overall.system


def score_system(
    gold: list[GoldRecord],
    predictions: list[PredictionRecord],
    index: AliasIndex,
) -> ComparisonResult:
    """Посчитать метрики одной системы против эталона.

    Args:
        gold: эталонные записи.
        predictions: ответы системы. Порядок не важен — сопоставление по коду.
        index: словарь алиасов для канонизации имён.

    Returns:
        Итог целиком и разбивка по языкам. Разбивка обязательна: на ней
        держится ответ на вопрос M2 о том, почему русские составы отстают.
    """
    if not predictions:
        raise ValueError("Пустой список предсказаний: нечего сравнивать")

    system = predictions[0].system
    by_code = {record.code: record for record in predictions}
    overall = SystemScore(system=system)
    by_lang: dict[str, SystemScore] = defaultdict(lambda: SystemScore(system=system))

    for record in gold:
        lang_score = by_lang[record.lang]
        prediction = by_code.get(record.code)

        overall.products += 1
        lang_score.products += 1

        if prediction is None:
            # Нет ответа — это тоже результат, и он идёт в метрику как пустой
            # ответ: все эталонные ингредиенты попадают в FN. Просто пропустить
            # продукт значило бы посчитать метрику только там, где система
            # справилась, и молчание выглядело бы как безошибочность.
            overall.missing += 1
            lang_score.missing += 1
            logger.debug(
                "Нет предсказания по продукту — засчитано как пустой ответ",
                extra=safe_extra(system=system, code=record.code),
            )

        predicted = prediction.extraction if prediction is not None else ExtractionResult()

        gold_names = name_set(record.extraction, index, lang=record.lang)
        pred_names = name_set(predicted, index, lang=record.lang)
        gold_typed = typed_set(record.extraction, index, lang=record.lang)
        pred_typed = typed_set(predicted, index, lang=record.lang)
        gold_sugar = sugar_forms(record.extraction, index, lang=record.lang)
        pred_sugar = sugar_forms(predicted, index, lang=record.lang)

        for score in (overall, lang_score):
            score.ingredients.add(gold_names, pred_names)
            score.typed.add(gold_typed, pred_typed)
            score.sugar.add(gold_sugar, pred_sugar)
            if prediction is not None:
                score.input_tokens += prediction.input_tokens
                score.output_tokens += prediction.output_tokens
                if prediction.latency_s:
                    score.latencies.append(prediction.latency_s)

    logger.info(
        "Метрики посчитаны",
        extra=safe_extra(
            system=system,
            products=overall.products,
            missing=overall.missing,
            f1_ingredients=round(overall.ingredients.f1, 3),
            f1_typed=round(overall.typed.f1, 3),
            sugar_accuracy=round(overall.sugar.accuracy, 3),
        ),
    )
    return ComparisonResult(overall=overall, by_lang=dict(by_lang))


def format_comparison(results: list[ComparisonResult]) -> str:
    """Таблица сравнения систем в markdown.

    Стоимость намеренно рядом с качеством: система на пункт лучше и в сто раз
    дороже — это другой ответ на вопрос «что брать».
    """
    lines = [
        "| Система | F1 ингр. | F1 типов | Точность форм сахара | MAE | Нет ответа | Токены |",
        "|---|---|---|---|---|---|---|",
    ]
    for result in results:
        score = result.overall
        lines.append(
            f"| {score.system} "
            f"| {score.ingredients.f1:.3f} "
            f"| {score.typed.f1:.3f} "
            f"| {score.sugar.accuracy:.3f} "
            f"| {score.sugar.mae:.2f} "
            f"| {score.missing} "
            f"| {score.input_tokens + score.output_tokens} |"
        )
    return "\n".join(lines)


def format_by_language(result: ComparisonResult) -> str:
    """Разбивка одной системы по языкам.

    Размер выборки в таблице обязателен: на 20 продуктах доверительный
    интервал шире многих наблюдаемых разниц, и число без него вводит
    в заблуждение.
    """
    lines = [
        f"### {result.system} — по языкам",
        "",
        "| Язык | Продуктов | F1 ингр. | F1 типов | Форм сахара точно | Пропустил весь сахар |",
        "|---|---|---|---|---|---|",
    ]
    for lang in sorted(result.by_lang):
        score = result.by_lang[lang]
        lines.append(
            f"| {lang} "
            f"| {score.products} "
            f"| {score.ingredients.f1:.3f} "
            f"| {score.typed.f1:.3f} "
            f"| {score.sugar.accuracy:.3f} "
            f"| {score.sugar.missed_all} |"
        )
    return "\n".join(lines)
