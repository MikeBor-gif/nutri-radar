"""Проверка решаемости задачи — до того, как строить сравнение подходов.

Смысл в одном: три числа ниже способны обесценить всю таблицу M4, и узнать
об этом надо раньше, чем она появится.

**Число 1: baseline большинства класса.** Без него accuracy нечитаема.
44,3% продуктов имеют оценку «e», значит вырожденная модель «всегда e»
даёт 44,3% — и подход с точностью 45% не работает, хотя выглядит рабочим.

**Число 2: точность предсказания языка по тем же признакам.** Корпус
смещён по странам (fr 48%, ru 0,2%), а продуктовая корзина по странам
разная. Модель может выучить «текст по-французски → скорее d», ничего
не узнав про состав. Язык по тексту предсказывается почти идеально —
это ожидаемо и само по себе не приговор. Приговор — число 3.

**Число 3: точность внутри каждого языка отдельно.** Здесь язык постоянен
и предиктором быть не может. Если внутриязыковая точность рушится до
базлайна, значит межъязыковая разница и была всем сигналом: модель
различала страны, а не составы. Если держится — модель читает состав.

Проверка идёт на подвыборке, а не на полном корпусе: она отвечает
на вопрос «есть ли сигнал вообще», и ради этого ждать обучения на
131 тысяче документов незачем.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd
from sklearn.metrics import accuracy_score

from nutri_radar.analytics.dataset import majority_baseline, split
from nutri_radar.analytics.features import (
    LANG_COLUMN,
    TEXT_COLUMN,
    build_pipeline,
    fit_predict,
)
from nutri_radar.config import Settings, get_settings
from nutri_radar.logging import safe_extra

logger = logging.getLogger(__name__)

# Сколько языков проверяется по отдельности. Языки с горсткой продуктов
# дают доверительный интервал шире любой наблюдаемой разницы, и включать
# их значит публиковать шум.
MIN_LANG_PRODUCTS = 300


@dataclass
class SanityResult:
    """Итог проверки. Читается сверху вниз, выводы делает человек."""

    target: str
    products: int
    baseline_label: str
    baseline: float
    overall_accuracy: float
    language_accuracy: float
    per_language: dict[str, tuple[int, float, float]] = field(default_factory=dict)

    @property
    def learned_language_not_content(self) -> bool:
        """Похоже ли, что модель выучила страну, а не состав.

        Признак: внутри языков модель не обгоняет свой внутриязыковой
        базлайн. Тогда весь её сигнал был межъязыковой разницей.
        """
        if not self.per_language:
            return False
        return all(
            accuracy - baseline <= 0.01 for _, (_, accuracy, baseline) in self.per_language.items()
        )


def run_sanity_check(
    frame: pd.DataFrame,
    target: str,
    settings: Settings | None = None,
    sample_size: int = 20_000,
) -> SanityResult:
    """Посчитать три числа проверки.

    Args:
        frame: подготовленный под задачу набор.
        target: что предсказываем.
        settings: настройки.
        sample_size: сколько строк брать. Проверка отвечает на вопрос
            «есть ли сигнал», и полный корпус для этого избыточен.
    """
    settings = settings or get_settings()

    if len(frame) > sample_size:
        frame = frame.sample(n=sample_size, random_state=settings.analytics.random_seed)
        logger.info("Проверка идёт на подвыборке", extra=safe_extra(rows=len(frame)))

    train, test = split(frame, target, settings)
    baseline_label, baseline = majority_baseline(frame, target)

    # Сигнал по задаче.
    predicted, _, _ = fit_predict(build_pipeline(settings), train, test, target)
    overall = float(accuracy_score(test[target].astype(str), pd.Series(predicted).astype(str)))

    # Тот же признаковый аппарат, но предсказывается язык. Если язык
    # предсказывается лучше задачи — признаки несут прежде всего его.
    lang_pipeline = build_pipeline(settings)
    lang_pipeline.fit(train[TEXT_COLUMN], train[LANG_COLUMN])
    lang_predicted = lang_pipeline.predict(test[TEXT_COLUMN])
    language_accuracy = float(
        accuracy_score(test[LANG_COLUMN].astype(str), pd.Series(lang_predicted).astype(str))
    )

    # Главное число: внутри языка язык предиктором быть не может.
    per_language: dict[str, tuple[int, float, float]] = {}
    for lang, group in test.groupby(LANG_COLUMN):
        if len(group) < MIN_LANG_PRODUCTS:
            continue
        mask = test[LANG_COLUMN] == lang
        lang_accuracy = float(
            accuracy_score(
                group[target].astype(str),
                pd.Series(predicted).astype(str)[mask.to_numpy()],
            )
        )
        _, lang_baseline = majority_baseline(group, target)
        per_language[str(lang)] = (len(group), lang_accuracy, lang_baseline)

    result = SanityResult(
        target=target,
        products=len(frame),
        baseline_label=baseline_label,
        baseline=baseline,
        overall_accuracy=overall,
        language_accuracy=language_accuracy,
        per_language=dict(sorted(per_language.items())),
    )
    logger.info(
        "Проверка решаемости завершена",
        extra=safe_extra(
            target=target,
            accuracy=round(overall, 4),
            baseline=round(baseline, 4),
            language_accuracy=round(language_accuracy, 4),
            suspicious=result.learned_language_not_content,
        ),
    )
    return result


def format_sanity(result: SanityResult) -> str:
    """Человекочитаемый итог. Выводы формулирует человек, не функция."""
    lines = [
        f"### Проверка решаемости: {result.target}",
        "",
        f"Проверено на {result.products} продуктах.",
        "",
        "| Величина | Значение |",
        "|---|---|",
        f"| Baseline большинства класса («{result.baseline_label}») | {result.baseline:.1%} |",
        f"| Точность модели по задаче | {result.overall_accuracy:.1%} |",
        f"| Обгон базлайна | {(result.overall_accuracy - result.baseline) * 100:+.1f} пункта |",
        f"| Точность предсказания **языка** теми же признаками | {result.language_accuracy:.1%} |",
        "",
        "Внутри языков (там язык предиктором быть не может):",
        "",
        "| Язык | Продуктов | Точность | Базлайн языка | Обгон |",
        "|---|---|---|---|---|",
    ]
    for lang, (products, accuracy, baseline) in result.per_language.items():
        lines.append(
            f"| {lang} | {products} | {accuracy:.1%} | {baseline:.1%} "
            f"| {(accuracy - baseline) * 100:+.1f} |"
        )

    lines += [
        "",
        "**Как читать.** Высокая точность по языку сама по себе не приговор: "
        "язык по тексту определяется почти идеально, и это ожидаемо. Приговор — "
        "последний столбец. Если внутри языков обгон базлайна около нуля, "
        "значит весь сигнал модели был межъязыковой разницей: она различала "
        "страны, а не составы, и таблицу сравнения подходов читать как оценку "
        "качества нельзя.",
    ]
    if result.learned_language_not_content:
        lines += [
            "",
            "> **ВНИМАНИЕ: внутри языков модель не обгоняет базлайн.** "
            "Признаки несут язык, а не состав. Числа майлстоуна ниже "
            "измеряют смещение корпуса по странам.",
        ]
    return "\n".join(lines)
