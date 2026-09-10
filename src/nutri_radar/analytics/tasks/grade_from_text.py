"""Предсказание `nutriscore_grade` по тексту состава.

Здесь живут все три подхода майлстоуна. Держать их в одном модуле, а не
разносить по файлам, — сознательный выбор: они обязаны считаться на одном
сплите и на одной подвыборке, и разнесённые по модулям они разъедутся
первым же изменением seed.

**Правило общей подвыборки.** Бриф запрещает гонять через LLM больше
нескольких тысяч продуктов, а TF-IDF учится на всех 105 тысячах. Сравнить
точность TF-IDF на полном тесте с точностью LLM на тысяче — значит
сравнить две разные задачи. Поэтому каждый подход считается дважды: на
полном тесте (там, где это возможно) и на общей подвыборке, одинаковой
для всех трёх и воспроизводимой по seed.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from nutri_radar.analytics.dataset import common_subset, majority_baseline, split
from nutri_radar.analytics.embeddings import embed_frame
from nutri_radar.analytics.features import (
    LANG_COLUMN,
    SET_COMMON,
    SET_FULL,
    TEXT_COLUMN,
    Score,
    build_pipeline,
    fit_predict,
    save_score,
    score_predictions,
)
from nutri_radar.analytics.prompts import grade_schema, load_prompt
from nutri_radar.config import Settings, get_settings
from nutri_radar.errors import ExtractionError, LLMUnavailableError
from nutri_radar.llm.ports import EmbeddingModel, StructuredLLM
from nutri_radar.logging import safe_extra

logger = logging.getLogger(__name__)

TFIDF_SYSTEM = "tfidf+logreg"
# Тот же TF-IDF, но обученный на том же урезанном train, что и эмбеддинги.
# Существует ради одного: без него разницу подходов невозможно отличить
# от разницы в размере обучающей выборки.
TFIDF_SMALL_SYSTEM = "tfidf+logreg (train 25k)"
EMBED_SYSTEM = "bge-m3+logreg"


def run_tfidf(
    frame: pd.DataFrame,
    target: str,
    settings: Settings | None = None,
) -> tuple[Score, Score]:
    """Обучить TF-IDF + логрегрессию и посчитать обе метрики.

    Модель обучается один раз на полном train. Предсказания на общей
    подвыборке — это срез предсказаний полного теста, а не второй прогон:
    иначе два числа различались бы ещё и случайностью обучения.

    Returns:
        Пара «результат на полном тесте, результат на общей подвыборке».
    """
    settings = settings or get_settings()
    train, test = split(frame, target, settings)
    _, baseline_full = majority_baseline(test, target)

    pipeline = build_pipeline(settings)
    predicted, fit_seconds, predict_seconds = fit_predict(pipeline, train, test, target)

    full = score_predictions(
        TFIDF_SYSTEM,
        test[target],
        predicted,
        baseline=baseline_full,
        langs=test[LANG_COLUMN],
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
    )

    subset = common_subset(test, target, settings)
    # Срез по коду, а не повторное предсказание: подвыборка обязана быть
    # тем же самым ответом модели, иначе разница двух чисел включит
    # ещё и случайность прогона.
    lookup = pd.Series(pd.Series(predicted).astype(str).to_numpy(), index=test["code"].to_numpy())
    subset_predicted = lookup.loc[subset["code"].to_numpy()]
    _, baseline_common = majority_baseline(subset, target)

    common = score_predictions(
        TFIDF_SYSTEM,
        subset[target],
        subset_predicted,
        baseline=baseline_common,
        langs=subset[LANG_COLUMN],
        fit_seconds=fit_seconds,
        # Время инференса пересчитывается на размер подвыборки: колонка
        # «секунд на 1000 продуктов» иначе окажется посчитана на разных
        # знаменателях и станет несравнимой между подходами.
        predict_seconds=predict_seconds / len(test) * len(subset),
    )

    save_score(full, target, SET_FULL)
    save_score(common, target, SET_COMMON)
    logger.info(
        "TF-IDF прогон завершён",
        extra=safe_extra(
            target=target,
            accuracy_full=round(full.accuracy, 4),
            accuracy_common=round(common.accuracy, 4),
            lift_full=round(full.lift, 1),
        ),
    )
    return full, common


def format_score(score: Score, dataset: str) -> str:
    """Короткая сводка одного результата для консоли."""
    lines = [
        f"{score.system} на множестве «{dataset}»:",
        f"  продуктов:     {score.products}",
        f"  accuracy:      {score.accuracy:.1%}  (baseline {score.baseline:.1%},"
        f" обгон {score.lift:+.1f} пункта)",
        f"  macro-F1:      {score.macro_f1:.3f}",
        f"  обучение:      {score.fit_seconds:.1f} с",
        f"  инференс:      {score.seconds_per_1000:.2f} с на 1000 продуктов",
    ]
    if score.by_lang:
        lines.append("  по языкам:")
        for lang, (products, accuracy) in score.by_lang.items():
            lines.append(f"    {lang}: {accuracy:.1%} ({products})")
    return "\n".join(lines)


def _score_pair(
    system: str,
    test: pd.DataFrame,
    predicted: np.ndarray | pd.Series,
    subset: pd.DataFrame,
    target: str,
    *,
    fit_seconds: float,
    predict_seconds: float,
    scores_root: Path | None = None,
) -> tuple[Score, Score]:
    """Посчитать метрики на полном тесте и на общей подвыборке.

    Подвыборка — срез тех же предсказаний, а не второй прогон: иначе разница
    двух чисел включала бы ещё и случайность обучения.
    """
    values = pd.Series(predicted).astype(str)
    _, baseline_full = majority_baseline(test, target)
    full = score_predictions(
        system,
        test[target],
        values,
        baseline=baseline_full,
        langs=test[LANG_COLUMN],
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
    )

    lookup = pd.Series(values.to_numpy(), index=test["code"].to_numpy())
    subset_predicted = lookup.loc[subset["code"].to_numpy()]
    _, baseline_common = majority_baseline(subset, target)
    common = score_predictions(
        system,
        subset[target],
        subset_predicted,
        baseline=baseline_common,
        langs=subset[LANG_COLUMN],
        fit_seconds=fit_seconds,
        # Время приводится к размеру подвыборки: колонка «секунд на 1000»
        # иначе окажется посчитана на разных знаменателях.
        predict_seconds=predict_seconds / len(test) * len(subset),
    )
    save_score(full, target, SET_FULL, scores_root)
    save_score(common, target, SET_COMMON, scores_root)
    return full, common


def small_train(train: pd.DataFrame, target: str, settings: Settings) -> pd.DataFrame:
    """Урезанный train — тот же, на котором учатся эмбеддинги.

    Стратифицирован и воспроизводим по seed, поэтому TF-IDF и эмбеддинги
    видят ровно одни и те же продукты. Без этого их сравнение включало бы
    разницу в размере выборки и выдавало бы её за разницу подходов.
    """
    size = settings.analytics.embed_train_size
    if len(train) <= size:
        return train

    from sklearn.model_selection import train_test_split as _split

    counts = train[target].value_counts()
    rare = counts[counts < 2].index.tolist()
    pool = train[~train[target].isin(rare)] if rare else train
    subset, _ = _split(
        pool,
        train_size=size,
        random_state=settings.analytics.random_seed,
        stratify=pool[target],
    )
    return subset.reset_index(drop=True)


def run_tfidf_small(
    frame: pd.DataFrame,
    target: str,
    settings: Settings | None = None,
) -> tuple[Score, Score]:
    """TF-IDF на урезанном train — контроль для сравнения с эмбеддингами."""
    settings = settings or get_settings()
    train, test = split(frame, target, settings)
    subset = common_subset(test, target, settings)

    predicted, fit_seconds, predict_seconds = fit_predict(
        build_pipeline(settings), small_train(train, target, settings), test, target
    )
    return _score_pair(
        TFIDF_SMALL_SYSTEM,
        test,
        predicted,
        subset,
        target,
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
    )


async def run_embeddings(
    frame: pd.DataFrame,
    target: str,
    model: EmbeddingModel,
    settings: Settings | None = None,
) -> tuple[Score, Score]:
    """Эмбеддинги `bge-m3` + логрегрессия.

    Train урезан по факту замера скорости: 0,099 с на продукт означает
    219 минут GPU на полный корпус. Тест векторизуется целиком — иначе
    сравнение уедет на другое множество и перестанет быть сравнением.

    Время векторизации входит в `predict_seconds`: для этого подхода оно
    и есть стоимость инференса, и прятать его в «подготовку данных» значило
    бы сравнивать TF-IDF с эмбеддингами, забыв про GPU-часы.
    """
    settings = settings or get_settings()
    train, test = split(frame, target, settings)
    train = small_train(train, target, settings)
    subset = common_subset(test, target, settings)

    started = time.perf_counter()
    train_vectors = await embed_frame(train, model)
    embed_train_seconds = time.perf_counter() - started

    started = time.perf_counter()
    test_vectors = await embed_frame(test, model)
    embed_test_seconds = time.perf_counter() - started

    cfg = settings.analytics
    classifier = LogisticRegression(
        C=cfg.logreg_c, max_iter=cfg.logreg_max_iter, class_weight=cfg.class_weight
    )
    started = time.perf_counter()
    classifier.fit(train_vectors, train[target])
    fit_seconds = time.perf_counter() - started

    started = time.perf_counter()
    predicted = classifier.predict(test_vectors)
    predict_seconds = time.perf_counter() - started

    logger.info(
        "Эмбеддинги: прогон завершён",
        extra=safe_extra(
            model=model.model_name,
            train=len(train),
            test=len(test),
            embed_train_min=round(embed_train_seconds / 60, 1),
            embed_test_min=round(embed_test_seconds / 60, 1),
            fit_seconds=round(fit_seconds, 1),
        ),
    )
    return _score_pair(
        EMBED_SYSTEM,
        test,
        predicted,
        subset,
        target,
        # Векторизация train — часть обучения этого подхода, а не бесплатная
        # подготовка: без неё модель не существует.
        fit_seconds=fit_seconds + embed_train_seconds,
        predict_seconds=predict_seconds + embed_test_seconds,
    )


__all__ = [
    "EMBED_SYSTEM",
    "TEXT_COLUMN",
    "TFIDF_SMALL_SYSTEM",
    "TFIDF_SYSTEM",
    "format_score",
    "run_embeddings",
    "run_tfidf",
    "run_tfidf_small",
    "run_zero_shot",
    "zero_shot_system",
]


ZERO_SHOT_PREDICTIONS_DIR = Path("data/analytics/zero_shot")


def zero_shot_system(model: str, version: str) -> str:
    """Имя подхода в таблице: модель вместе с версией промпта.

    Версия входит в имя, потому что она — часть подхода, а не деталь
    запуска. Без неё два прогона по разным промптам перезаписали бы друг
    друга, и сравнение версий (то самое, ради которого M2 держал три)
    оказалось бы невозможным.
    """
    return f"{model} ({version})"


def zero_shot_path(
    target: str, model: str, root: Path | None = None, version: str | None = None
) -> Path:
    """Файл предсказаний zero-shot. Модель и версия промпта — часть пути."""
    name = zero_shot_system(model, version) if version else model
    safe = name.replace(":", "_").replace("/", "_").replace(" ", "_")
    return (root or ZERO_SHOT_PREDICTIONS_DIR) / target / f"{safe}.jsonl"


def _as_float(value: object) -> float:
    """Число из записи прогона, прочитанной с диска.

    Записи JSONL типизированы как `object`: они пришли из файла, а не из
    кода. Отсутствие поля и `null` считаются нулём — так же, как раньше
    делал `or 0.0`. А вот непригодное значение поднимает `ValueError`
    и не превращается в тихий ноль: стоимость прогона входит в обязательные
    метрики проекта, и занизить её молча хуже, чем упасть.
    """
    if value is None:
        return 0.0
    if isinstance(value, bool):
        # bool — подкласс int, и `True` дал бы 1.0. В поле стоимости это
        # почти наверняка испорченные данные, а не единица.
        raise ValueError(f"ожидалось число, получено {value!r}")
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return float(value) if value.strip() else 0.0
    raise ValueError(f"ожидалось число, получено {type(value).__name__}")


def _as_int(value: object) -> int:
    """Целое из записи прогона. Правила те же, что у `_as_float`."""
    return int(_as_float(value))


def _read_zero_shot(path: Path) -> dict[str, dict[str, object]]:
    """Прочитать уже посчитанное. Нет файла — пусто, а не ошибка."""
    if not path.exists():
        return {}
    done: dict[str, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            done[str(record["code"])] = record
    return done


async def run_zero_shot(
    frame: pd.DataFrame,
    target: str,
    llm: StructuredLLM,
    settings: Settings | None = None,
    *,
    root: Path | None = None,
    scores_root: Path | None = None,
) -> Score:
    """LLM zero-shot на общей подвыборке.

    Считается только на подвыборке: бриф запрещает гонять через модель
    больше нескольких тысяч продуктов, и полный тест в 26 388 продуктов
    при наблюдаемых секундах на продукт — это часы без нового знания.

    Прогресс пишется после каждого продукта. Прогон на тысяче составов
    идёт десятки минут, и обрыв не должен стоить всей работы.

    Отказы модели считаются ошибкой, а не выбрасываются: система, которая
    промолчала на трудных составах, иначе выглядела бы точнее той, что
    ответила на всех.
    """
    settings = settings or get_settings()
    _, test = split(frame, target, settings)
    subset = common_subset(test, target, settings)

    version = (
        settings.analytics.grade_prompt_version
        if target == "nutriscore_grade"
        else settings.analytics.nova_prompt_version
    )
    prompt = load_prompt(version)
    schema = grade_schema(sorted(frame[target].astype(str).unique()))

    system = zero_shot_system(llm.model_name, version)
    path = zero_shot_path(target, llm.model_name, root, version)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = _read_zero_shot(path)
    logger.info(
        "Zero-shot прогон начат",
        extra=safe_extra(
            model=llm.model_name,
            prompt_version=version,
            products=len(subset),
            already_done=len(done),
        ),
    )

    started = time.perf_counter()
    with path.open("a", encoding="utf-8") as sink:
        for position, row in enumerate(subset.itertuples(index=False), start=1):
            code = str(row.code)
            if code in done:
                continue

            rendered = prompt.render(getattr(row, TEXT_COLUMN), lang=row.lang)
            try:
                response = await llm.generate(rendered, json_schema=schema)
                predicted = str(response.raw_json.get("grade", ""))
                record = {
                    "code": code,
                    "predicted": predicted,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "latency_s": response.latency_s,
                }
            except (LLMUnavailableError, ExtractionError) as exc:
                # Пустая строка не совпадёт ни с одним классом и попадёт
                # в метрику как ошибка. Это и есть честный учёт отказа.
                logger.warning(
                    "Модель не ответила по продукту",
                    extra=safe_extra(code=code, error=type(exc).__name__),
                )
                record = {
                    "code": code,
                    "predicted": "",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "latency_s": 0.0,
                }

            done[code] = record
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()

            if position % 50 == 0:
                elapsed = time.perf_counter() - started
                logger.info(
                    "Zero-shot идёт",
                    extra=safe_extra(
                        done=position,
                        total=len(subset),
                        elapsed_min=round(elapsed / 60, 1),
                    ),
                )

    predict_seconds = sum(_as_float(r.get("latency_s")) for r in done.values())
    predicted_series = pd.Series([str(done[str(c)]["predicted"]) for c in subset["code"]])
    _, baseline = majority_baseline(subset, target)

    score = score_predictions(
        system,
        subset[target],
        predicted_series,
        baseline=baseline,
        langs=subset[LANG_COLUMN],
        # Zero-shot ничего не обучает — обучение стоит ноль, и это
        # часть ответа на вопрос «что брать».
        fit_seconds=0.0,
        predict_seconds=predict_seconds,
        input_tokens=sum(_as_int(r.get("input_tokens")) for r in done.values()),
        output_tokens=sum(_as_int(r.get("output_tokens")) for r in done.values()),
    )
    save_score(score, target, SET_COMMON, scores_root)
    refusals = sum(1 for r in done.values() if not r.get("predicted"))
    logger.info(
        "Zero-shot прогон завершён",
        extra=safe_extra(
            model=system,
            accuracy=round(score.accuracy, 4),
            refusals=refusals,
            minutes=round(predict_seconds / 60, 1),
        ),
    )
    return score
