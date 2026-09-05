"""Признаки и метрики: общая машинерия для всех подходов M4.

**Признаки собираются только из текста состава.** Ни одного нутриента —
`nutriscore_grade` вычисляется по ним, и подмешать их значит превратить
задачу в пересчёт формулы. Ограничение держится структурно: сюда приходит
`Series` со строками, а не датафрейм, и взять лишнюю колонку неоткуда.

**Почему символьные n-граммы, а не слова.** Корпус многоязычный: fr 48%,
en 33%, de 17%. Словарная токенизация развела бы `sucre`, `Zucker` и `сахар`
по трём независимым признакам, хотя это одно и то же вещество, и модель
училась бы каждому языку с нуля. `char_wb` держит n-граммы внутри слов,
поэтому ловит общие корни (`gluco`, `lecith`, `-ose`, `-syrup`), которые
у европейских языков совпадают.

**Почему `class_weight="balanced"`.** 44% продуктов имеют оценку «e».
Без балансировки логрегрессия сходится к «всегда e»: это её локальный
оптимум по accuracy, и он даёт macro-F1 около 0,12. Балансировка меняет
цену ошибки на редких классах и заставляет модель их различать.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from nutri_radar.config import Settings, get_settings
from nutri_radar.logging import safe_extra

logger = logging.getLogger(__name__)

TEXT_COLUMN = "ingredients_text"
LANG_COLUMN = "lang"

# Результаты подходов лежат файлами, а не в памяти одного прогона: каждый
# подход считается своей командой и своим временем (TF-IDF — минуты,
# эмбеддинги и LLM — часы), и держать их в одном запуске значило бы терять
# всё при обрыве. Сравнение потом читает каталог — тот же порядок, что
# у предсказаний систем в evals.
SCORES_DIR = Path("data/analytics/scores")

# Множество, на котором посчитан результат. Их ровно два, и путать их нельзя:
# точность на полном тесте и точность на общей подвыборке — разные числа.
SET_FULL = "full"
SET_COMMON = "common"


@dataclass
class Score:
    """Метрики одного подхода на одном множестве.

    `accuracy` без `baseline` рядом непригодна для чтения: 45% выглядит
    работающей моделью ровно до тех пор, пока не выяснится, что «всегда e»
    даёт 44,3%. Поэтому базлайн — поле результата, а не примечание к нему.
    """

    system: str
    products: int
    accuracy: float
    macro_f1: float
    baseline: float
    labels: list[str] = field(default_factory=list)
    matrix: list[list[int]] = field(default_factory=list)
    by_lang: dict[str, tuple[int, float]] = field(default_factory=dict)
    fit_seconds: float = 0.0
    predict_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def lift(self) -> float:
        """Насколько подход обгоняет «всегда самый частый класс», в пунктах.

        Единственное честное краткое число майлстоуна: подход, чей lift
        около нуля, не работает, какой бы ни была его accuracy.
        """
        return (self.accuracy - self.baseline) * 100

    @property
    def seconds_per_1000(self) -> float:
        """Время инференса на 1000 продуктов — общая единица стоимости.

        Секунды CPU, GPU-часы и токены не складываются, поэтому все подходы
        приводятся к одному: сколько занимает предсказание тысячи продуктов
        на этом железе.
        """
        if self.products == 0:
            return 0.0
        return self.predict_seconds / self.products * 1000


def build_pipeline(settings: Settings | None = None) -> Pipeline:
    """Собрать TF-IDF + логистическую регрессию.

    Пайплайн, а не два объекта: словарь обязан строиться только по train.
    Если вызвать `fit_transform` на всех данных, тестовые тексты попадут
    в словарь — утечка, которая поднимает точность и ничего не значит.
    """
    settings = settings or get_settings()
    cfg = settings.analytics

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer=cfg.tfidf_analyzer,
                    ngram_range=(cfg.tfidf_ngram_min, cfg.tfidf_ngram_max),
                    min_df=cfg.tfidf_min_df,
                    max_features=cfg.tfidf_max_features,
                    lowercase=True,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                # `n_jobs` не передаётся: с sklearn 1.8 он не действует
                # и печатает FutureWarning. Распараллеливание здесь и так
                # идёт внутри BLAS.
                LogisticRegression(
                    C=cfg.logreg_c,
                    max_iter=cfg.logreg_max_iter,
                    class_weight=cfg.class_weight,
                ),
            ),
        ]
    )


def score_predictions(
    system: str,
    truth: pd.Series,
    predicted: np.ndarray | pd.Series,
    *,
    baseline: float,
    langs: pd.Series | None = None,
    fit_seconds: float = 0.0,
    predict_seconds: float = 0.0,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> Score:
    """Посчитать метрики по готовым предсказаниям.

    Отделено от обучения намеренно: LLM zero-shot ничего не обучает, но
    метрики ей нужны те же самые. Считать их вторым кодом значило бы
    получить два числа, различающихся способом округления.
    """
    truth_values = truth.astype(str).to_numpy()
    predicted_values = pd.Series(predicted).astype(str).to_numpy()
    labels = sorted(set(truth_values) | set(predicted_values))

    by_lang: dict[str, tuple[int, float]] = {}
    if langs is not None:
        frame = pd.DataFrame(
            {"lang": langs.to_numpy(), "truth": truth_values, "pred": predicted_values}
        )
        for lang, group in frame.groupby("lang"):
            by_lang[str(lang)] = (
                len(group),
                float(accuracy_score(group["truth"], group["pred"])),
            )

    score = Score(
        system=system,
        products=len(truth_values),
        accuracy=float(accuracy_score(truth_values, predicted_values)),
        macro_f1=float(f1_score(truth_values, predicted_values, average="macro", zero_division=0)),
        baseline=baseline,
        labels=labels,
        matrix=confusion_matrix(truth_values, predicted_values, labels=labels).tolist(),
        by_lang=dict(sorted(by_lang.items())),
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    logger.info(
        "Метрики посчитаны",
        extra=safe_extra(
            system=system,
            products=score.products,
            accuracy=round(score.accuracy, 4),
            macro_f1=round(score.macro_f1, 4),
            lift=round(score.lift, 1),
        ),
    )
    return score


def fit_predict(
    pipeline: Pipeline,
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
) -> tuple[np.ndarray, float, float]:
    """Обучить на train и предсказать на test, замерив оба времени.

    Returns:
        Тройка «предсказания, секунды обучения, секунды инференса».
    """
    started = time.perf_counter()
    pipeline.fit(train[TEXT_COLUMN], train[target])
    fit_seconds = time.perf_counter() - started
    logger.info(
        "Модель обучена",
        extra=safe_extra(target=target, rows=len(train), seconds=round(fit_seconds, 1)),
    )

    started = time.perf_counter()
    predicted = pipeline.predict(test[TEXT_COLUMN])
    predict_seconds = time.perf_counter() - started
    logger.info(
        "Предсказание выполнено",
        extra=safe_extra(rows=len(test), seconds=round(predict_seconds, 1)),
    )
    return predicted, fit_seconds, predict_seconds


def score_path(target: str, dataset: str, system: str, root: Path | None = None) -> Path:
    """Файл результата одного подхода на одном множестве.

    Имя модели уезжает в путь, поэтому двоеточие и слеши заменяются:
    `qwen2.5:3b-instruct-q4_K_M` иначе не станет именем файла на Windows.
    """
    safe = system.replace(":", "_").replace("/", "_")
    return (root or SCORES_DIR) / target / dataset / f"{safe}.json"


def save_score(score: Score, target: str, dataset: str, root: Path | None = None) -> Path:
    """Сохранить результат подхода."""
    path = score_path(target, dataset, score.system, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(score), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(
        "Результат сохранён",
        extra=safe_extra(path=str(path), system=score.system, dataset=dataset),
    )
    return path


def load_scores(target: str, dataset: str, root: Path | None = None) -> list[Score]:
    """Прочитать все результаты одной задачи на одном множестве.

    Имя подхода берётся из содержимого файла, а не из его имени:
    переименование файла не должно переименовывать подход в таблице.
    """
    directory = (root or SCORES_DIR) / target / dataset
    if not directory.exists():
        logger.info("Результатов нет", extra=safe_extra(path=str(directory)))
        return []

    scores: list[Score] = []
    for file in sorted(directory.glob("*.json")):
        data = json.loads(file.read_text(encoding="utf-8"))
        # `by_lang` в JSON становится словарём списков — возвращаем кортежи,
        # иначе сравнение на равенство между прогонами начнёт врать.
        data["by_lang"] = {k: tuple(v) for k, v in dict(data.get("by_lang", {})).items()}
        scores.append(Score(**data))
    logger.debug(
        "Результаты прочитаны",
        extra=safe_extra(path=str(directory), systems=[s.system for s in scores]),
    )
    return scores
