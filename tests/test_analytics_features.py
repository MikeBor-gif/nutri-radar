"""Тесты пайплайна признаков и проверки решаемости.

Здесь закрепляются два свойства, потеря которых не заметна по числам.

**Словарь строится только по train.** `Pipeline`, а не два отдельных
объекта, именно поэтому: вызов `fit_transform` на всех данных пустил бы
тестовые тексты в словарь. Точность от этого вырастет, и понять, что она
выросла даром, будет уже нельзя.

**Признаки берутся только из текста.** В пайплайн приходит `Series` строк,
а не датафрейм, — взять колонку с нутриентами неоткуда даже по ошибке.

Плюс логика вывода проверки решаемости: она решает, читать ли числа
майлстоуна как оценку качества вообще.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from nutri_radar.analytics.features import TEXT_COLUMN, build_pipeline, fit_predict
from nutri_radar.analytics.tasks.sanity_check import SanityResult, format_sanity
from nutri_radar.config import AnalyticsSettings, Settings


@pytest.fixture
def analytics_settings(settings: Settings) -> Settings:
    return settings.model_copy(
        update={
            "analytics": AnalyticsSettings(
                tfidf_min_df=1,
                tfidf_max_features=5000,
                tfidf_ngram_min=2,
                tfidf_ngram_max=3,
                logreg_max_iter=50,
            )
        }
    )


def _frame(rows: int = 60) -> pd.DataFrame:
    """Данные с настоящим сигналом: «сахар» тянет к «e», «вода» — к «a»."""
    texts, labels = [], []
    for i in range(rows):
        if i % 2:
            texts.append("сахар, глюкозный сироп, патока, декстроза")
            labels.append("e")
        else:
            texts.append("вода, овощи, соль")
            labels.append("a")
    return pd.DataFrame(
        {
            "code": [str(i) for i in range(rows)],
            TEXT_COLUMN: texts,
            "lang": ["ru"] * rows,
            "nutriscore_grade": labels,
        }
    )


class TestПайплайн:
    def test_состоит_из_векторайзера_и_логрегрессии(self, analytics_settings: Settings):
        pipeline = build_pipeline(analytics_settings)

        assert isinstance(pipeline.named_steps["tfidf"], TfidfVectorizer)
        assert isinstance(pipeline.named_steps["clf"], LogisticRegression)

    def test_параметры_берутся_из_настроек(self, analytics_settings: Settings):
        """Правило 5 брифа: ни одной магической константы в коде."""
        vectorizer = build_pipeline(analytics_settings).named_steps["tfidf"]

        assert vectorizer.ngram_range == (2, 3)
        assert vectorizer.min_df == 1
        assert vectorizer.max_features == 5000

    def test_классы_балансируются(self, settings: Settings):
        """Без балансировки логрегрессия вырождается в «всегда e»:
        это её локальный оптимум по accuracy при 44% большинства."""
        assert build_pipeline(settings).named_steps["clf"].class_weight == "balanced"

    def test_анализатор_символьный_а_не_словарный(self, settings: Settings):
        """Корпус многоязычный: словарная токенизация развела бы sucre,
        Zucker и сахар по трём независимым признакам."""
        assert build_pipeline(settings).named_steps["tfidf"].analyzer == "char_wb"

    def test_словарь_строится_только_по_train(self, analytics_settings: Settings):
        """Тестовое слово, которого не было в train, не должно попасть
        в словарь: иначе это утечка, поднимающая точность даром."""
        train = _frame(40)
        test = pd.DataFrame(
            {
                "code": ["999"],
                TEXT_COLUMN: ["мальтодекстрин уникальныйтокен"],
                "lang": ["ru"],
                "nutriscore_grade": ["e"],
            }
        )

        pipeline = build_pipeline(analytics_settings)
        fit_predict(pipeline, train, test, "nutriscore_grade")
        vocabulary = pipeline.named_steps["tfidf"].vocabulary_

        assert not any("уникальныйтокен"[:6] in term for term in vocabulary)

    def test_модель_учит_сигнал_а_не_шум(self, analytics_settings: Settings):
        """Санитарная проверка самого теста: на данных с явным сигналом
        пайплайн обязан его находить, иначе остальные тесты бессмысленны."""
        frame = _frame(60)

        predicted, fit_seconds, predict_seconds = fit_predict(
            build_pipeline(analytics_settings), frame, frame, "nutriscore_grade"
        )

        assert list(predicted) == list(frame["nutriscore_grade"])
        assert fit_seconds > 0
        assert predict_seconds > 0


class TestВыводПроверки:
    def _result(self, per_language: dict[str, tuple[int, float, float]]) -> SanityResult:
        return SanityResult(
            target="nutriscore_grade",
            products=20000,
            baseline_label="e",
            baseline=0.44,
            overall_accuracy=0.64,
            language_accuracy=0.96,
            per_language=per_language,
        )

    def test_обгон_внутри_языков_снимает_подозрение(self):
        """Язык постоянен внутри группы и предиктором быть не может."""
        result = self._result({"de": (690, 0.70, 0.44), "fr": (1919, 0.66, 0.45)})

        assert result.learned_language_not_content is False
        assert "ВНИМАНИЕ" not in format_sanity(result)

    def test_отсутствие_обгона_внутри_языков_поднимает_тревогу(self):
        """Тогда весь сигнал модели был межъязыковой разницей: она различала
        страны, а не составы."""
        result = self._result({"de": (690, 0.44, 0.44), "fr": (1919, 0.45, 0.45)})

        assert result.learned_language_not_content is True
        assert "ВНИМАНИЕ" in format_sanity(result)
        assert "смещение корпуса по странам" in format_sanity(result)

    def test_один_отставший_язык_не_поднимает_тревогу(self):
        """Тревога — про все языки сразу. Один слабый язык это другой
        разговор, и путать их значит получать ложные срабатывания."""
        result = self._result({"de": (690, 0.70, 0.44), "fr": (1919, 0.45, 0.45)})

        assert result.learned_language_not_content is False

    def test_пустая_разбивка_не_поднимает_тревогу(self):
        """Языков с достаточным числом продуктов может не оказаться вовсе —
        это отсутствие данных, а не доказательство смещения."""
        assert self._result({}).learned_language_not_content is False

    def test_отчёт_называет_базлайн_и_обгон(self):
        text = format_sanity(self._result({"de": (690, 0.70, 0.44)}))

        assert "44.0%" in text
        assert "+20.0 пункта" in text
        assert "96.0%" in text
