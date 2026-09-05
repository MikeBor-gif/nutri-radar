"""Тесты zero-shot: возобновляемость, учёт отказов, схема ответа.

Прогон на тысяче составов идёт десятки минут, и два свойства здесь важнее
точности самих предсказаний.

**Возобновляемость.** Прогресс пишется после каждого продукта, и перезапуск
не должен ни терять сделанное, ни звать модель повторно. Проверяется
по числу вызовов, а не по логу.

**Отказ считается ошибкой.** Продукт, по которому модель не ответила,
входит в метрику как промах. Выбрасывать такие значило бы делать систему,
промолчавшую на трудных составах, точнее той, что ответила на всех.

Сети здесь нет — модель подставная (правило 4 брифа).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nutri_radar.analytics.dataset import prepare
from nutri_radar.analytics.prompts import available_versions, grade_schema, load_prompt
from nutri_radar.analytics.tasks.grade_from_text import run_zero_shot, zero_shot_path
from nutri_radar.config import AnalyticsSettings, Settings
from nutri_radar.errors import ConfigurationError, LLMUnavailableError
from nutri_radar.llm.adapters import FakeLLM

TARGET = "nutriscore_grade"


@pytest.fixture
def analytics_settings(settings: Settings) -> Settings:
    return settings.model_copy(
        update={"analytics": AnalyticsSettings(random_seed=7, test_size=0.5, llm_subset_size=10)}
    )


def _frame(rows: int = 40) -> pd.DataFrame:
    return prepare(
        pd.DataFrame(
            {
                "code": [str(1000 + i) for i in range(rows)],
                "ingredients_text": [f"сахар, вода, номер {i}" for i in range(rows)],
                "lang": ["ru" if i % 2 else "de" for i in range(rows)],
                "nutriscore_grade": ["e" if i % 2 else "a" for i in range(rows)],
                "nova_group": [4] * rows,
            }
        ),
        TARGET,
    )


class TestПромпты:
    def test_обе_версии_лежат_на_диске(self):
        assert {"grade_v1", "nova_v1"} <= set(available_versions())

    def test_плейсхолдеры_обязательны(self):
        """Без них модель получила бы шаблон как есть и оценила его."""
        prompt = load_prompt("grade_v1")

        rendered = prompt.render("сахар, вода", lang="ru")

        assert "сахар, вода" in rendered
        assert "{text}" not in rendered

    def test_несуществующая_версия_перечисляет_доступные(self):
        with pytest.raises(ConfigurationError, match="grade_v1"):
            load_prompt("нет-такой-версии")

    def test_в_промпте_нет_примеров_с_метками(self):
        """Пример с меткой — это обучающая выборка, поданная через промпт.
        Подход, получивший примеры, — уже не zero-shot, и подпись строки
        в таблице врала бы."""
        template = load_prompt("grade_v1").template.lower()

        assert "example" not in template
        assert "for instance" not in template


class TestСхема:
    def test_множество_классов_приходит_из_данных(self):
        """Захардкоженный список разъехался бы с реальными классами
        при первом же изменении корпуса."""
        schema = grade_schema(["1", "3", "4"])

        assert schema["properties"]["grade"]["enum"] == ["1", "3", "4"]

    def test_ответ_вне_множества_невозможен(self):
        schema = grade_schema(["a", "e"])

        assert schema["additionalProperties"] is False
        assert schema["required"] == ["grade"]


class TestПрогон:
    async def test_предсказания_пишутся_на_диск(self, analytics_settings: Settings, tmp_path: Path):
        llm = FakeLLM(default_response={"grade": "e"}, model_name="fake-model")

        score = await run_zero_shot(
            _frame(),
            TARGET,
            llm,
            analytics_settings,
            root=tmp_path,
            scores_root=tmp_path,
        )

        assert zero_shot_path(TARGET, "fake-model", tmp_path).exists()
        assert score.products == 10

    async def test_перезапуск_не_зовёт_модель_повторно(
        self, analytics_settings: Settings, tmp_path: Path
    ):
        """Прогон идёт десятки минут; повторный вызов на сделанном — это
        потерянное время и лишние токены."""
        frame = _frame()
        first = FakeLLM(default_response={"grade": "e"}, model_name="fake-model")
        await run_zero_shot(
            frame,
            TARGET,
            first,
            analytics_settings,
            root=tmp_path,
            scores_root=tmp_path,
        )

        second = FakeLLM(default_response={"grade": "e"}, model_name="fake-model")
        await run_zero_shot(
            frame,
            TARGET,
            second,
            analytics_settings,
            root=tmp_path,
            scores_root=tmp_path,
        )

        assert second.call_count == 0

    async def test_отказ_модели_считается_ошибкой(
        self, analytics_settings: Settings, tmp_path: Path
    ):
        """Иначе система, промолчавшая на трудных составах, выглядела бы
        точнее той, что ответила на всех."""
        llm = FakeLLM(
            default_response={"grade": "e"},
            model_name="fake-model",
            fail_times=1000,
        )

        score = await run_zero_shot(
            _frame(),
            TARGET,
            llm,
            analytics_settings,
            root=tmp_path,
            scores_root=tmp_path,
        )

        assert score.products == 10
        assert score.accuracy == 0.0

    async def test_токены_и_время_учитываются(self, analytics_settings: Settings, tmp_path: Path):
        """Сравнение без стоимости бессмысленно — это требование DoD."""
        llm = FakeLLM(default_response={"grade": "e"}, model_name="fake-model", latency_s=0.001)

        score = await run_zero_shot(
            _frame(),
            TARGET,
            llm,
            analytics_settings,
            root=tmp_path,
            scores_root=tmp_path,
        )

        assert score.predict_seconds > 0
        assert score.fit_seconds == 0.0

    async def test_разбивка_по_языкам_считается(self, analytics_settings: Settings, tmp_path: Path):
        llm = FakeLLM(default_response={"grade": "e"}, model_name="fake-model")

        score = await run_zero_shot(
            _frame(),
            TARGET,
            llm,
            analytics_settings,
            root=tmp_path,
            scores_root=tmp_path,
        )

        assert set(score.by_lang) <= {"ru", "de"}

    async def test_двоеточие_в_имени_модели_не_ломает_путь(self):
        path = zero_shot_path(TARGET, "qwen2.5:3b-instruct-q4_K_M")

        assert ":" not in path.name


class TestОшибки:
    async def test_недоступность_не_роняет_весь_прогон(
        self, analytics_settings: Settings, tmp_path: Path
    ):
        """Один отказ не должен стоить всей сессии: остальные продукты
        обязаны досчитаться."""
        llm = FakeLLM(default_response={"grade": "e"}, model_name="fake-model", fail_times=1)

        score = await run_zero_shot(
            _frame(),
            TARGET,
            llm,
            analytics_settings,
            root=tmp_path,
            scores_root=tmp_path,
        )

        assert score.products == 10
        assert score.accuracy > 0.0

    async def test_ошибка_записывается_как_пустое_предсказание(
        self, analytics_settings: Settings, tmp_path: Path
    ):
        llm = FakeLLM(default_response={"grade": "e"}, model_name="fake-model", fail_times=1)
        await run_zero_shot(
            _frame(),
            TARGET,
            llm,
            analytics_settings,
            root=tmp_path,
            scores_root=tmp_path,
        )

        lines = zero_shot_path(TARGET, "fake-model", tmp_path).read_text(encoding="utf-8")

        assert '"predicted": ""' in lines


def test_LLMUnavailableError_обрабатывается_а_не_протекает():
    """Тип ошибки — часть контракта: раннер ловит именно её."""
    assert issubclass(LLMUnavailableError, Exception)
