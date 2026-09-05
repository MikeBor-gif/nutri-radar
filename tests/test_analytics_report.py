"""Тесты метрик и отчёта.

Ошибка здесь не роняет прогон и не красит сборку: она молча даёт неверное
число, на которое потом ссылается вывод майлстоуна. Поэтому проверяется
арифметика на вырожденных случаях, где ответ известен заранее.

Ключевое, что здесь закрепляется:

* **`lift` считается от базлайна**, а не от нуля. Подход с accuracy 45%
  при базлайне 44% не работает, и число обязано это показывать.
* **Стоимость приводится к общей единице.** «Секунд на 1000 продуктов» —
  единственное, что сравнимо между CPU-инференсом и GPU-прогоном.
* **Две таблицы не смешиваются.** Полный тест и общая подвыборка — разные
  множества, и результат, сохранённый в одно, не должен всплыть в другом.

Сети, БД и GPU здесь нет.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nutri_radar.analytics.features import (
    SET_COMMON,
    SET_FULL,
    Score,
    load_scores,
    save_score,
    score_path,
    score_predictions,
)
from nutri_radar.analytics.report import (
    build_report,
    format_by_language,
    format_comparison,
    plot_accuracy_vs_cost,
    plot_confusion,
)

TARGET = "nutriscore_grade"


def _score(system: str = "tfidf+logreg", **overrides: object) -> Score:
    defaults: dict[str, object] = {
        "system": system,
        "products": 1000,
        "accuracy": 0.62,
        "macro_f1": 0.41,
        "baseline": 0.44,
        "labels": ["a", "e"],
        "matrix": [[10, 2], [3, 985]],
        "by_lang": {"de": (300, 0.66), "fr": (700, 0.60)},
        "fit_seconds": 120.0,
        "predict_seconds": 2.0,
    }
    defaults.update(overrides)
    return Score(**defaults)  # type: ignore[arg-type]


class TestМетрики:
    def test_идеальное_совпадение_даёт_единицу(self):
        truth = pd.Series(["a", "b", "c", "d", "e"])

        score = score_predictions("s", truth, truth, baseline=0.2)

        assert score.accuracy == 1.0
        assert score.macro_f1 == 1.0

    def test_полный_промах_даёт_ноль(self):
        truth = pd.Series(["a", "a", "a"])
        predicted = pd.Series(["e", "e", "e"])

        score = score_predictions("s", truth, predicted, baseline=1.0)

        assert score.accuracy == 0.0

    def test_пустые_ответы_считаются_ошибкой_а_не_выбрасываются(self):
        """Отказ модели — это ошибка. Иначе система, промолчавшая на трудных
        составах, выглядела бы точнее той, что ответила на всех."""
        truth = pd.Series(["a", "b", "c", "d"])
        predicted = pd.Series(["a", "", "", ""])

        score = score_predictions("s", truth, predicted, baseline=0.25)

        assert score.accuracy == 0.25
        assert score.products == 4

    def test_обгон_считается_от_базлайна(self):
        """45% при базлайне 44% — это не работающая модель, а почти ноль."""
        truth = pd.Series(["e"] * 45 + ["a"] * 55)
        predicted = pd.Series(["e"] * 100)

        score = score_predictions("s", truth, predicted, baseline=0.45)

        assert score.accuracy == pytest.approx(0.45)
        assert score.lift == pytest.approx(0.0, abs=1e-9)

    def test_разбивка_по_языкам_считается_независимо(self):
        truth = pd.Series(["a", "a", "e", "e"])
        predicted = pd.Series(["a", "a", "a", "a"])
        langs = pd.Series(["ru", "ru", "de", "de"])

        score = score_predictions("s", truth, predicted, baseline=0.5, langs=langs)

        assert score.by_lang["ru"] == (2, 1.0)
        assert score.by_lang["de"] == (2, 0.0)

    def test_стоимость_приводится_к_тысяче_продуктов(self):
        score = _score(products=500, predict_seconds=5.0)

        assert score.seconds_per_1000 == pytest.approx(10.0)

    def test_нулевое_число_продуктов_не_делит_на_ноль(self):
        assert _score(products=0).seconds_per_1000 == 0.0

    def test_матрица_ошибок_квадратная_по_всем_меткам(self):
        """Класс, которого нет в предсказаниях, обязан остаться строкой:
        иначе пропущенный целиком класс исчезнет из картины."""
        truth = pd.Series(["a", "b", "c"])
        predicted = pd.Series(["a", "a", "a"])

        score = score_predictions("s", truth, predicted, baseline=0.33)

        assert score.labels == ["a", "b", "c"]
        assert len(score.matrix) == 3
        assert all(len(row) == 3 for row in score.matrix)


class TestХранениеРезультатов:
    def test_запись_и_чтение_совпадают(self, tmp_path: Path):
        save_score(_score(), TARGET, SET_FULL, tmp_path)

        restored = load_scores(TARGET, SET_FULL, tmp_path)

        assert len(restored) == 1
        assert restored[0].system == "tfidf+logreg"
        assert restored[0].by_lang["de"] == (300, 0.66)

    def test_множества_не_смешиваются(self, tmp_path: Path):
        """Полный тест и общая подвыборка — разные числа. Результат одного
        не должен всплывать в таблице другого."""
        save_score(_score(accuracy=0.62), TARGET, SET_FULL, tmp_path)
        save_score(_score(accuracy=0.58), TARGET, SET_COMMON, tmp_path)

        assert load_scores(TARGET, SET_FULL, tmp_path)[0].accuracy == 0.62
        assert load_scores(TARGET, SET_COMMON, tmp_path)[0].accuracy == 0.58

    def test_двоеточие_в_имени_модели_не_ломает_путь(self, tmp_path: Path):
        """`qwen2.5:3b-instruct-q4_K_M` иначе не станет именем файла в Windows."""
        path = score_path(TARGET, SET_COMMON, "qwen2.5:3b-instruct-q4_K_M", tmp_path)

        assert ":" not in path.name

    def test_имя_подхода_берётся_из_записи_а_не_из_файла(self, tmp_path: Path):
        """Переименование файла не должно переименовывать подход в таблице."""
        save_score(_score(system="настоящее-имя"), TARGET, SET_FULL, tmp_path)
        found = next((tmp_path / TARGET / SET_FULL).glob("*.json"))
        found.rename(found.with_name("как-угодно.json"))

        assert load_scores(TARGET, SET_FULL, tmp_path)[0].system == "настоящее-имя"

    def test_отсутствующий_каталог_не_роняет_отчёт(self, tmp_path: Path):
        assert load_scores(TARGET, SET_FULL, tmp_path / "нет-такого") == []


class TestТаблицы:
    def test_подходы_идут_по_убыванию_точности(self):
        text = format_comparison(
            [_score("слабый", accuracy=0.50), _score("сильный", accuracy=0.70)],
            "Тест",
            "примечание",
        )

        assert text.index("сильный") < text.index("слабый")

    def test_базлайн_называется_рядом_с_таблицей(self):
        text = format_comparison([_score(baseline=0.44)], "Тест", "примечание")

        assert "44.0%" in text
        assert "обгоном около нуля не работает" in text

    def test_пустой_список_не_роняет_отчёт(self):
        assert "Результатов нет" in format_comparison([], "Тест", "примечание")

    def test_отсутствие_токенов_показывается_прочерком(self):
        """У TF-IDF токенов нет вовсе, и ноль здесь читался бы как измерение."""
        text = format_comparison([_score(input_tokens=0, output_tokens=0)], "Тест", "п")

        assert "| — |" in text

    def test_языковая_таблица_показывает_размер_группы(self):
        text = format_by_language([_score()], "По языкам")

        assert "(300)" in text
        assert "(700)" in text

    def test_языковая_таблица_пуста_если_разбивки_нет(self):
        assert format_by_language([_score(by_lang={})], "По языкам") == ""


class TestОтчёт:
    def test_отчёт_собирается_из_сохранённого(self, tmp_path: Path):
        save_score(_score(), TARGET, SET_FULL, tmp_path)
        save_score(_score(system="bge-m3+logreg"), TARGET, SET_COMMON, tmp_path)

        text = build_report(TARGET, tmp_path)

        assert "На полном тесте" in text
        assert "На общей подвыборке" in text
        assert "bge-m3+logreg" in text

    def test_отчёт_напоминает_про_отсутствие_нутриентов(self, tmp_path: Path):
        save_score(_score(), TARGET, SET_FULL, tmp_path)

        assert "Ни одного нутриента" in build_report(TARGET, tmp_path)

    def test_пустые_результаты_не_роняют_сборку(self, tmp_path: Path):
        assert "Результатов нет" in build_report(TARGET, tmp_path)


class TestГрафики:
    def test_график_стоимости_рисуется(self, tmp_path: Path):
        path = plot_accuracy_vs_cost([_score()], TARGET, tmp_path / "cost.png")

        assert path.exists()
        assert path.stat().st_size > 0

    def test_нулевая_стоимость_не_ломает_логарифм(self, tmp_path: Path):
        """У TF-IDF инференс быстрее миллисекунды на продукт; ноль
        на логарифмической оси не рисуется вовсе."""
        path = plot_accuracy_vs_cost(
            [_score(predict_seconds=0.0)], TARGET, tmp_path / "zero.png"
        )

        assert path.exists()

    def test_матрица_ошибок_рисуется(self, tmp_path: Path):
        path = plot_confusion(_score(), TARGET, tmp_path / "cm.png")

        assert path.exists()

    def test_пробелы_и_двоеточия_в_имени_не_ломают_путь(self, tmp_path: Path):
        path = plot_confusion(_score(system="tfidf+logreg (train 25k)"), TARGET)

        assert " " not in path.name
        path.unlink(missing_ok=True)
