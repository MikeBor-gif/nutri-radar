"""Тесты гейта качества.

Гейт — точка входа для CI, и требование к нему жёсткое: **ни GPU, ни БД,
ни сети**. Эти тесты заодно и доказывают его: всё, что здесь есть, — три файла
во временном каталоге. Если однажды гейт потянется в базу за словарём или
за предсказаниями, тест упадёт первым.

Второе, что здесь закрепляется, — гейт **детерминирован**. Он не вызывает
модель и не пересчитывает предсказания, поэтому не может покраснеть от
разброса между прогонами (ADR-018). Просадка проверяется на искусственной:
эталон один и тот же, предсказания заменяются на заведомо худшие, и порог
из настроек либо срабатывает, либо нет.

И третье: пустой эталон — не провал, а пропуск. Разметку ведёт человек
(правило 6), и держать сборку красной, пока она идёт, значит приучить всех
на неё не смотреть.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nutri_radar.config import EvalsSettings, Settings
from nutri_radar.evals.gate import (
    collect_current,
    format_gate,
    load_baseline,
    metrics_snapshot,
    run_gate,
    write_baseline,
)
from nutri_radar.evals.schemas import GoldRecord, PredictionRecord, write_jsonl
from nutri_radar.extract.schemas import ExtractionResult, Ingredient, IngredientKind

SYSTEM = "test-system"
PRODUCTS = 50


# Имена намеренно бессмысленные: настоящий словарь алиасов читается с диска,
# и совпадение с реальной формой сахара сделало бы числа зависимыми от того,
# что человек допишет в словарь завтра.
def _name(number: int) -> str:
    return f"вещество-{number}"


def _gold(count: int = PRODUCTS) -> list[GoldRecord]:
    return [
        GoldRecord(
            code=str(number),
            lang="ru" if number % 2 else "de",
            ingredients_text="состав для теста",
            extraction=ExtractionResult(
                ingredients=[Ingredient(canonical_name=_name(number), kind=IngredientKind.BASE)]
            ),
            annotator="tester",
        )
        for number in range(count)
    ]


def _predictions(correct: int, *, system: str = SYSTEM) -> list[PredictionRecord]:
    """Верные ответы по первым `correct` продуктам. По остальным — молчание."""
    return [
        PredictionRecord(
            code=str(number),
            system=system,
            extraction=ExtractionResult(
                ingredients=[Ingredient(canonical_name=_name(number), kind=IngredientKind.BASE)]
            ),
        )
        for number in range(correct)
    ]


@pytest.fixture
def gate_settings(settings: Settings) -> Settings:
    """Порог задаётся явно: значение из `.env` разработчика тест бы расшатало."""
    return settings.model_copy(update={"evals": EvalsSettings(max_f1_drop=3.0)})


@pytest.fixture
def paths(tmp_path: Path) -> dict[str, Path]:
    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    return {
        "gold": tmp_path / "gold.jsonl",
        "predictions": predictions_dir,
        "baseline": tmp_path / "baseline.json",
    }


def _write(paths: dict[str, Path], *, gold: list, predictions: list) -> None:
    write_jsonl(paths["gold"], gold)
    if predictions:
        write_jsonl(paths["predictions"] / f"{SYSTEM}.jsonl", predictions)


def _gate(settings: Settings, paths: dict[str, Path]):
    return run_gate(
        settings,
        gold_path=paths["gold"],
        predictions_dir=paths["predictions"],
        baseline_path=paths["baseline"],
    )


class TestСнимокМетрик:
    def test_сторожатся_только_величины_качества(self, paths: dict[str, Path]):
        """Токены и латентность меняются от железа — падать из-за них незачем."""
        _write(paths, gold=_gold(4), predictions=_predictions(4))

        results = collect_current(_gold(4), paths["predictions"])
        snapshot = metrics_snapshot(results[SYSTEM])

        assert set(snapshot) == {"f1_ingredients", "f1_typed", "sugar_accuracy"}
        assert snapshot["f1_ingredients"] == 1.0

    def test_отсутствующий_каталог_предсказаний_не_роняет_прогон(self):
        assert collect_current(_gold(2), Path("нет-такого-каталога")) == {}


class TestБазлайн:
    def test_запись_и_чтение_совпадают(self, tmp_path: Path):
        snapshot = {SYSTEM: {"f1_ingredients": 0.812, "f1_typed": 0.5, "sugar_accuracy": 0.9}}
        path = write_baseline(snapshot, tmp_path / "baseline.json")

        assert load_baseline(path) == snapshot

    def test_файл_читаем_человеком(self, tmp_path: Path):
        """Базлайн лежит в git и правится глазами — он обязан быть читаемым."""
        path = write_baseline({SYSTEM: {"f1_ingredients": 0.5}}, tmp_path / "baseline.json")

        assert "\n" in path.read_text(encoding="utf-8")
        assert json.loads(path.read_text(encoding="utf-8")) == {SYSTEM: {"f1_ingredients": 0.5}}

    def test_отсутствие_файла_не_ошибка(self, tmp_path: Path):
        assert load_baseline(tmp_path / "нет-такого.json") == {}


class TestПропускиГейта:
    def test_пустой_эталон_это_пропуск_а_не_провал(
        self, gate_settings: Settings, paths: dict[str, Path]
    ):
        """Разметку ведёт человек — красная сборка на всё время разметки вредна."""
        _write(paths, gold=[], predictions=_predictions(4))

        result = _gate(gate_settings, paths)

        assert result.passed is True
        assert result.skipped is True
        assert "разметка" in result.reason

    def test_нет_предсказаний_это_пропуск(self, gate_settings: Settings, paths: dict[str, Path]):
        _write(paths, gold=_gold(4), predictions=[])

        result = _gate(gate_settings, paths)

        assert result.skipped is True
        assert "предсказаний" in result.reason

    def test_первый_прогон_без_базлайна_показывает_метрики(
        self, gate_settings: Settings, paths: dict[str, Path]
    ):
        """Сравнивать не с чем, но числа человек должен увидеть."""
        _write(paths, gold=_gold(4), predictions=_predictions(4))

        result = _gate(gate_settings, paths)

        assert result.passed is True
        assert result.skipped is True
        assert result.current[SYSTEM]["f1_ingredients"] == 1.0

    def test_пустой_файл_предсказаний_не_роняет_прогон(
        self, gate_settings: Settings, paths: dict[str, Path]
    ):
        """Иначе одна пустая система уронила бы сравнение остальных трёх."""
        _write(paths, gold=_gold(4), predictions=_predictions(4))
        (paths["predictions"] / "пустая-система.jsonl").write_text("", encoding="utf-8")

        result = _gate(gate_settings, paths)

        assert set(result.current) == {SYSTEM}


class TestСрабатываниеНаПросадке:
    def test_метрики_на_месте_гейт_зелёный(self, gate_settings: Settings, paths: dict[str, Path]):
        _write(paths, gold=_gold(), predictions=_predictions(PRODUCTS))
        write_baseline(
            {SYSTEM: {"f1_ingredients": 1.0, "f1_typed": 1.0, "sugar_accuracy": 1.0}},
            paths["baseline"],
        )

        result = _gate(gate_settings, paths)

        assert result.passed is True
        assert result.regressions == []

    def test_мелкая_просадка_в_пределах_порога_проходит(
        self, gate_settings: Settings, paths: dict[str, Path]
    ):
        """Один продукт из пятидесяти — это около пункта F1, шум такого масштаба
        не должен ронять сборку."""
        _write(paths, gold=_gold(), predictions=_predictions(PRODUCTS - 1))
        write_baseline(
            {SYSTEM: {"f1_ingredients": 1.0, "f1_typed": 1.0, "sugar_accuracy": 1.0}},
            paths["baseline"],
        )

        result = _gate(gate_settings, paths)

        assert result.current[SYSTEM]["f1_ingredients"] == pytest.approx(0.9899, abs=1e-4)
        assert result.passed is True

    def test_просадка_больше_порога_роняет_гейт(
        self, gate_settings: Settings, paths: dict[str, Path]
    ):
        """Десять продуктов из пятидесяти — это одиннадцать пунктов F1."""
        _write(paths, gold=_gold(), predictions=_predictions(PRODUCTS - 10))
        write_baseline(
            {SYSTEM: {"f1_ingredients": 1.0, "f1_typed": 1.0, "sugar_accuracy": 1.0}},
            paths["baseline"],
        )

        result = _gate(gate_settings, paths)

        assert result.passed is False
        metrics = {regression.metric for regression in result.regressions}
        assert "f1_ingredients" in metrics

    def test_просадка_называет_было_и_стало(self, gate_settings: Settings, paths: dict[str, Path]):
        """Красный гейт без чисел заставляет лезть в код — это лишний час."""
        _write(paths, gold=_gold(), predictions=_predictions(1))
        write_baseline({SYSTEM: {"f1_ingredients": 1.0}}, paths["baseline"])

        result = _gate(gate_settings, paths)
        regression = result.regressions[0]

        assert regression.baseline == 1.0
        assert regression.drop_points > 3.0
        assert "было" in str(regression) and "стало" in str(regression)

    def test_порог_берётся_из_настроек(self, gate_settings: Settings, paths: dict[str, Path]):
        """Магической константы в коде нет — правило 5 брифа."""
        _write(paths, gold=_gold(), predictions=_predictions(PRODUCTS - 10))
        write_baseline({SYSTEM: {"f1_ingredients": 1.0}}, paths["baseline"])

        tolerant = gate_settings.model_copy(update={"evals": EvalsSettings(max_f1_drop=50.0)})

        assert _gate(gate_settings, paths).passed is False
        assert _gate(tolerant, paths).passed is True

    def test_исчезнувшая_система_это_регрессия(
        self, gate_settings: Settings, paths: dict[str, Path]
    ):
        """Сравнение обещало четыре системы, а показывает три — это ухудшение."""
        _write(paths, gold=_gold(4), predictions=_predictions(4))
        write_baseline(
            {
                SYSTEM: {"f1_ingredients": 1.0},
                "пропавшая-система": {"f1_ingredients": 0.7},
            },
            paths["baseline"],
        )

        result = _gate(gate_settings, paths)

        assert result.passed is False
        assert [r.system for r in result.regressions] == ["пропавшая-система"]

    def test_новая_система_гейт_не_роняет(self, gate_settings: Settings, paths: dict[str, Path]):
        """Добавить систему в сравнение — это улучшение, а не регрессия."""
        _write(paths, gold=_gold(4), predictions=_predictions(4))
        write_jsonl(paths["predictions"] / "новая.jsonl", _predictions(4, system="новая-система"))
        write_baseline({SYSTEM: {"f1_ingredients": 1.0}}, paths["baseline"])

        result = _gate(gate_settings, paths)

        assert result.passed is True
        assert set(result.current) == {SYSTEM, "новая-система"}

    def test_гейт_детерминирован(self, gate_settings: Settings, paths: dict[str, Path]):
        """Дважды по тем же файлам — те же числа. Модель не вызывается."""
        _write(paths, gold=_gold(), predictions=_predictions(PRODUCTS - 5))
        write_baseline({SYSTEM: {"f1_ingredients": 1.0}}, paths["baseline"])

        assert _gate(gate_settings, paths).current == _gate(gate_settings, paths).current


class TestВыводДляCI:
    def test_пропуск_объясняет_причину(self, gate_settings: Settings, paths: dict[str, Path]):
        _write(paths, gold=[], predictions=_predictions(4))

        text = format_gate(_gate(gate_settings, paths), 3.0)

        assert "ГЕЙТ ПРОПУЩЕН" in text

    def test_зелёный_показывает_метрики_и_порог(
        self, gate_settings: Settings, paths: dict[str, Path]
    ):
        _write(paths, gold=_gold(), predictions=_predictions(PRODUCTS))
        write_baseline({SYSTEM: {"f1_ingredients": 1.0}}, paths["baseline"])

        text = format_gate(_gate(gate_settings, paths), 3.0)

        assert "ГЕЙТ ПРОЙДЕН" in text
        assert "Порог просадки: 3.0" in text
        assert SYSTEM in text

    def test_красный_перечисляет_просадки(self, gate_settings: Settings, paths: dict[str, Path]):
        _write(paths, gold=_gold(), predictions=_predictions(1))
        write_baseline({SYSTEM: {"f1_ingredients": 1.0}}, paths["baseline"])

        text = format_gate(_gate(gate_settings, paths), 3.0)

        assert "ГЕЙТ НЕ ПРОЙДЕН" in text
        assert "f1_ingredients" in text
