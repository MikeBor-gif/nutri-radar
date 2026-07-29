"""Тесты метрик сравнения систем.

Данные здесь синтетические и с заранее известным ответом — это принципиально.
Метрика, проверенная на живом прогоне, проверяет заодно и модель, и словарь,
и корпус; когда число разойдётся с ожиданием, будет непонятно, что именно
сломалось. На синтетике же ответ считается руками и не зависит ни от чего.

Отдельно закрепляется правило сопоставления из ADR: имя канонизируется
словарём, а **тип берётся тот, что система реально выдала**. Если однажды
кто-то «улучшит» метрику, прогнав через словарь и типы, F1 по типам уедет
к единице — и вот этот тест покраснеет раньше, чем число попадёт в README.

Сети здесь нет и быть не может: и эталон, и предсказания — обычные объекты
в памяти, словарь собирается вручную.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from nutri_radar.evals.metrics import (
    ComparisonResult,
    PrfScore,
    SugarScore,
    format_by_language,
    format_comparison,
    match_key,
    name_set,
    score_system,
    sugar_forms,
    typed_set,
)
from nutri_radar.evals.schemas import GoldRecord, PredictionRecord
from nutri_radar.extract.normalize import ANY_LANG, AliasEntry, AliasIndex
from nutri_radar.extract.schemas import ExtractionResult, Ingredient, IngredientKind

SYSTEM = "test-system"


@pytest.fixture
def index() -> AliasIndex:
    """Маленький словарь: три языка сводятся к двум каноническим именам."""
    return AliasIndex(
        [
            AliasEntry("glucose syrup", ANY_LANG, "glucose syrup", IngredientKind.SUGAR),
            AliasEntry("сироп глюкозы", "ru", "glucose syrup", IngredientKind.SUGAR),
            AliasEntry("Glukosesirup", "de", "glucose syrup", IngredientKind.SUGAR),
            AliasEntry("sugar", ANY_LANG, "sugar", IngredientKind.SUGAR),
            AliasEntry("сахар", "ru", "sugar", IngredientKind.SUGAR),
        ]
    )


def _extraction(*items: tuple[str, IngredientKind]) -> ExtractionResult:
    return ExtractionResult(
        ingredients=[Ingredient(canonical_name=name, kind=kind) for name, kind in items]
    )


def _gold(code: str, lang: str, *items: tuple[str, IngredientKind]) -> GoldRecord:
    return GoldRecord(
        code=code,
        lang=lang,
        ingredients_text="состав для теста",
        extraction=_extraction(*items),
        annotator="tester",
        annotated_at=datetime(2026, 7, 29, tzinfo=UTC),
    )


def _prediction(
    code: str,
    *items: tuple[str, IngredientKind],
    system: str = SYSTEM,
    input_tokens: int = 0,
    output_tokens: int = 0,
    latency_s: float = 0.0,
) -> PredictionRecord:
    return PredictionRecord(
        code=code,
        system=system,
        extraction=_extraction(*items),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_s=latency_s,
    )


class TestКлючСопоставления:
    def test_алиас_ведёт_к_каноническому_имени(self, index: AliasIndex):
        assert match_key("сироп глюкозы", index, lang="ru") == "glucose syrup"
        assert match_key("Glukosesirup", index, lang="de") == "glucose syrup"

    def test_незнакомое_имя_совпадает_само_с_собой(self, index: AliasIndex):
        """Словарь неполон по построению — без отката имя не нашло бы себя."""
        assert match_key("Mystery Crunch", index) == match_key("mystery  crunch", index)

    def test_пустое_имя_даёт_пустой_ключ(self, index: AliasIndex):
        assert match_key("   ", index) == ""

    def test_пустые_имена_не_попадают_в_множества(self, index: AliasIndex):
        extraction = ExtractionResult(
            ingredients=[Ingredient(canonical_name="  ", kind=IngredientKind.BASE)]
        )

        assert name_set(extraction, index) == set()
        assert typed_set(extraction, index) == set()


class TestТипБерётсяУСистемы:
    """Ключевое место методики. Здесь легко незаметно смошенничать."""

    def test_имя_канонизируется_словарём(self, index: AliasIndex):
        extraction = _extraction(("сироп глюкозы", IngredientKind.FLAVOURING))

        assert name_set(extraction, index, lang="ru") == {"glucose syrup"}

    def test_тип_остаётся_тот_что_выдала_система(self, index: AliasIndex):
        """Словарь знает, что это sugar. Метрика обязана видеть flavouring."""
        extraction = _extraction(("сироп глюкозы", IngredientKind.FLAVOURING))

        assert typed_set(extraction, index, lang="ru") == {("glucose syrup", "flavouring")}

    def test_форма_сахара_не_засчитывается_по_словарю(self, index: AliasIndex):
        """Система назвала сироп ароматизатором — значит сахара она не нашла."""
        extraction = _extraction(("сироп глюкозы", IngredientKind.FLAVOURING))

        assert sugar_forms(extraction, index, lang="ru") == 0

    def test_разные_написания_одной_формы_считаются_одной(self, index: AliasIndex):
        extraction = _extraction(
            ("glucose syrup", IngredientKind.SUGAR),
            ("сироп глюкозы", IngredientKind.SUGAR),
            ("Glukosesirup", IngredientKind.SUGAR),
        )

        assert sugar_forms(extraction, index) == 1


class TestPrfScore:
    def test_идеальное_совпадение_даёт_единицу(self):
        score = PrfScore()
        score.add({"a", "b"}, {"a", "b"})

        assert score.precision == 1.0
        assert score.recall == 1.0
        assert score.f1 == 1.0

    def test_пустое_предсказание_даёт_нулевой_recall(self):
        score = PrfScore()
        score.add({"a", "b"}, set())

        assert score.recall == 0.0
        assert score.f1 == 0.0
        assert score.fn == 2

    def test_лишние_имена_бьют_по_precision_но_не_по_recall(self):
        score = PrfScore()
        score.add({"a"}, {"a", "b", "c"})

        assert score.recall == 1.0
        assert score.precision == pytest.approx(1 / 3)
        assert score.f1 == pytest.approx(0.5)

    def test_накопление_микро_а_не_макро(self):
        """Длинный состав весит больше короткого — в этом смысл микро-усреднения."""
        score = PrfScore()
        score.add({"a"}, {"a"})  # короткий состав, всё верно
        score.add(set("bcdefghijk"), set())  # длинный состав, всё мимо

        # Макро дало бы (1.0 + 0.0) / 2 = 0.5, микро — 1 из 11.
        assert score.recall == pytest.approx(1 / 11)

    def test_пустые_множества_не_делят_на_ноль(self):
        score = PrfScore()
        score.add(set(), set())

        assert (score.precision, score.recall, score.f1) == (0.0, 0.0, 0.0)


class TestSugarScore:
    def test_точное_попадание_и_средняя_ошибка(self):
        score = SugarScore()
        score.add(2, 2)
        score.add(3, 1)

        assert score.accuracy == 0.5
        assert score.mae == 1.0

    def test_пропущенный_весь_сахар_считается_отдельно(self):
        """Прямой ответ на вопрос M2 о 44% составов без единой формы."""
        score = SugarScore()
        score.add(2, 0)
        score.add(0, 0)

        assert score.missed_all == 1
        assert score.gold_zero == 1

    def test_ноль_у_обоих_не_считается_промахом(self):
        score = SugarScore()
        score.add(0, 0)

        assert score.missed_all == 0
        assert score.accuracy == 1.0

    def test_пустая_метрика_не_делит_на_ноль(self):
        score = SugarScore()

        assert score.accuracy == 0.0
        assert score.mae == 0.0


class TestScoreSystem:
    def test_идеальная_система_даёт_единицу_везде(self, index: AliasIndex):
        gold = [_gold("1", "ru", ("сахар", IngredientKind.SUGAR))]
        predictions = [_prediction("1", ("sugar", IngredientKind.SUGAR))]

        result = score_system(gold, predictions, index)

        assert result.system == SYSTEM
        assert result.overall.ingredients.f1 == 1.0
        assert result.overall.typed.f1 == 1.0
        assert result.overall.sugar.accuracy == 1.0
        assert result.overall.missing == 0

    def test_сопоставление_идёт_по_коду_а_не_по_порядку(self, index: AliasIndex):
        gold = [
            _gold("1", "ru", ("сахар", IngredientKind.SUGAR)),
            _gold("2", "de", ("Glukosesirup", IngredientKind.SUGAR)),
        ]
        predictions = [
            _prediction("2", ("glucose syrup", IngredientKind.SUGAR)),
            _prediction("1", ("sugar", IngredientKind.SUGAR)),
        ]

        result = score_system(gold, predictions, index)

        assert result.overall.ingredients.f1 == 1.0

    def test_алиасы_срабатывают_поверх_языка(self, index: AliasIndex):
        """Эталон на русском, ответ модели на английском — это одна сущность."""
        gold = [_gold("1", "ru", ("сироп глюкозы", IngredientKind.SUGAR))]
        predictions = [_prediction("1", ("Glucose Syrup", IngredientKind.SUGAR))]

        result = score_system(gold, predictions, index)

        assert result.overall.ingredients.tp == 1
        assert result.overall.ingredients.f1 == 1.0

    def test_пустое_предсказание_обнуляет_recall(self, index: AliasIndex):
        gold = [_gold("1", "ru", ("сахар", IngredientKind.SUGAR))]
        predictions = [_prediction("1")]

        result = score_system(gold, predictions, index)

        assert result.overall.ingredients.recall == 0.0
        assert result.overall.sugar.missed_all == 1

    def test_молчание_не_даёт_бонуса(self, index: AliasIndex):
        """Продукт без ответа идёт в метрику как пустой ответ, а не мимо неё.

        Иначе система, ответившая на один продукт из двух, получила бы recall
        по той половине, где справилась, и обошла бы честную систему.
        """
        gold = [
            _gold("1", "ru", ("сахар", IngredientKind.SUGAR)),
            _gold("2", "ru", ("сироп глюкозы", IngredientKind.SUGAR)),
        ]
        predictions = [_prediction("1", ("sugar", IngredientKind.SUGAR))]

        result = score_system(gold, predictions, index)

        assert result.overall.products == 2
        assert result.overall.missing == 1
        assert result.overall.ingredients.recall == 0.5

    def test_разбивка_по_языкам_считается_независимо(self, index: AliasIndex):
        """На этой разбивке держится ответ на второй вопрос M2."""
        gold = [
            _gold("1", "ru", ("сахар", IngredientKind.SUGAR)),
            _gold("2", "de", ("Glukosesirup", IngredientKind.SUGAR)),
        ]
        predictions = [
            _prediction("1"),  # на русском промахнулась целиком
            _prediction("2", ("glucose syrup", IngredientKind.SUGAR)),  # на немецком верно
        ]

        result = score_system(gold, predictions, index)

        assert set(result.by_lang) == {"ru", "de"}
        assert result.by_lang["ru"].ingredients.f1 == 0.0
        assert result.by_lang["de"].ingredients.f1 == 1.0
        assert result.by_lang["ru"].products == 1
        assert result.by_lang["de"].products == 1
        # Итог целиком — не среднее по языкам, а сумма TP/FP/FN.
        assert result.overall.ingredients.recall == 0.5

    def test_язык_без_ошибок_не_наследует_чужие(self, index: AliasIndex):
        """Счётчики языков не должны быть одним и тем же объектом."""
        gold = [
            _gold("1", "ru", ("сахар", IngredientKind.SUGAR)),
            _gold("2", "de", ("Glukosesirup", IngredientKind.SUGAR)),
        ]
        predictions = [_prediction("2", ("glucose syrup", IngredientKind.SUGAR))]

        result = score_system(gold, predictions, index)

        assert result.by_lang["ru"].missing == 1
        assert result.by_lang["de"].missing == 0

    def test_типы_сравниваются_отдельно_от_имён(self, index: AliasIndex):
        """Имя угадано, тип — нет: F1 по именам единица, по типам ноль."""
        gold = [_gold("1", "ru", ("сахар", IngredientKind.SUGAR))]
        predictions = [_prediction("1", ("sugar", IngredientKind.FLAVOURING))]

        result = score_system(gold, predictions, index)

        assert result.overall.ingredients.f1 == 1.0
        assert result.overall.typed.f1 == 0.0

    def test_стоимость_и_время_суммируются(self, index: AliasIndex):
        gold = [
            _gold("1", "ru", ("сахар", IngredientKind.SUGAR)),
            _gold("2", "ru", ("сахар", IngredientKind.SUGAR)),
        ]
        predictions = [
            _prediction("1", input_tokens=100, output_tokens=10, latency_s=1.0),
            _prediction("2", input_tokens=200, output_tokens=20, latency_s=3.0),
        ]

        result = score_system(gold, predictions, index)

        assert result.overall.input_tokens == 300
        assert result.overall.output_tokens == 30
        assert result.overall.median_latency == 2.0

    def test_молчание_не_портит_медиану_времени(self, index: AliasIndex):
        """Нулевая латентность отсутствующего ответа — не измерение."""
        gold = [
            _gold("1", "ru", ("сахар", IngredientKind.SUGAR)),
            _gold("2", "ru", ("сахар", IngredientKind.SUGAR)),
        ]
        predictions = [_prediction("1", latency_s=5.0)]

        result = score_system(gold, predictions, index)

        assert result.overall.median_latency == 5.0

    def test_лишнее_предсказание_вне_эталона_игнорируется(self, index: AliasIndex):
        """Метрика считается по эталону: чего человек не размечал, того нет."""
        gold = [_gold("1", "ru", ("сахар", IngredientKind.SUGAR))]
        predictions = [
            _prediction("1", ("sugar", IngredientKind.SUGAR)),
            _prediction("999", ("sugar", IngredientKind.SUGAR)),
        ]

        result = score_system(gold, predictions, index)

        assert result.overall.products == 1
        assert result.overall.ingredients.f1 == 1.0

    def test_пустой_список_предсказаний_отбивается(self, index: AliasIndex):
        with pytest.raises(ValueError, match="нечего сравнивать"):
            score_system([_gold("1", "ru", ("сахар", IngredientKind.SUGAR))], [], index)

    def test_пустой_эталон_не_роняет_подсчёт(self, index: AliasIndex):
        result = score_system([], [_prediction("1", ("sugar", IngredientKind.SUGAR))], index)

        assert result.overall.products == 0
        assert result.by_lang == {}


class TestТаблицы:
    @pytest.fixture
    def result(self, index: AliasIndex) -> ComparisonResult:
        gold = [
            _gold("1", "ru", ("сахар", IngredientKind.SUGAR)),
            _gold("2", "de", ("Glukosesirup", IngredientKind.SUGAR)),
        ]
        predictions = [
            _prediction("1", ("sugar", IngredientKind.SUGAR), input_tokens=100, output_tokens=10)
        ]
        return score_system(gold, predictions, index)

    def test_сравнение_содержит_систему_и_стоимость(self, result: ComparisonResult):
        table = format_comparison([result])

        assert SYSTEM in table
        # Токены рядом с качеством: сравнение без цены бессмысленно.
        assert "110" in table

    def test_разбивка_по_языкам_показывает_размер_выборки(self, result: ComparisonResult):
        table = format_by_language(result)

        assert "ru" in table
        assert "de" in table
        # Без числа продуктов рядом метрика на 20 продуктах вводит в заблуждение.
        assert "Продуктов" in table
