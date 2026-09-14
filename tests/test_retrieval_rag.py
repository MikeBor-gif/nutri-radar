"""Тесты RAG и метрик подтверждённости.

Главное свойство здесь — **отказ «не знаю»**, и проверяется оно не тем,
что модель послушалась просьбы в промпте, а тем, что модель **не была
вызвана вовсе**. Инструмент прозрачности, выдающий правдоподобный текст
на любой вопрос, хуже отсутствия инструмента: он выглядит одинаково
уверенно и когда данные есть, и когда их нет.

Второе — **выдуманные штрихкоды**. Ссылка на продукт, которого не было
в выдаче, выглядит как подтверждённое утверждение, а проверить его может
только тот, кто пойдёт в базу. Такие ссылки обязаны попадать в метрику,
а не отфильтровываться по дороге.

Сети и БД здесь нет: поиск подставной, модель подставная (правило 4).
"""

from __future__ import annotations

import pytest

from nutri_radar.config import RetrievalSettings, Settings
from nutri_radar.llm.adapters import FakeLLM
from nutri_radar.retrieval.metrics import (
    GoldQuery,
    RetrievalReport,
    score_grounding,
    score_recall,
)
from nutri_radar.retrieval.rag import (
    REFUSAL,
    RagAnswer,
    answer,
    answer_schema,
    format_products,
    language_rule,
    relevant_hits,
)
from nutri_radar.retrieval.search import SearchHit, SearchResult


@pytest.fixture
def retrieval_settings(settings: Settings) -> Settings:
    return settings.model_copy(update={"retrieval": RetrievalSettings(top_k=5)})


def _hit(code: str = "3017620425035", distance: float = 0.2, **overrides: object) -> SearchHit:
    defaults: dict[str, object] = {
        "code": code,
        "product_name": "Молочный шоколад",
        "brands": "Alpen Gold",
        "ingredients_text": "Сахар, какао тёртое, пальмовое масло",
        "lang": "ru",
        "nutriscore_grade": "e",
        "nova_group": 4,
        "distance": distance,
    }
    defaults.update(overrides)
    return SearchHit(**defaults)  # type: ignore[arg-type]


def _result(*hits: SearchHit, query: str = "шоколад") -> SearchResult:
    return SearchResult(query=query, hits=list(hits))


class TestОтбор:
    def test_далёкие_результаты_отсеиваются(self):
        """Векторный поиск возвращает ближайшее всегда — даже когда ничего
        подходящего нет. Без порога «не знаю» не наступит никогда."""
        result = _result(_hit(distance=0.1), _hit("2", distance=0.7))

        assert [hit.code for hit in relevant_hits(result)] == ["3017620425035"]

    def test_граница_включается(self):
        assert len(relevant_hits(_result(_hit(distance=0.5)))) == 1

    def test_пустая_выдача_даёт_пустой_отбор(self):
        assert relevant_hits(_result()) == []


class TestОтказ:
    async def test_пустая_выдача_даёт_отказ(self, retrieval_settings: Settings):
        llm = FakeLLM(default_response={"answer": "что-нибудь", "sources": []})

        got = await answer("вопрос", _result(), llm, retrieval_settings)

        assert got.refused is True
        assert got.text == REFUSAL

    async def test_при_отказе_модель_не_вызывается(self, retrieval_settings: Settings):
        """Ключевое: отказ формируется кодом. Просить модель не выдумывать
        и надеяться — не проверяемое свойство системы."""
        llm = FakeLLM(default_response={"answer": "выдумка", "sources": []})

        await answer("вопрос", _result(), llm, retrieval_settings)

        assert llm.call_count == 0

    async def test_нерелевантная_выдача_даёт_отказ(self, retrieval_settings: Settings):
        """Нашлось пять продуктов, но все далеко — это тот же «не знаю»."""
        llm = FakeLLM(default_response={"answer": "текст", "sources": []})

        got = await answer("вопрос", _result(_hit(distance=0.9)), llm, retrieval_settings)

        assert got.refused is True
        assert llm.call_count == 0

    async def test_недоступность_модели_даёт_отказ_а_не_падение(self, retrieval_settings: Settings):
        llm = FakeLLM(default_response={"answer": "текст", "sources": []}, fail_times=99)

        got = await answer("вопрос", _result(_hit()), llm, retrieval_settings)

        assert got.refused is True
        assert got.text == REFUSAL

    async def test_пустой_ответ_модели_это_тоже_отказ(self, retrieval_settings: Settings):
        """Показать пользователю пустоту хуже, чем сказать «не знаю»."""
        llm = FakeLLM(default_response={"answer": "   ", "sources": []})

        got = await answer("вопрос", _result(_hit()), llm, retrieval_settings)

        assert got.refused is True


class TestОтвет:
    async def test_ответ_строится_по_найденному(self, retrieval_settings: Settings):
        llm = FakeLLM(
            default_response={
                "answer": "В составе [3017620425035] есть пальмовое масло.",
                "sources": ["3017620425035"],
            }
        )

        got = await answer("вопрос", _result(_hit()), llm, retrieval_settings)

        assert got.refused is False
        assert got.cited == ["3017620425035"]
        assert got.cited_outside_sources == []

    async def test_выдуманный_штрихкод_виден(self, retrieval_settings: Settings):
        """Самый опасный отказ: утверждение выглядит подтверждённым,
        а проверить может только тот, кто пойдёт в базу."""
        llm = FakeLLM(
            default_response={
                "answer": "Смотрите [9999999999999].",
                "sources": ["9999999999999"],
            }
        )

        got = await answer("вопрос", _result(_hit()), llm, retrieval_settings)

        assert got.cited_outside_sources == ["9999999999999"]

    async def test_выдуманное_не_фильтруется_а_остаётся_в_ответе(
        self, retrieval_settings: Settings
    ):
        """Это факт о работе системы, и он обязан попасть в метрику,
        а не быть тихо вычищен."""
        llm = FakeLLM(default_response={"answer": "Смотрите [9999999999999].", "sources": []})

        got = await answer("вопрос", _result(_hit()), llm, retrieval_settings)

        assert "9999999999999" in got.text

    async def test_версия_промпта_и_модель_записываются(self, retrieval_settings: Settings):
        llm = FakeLLM(default_response={"answer": "текст", "sources": []})

        got = await answer("вопрос", _result(_hit()), llm, retrieval_settings)

        assert got.prompt_version == "rag_v1"
        assert got.model_name == "fake-model"


class TestБлокПродуктов:
    def test_штрихкод_подаётся_в_той_же_форме_что_требуется_в_ответе(self):
        """Чем ближе форма ссылки в промпте к требуемой, тем реже модель
        изобретает свою."""
        assert "[3017620425035]" in format_products([_hit()])

    def test_состав_попадает_в_промпт(self):
        assert "пальмовое масло" in format_products([_hit()])

    def test_продукт_без_названия_не_роняет_сборку(self):
        assert "без названия" in format_products([_hit(product_name=None)])


class TestПромпт:
    def test_запрещает_оценочные_ярлыки(self):
        """Границы продукта из брифа: описываем, а не судим."""
        from nutri_radar.retrieval.prompts import load_prompt

        template = load_prompt("rag_v1").template.lower()

        assert "describe, do not judge" in template
        assert "do not know" in template

    def test_требует_ссылок_на_штрихкоды(self):
        from nutri_radar.retrieval.prompts import load_prompt

        assert "barcode" in load_prompt("rag_v1").template.lower()


class TestМетрики:
    def test_recall_считается_по_множеству_а_не_по_позиции(self):
        """«Нашлось в первых пяти» — про попадание, а не про порядок."""
        gold = GoldQuery(query="q", expected=["1", "2"])
        result = _result(_hit("2"), _hit("9"), _hit("1"))

        score = score_recall(gold, result)

        assert score.found == 2
        assert score.recall == 1.0

    def test_пропуск_снижает_recall(self):
        gold = GoldQuery(query="q", expected=["1", "2"])

        assert score_recall(gold, _result(_hit("1"))).recall == 0.5

    def test_запрос_без_эталона_не_делит_на_ноль(self):
        assert score_recall(GoldQuery(query="q"), _result(_hit())).recall == 0.0

    def test_подтверждённость_считается_кодом(self):
        """Никакого суждения о смысле: штрихкод либо был в выдаче, либо нет."""
        answer_obj = RagAnswer(
            question="q",
            # Настоящие штрихкоды: регулярка ждёт 6-14 цифр, потому что
            # «[1]» в тексте — это сноска или список, а не ссылка на продукт.
            text="Есть [3017620425035] и [9999999999999].",
            sources=["3017620425035"],
        )

        score = score_grounding(answer_obj)

        assert score.cited == 2
        assert score.invented == 1
        assert score.grounded_share == 0.5

    def test_короткое_число_в_скобках_не_считается_ссылкой(self):
        """«[1]» в тексте — это сноска или пункт списка, а не штрихкод."""
        score = score_grounding(
            RagAnswer(question="q", text="пункт [1] списка", sources=["3017620425035"])
        )

        assert score.cited == 0

    def test_ответ_без_ссылок_даёт_нулевую_подтверждённость(self):
        score = score_grounding(
            RagAnswer(question="q", text="просто текст", sources=["3017620425035"])
        )

        assert score.has_citations is False
        assert score.grounded_share == 0.0

    def test_отказы_не_входят_в_среднюю_подтверждённость(self):
        """Иначе система, отказывающая всегда, показала бы идеальную
        подтверждённость — при нулевой пользе."""
        report = RetrievalReport(
            k=5,
            groundings=[
                score_grounding(RagAnswer(question="a", text=REFUSAL, refused=True)),
                score_grounding(
                    RagAnswer(
                        question="b",
                        text="[3017620425035]",
                        sources=["3017620425035"],
                    )
                ),
            ],
        )

        assert report.refusal_share == 0.5
        assert report.mean_grounded == 1.0

    def test_пустой_отчёт_не_делит_на_ноль(self):
        report = RetrievalReport(k=5)

        assert report.mean_recall == 0.0
        assert report.mean_grounded == 0.0
        assert report.refusal_share == 0.0


class TestЭталонныеЗапросы:
    def test_запрос_без_эталона_негоден(self):
        assert GoldQuery(query="q", expected=[]).is_usable is False

    def test_пустая_формулировка_негодна(self):
        assert GoldQuery(query="  ", expected=["1"]).is_usable is False

    def test_полный_запрос_годен(self):
        assert GoldQuery(query="q", expected=["1"]).is_usable is True

    def test_отсутствие_файла_объясняет_чья_это_работа(self, tmp_path):
        from nutri_radar.retrieval.metrics import read_queries

        with pytest.raises(FileNotFoundError, match="человек"):
            read_queries(tmp_path / "нет-такого.jsonl")

    def test_запись_и_чтение_совпадают(self, tmp_path):
        from nutri_radar.retrieval.metrics import append_query, read_queries

        path = tmp_path / "queries.jsonl"
        append_query(GoldQuery(query="шоколад", expected=["1"], author="Mikhail"), path)
        append_query(GoldQuery(query="йогурт", expected=["2"], author="Mikhail"), path)

        queries = read_queries(path)

        assert [item.query for item in queries] == ["шоколад", "йогурт"]
        assert queries[0].author == "Mikhail"


class TestПринуждениеЯзыка:
    """Поле `language` в схеме — структурное принуждение из `rag_v2`.

    Проверяется свойство, которое однажды уже разошлось с описанием:
    промпт требовал поле, а схема его не допускала, и для латинских
    вопросов принуждения не было ни в каком виде. Проверять надо именно
    наличие поля, а не текст промпта: просьба в промпте свойством
    системы не является — это и есть измеренный вывод майлстоуна.
    """

    def test_кириллический_вопрос_сужается_до_русского(self) -> None:
        rule, allowed = language_rule("шоколад с пальмовым маслом")
        assert allowed == ["ru"]
        assert "Russian" in rule

    def test_латинский_вопрос_не_навязывает_английский(self) -> None:
        """Навязать немецкому вопросу английский было бы хуже `rag_v1`."""
        _, allowed = language_rule("Schokolade mit Haselnüssen")
        assert allowed == []

    def test_поле_есть_и_для_латиницы(self) -> None:
        """Список пуст — но поле остаётся, свободной строкой.

        «Поля нет» и «поле есть, значения любые» — разные вещи, и первая
        версия молча выбирала первое там, где описание обещало второе.
        """
        schema = answer_schema([], require_language=True)
        assert "language" in schema["properties"]
        assert "enum" not in schema["properties"]["language"]
        assert "language" in schema["required"]

    def test_поле_идёт_первым(self) -> None:
        """Порядок значим: назвать язык надо ДО первого слова ответа."""
        schema = answer_schema(["ru"], require_language=True)
        assert list(schema["properties"]) == ["language", "answer", "sources"]

    def test_без_принуждения_схема_как_в_rag_v1(self) -> None:
        """`rag_v1` не должен измениться: на нём посчитаны опубликованные числа."""
        schema = answer_schema(["ru"], require_language=False)
        assert list(schema["properties"]) == ["answer", "sources"]
        assert schema["required"] == ["answer", "sources"]

    def test_требование_про_json_живёт_в_том_же_тексте(self) -> None:
        """Промпт и схему возвращает одна функция — разойтись им негде."""
        for question in ("шоколад", "chocolate", "Schokolade mit Nüssen"):
            rule, _ = language_rule(question)
            assert "`language`" in rule
