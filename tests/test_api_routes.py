"""Тесты HTTP-API.

Точка входа тонкая, и проверяется здесь ровно то, что она добавляет
поверх слайсов, а не работа самих слайсов (у них свои тесты).

**Коды ответов различают, чей отказ.** 400 — ввод не похож на штрихкод,
404 — такого продукта нет, 503 — среда не готова, 504 — не уложились
в таймаут. Свалить всё в 500 значило бы лишить клиента возможности решить,
имеет ли смысл повторить запрос.

**Отказ «не знаю» доезжает до клиента отказом**, а не пустым ответом:
инструмент прозрачности, выдающий правдоподобный текст на любой вопрос,
хуже отсутствия инструмента.

**Атрибуция ODbL есть в ответе.** Это лицензионное обязательство, и оно
обязано ломать тест, а не ждать вычитки.

Ни сети, ни БД: конвейер и здоровье среды подставные (правило 4 брифа).
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

from nutri_radar.api import create_app
from nutri_radar.api.routes import agent as agent_route
from nutri_radar.api.routes import ask as ask_route
from nutri_radar.api.routes import products as products_route
from nutri_radar.api.routes import search as search_route
from nutri_radar.config import ApiSettings, Settings
from nutri_radar.errors import DatabaseError, LLMUnavailableError
from nutri_radar.health import CheckResult, CheckStatus, HealthReport
from nutri_radar.retrieval.pipeline import AskResult
from nutri_radar.retrieval.product_card import CardSource, ProductCard
from nutri_radar.retrieval.rag import REFUSAL, RagAnswer
from nutri_radar.retrieval.search import SearchFilters, SearchHit, SearchResult

CODE = "3017620425035"


@pytest.fixture
def api_settings(settings: Settings) -> Settings:
    return settings.model_copy(
        update={
            "api": ApiSettings(
                # Короткие таймауты: тест проверяет их срабатывание,
                # и ждать минуту на каждую проверку незачем.
                request_timeout_s=0.2,
                agent_timeout_s=0.2,
            )
        }
    )


@pytest.fixture
def client(api_settings: Settings) -> Iterator[TestClient]:
    """Клиент с поднятым lifespan: общий HTTP-клиент создаётся в нём."""
    with TestClient(create_app(api_settings)) as test_client:
        yield test_client


def _card(**overrides: object) -> ProductCard:
    defaults: dict[str, object] = {
        "code": CODE,
        "source": CardSource.CORPUS,
        "product_name": "Молочный шоколад",
        "distinct_sugar_forms": 3,
        "e_additives_count": 1,
        "extraction_model": "qwen2.5:3b-instruct-q4_K_M",
        "extraction_prompt_version": "v3",
    }
    defaults.update(overrides)
    return ProductCard(**defaults)  # type: ignore[arg-type]


def _hit() -> SearchHit:
    return SearchHit(
        code=CODE,
        product_name="Молочный шоколад",
        brands="Alpen Gold",
        ingredients_text="Сахар, какао тёртое",
        lang="ru",
        nutriscore_grade="e",
        nova_group=4,
        distance=0.2,
    )


class TestКорень:
    def test_атрибуция_и_дисклеймер_в_ответе(self, client: TestClient) -> None:
        body = client.get("/").json()

        assert "Open Food Facts" in body["attribution"]
        assert "ODbL" in body["attribution"]
        assert "краудсорсинговые" in body["disclaimer"]

    def test_атрибуция_в_описании_openapi(self, client: TestClient) -> None:
        # `/docs` — первое, что открывает потребитель API, и лицензия
        # требует, чтобы источник был виден там, а не в подвале README.
        description = client.get("/openapi.json").json()["info"]["description"]
        assert "ODbL" in description


class TestГотовность:
    def test_готовая_среда_даёт_200(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _ok(*_args: object, **_kwargs: object) -> HealthReport:
            return HealthReport(
                checks=[CheckResult(name="postgres", status=CheckStatus.OK, detail="ok")]
            )

        monkeypatch.setattr(products_route, "check_health", _ok)
        response = client.get("/healthz")

        assert response.status_code == 200
        assert response.json()["healthy"] is True

    def test_неготовая_среда_даёт_503(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Оркестратору нужен код, а не разбор тела ответа.

        200 с «healthy: false» внутри означал бы, что контейнер считается
        живым, и трафик пошёл бы в сервис без базы.
        """

        async def _fail(*_args: object, **_kwargs: object) -> HealthReport:
            return HealthReport(
                checks=[CheckResult(name="postgres", status=CheckStatus.FAIL, detail="нет")]
            )

        monkeypatch.setattr(products_route, "check_health", _fail)
        response = client.get("/healthz")

        assert response.status_code == 503
        assert response.json()["healthy"] is False

    def test_предупреждение_не_роняет_проверку(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Отсутствие ключа Anthropic — штатное состояние проекта."""

        async def _warn(*_args: object, **_kwargs: object) -> HealthReport:
            return HealthReport(
                checks=[CheckResult(name="anthropic", status=CheckStatus.WARN, detail="нет ключа")]
            )

        monkeypatch.setattr(products_route, "check_health", _warn)
        assert client.get("/healthz").status_code == 200


class TestКарточка:
    def test_не_штрихкод_даёт_400(self, client: TestClient) -> None:
        # 400, а не 404: «это не штрихкод» и «такого штрихкода нет» —
        # разные ответы, и лечатся они по-разному.
        response = client.get("/products/abcdefgh")
        assert response.status_code == 400

    def test_ненайденный_продукт_даёт_404(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _none(*_args: object, **_kwargs: object) -> ProductCard | None:
            return None

        monkeypatch.setattr(products_route, "load_card", _none)
        assert client.get(f"/products/{CODE}").status_code == 404

    def test_карточка_отдаёт_формы_сахара_и_источник(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _load(*_args: object, **_kwargs: object) -> ProductCard:
            return _card()

        monkeypatch.setattr(products_route, "load_card", _load)

        body = client.get(f"/products/{CODE}").json()

        assert body["distinct_sugar_forms"] == 3
        assert body["source"] == "corpus"
        assert "из базы проекта" in body["source_label"]
        assert "ODbL" in body["attribution"]
        assert body["extraction_model"] == "qwen2.5:3b-instruct-q4_K_M"

    def test_продукт_без_разбора_не_получает_ноль(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`None` и `0` — разные утверждения, и клиент должен видеть разницу."""

        async def _card(*_args: object, **_kwargs: object) -> ProductCard:
            return ProductCard(code=CODE, source=CardSource.OPENFOODFACTS)

        monkeypatch.setattr(products_route, "load_card", _card)
        body = client.get(f"/products/{CODE}").json()

        assert body["distinct_sugar_forms"] is None

    def test_недоступная_база_даёт_503(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _boom(*_args: object, **_kwargs: object) -> ProductCard:
            raise DatabaseError("Postgres недоступен (подделка для теста)")

        monkeypatch.setattr(products_route, "load_card", _boom)
        response = client.get(f"/products/{CODE}")

        assert response.status_code == 503
        assert response.json()["error"] == "DatabaseError"
        # Идентификатор запроса возвращается клиенту: без него жалобу
        # «у меня не работает» невозможно найти в логе.
        assert response.json()["request_id"]


class TestПоиск:
    def test_фильтры_доезжают_до_слайса(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, object] = {}

        async def _search(
            query: str, *, client: object, settings: object, limit: object, filters: SearchFilters
        ) -> SearchResult:
            seen["query"] = query
            seen["limit"] = limit
            seen["filters"] = filters
            return SearchResult(query=query, hits=[_hit()], filters=filters)

        monkeypatch.setattr(search_route, "search_by_text", _search)
        response = client.post(
            "/search",
            json={"query": "шоколад", "limit": 3, "lang": "ru", "grade_in": ["a", "b"]},
        )

        assert response.status_code == 200
        assert seen["limit"] == 3
        assert seen["filters"] == SearchFilters(lang="ru", grade_in=("a", "b"))
        assert response.json()["hits"][0]["code"] == CODE

    def test_пустой_запрос_отбивается_валидацией(self, client: TestClient) -> None:
        assert client.post("/search", json={"query": ""}).status_code == 422

    def test_близость_понятнее_расстояния(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _search(query: str, **_kwargs: object) -> SearchResult:
            return SearchResult(query=query, hits=[_hit()])

        monkeypatch.setattr(search_route, "search_by_text", _search)
        hit = client.post("/search", json={"query": "шоколад"}).json()["hits"][0]

        # distance=0.2 -> similarity=0.8. Наружу отдаётся то, что читается.
        assert hit["similarity"] == pytest.approx(0.8)
        assert "distance" not in hit


class TestОтвет:
    def _patch_ask(self, monkeypatch: pytest.MonkeyPatch, answer: RagAnswer) -> None:
        async def _ask(question: str, **_kwargs: object) -> AskResult:
            return AskResult(
                search=SearchResult(query=question, hits=[_hit()]),
                answer=answer,
                retrieval_latency_s=0.1,
            )

        monkeypatch.setattr(ask_route, "pipeline_ask", _ask)

    def test_ответ_отдаётся_со_ссылками(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch_ask(
            monkeypatch,
            RagAnswer(
                question="что в составе",
                text=f"В составе есть сахар [{CODE}].",
                sources=[CODE],
                model_name="test-model:3b",
            ),
        )
        body = client.post("/ask", json={"question": "что в составе"}).json()

        assert body["sources"] == [CODE]
        assert body["cited"] == [CODE]
        assert body["invented"] == []
        assert body["refused"] is False

    def test_отказ_доезжает_отказом(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch_ask(
            monkeypatch,
            RagAnswer(question="чего нет", text=REFUSAL, refused=True),
        )
        body = client.post("/ask", json={"question": "чего нет"}).json()

        assert body["refused"] is True
        assert body["answer"] == REFUSAL

    def test_выдуманные_ссылки_видны_клиенту(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Самый опасный вид ошибки не прячется, а отдаётся полем.

        Ссылка на продукт, которого не было в выдаче, выглядит как
        подтверждённое утверждение, а проверить его может только тот,
        кто пойдёт в базу.
        """
        self._patch_ask(
            monkeypatch,
            RagAnswer(
                question="что в составе",
                text="Смотрите [9999999999999].",
                sources=[CODE],
            ),
        )
        body = client.post("/ask", json={"question": "что в составе"}).json()

        assert body["invented"] == ["9999999999999"]

    def test_таймаут_даёт_504_а_не_зависание(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _slow(*_args: object, **_kwargs: object) -> AskResult:
            await asyncio.sleep(5)
            raise AssertionError("не должно быть достигнуто")

        monkeypatch.setattr(ask_route, "pipeline_ask", _slow)
        response = client.post("/ask", json={"question": "долгий вопрос"})

        # 504, а не 503: сервис жив, не уложился конкретный запрос.
        assert response.status_code == 504

    def test_упавшая_модель_даёт_503(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _boom(*_args: object, **_kwargs: object) -> AskResult:
            raise LLMUnavailableError("Ollama недоступна (подделка для теста)")

        monkeypatch.setattr(ask_route, "pipeline_ask", _boom)
        response = client.post("/ask", json={"question": "вопрос"})

        assert response.status_code == 503
        assert "Ollama" in response.json()["hint"]


class TestАгент:
    def test_протокол_отдаётся_целиком(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Агент без видимого протокола — чёрный ящик.

        По одному ответу нельзя понять, решал он задачу или перебирал
        инструменты, а это и есть то, что M6 измерял.
        """
        from nutri_radar.agent.loop import AgentRun, Step
        from nutri_radar.agent.tools import ToolResult

        run = AgentRun(
            question="что за продукт",
            answer="Это шоколад.",
            stop_reason="ответ",
            model_name="test-model:3b",
            steps=[
                Step(
                    number=1,
                    thought="надо посмотреть штрихкод",
                    action="lookup_barcode",
                    arguments={"barcode": CODE},
                    result=ToolResult(ok=True, content="нашёл"),
                ),
                Step(number=2, thought="готово", action="final_answer"),
            ],
        )

        async def _run(*_args: object, **_kwargs: object) -> AgentRun:
            return run

        monkeypatch.setattr(agent_route, "run_agent", _run)
        body = client.post("/agent/ask", json={"question": "что за продукт"}).json()

        assert body["stop_reason"] == "ответ"
        assert [step["action"] for step in body["steps"]] == ["lookup_barcode", "final_answer"]
        assert body["steps"][0]["ok"] is True
        assert body["steps"][1]["is_final"] is True
        assert body["tool_calls"] == 1

    def test_таймаут_агента_даёт_504(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """На локальной модели это штатный исход, а не поломка."""

        async def _slow(*_args: object, **_kwargs: object) -> object:
            await asyncio.sleep(5)
            raise AssertionError("не должно быть достигнуто")

        monkeypatch.setattr(agent_route, "run_agent", _slow)
        response = client.post("/agent/ask", json={"question": "долгий вопрос"})

        assert response.status_code == 504
        assert "не уложился" in response.json()["detail"]

    def test_пустой_вопрос_отбивается(self, client: TestClient) -> None:
        assert client.post("/agent/ask", json={"question": ""}).status_code == 422


class TestИдентификаторЗапроса:
    def test_возвращается_в_заголовке(self, client: TestClient) -> None:
        assert client.get("/").headers.get("X-Request-ID")

    def test_идентификатор_клиента_сохраняется(self, client: TestClient) -> None:
        # Сквозной идентификатор: клиент, у которого свой трейсинг, должен
        # находить свой запрос в нашем логе по своему же значению.
        # Значение латиницей: заголовки HTTP по стандарту ASCII.
        response = client.get("/", headers={"X-Request-ID": "client-trace-42"})
        assert response.headers["X-Request-ID"] == "client-trace-42"


class TestОбщийКлиент:
    def test_клиент_один_на_приложение(self, client: TestClient) -> None:
        """Клиент на запрос означал бы новое TCP-соединение каждый раз."""
        app_client = client.app.state.http_client  # type: ignore[union-attr]
        assert isinstance(app_client, httpx.AsyncClient)
