"""Тесты общей сборки конвейера.

Проверяемых свойства два, и оба про то, ради чего сборка вынесена из CLI.

**Порядок работы с моделями.** Векторизация обязана закончиться и освободить
видеопамять до того, как начнётся генерация: на 6 ГБ VRAM обе модели вместе
не помещаются. В CLI это делалось руками, здесь — очередью, и тест смотрит
на фактический порядок запросов к Ollama, а не на намерения кода.

**Отказ остаётся отказом.** Пустая выдача обязана давать «не знаю» без вызова
модели — свойство M5, и оно не должно потеряться при переезде сборки.

Сеть подменена целиком через `MockTransport`, поиск подставной: тест не ходит
ни в Ollama, ни в Postgres (правило 4 брифа).
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx
import pytest

from nutri_radar.config import RetrievalSettings, Settings
from nutri_radar.retrieval import pipeline
from nutri_radar.retrieval.rag import REFUSAL
from nutri_radar.retrieval.search import SearchFilters, SearchHit, SearchResult

# Протокол обращений к Ollama: что именно спросили и в каком порядке.
Journal = list[str]


def _hit(code: str = "3017620425035") -> SearchHit:
    return SearchHit(
        code=code,
        product_name="Молочный шоколад",
        brands="Alpen Gold",
        ingredients_text="Сахар, какао тёртое, пальмовое масло",
        lang="ru",
        nutriscore_grade="e",
        nova_group=4,
        distance=0.1,
    )


@pytest.fixture
def pipeline_settings(settings: Settings) -> Settings:
    return settings.model_copy(update={"retrieval": RetrievalSettings(top_k=5)})


@pytest.fixture
def journal() -> Journal:
    return []


@pytest.fixture
def ollama(journal: Journal, pipeline_settings: Settings) -> Iterator[httpx.AsyncClient]:
    """Ollama, которая отвечает и записывает, о чём её спросили."""
    dim = pipeline_settings.ollama.embedding_dim

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if request.url.path == "/api/embed":
            # Выгрузка отличается от векторизации пустым входом и keep_alive=0.
            if not payload.get("input"):
                journal.append("unload")
                return httpx.Response(200, json={"embeddings": []})
            journal.append("embed")
            vectors = [[0.1] * dim for _ in payload["input"]]
            return httpx.Response(200, json={"embeddings": vectors})

        assert request.url.path == "/api/chat", f"Неожиданный запрос: {request.url}"
        journal.append("chat")
        content = json.dumps({"answer": "В составе есть сахар [3017620425035].", "sources": []})
        return httpx.Response(
            200,
            json={
                "message": {"content": content},
                "prompt_eval_count": 100,
                "eval_count": 20,
            },
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://ollama.invalid:11434"
    )
    yield client


@pytest.fixture
def fake_search(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Подменить векторный поиск: БД в тестах нет."""
    calls: list[dict[str, object]] = []
    hits = [_hit()]

    async def _search(
        embedding: list[float],
        *,
        query: str = "",
        limit: int | None = None,
        filters: SearchFilters | None = None,
        settings: Settings | None = None,
    ) -> SearchResult:
        calls.append({"dim": len(embedding), "query": query, "limit": limit, "filters": filters})
        return SearchResult(query=query, hits=list(hits), filters=filters or SearchFilters())

    monkeypatch.setattr(pipeline, "search_vectors", _search)
    return calls


class TestВекторизация:
    async def test_пустой_вход_не_идёт_в_сеть(
        self, ollama: httpx.AsyncClient, pipeline_settings: Settings, journal: Journal
    ) -> None:
        assert await pipeline.embed_texts([], client=ollama, settings=pipeline_settings) == []
        assert journal == []

    async def test_вектор_имеет_объявленную_размерность(
        self, ollama: httpx.AsyncClient, pipeline_settings: Settings
    ) -> None:
        vectors = await pipeline.embed_texts(["шоколад"], client=ollama, settings=pipeline_settings)
        assert len(vectors) == 1
        assert len(vectors[0]) == pipeline_settings.ollama.embedding_dim


class TestПоискПоТексту:
    async def test_запрос_векторизуется_и_уходит_в_поиск(
        self,
        ollama: httpx.AsyncClient,
        pipeline_settings: Settings,
        fake_search: list[dict[str, object]],
        journal: Journal,
    ) -> None:
        result = await pipeline.search_by_text(
            "шоколад без пальмового масла", client=ollama, settings=pipeline_settings
        )

        assert journal == ["embed"]
        assert len(fake_search) == 1
        assert fake_search[0]["dim"] == pipeline_settings.ollama.embedding_dim
        assert fake_search[0]["query"] == "шоколад без пальмового масла"
        assert result.codes == ["3017620425035"]

    async def test_фильтры_доходят_до_поиска(
        self,
        ollama: httpx.AsyncClient,
        pipeline_settings: Settings,
        fake_search: list[dict[str, object]],
    ) -> None:
        filters = SearchFilters(lang="ru", grade_in=("a", "b"))
        await pipeline.search_by_text(
            "шоколад", client=ollama, settings=pipeline_settings, limit=3, filters=filters
        )

        assert fake_search[0]["filters"] == filters
        assert fake_search[0]["limit"] == 3


class TestОтвет:
    async def test_модель_эмбеддингов_выгружается_до_генерации(
        self,
        ollama: httpx.AsyncClient,
        pipeline_settings: Settings,
        fake_search: list[dict[str, object]],
        journal: Journal,
    ) -> None:
        """Главное свойство сборки: две модели не оказываются в VRAM вместе."""
        await pipeline.ask("что в составе", client=ollama, settings=pipeline_settings)

        assert journal == ["embed", "unload", "chat"], (
            f"Между векторизацией и генерацией обязана быть выгрузка: фактический порядок {journal}"
        )

    async def test_ответ_возвращается_вместе_с_выдачей(
        self,
        ollama: httpx.AsyncClient,
        pipeline_settings: Settings,
        fake_search: list[dict[str, object]],
    ) -> None:
        # Ответ без выдачи непроверяем: нельзя понять, сослалась модель
        # на найденное или сочинила штрихкод.
        outcome = await pipeline.ask("что в составе", client=ollama, settings=pipeline_settings)

        assert outcome.search.codes == ["3017620425035"]
        assert outcome.answer.sources == ["3017620425035"]
        assert outcome.answer.cited_outside_sources == []
        assert not outcome.answer.refused
        assert outcome.retrieval_latency_s >= 0

    async def test_пустая_выдача_даёт_отказ_без_вызова_модели(
        self,
        ollama: httpx.AsyncClient,
        pipeline_settings: Settings,
        monkeypatch: pytest.MonkeyPatch,
        journal: Journal,
    ) -> None:
        async def _empty(
            embedding: list[float],
            *,
            query: str = "",
            limit: int | None = None,
            filters: SearchFilters | None = None,
            settings: Settings | None = None,
        ) -> SearchResult:
            return SearchResult(query=query, hits=[])

        monkeypatch.setattr(pipeline, "search_vectors", _empty)

        outcome = await pipeline.ask("чего в базе нет", client=ollama, settings=pipeline_settings)

        assert outcome.answer.refused
        assert outcome.answer.text == REFUSAL
        # Модель не вызывалась вовсе — отказ сформирован кодом, а не просьбой
        # в промпте. Это то же свойство, что проверяет M5, и оно не должно
        # было потеряться при переезде сборки.
        assert "chat" not in journal
