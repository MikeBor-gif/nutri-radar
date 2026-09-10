"""Тесты выбора провайдера модели.

Проверяемое свойство одно и оно было заявлено проектом задолго до этого
модуля: **смена модели — строка в `.env`, а не правка кода**. Утверждение
повторяется в ADR-030 и в README, и до M7 оно держалось на том, что каждый
composition root собирал адаптер сам. Когда сборка конвейера переехала
в слайс, утверждение молча перестало быть верным для поиска и RAG: код
работал, просто на Ollama независимо от `LLM__PROVIDER`.

Поэтому тест смотрит не на то, что фабрика «что-то возвращает», а на то,
**что именно** она возвращает при каждом значении настройки, и что слайс
ходит через неё, а не мимо.

Сети здесь нет: адаптеры создаются, но никуда не обращаются.
"""

from __future__ import annotations

import typing
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from nutri_radar.config import AnthropicSettings, LLMSettings, Settings
from nutri_radar.errors import ConfigurationError
from nutri_radar.llm.adapters.anthropic import AnthropicLLM
from nutri_radar.llm.adapters.ollama import OllamaEmbeddings, OllamaLLM
from nutri_radar.llm.factory import build_embeddings, build_llm
from nutri_radar.llm.ports import EmbeddingModel, StructuredLLM


@pytest.fixture
def client() -> httpx.AsyncClient:
    """Клиент, который никуда не сходит: транспорт отказывает на любом запросе."""

    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("фабрика не должна ходить в сеть при сборке")

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TestВыборПровайдера:
    def test_ollama_по_умолчанию(self, settings: Settings, client: httpx.AsyncClient) -> None:
        llm = build_llm(settings, client)

        assert isinstance(llm, OllamaLLM)
        assert llm.model_name == settings.ollama.model

    def test_anthropic_по_настройке(self, settings: Settings, client: httpx.AsyncClient) -> None:
        """Ключевое свойство: значение в .env действительно меняет модель."""
        cloud = settings.model_copy(
            update={
                "llm": LLMSettings(provider="anthropic"),
                "anthropic": AnthropicSettings(api_key=SecretStr("sk-ant-подделка-для-теста")),
            }
        )

        llm = build_llm(cloud, client)

        assert isinstance(llm, AnthropicLLM)
        assert llm.model_name == cloud.anthropic.model

    def test_облако_без_ключа_отказывает_понятно(self, settings: Settings) -> None:
        """Отказ до первого запроса, а не на сотом продукте."""
        cloud = settings.model_copy(update={"llm": LLMSettings(provider="anthropic")})

        with pytest.raises(ConfigurationError, match="ANTHROPIC__API_KEY"):
            build_llm(cloud)

    def test_неизвестный_провайдер_отказывает(
        self, settings: Settings, client: httpx.AsyncClient
    ) -> None:
        # Настройки валидируют значение, поэтому подсовываем его в обход
        # валидатора: проверяется ветка защиты самой фабрики.
        broken = settings.model_copy(update={"llm": LLMSettings.model_construct(provider="gpt")})

        with pytest.raises(ConfigurationError, match="Неизвестный провайдер"):
            build_llm(broken, client)

    def test_локальному_провайдеру_нужен_клиент(self, settings: Settings) -> None:
        """Клиент не создаётся внутри: пул должен жить столько же, сколько
        точка входа, иначе каждый запрос открывает новое соединение."""
        with pytest.raises(ConfigurationError, match="HTTP-клиент"):
            build_llm(settings)


class TestЭмбеддинги:
    def test_собираются_с_настроенной_моделью(
        self, settings: Settings, client: httpx.AsyncClient
    ) -> None:
        model = build_embeddings(settings, client)

        assert isinstance(model, OllamaEmbeddings)
        assert model.model_name == settings.ollama.embedding_model
        assert model.dimensions == settings.ollama.embedding_dim

    def test_без_клиента_отказывают(self, settings: Settings) -> None:
        with pytest.raises(ConfigurationError, match="HTTP-клиент"):
            build_embeddings(settings)

    def test_облачный_провайдер_не_меняет_эмбеддинги(
        self, settings: Settings, client: httpx.AsyncClient
    ) -> None:
        """У Anthropic эмбеддингов нет, и корпус векторизован bge-m3.

        Векторы разных моделей несравнимы: посчитать запрос чужой моделью
        значит получить правдоподобный мусор без сообщения об ошибке.
        """
        cloud = settings.model_copy(
            update={
                "llm": LLMSettings(provider="anthropic"),
                "anthropic": AnthropicSettings(api_key=SecretStr("sk-ant-подделка-для-теста")),
            }
        )

        model = build_embeddings(cloud, client)

        assert model.model_name == settings.ollama.embedding_model


class TestТипыВозврата:
    def test_фабрика_объявляет_порты_а_не_адаптеры(self) -> None:
        """Аннотация — часть границы.

        Вернув конкретный тип, фабрика позволила бы вызывающему завязаться
        на частности реализации, и запрет на импорт адаптера в слайсе стал
        бы формальностью.
        """
        # `get_type_hints`, а не `signature`: в модуле включён
        # `from __future__ import annotations`, и сырые аннотации остаются
        # строками — сравнение с типом всегда было бы ложным.
        assert typing.get_type_hints(build_llm)["return"] is StructuredLLM
        assert typing.get_type_hints(build_embeddings)["return"] is EmbeddingModel


class TestГраницаСлайсов:
    def test_ни_один_слайс_не_импортирует_конкретный_адаптер(self) -> None:
        """Правило `ARCHITECTURE.md`, проверяемое кодом, а не вычиткой.

        Composition root — CLI каждого слайса и точки входа — выбирать
        реализацию вправе. Модули логики внутри слайса — нет: собранный
        руками `OllamaLLM` молча игнорирует `LLM__PROVIDER`, и переключение
        провайдера ломается без единой ошибки.
        """
        root = Path(__file__).resolve().parents[1] / "src" / "nutri_radar"
        # Общий слой моделей и composition roots исключены осознанно:
        # им выбирать реализацию положено.
        entry_points = {"api", "bot", "mcp_server"}
        offenders = []

        for path in root.rglob("*.py"):
            relative = path.relative_to(root)
            top = relative.parts[0]
            if top == "llm" or top in entry_points or path.name == "cli.py":
                continue
            if "llm.adapters" in path.read_text(encoding="utf-8"):
                offenders.append(str(relative))

        assert offenders == [], (
            "Слайс импортирует конкретный адаптер вместо порта или фабрики: "
            f"{offenders}. Используйте nutri_radar.llm.factory."
        )
