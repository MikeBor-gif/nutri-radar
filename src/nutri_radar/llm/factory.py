"""Выбор конкретной реализации модели по настройкам.

**Зачем модуль в общем слое.** Слайсу нужна модель, но знать, какая именно,
он не должен: `ARCHITECTURE.md` запрещает импорт конкретного адаптера внутри
слайса, и не из формализма. Провайдер переключается через `LLM__PROVIDER`
в `.env`, и слайс, собравший `OllamaLLM` руками, это переключение молча
ломает — код продолжает работать, просто на другой модели, чем написано
в конфиге.

До M7 выбор жил в `extract/cli.py`. Пока потребитель был один — composition
root, — этого хватало. На M7 модель понадобилась общей сборке конвейера
(`retrieval/pipeline.py`), а это уже слайс, и место выбору стало здесь.

**Функции возвращают порты, а не адаптеры.** Аннотация — `StructuredLLM`
и `EmbeddingModel`; вызывающий не видит, что под ними, и не может случайно
завязаться на частности реализации.
"""

from __future__ import annotations

import logging

import anthropic
import httpx

from nutri_radar.config import Settings, get_settings
from nutri_radar.errors import ConfigurationError
from nutri_radar.llm.adapters.anthropic import AnthropicLLM
from nutri_radar.llm.adapters.ollama import OllamaEmbeddings, OllamaLLM
from nutri_radar.llm.ports import EmbeddingModel, StructuredLLM
from nutri_radar.logging import safe_extra

logger = logging.getLogger(__name__)


def build_llm(
    settings: Settings | None = None, client: httpx.AsyncClient | None = None
) -> StructuredLLM:
    """Собрать модель генерации по `LLM__PROVIDER`.

    Args:
        settings: настройки.
        client: HTTP-клиент для Ollama. Для облачного провайдера не нужен —
            у него свой SDK со своим транспортом.

    Raises:
        ConfigurationError: провайдер неизвестен или для облачного не задан
            ключ.
    """
    settings = settings or get_settings()

    match settings.llm.provider:
        case "ollama":
            if client is None:
                raise ConfigurationError(
                    "Для LLM__PROVIDER=ollama нужен HTTP-клиент: адаптер ходит "
                    "в Ollama по HTTP и не создаёт клиент сам, чтобы пул "
                    "соединений жил столько же, сколько точка входа."
                )
            logger.debug("Выбран провайдер ollama", extra=safe_extra(model=settings.ollama.model))
            return OllamaLLM(client, settings.ollama)

        case "anthropic":
            # Ключ проверяем здесь, а не внутри адаптера: выбор провайдера —
            # единственное место, которое знает про облако, и отказ должен
            # быть понятным до первого запроса, а не на сотом продукте.
            if settings.anthropic.api_key is None:
                raise ConfigurationError(
                    "LLM__PROVIDER=anthropic, но ANTHROPIC__API_KEY не задан. "
                    "Укажите ключ в .env или переключитесь на LLM__PROVIDER=ollama."
                )
            logger.debug(
                "Выбран провайдер anthropic", extra=safe_extra(model=settings.anthropic.model)
            )
            return AnthropicLLM(
                anthropic.AsyncAnthropic(
                    api_key=settings.anthropic.api_key.get_secret_value(),
                    timeout=settings.anthropic.timeout_s,
                ),
                settings.anthropic,
            )

        case unknown:
            raise ConfigurationError(f"Неизвестный провайдер LLM: {unknown}")


def build_embeddings(
    settings: Settings | None = None, client: httpx.AsyncClient | None = None
) -> EmbeddingModel:
    """Собрать модель эмбеддингов.

    Провайдер здесь всегда локальный, и это не недосмотр: у Anthropic нет
    эмбеддингов вовсе. Функция существует ради того же, ради чего `build_llm`, —
    чтобы слайс не импортировал конкретный адаптер. Когда появится второй
    источник векторов, менять придётся это место, а не каждый вызов.

    Важнее другое: **векторы разных моделей несравнимы.** Корпус
    векторизован `bge-m3`, и запрос обязан считаться той же моделью —
    иначе поиск вернёт правдоподобный мусор, не сообщив об этом.
    """
    settings = settings or get_settings()
    if client is None:
        raise ConfigurationError(
            "Для модели эмбеддингов нужен HTTP-клиент: адаптер ходит в Ollama "
            "по HTTP и не создаёт клиент сам."
        )

    logger.debug(
        "Собрана модель эмбеддингов",
        extra=safe_extra(model=settings.ollama.embedding_model, dim=settings.ollama.embedding_dim),
    )
    return OllamaEmbeddings(client, settings.ollama)
