"""Конкретные реализации порта `StructuredLLM`.

Слайсы их НЕ импортируют: провайдера выбирает composition root по `.env`.
"""

from __future__ import annotations

from nutri_radar.llm.adapters.fake import FakeLLM
from nutri_radar.llm.adapters.ollama import OllamaEmbeddings, OllamaLLM

__all__ = ["FakeLLM", "OllamaEmbeddings", "OllamaLLM"]
