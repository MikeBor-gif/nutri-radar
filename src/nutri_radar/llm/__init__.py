"""Доступ к языковым моделям: порты и адаптеры.

Общий модуль, а не часть `extract`: потребителей четыре — `extract` (M2),
`evals` (M3), `analytics` (M4), `agent` (M6). Внутри одного слайса он создал бы
перекрёстные зависимости между слайсами (см. ARCHITECTURE.md).
"""

from __future__ import annotations

from nutri_radar.llm.models import LLMResponse, TokenUsage
from nutri_radar.llm.ports import StructuredLLM

__all__ = ["LLMResponse", "StructuredLLM", "TokenUsage"]
