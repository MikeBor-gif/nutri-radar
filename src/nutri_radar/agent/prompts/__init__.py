"""Промпты агента, версионируемые отдельно.

Тот же довод, что в M4 и M5: общий каталог заставил бы версионировать
их вместе с промптами других задач, и правка одной ломала бы сравнимость
чисел другой.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from nutri_radar.errors import ConfigurationError
from nutri_radar.logging import safe_extra

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent
_SUFFIX = ".md"


@dataclass(frozen=True)
class AgentPrompt:
    """Шаблон промпта одной версии."""

    version: str
    template: str

    def render(self, *, question: str, tools: str, max_steps: int) -> str:
        return self.template.format(question=question, tools=tools, max_steps=max_steps)


def available_versions() -> list[str]:
    return sorted(path.stem for path in _PROMPTS_DIR.glob(f"*{_SUFFIX}"))


@cache
def load_prompt(version: str) -> AgentPrompt:
    """Загрузить промпт по версии."""
    path = _PROMPTS_DIR / f"{version}{_SUFFIX}"
    if not path.exists():
        available = available_versions()
        logger.error(
            "Запрошена несуществующая версия промпта агента",
            extra=safe_extra(requested=version, available=available),
        )
        raise ConfigurationError(
            f"Версия промпта {version!r} не найдена. Доступны: {', '.join(available)}"
        )

    template = path.read_text(encoding="utf-8").strip()
    for placeholder in ("{question}", "{tools}", "{max_steps}"):
        if placeholder not in template:
            raise ConfigurationError(f"В промпте {version!r} нет плейсхолдера {placeholder}")
    return AgentPrompt(version=version, template=template)
