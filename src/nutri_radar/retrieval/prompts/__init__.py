"""Промпты RAG, версионируемые отдельно от промптов извлечения и оценки.

Общий каталог заставил бы версионировать их вместе, и правка одной задачи
ломала бы сравнимость чисел другой — тот же довод, что в M4.
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
class RagPrompt:
    """Шаблон промпта одной версии."""

    version: str
    template: str

    @property
    def forces_language(self) -> bool:
        """Требует ли эта версия правила о языке ответа отдельным блоком.

        Определяется наличием плейсхолдера, а не списком версий в коде:
        список пришлось бы править в двух местах при каждой новой версии,
        и он разошёлся бы с файлами на диске — ровно так ломается
        сравнимость замеров.
        """
        return "{language_rule}" in self.template

    def render(self, *, question: str, products: str, language_rule: str = "") -> str:
        """Подставить вопрос, продукты и правило о языке.

        Версии без `{language_rule}` просто игнорируют лишний аргумент:
        `str.format` не возражает против неиспользованных ключей. Это
        сознательно — иначе вызывающий код ветвился бы по версии промпта,
        а решать, что делать с плейсхолдером, должен шаблон.
        """
        return self.template.format(
            question=question, products=products, language_rule=language_rule
        )


def available_versions() -> list[str]:
    return sorted(path.stem for path in _PROMPTS_DIR.glob(f"*{_SUFFIX}"))


@cache
def load_prompt(version: str) -> RagPrompt:
    """Загрузить промпт по версии.

    Raises:
        ConfigurationError: версии нет на диске. Сообщение перечисляет
            доступные — иначе опечатка даёт непонятный отказ.
    """
    path = _PROMPTS_DIR / f"{version}{_SUFFIX}"
    if not path.exists():
        available = available_versions()
        logger.error(
            "Запрошена несуществующая версия промпта RAG",
            extra=safe_extra(requested=version, available=available),
        )
        raise ConfigurationError(
            f"Версия промпта {version!r} не найдена. Доступны: {', '.join(available)}"
        )

    template = path.read_text(encoding="utf-8").strip()
    for placeholder in ("{question}", "{products}"):
        if placeholder not in template:
            raise ConfigurationError(
                f"В промпте {version!r} нет плейсхолдера {placeholder}: "
                "модель получит шаблон как есть"
            )
    return RagPrompt(version=version, template=template)
