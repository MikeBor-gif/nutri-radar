"""Промпты zero-shot для M4.

Отдельные от промптов извлечения (M2) намеренно: там задача — вытащить
структуру из текста, здесь — поставить оценку. Общий каталог заставил бы
версионировать их вместе, и правка одной задачи ломала бы сравнимость
чисел другой.

**Zero-shot, а не few-shot.** Ни одного примера из train в промпте нет.
Пример с меткой — это обучающая выборка, поданная через промпт, и подход,
получивший десять примеров, сравнивается с логрегрессией, получившей
105 тысяч, уже не как «модель без обучения». Подпись строки в таблице
обязана соответствовать тому, что реально происходило.

**Схема с `enum`, а не просьба в промпте.** Ответ вне множества классов
физически невозможен, поэтому отказы модели не превращаются в отдельный
разбор строк вроде «grade B» или «probably d».
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from nutri_radar.errors import ConfigurationError
from nutri_radar.logging import safe_extra

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent
_TEMPLATE_SUFFIX = ".md"


@dataclass(frozen=True)
class GradePrompt:
    """Шаблон промпта одной версии."""

    version: str
    template: str

    def render(self, text: str, *, lang: str) -> str:
        return self.template.format(text=text, lang=lang or "unknown")


def available_versions() -> list[str]:
    return sorted(path.stem for path in _PROMPTS_DIR.glob(f"*{_TEMPLATE_SUFFIX}"))


@cache
def load_prompt(version: str) -> GradePrompt:
    """Загрузить промпт по версии.

    Raises:
        ConfigurationError: версии нет на диске. Сообщение перечисляет
            доступные — иначе опечатка даёт непонятный отказ.
    """
    path = _PROMPTS_DIR / f"{version}{_TEMPLATE_SUFFIX}"
    if not path.exists():
        available = available_versions()
        logger.error(
            "Запрошена несуществующая версия промпта",
            extra=safe_extra(requested=version, available=available),
        )
        raise ConfigurationError(
            f"Версия промпта {version!r} не найдена. Доступны: {', '.join(available)}"
        )

    template = path.read_text(encoding="utf-8").strip()
    for placeholder in ("{text}", "{lang}"):
        if placeholder not in template:
            raise ConfigurationError(
                f"В промпте {version!r} нет плейсхолдера {placeholder}: "
                "состав или язык не подставятся, и модель получит шаблон как есть"
            )
    return GradePrompt(version=version, template=template)


def grade_schema(labels: list[str]) -> dict[str, Any]:
    """JSON-схема ответа: одно поле с закрытым множеством значений.

    Множество приходит из данных, а не зашито здесь: у `nutriscore_grade`
    это `a`–`e`, у `nova_group` — `1`, `3`, `4`, и захардкоженный список
    разъехался бы с реальными классами при первом же изменении корпуса.
    """
    return {
        "type": "object",
        "properties": {"grade": {"type": "string", "enum": sorted(labels)}},
        "required": ["grade"],
        "additionalProperties": False,
    }
