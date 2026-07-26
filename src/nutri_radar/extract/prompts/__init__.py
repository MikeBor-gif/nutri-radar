"""Версионируемые промпты извлечения.

Бриф требует версионировать промпт и писать версию в БД рядом с результатом,
а DoD M3 — сравнить не менее трёх версий. Поэтому инфраструктура появляется
сразу, а не достраивается задним числом.

Три версии заданы **разными гипотезами**, выведенными из живой проверки на
`qwen2.5:3b-instruct-q4_K_M`, а не косметическими правками:

* **v1** — минимальный промпт. Базовая линия. На нём наивно ожидалось, что
  модель сама разберётся; проверка показала ноль форм сахара на составе,
  где их три.
* **v2** — подробные правила по каждому виду с перечислением форм сахара.
  Проверено: на длинном составе с вложенными скобками извлечение **просело** —
  модель вернула мало ингредиентов, хотя нужные термины перечислены в промпте
  буквально.
* **v3** — короткий промпт с явным «перечисли ВСЕ ингредиенты, включая те, что
  в скобках», плюс краткий список форм сахара. Проверено: нашла **все три**
  формы сахара и развернула скобки (13 ингредиентов из 12), но почти всё
  классифицировала как `flavouring`.

Зафиксированный конфликт: подробные правила ломают извлечение, короткий промпт
ломает классификацию. Какая версия победит — решают метрики M3, а не мнение.

Все версии на английском и требуют английских канонических имён: русский
промпт заставлял модель переводить имена, а их предстоит сопоставлять
с таксономией OFF, где теги вида `en:sugar`.
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

# Версия по умолчанию не задаётся здесь: она приходит из настроек, иначе
# «версия промпта» перестала бы быть параметром прогона.
_TEMPLATE_SUFFIX = ".md"


@dataclass(frozen=True)
class Prompt:
    """Шаблон промпта конкретной версии."""

    version: str
    template: str

    def render(self, text: str, *, lang: str) -> str:
        """Подставить состав и язык.

        Язык передаётся как контекст: модель работает с пятью языками,
        и подсказка помогает ей не принять немецкий состав за английский.
        """
        return self.template.format(text=text, lang=lang or "unknown")

    @property
    def length(self) -> int:
        return len(self.template)


def available_versions() -> list[str]:
    """Все версии, доступные на диске. Нужен `evals` для перебора."""
    return sorted(path.stem for path in _PROMPTS_DIR.glob(f"*{_TEMPLATE_SUFFIX}"))


@cache
def load_prompt(version: str) -> Prompt:
    """Загрузить промпт по версии.

    Raises:
        ConfigurationError: версии нет на диске. Сообщение перечисляет
            доступные — иначе опечатка в `.env` даёт непонятный отказ.
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
    if "{text}" not in template:
        raise ConfigurationError(
            f"Шаблон промпта {version!r} не содержит плейсхолдер {{text}} — "
            "состав в него не подставится"
        )

    prompt = Prompt(version=version, template=template)
    logger.info(
        "Промпт загружен",
        extra=safe_extra(version=version, length=prompt.length),
    )
    return prompt
