"""Подготовка текста состава перед отправкой в модель.

Необходимость доказана на реальных данных корпуса, а не предположена.
В `products` встречается разметка прямо внутри состава:

    Hergestellt aus pasteurisierter <span class="allergen">Milch</span> in Bayern.

Без очистки модель возвращает аллерген вместе с тегом — проверено на живой
модели, ответ был `<span class="allergen">Milch</span>`.

**Скобки не разворачиваются намеренно.** Содержимое скобок — это состав
составного ингредиента, и модель его извлекает. Разворачивание потеряло бы
вложенность, которая несёт смысл: «Flour (Wheat Flour, Calcium)» — это мука,
состоящая из перечисленного, а не четыре независимых ингредиента.
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass

from nutri_radar.logging import safe_extra

logger = logging.getLogger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
# Хвосты вида «_» и «*» на границах, которыми в OFF помечают органику и сноски.
_EDGE_NOISE_RE = re.compile(r"^[\s\-–—_*.,;:]+|[\s\-–—_*.,;:]+$")

# Грубая оценка: сколько символов приходится на токен. Для смеси латиницы
# и кириллицы 3 — консервативная нижняя граница, то есть оценка токенов
# получается завышенной, и это безопаснее занижения.
_CHARS_PER_TOKEN = 3

# Доля контекста, которую можно отдать под сам состав. Остальное уходит
# на промпт и на ответ модели.
_TEXT_CONTEXT_SHARE = 0.4


@dataclass(frozen=True)
class PreparedText:
    """Результат подготовки."""

    cleaned: str
    original_length: int
    had_markup: bool
    too_long: bool

    @property
    def cleaned_length(self) -> int:
        return len(self.cleaned)

    @property
    def is_usable(self) -> bool:
        """Годится ли текст для отправки в модель."""
        return bool(self.cleaned) and not self.too_long


def max_text_length(num_ctx: int) -> int:
    """Сколько символов состава влезает в контекст модели.

    Считается от `num_ctx`, а не берётся константой: размер контекста —
    параметр конфигурации, и при его изменении предел должен меняться сам.
    """
    return int(num_ctx * _TEXT_CONTEXT_SHARE * _CHARS_PER_TOKEN)


def prepare_text(text: str | None, *, num_ctx: int) -> PreparedText:
    """Очистить текст состава и проверить, влезает ли он в контекст.

    Проверка длины обязательна: при превышении `num_ctx` Ollama обрезает вход
    **молча**, и ответ выглядит валидным, хотя разобрана только часть состава.
    Такой текст лучше не отправлять вовсе и пометить, чем получить правдоподобно
    выглядящий мусор.
    """
    original = text or ""
    had_markup = bool(_TAG_RE.search(original)) or "&" in original

    cleaned = html.unescape(original)
    cleaned = _TAG_RE.sub(" ", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    cleaned = _EDGE_NOISE_RE.sub("", cleaned).strip()

    limit = max_text_length(num_ctx)
    too_long = len(cleaned) > limit

    if had_markup:
        logger.debug(
            "Из состава снята разметка",
            extra=safe_extra(before=len(original), after=len(cleaned)),
        )
    if too_long:
        logger.warning(
            "Состав длиннее допустимого для контекста модели — в модель не отправляется",
            extra=safe_extra(length=len(cleaned), limit=limit, num_ctx=num_ctx),
        )

    return PreparedText(
        cleaned=cleaned,
        original_length=len(original),
        had_markup=had_markup,
        too_long=too_long,
    )


@dataclass
class PreprocessStats:
    """Сводка по прогону. Характеристика данных, её надо знать."""

    total: int = 0
    with_markup: int = 0
    too_long: int = 0
    empty: int = 0

    def add(self, prepared: PreparedText) -> None:
        self.total += 1
        if prepared.had_markup:
            self.with_markup += 1
        if prepared.too_long:
            self.too_long += 1
        if not prepared.cleaned:
            self.empty += 1

    def log_summary(self) -> None:
        if not self.total:
            return
        logger.info(
            "Предобработка составов завершена",
            extra=safe_extra(
                total=self.total,
                with_markup=self.with_markup,
                markup_share=f"{self.with_markup / self.total:.2%}",
                too_long=self.too_long,
                empty=self.empty,
            ),
        )
