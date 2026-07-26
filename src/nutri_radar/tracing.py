"""Порт трассировки и реализация-заглушка.

Langfuse появляется на M6 и живёт в отдельном compose-профиле (ADR-007).
Но вызовы трассировки пишутся с самого начала: иначе на M6 придётся
проходить по всему пайплайну и расставлять их задним числом.

По умолчанию работает `NoOpTracer` — отсутствие Langfuse не ломает пайплайн.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class Tracer(Protocol):
    """Минимальный контракт трассировщика.

    Сознательно узкий: только span. Всё, что нужно проекту — видеть шаги
    агента и прогоны извлечения. Расширять по факту потребности на M6,
    а не проектировать впрок.
    """

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[None]:
        """Открыть участок трассировки с произвольными атрибутами."""
        ...


class NoOpTracer:
    """Ничего не отправляет наружу, но пишет в лог на DEBUG.

    Логирование здесь не декоративное: без него на M6 будет непонятно,
    вызывалась ли трассировка вообще и с какими атрибутами.
    """

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[None]:
        logger.debug("span начат: %s", name, extra={"span": name, **attributes})
        started = time.perf_counter()
        try:
            yield
        except Exception as exc:
            logger.debug(
                "span прерван ошибкой: %s",
                name,
                extra={
                    "span": name,
                    "elapsed_s": round(time.perf_counter() - started, 4),
                    "error": type(exc).__name__,
                },
            )
            raise
        else:
            logger.debug(
                "span завершён: %s",
                name,
                extra={"span": name, "elapsed_s": round(time.perf_counter() - started, 4)},
            )
