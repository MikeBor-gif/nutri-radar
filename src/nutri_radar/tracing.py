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


class LangfuseTracer:
    """Реализация порта `Tracer` поверх Langfuse.

    **Отсутствие Langfuse не должно ронять агента.** Наблюдаемость — это
    удобство разбора, а не часть работы: прогон на двадцати вопросах,
    падающий из-за недоступного контейнера трассировки, теряет данные ради
    их же протоколирования. Поэтому любой отказ здесь логируется и
    проглатывается, а фабрика ниже при неудачном подключении возвращает
    заглушку.

    Атрибуты уходят в `metadata`: порт узкий и передаёт произвольные
    ключи, а раскладывать их по полям Langfuse значило бы завязать порт
    на конкретного вендора.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[None]:
        try:
            with self._client.start_as_current_observation(
                name=name, as_type="span", metadata=attributes or None
            ):
                yield
        except Exception as exc:
            # Ошибка самой трассировки не должна выглядеть как ошибка
            # трассируемого кода. Тело уже выполнено или не начиналось —
            # в обоих случаях лучше потерять запись, чем прогон.
            logger.warning(
                "Трассировка отказала — работаем без неё",
                extra={"span": name, "error": type(exc).__name__},
            )
            yield

    def flush(self) -> None:
        """Дослать накопленное. Прогон короткий, и без явного сброса
        последние спаны не успевают уйти до выхода процесса."""
        try:
            self._client.flush()
        except Exception as exc:
            logger.warning("Сброс трассировки не удался", extra={"error": type(exc).__name__})


def get_tracer(settings: Any = None) -> Tracer:
    """Собрать трассировщик по настройкам.

    Возвращает `NoOpTracer`, если Langfuse выключен, не настроен или
    недоступен. Проверка связи делается здесь один раз, а не при первом
    спане: узнать о неработающей трассировке в начале прогона полезнее,
    чем на середине.
    """
    from nutri_radar.config import get_settings

    settings = settings or get_settings()
    cfg = settings.langfuse

    if not cfg.is_configured:
        logger.info("Langfuse не настроен — трассировка в логи")
        return NoOpTracer()

    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=cfg.public_key,
            secret_key=cfg.secret_key.get_secret_value(),
            host=cfg.host,
            timeout=int(cfg.timeout_s),
        )
        if not client.auth_check():
            logger.warning("Langfuse отверг ключи — трассировка в логи")
            return NoOpTracer()
    except Exception as exc:
        logger.warning(
            "Langfuse недоступен — трассировка в логи",
            extra={"host": cfg.host, "error": type(exc).__name__},
        )
        return NoOpTracer()

    logger.info("Трассировка через Langfuse включена", extra={"host": cfg.host})
    return LangfuseTracer(client)
