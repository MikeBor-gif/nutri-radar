"""Настройка логирования.

Зависимости нет намеренно (ADR-010): структурный вывод — это ~40 строк
форматтера на стандартном `logging`. Тащить structlog ради JSON-строки
противоречит основному сигналу проекта: не брать тяжёлый инструмент там,
где хватает простого.

Два формата вывода:

* человекочитаемый — для разработки в терминале;
* JSON — для контейнеров и CI, где логи парсят машины.

Плюс фильтр вычищения секретов: `SecretStr` и так не раскрывается в repr,
но пароль может попасть в лог через собранный DSN или через текст ошибки
драйвера. Фильтр — вторая линия защиты (правило 10 брифа).
"""

from __future__ import annotations

import json
import logging
import re
import sys
from typing import Any

# Поля LogRecord, которые не являются пользовательскими данными: всё остальное,
# переданное через extra=, попадает в структурный вывод.
_STANDARD_RECORD_FIELDS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)

_REDACTED = "***"

# Пароль внутри URL вида postgresql+asyncpg://user:password@host:5432/db
_DSN_PASSWORD_RE = re.compile(r"(?P<prefix>://[^:/@\s]+:)(?P<secret>[^@\s]+)(?P<suffix>@)")


class SecretRedactingFilter(logging.Filter):
    """Заменяет известные секреты и пароли в DSN на `***`.

    Секреты регистрируются на старте приложения: фильтр не угадывает, что
    является секретом, а получает конкретные значения из настроек.

    Замена буквальная и контекст не учитывает — это осознанный выбор в пользу
    безопасности. Следствие: если пароль совпадает с подстрокой обычного текста
    (например пароль `nutri` при пользователе `nutri` и базе `nutri_radar`),
    затрётся и она, а диагностическое сообщение станет нечитаемым. Поэтому
    пароли в проекте не должны быть подстроками имён — см. `DatabaseSettings`.
    """

    def __init__(self, secrets: frozenset[str] = frozenset()) -> None:
        super().__init__()
        # Пустые и слишком короткие значения игнорируем: замена строки из двух
        # символов изрешетила бы весь лог.
        self._secrets = frozenset(s for s in secrets if len(s) >= 4)

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self._scrub(str(record.getMessage()))
        # getMessage уже подставил args в текст — иначе они подставятся повторно
        record.args = ()

        for key, value in list(record.__dict__.items()):
            if key not in _STANDARD_RECORD_FIELDS and isinstance(value, str):
                record.__dict__[key] = self._scrub(value)
        return True

    def _scrub(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, _REDACTED)
        return _DSN_PASSWORD_RE.sub(rf"\g<prefix>{_REDACTED}\g<suffix>", text)


class JsonFormatter(logging.Formatter):
    """Одна строка JSON на запись. Для контейнеров и CI."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_RECORD_FIELDS
        }
        if extras:
            payload["data"] = extras
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class HumanFormatter(logging.Formatter):
    """Читаемый формат для разработки: `HH:MM:SS LEVEL [logger] message {data}`."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-8s [%(name)s] %(message)s", datefmt="%H:%M:%S"
        )

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_RECORD_FIELDS
        }
        if extras:
            rendered = ", ".join(f"{key}={value!r}" for key, value in sorted(extras.items()))
            base = f"{base} {{{rendered}}}"
        return base


def setup_logging(
    level: str = "INFO",
    *,
    json_output: bool = False,
    secrets: frozenset[str] = frozenset(),
) -> None:
    """Настроить корневой логгер. Повторный вызов переконфигурирует его целиком.

    Args:
        level: уровень как строка (`DEBUG`, `INFO`, ...). Управляется `LOG_LEVEL`.
        json_output: JSON вместо человекочитаемого вывода.
        secrets: значения, которые нужно вычищать из сообщений.
    """
    resolved = logging.getLevelNamesMapping().get(level.upper())
    if resolved is None:
        raise ValueError(f"Неизвестный уровень логирования: {level!r}")

    # Сообщения в проекте на русском, а консоль Windows по умолчанию не UTF-8 —
    # без этого кириллица в логах превращается в мусор.
    stream = sys.stderr
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")

    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter() if json_output else HumanFormatter())
    handler.addFilter(SecretRedactingFilter(secrets))

    root = logging.getLogger()
    # Снимаем прежние обработчики: иначе повторный setup_logging (тесты, CLI)
    # начнёт дублировать каждую запись.
    for existing in root.handlers[:]:
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(resolved)

    # SQL-эхо SQLAlchemy управляется отдельным флагом конфига, а не LOG_LEVEL:
    # на DEBUG он забивает вывод целиком (см. db/session.py).
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
