"""Тесты логирования.

Главное здесь — проверка, что секреты не утекают в логи (правило 10 брифа).
Это не косметика: DSN с паролем и ключ облака попадают в сообщения об ошибках
драйверов, и без фильтра уехали бы в вывод CI.
"""

from __future__ import annotations

import json
import logging

import pytest

from nutri_radar.logging import (
    HumanFormatter,
    JsonFormatter,
    SecretRedactingFilter,
    setup_logging,
)


def _record(message: str, **extra: object) -> logging.LogRecord:
    record = logging.LogRecord(
        name="тест",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


class TestSetupLogging:
    def test_уровень_берётся_из_аргумента(self):
        setup_logging("WARNING")
        assert logging.getLogger().level == logging.WARNING

        setup_logging("DEBUG")
        assert logging.getLogger().level == logging.DEBUG

    def test_повторный_вызов_не_дублирует_обработчики(self):
        setup_logging("INFO")
        setup_logging("INFO")
        setup_logging("INFO")

        assert len(logging.getLogger().handlers) == 1

    def test_неизвестный_уровень_отбивается(self):
        with pytest.raises(ValueError, match="Неизвестный уровень"):
            setup_logging("НЕ_УРОВЕНЬ")

    def test_эхо_sqlalchemy_приглушено(self):
        """На DEBUG эхо SQL забило бы вывод целиком."""
        setup_logging("DEBUG")
        assert logging.getLogger("sqlalchemy.engine").level == logging.WARNING


class TestSecretRedaction:
    def test_зарегистрированный_секрет_вычищается_из_сообщения(self):
        f = SecretRedactingFilter(frozenset({"сверхсекрет"}))
        record = _record("токен сверхсекрет уехал в лог")

        f.filter(record)

        assert "сверхсекрет" not in record.getMessage()
        assert "***" in record.getMessage()

    def test_пароль_в_dsn_вычищается_даже_если_не_зарегистрирован(self):
        """Пароль может прийти из текста ошибки драйвера, а не из настроек."""
        f = SecretRedactingFilter(frozenset())
        record = _record("postgresql+asyncpg://nutri:неизвестный_пароль@localhost:5432/db")

        f.filter(record)

        assert "неизвестный_пароль" not in record.getMessage()
        assert "nutri" in record.getMessage(), "имя пользователя должно остаться читаемым"

    def test_секрет_вычищается_и_из_extra(self):
        f = SecretRedactingFilter(frozenset({"сверхсекрет"}))
        record = _record("сообщение", payload="внутри сверхсекрет лежит")

        f.filter(record)

        assert "сверхсекрет" not in record.payload

    def test_слишком_короткие_значения_игнорируются(self):
        """Замена строки из двух символов изрешетила бы весь лог."""
        f = SecretRedactingFilter(frozenset({"ab"}))
        record = _record("abrakadabra")

        f.filter(record)

        assert record.getMessage() == "abrakadabra"

    def test_подстановка_args_не_происходит_дважды(self):
        f = SecretRedactingFilter(frozenset())
        record = logging.LogRecord(
            name="тест",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="значение %s и ещё %s",
            args=("первое", "второе"),
            exc_info=None,
        )

        f.filter(record)

        assert record.getMessage() == "значение первое и ещё второе"


class TestFormatters:
    def test_json_формат_валиден_и_содержит_extra(self):
        rendered = JsonFormatter().format(_record("сообщение", stage="ingest", items=42))

        payload = json.loads(rendered)

        assert payload["message"] == "сообщение"
        assert payload["level"] == "INFO"
        assert payload["data"]["stage"] == "ingest"
        assert payload["data"]["items"] == 42

    def test_json_не_экранирует_кириллицу(self):
        rendered = JsonFormatter().format(_record("скрытые формы сахара"))
        assert "скрытые формы сахара" in rendered

    def test_человекочитаемый_формат_показывает_extra(self):
        rendered = HumanFormatter().format(_record("сообщение", stage="ingest"))

        assert "сообщение" in rendered
        assert "stage='ingest'" in rendered

    def test_секрет_не_доходит_до_вывода_через_setup_logging(self, caplog):
        """Сквозная проверка: фильтр действительно подключён к обработчику."""
        setup_logging("INFO", secrets=frozenset({"сверхсекрет"}))
        handler = logging.getLogger().handlers[0]

        record = _record("ключ сверхсекрет")
        for log_filter in handler.filters:
            log_filter.filter(record)

        assert "сверхсекрет" not in handler.formatter.format(record)
