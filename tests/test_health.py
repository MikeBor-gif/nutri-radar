"""Тесты health-check.

Ключевое поведение, которое проверяется:

* деградации (Ollama, ключ, миграции) дают WARN и НЕ роняют команду;
* отказы БД дают FAIL и код возврата 1;
* падение одной проверки не отменяет остальные — в отчёте всегда пять пунктов.

В сеть не ходим: Ollama подменена MockTransport, БД недоступна намеренно
(порт заведомо закрыт), что и является проверяемым сценарием отказа.
"""

from __future__ import annotations

import pytest

from nutri_radar.config import AnthropicSettings, DatabaseSettings, Settings
from nutri_radar.db.session import dispose_engine
from nutri_radar.health import CheckStatus, check_health


@pytest.fixture(autouse=True)
async def _reset_engine():
    """Движок кэшируется в модуле — между тестами его нужно сбрасывать."""
    await dispose_engine()
    yield
    await dispose_engine()


def _settings_with_unreachable_db(base: Settings) -> Settings:
    """Порт 1 закрыт всегда — подключение гарантированно не установится.

    Хост задан числом, а не `localhost`: последний разрешается сразу в ::1
    и 127.0.0.1, и попытки идут по очереди, удваивая ожидание.
    """
    return base.model_copy(
        update={"db": DatabaseSettings(**{**base.db.model_dump(), "host": "127.0.0.1", "port": 1})}
    )


class TestUnreachableDatabase:
    async def test_недоступная_бд_даёт_fail_и_код_1(self, settings, fake_ollama):
        report = await check_health(
            _settings_with_unreachable_db(settings), http_client=fake_ollama
        )

        assert report.is_healthy is False
        assert report.exit_code == 1
        assert len(report.failures) >= 1

    async def test_падение_бд_не_отменяет_остальные_проверки(self, settings, fake_ollama):
        """Иначе после первой ошибки непонятно, что ещё сломано."""
        report = await check_health(
            _settings_with_unreachable_db(settings), http_client=fake_ollama
        )

        assert len(report.checks) == 5
        # Ollama проверена, несмотря на отказ БД
        ollama = next(c for c in report.checks if "Ollama" in c.name)
        assert ollama.status is CheckStatus.OK

    async def test_ошибка_бд_не_раскрывает_пароль(self, settings, fake_ollama):
        broken = _settings_with_unreachable_db(settings)
        report = await check_health(broken, http_client=fake_ollama)

        password = broken.db.password.get_secret_value()
        for check in report.checks:
            assert password not in check.detail


class TestOllamaDegradation:
    async def test_недоступная_ollama_даёт_warn_но_не_роняет(self, settings, unavailable_ollama):
        report = await check_health(
            _settings_with_unreachable_db(settings), http_client=unavailable_ollama
        )

        ollama = next(c for c in report.checks if "Ollama" in c.name)
        assert ollama.status is CheckStatus.WARN
        assert "не критично" in ollama.detail

    async def test_отсутствие_моделей_даёт_warn_с_подсказкой(
        self, settings, fake_ollama_without_models
    ):
        report = await check_health(
            _settings_with_unreachable_db(settings), http_client=fake_ollama_without_models
        )

        ollama = next(c for c in report.checks if "Ollama" in c.name)
        assert ollama.status is CheckStatus.WARN
        assert "ollama pull" in ollama.detail

    async def test_модель_опознаётся_несмотря_на_тег_latest(self, settings, fake_ollama):
        """Ollama отдаёт `имя:latest`, в конфиге тег может быть опущен."""
        report = await check_health(
            _settings_with_unreachable_db(settings), http_client=fake_ollama
        )

        ollama = next(c for c in report.checks if "Ollama" in c.name)
        assert ollama.status is CheckStatus.OK


class TestAnthropicKey:
    async def test_отсутствие_ключа_даёт_warn(self, settings, fake_ollama):
        report = await check_health(
            _settings_with_unreachable_db(settings), http_client=fake_ollama
        )

        key_check = next(c for c in report.checks if "Anthropic" in c.name)
        assert key_check.status is CheckStatus.WARN

    async def test_заданный_ключ_даёт_ok_и_не_печатается(self, settings, fake_ollama):
        with_key = settings.model_copy(
            update={"anthropic": AnthropicSettings(api_key="sk-секрет-для-теста")}
        )
        report = await check_health(
            _settings_with_unreachable_db(with_key), http_client=fake_ollama
        )

        key_check = next(c for c in report.checks if "Anthropic" in c.name)
        assert key_check.status is CheckStatus.OK
        assert "sk-секрет-для-теста" not in key_check.detail


class TestReportShape:
    async def test_в_отчёте_ровно_пять_проверок(self, settings, fake_ollama):
        report = await check_health(
            _settings_with_unreachable_db(settings), http_client=fake_ollama
        )
        assert len(report.checks) == 5

    async def test_отчёт_сериализуется_в_json(self, settings, fake_ollama):
        report = await check_health(
            _settings_with_unreachable_db(settings), http_client=fake_ollama
        )

        payload = report.model_dump_json()
        assert '"checks"' in payload
        assert '"status"' in payload

    async def test_только_warn_даёт_код_0(self, settings, fake_ollama, monkeypatch):
        """Здоровье определяется отсутствием FAIL, а не отсутствием WARN."""
        report = await check_health(
            _settings_with_unreachable_db(settings), http_client=fake_ollama
        )
        # Искусственно убираем FAIL и проверяем правило
        report.checks = [c for c in report.checks if c.status is not CheckStatus.FAIL]

        assert report.warnings
        assert report.is_healthy is True
        assert report.exit_code == 0


@pytest.mark.integration
class TestAgainstLiveDatabase:
    """Требует поднятого Postgres: docker compose up -d db."""

    async def test_на_свежесмигрированной_базе_нет_отказов(
        self, integration_settings, migrated_database, fake_ollama
    ):
        report = await check_health(integration_settings, http_client=fake_ollama)

        assert report.failures == [], [c.model_dump() for c in report.failures]
        assert report.exit_code == 0

        vector = next(c for c in report.checks if "vector" in c.name)
        assert vector.status is CheckStatus.OK

        alembic_check = next(c for c in report.checks if "Alembic" in c.name)
        assert alembic_check.status is CheckStatus.OK
        assert "head" in alembic_check.detail
