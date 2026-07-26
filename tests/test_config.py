"""Тесты конфигурации.

Отдельно проверяется актуальность `.env.example`: без этого теста файл
устареет через два майлстоуна, и новый разработчик не сможет поднять проект.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from nutri_radar.config import (
    AnthropicSettings,
    DatabaseSettings,
    IngestSettings,
    OllamaSettings,
    Settings,
    get_settings,
)
from nutri_radar.errors import ConfigurationError

ENV_EXAMPLE = Path(".env.example")


class TestNestedPrefixes:
    def test_вложенные_группы_читаются_с_префиксом(self, monkeypatch):
        monkeypatch.setenv("DB__HOST", "db.example")
        monkeypatch.setenv("DB__PORT", "6543")
        monkeypatch.setenv("OLLAMA__NUM_CTX", "16384")

        settings = Settings()

        assert settings.db.host == "db.example"
        assert settings.db.port == 6543
        assert settings.ollama.num_ctx == 16384

    def test_dsn_собирается_из_частей(self):
        db = DatabaseSettings(
            host="h",
            port=1234,
            user="u",
            password="p",
            name="n",
        )
        assert db.dsn == "postgresql+asyncpg://u:p@h:1234/n"


class TestSecrets:
    """Правило 10 брифа: секреты не попадают в код, логи и коммиты."""

    def test_пароль_не_раскрывается_в_repr_и_str(self):
        db = DatabaseSettings(password="сверхсекретное_значение")

        assert "сверхсекретное_значение" not in repr(db)
        assert "сверхсекретное_значение" not in str(db)
        assert "сверхсекретное_значение" not in repr(db.password)
        # Само значение доступно только через явный вызов
        assert db.password.get_secret_value() == "сверхсекретное_значение"

    def test_safe_dsn_не_содержит_пароля_а_dsn_содержит(self):
        db = DatabaseSettings(password="пароль_из_теста")

        assert "пароль_из_теста" not in db.safe_dsn
        assert "***" in db.safe_dsn
        assert "пароль_из_теста" in db.dsn

    def test_describe_не_содержит_секретов(self):
        settings = Settings(
            db=DatabaseSettings(password="пароль_бд"),
            anthropic=AnthropicSettings(api_key="ключ_облака"),
        )

        rendered = repr(settings.describe())

        assert "пароль_бд" not in rendered
        assert "ключ_облака" not in rendered

    def test_secret_values_собирает_оба_секрета(self):
        settings = Settings(
            db=DatabaseSettings(password="пароль_бд"),
            anthropic=AnthropicSettings(api_key="ключ_облака"),
        )

        assert settings.secret_values() == frozenset({"пароль_бд", "ключ_облака"})

    def test_пустой_ключ_anthropic_считается_отсутствующим(self):
        """В .env.example ключ объявлен пустым — это НЕ заданный ключ."""
        assert AnthropicSettings(api_key="").api_key is None
        assert AnthropicSettings(api_key="   ").is_configured is False
        assert AnthropicSettings(api_key=None).is_configured is False
        assert AnthropicSettings(api_key="sk-настоящий").is_configured is True


class TestValidators:
    def test_маленький_num_ctx_отбивается(self):
        with pytest.raises(ValidationError, match="слишком мал"):
            OllamaSettings(num_ctx=1024)

    def test_граничное_значение_num_ctx_проходит(self):
        assert OllamaSettings(num_ctx=2048).num_ctx == 2048

    def test_нулевой_параллелизм_отбивается(self):
        with pytest.raises(ValidationError, match=">= 1"):
            OllamaSettings(max_concurrency=0)

    def test_пустой_список_языков_отбивается(self):
        with pytest.raises(ValidationError, match="languages пуст"):
            IngestSettings(languages=[])


class TestErrorWrapping:
    def test_невалидная_настройка_даёт_ConfigurationError(self, monkeypatch):
        """Наружу не должен протекать ValidationError pydantic."""
        monkeypatch.setenv("LOG_LEVEL", "НЕ_УРОВЕНЬ")
        get_settings.cache_clear()

        with pytest.raises(ConfigurationError) as exc_info:
            get_settings()

        message = str(exc_info.value)
        # Сообщение должно указывать конкретный ключ и куда смотреть
        assert "log_level" in message
        assert ".env.example" in message

        get_settings.cache_clear()

    def test_настройки_кэшируются(self):
        get_settings.cache_clear()
        assert get_settings() is get_settings()
        get_settings.cache_clear()


class TestEnvExampleIsCurrent:
    """`.env.example` — эталон для нового разработчика, он обязан быть полным."""

    @staticmethod
    def _keys_from_env_example() -> set[str]:
        assert ENV_EXAMPLE.exists(), f"Не найден {ENV_EXAMPLE}"
        keys: set[str] = set()
        for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            match = re.match(r"^([A-Z0-9_]+)=", stripped)
            if match:
                keys.add(match.group(1))
        return keys

    @staticmethod
    def _expected_keys() -> set[str]:
        """Ключи, выведенные из полей `Settings` и префиксов вложенных групп."""
        expected: set[str] = set()
        for field_name, field in Settings.model_fields.items():
            nested = field.annotation
            assert isinstance(nested, type) and issubclass(nested, BaseModel), field_name
            prefix = getattr(nested, "model_config", {}).get("env_prefix", "") or ""
            for nested_field in nested.model_fields:
                expected.add(f"{prefix}{nested_field}".upper())
        return expected

    def test_каждое_поле_настроек_присутствует_в_env_example(self):
        missing = self._expected_keys() - self._keys_from_env_example()
        assert not missing, (
            "В .env.example отсутствуют ключи: "
            f"{sorted(missing)}. Добавили параметр в config.py — добавьте и туда."
        )

    def test_в_env_example_нет_лишних_ключей(self):
        extra = self._keys_from_env_example() - self._expected_keys()
        assert not extra, (
            f"В .env.example есть ключи, которых нет в Settings: {sorted(extra)}. "
            "Скорее всего параметр удалили из конфига, а из примера — нет."
        )

    def test_в_env_example_нет_настоящих_секретов(self):
        """В репозиторий не должен уехать заполненный ключ."""
        content = ENV_EXAMPLE.read_text(encoding="utf-8")
        assert not re.search(r"^ANTHROPIC__API_KEY=\s*sk-", content, re.MULTILINE), (
            "В .env.example похоже попал настоящий ключ Anthropic"
        )
