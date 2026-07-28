"""Тесты версионируемых промптов.

Версия промпта уезжает в БД рядом с результатом, и DoD M3 требует сравнить
не менее трёх версий. Поэтому проверяется не «промпт красивый», а то, без чего
сравнение сломается: версии существуют, различаются и корректно подставляют
состав.
"""

from __future__ import annotations

import pytest

from nutri_radar.errors import ConfigurationError
from nutri_radar.extract.prompts import available_versions, load_prompt

EXPECTED_VERSIONS = ["v1", "v2", "v3"]


class TestДоступныеВерсии:
    def test_все_три_версии_на_месте(self):
        """Три версии заданы разными гипотезами, а не косметическими правками."""
        assert available_versions() == EXPECTED_VERSIONS

    @pytest.mark.parametrize("version", EXPECTED_VERSIONS)
    def test_версия_загружается(self, version: str):
        prompt = load_prompt(version)

        assert prompt.version == version
        assert prompt.length > 0

    def test_версии_действительно_разные(self):
        """Иначе сравнение версий в M3 не имело бы смысла."""
        templates = {load_prompt(version).template for version in EXPECTED_VERSIONS}

        assert len(templates) == len(EXPECTED_VERSIONS)

    def test_несуществующая_версия_отбивается_со_списком(self):
        """Опечатка в .env не должна давать непонятный отказ."""
        with pytest.raises(ConfigurationError, match="Доступны: v1, v2, v3"):
            load_prompt("v42")


class TestПодстановка:
    @pytest.mark.parametrize("version", EXPECTED_VERSIONS)
    def test_состав_и_язык_подставляются(self, version: str):
        rendered = load_prompt(version).render("Sugar, Palm Oil", lang="de")

        assert "Sugar, Palm Oil" in rendered
        assert "{text}" not in rendered
        assert "de" in rendered

    def test_пустой_язык_становится_unknown(self):
        rendered = load_prompt("v1").render("Sugar", lang="")

        assert "unknown" in rendered

    @pytest.mark.parametrize("version", EXPECTED_VERSIONS)
    def test_промпт_на_английском(self, version: str):
        """Русский промпт заставлял модель переводить канонические имена,
        а их предстоит сопоставлять с таксономией OFF (теги вида en:sugar)."""
        template = load_prompt(version).template
        cyrillic = [ch for ch in template if "а" <= ch.lower() <= "я"]

        assert not cyrillic, f"в промпте {version} есть кириллица: {set(cyrillic)}"
