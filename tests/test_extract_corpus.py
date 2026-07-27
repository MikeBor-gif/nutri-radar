"""Тесты отбора LLM-корпуса. Без БД — запрос к Postgres подменяется.

Главное свойство здесь — **детерминированность**. Если выборка меняется между
запусками, сравнение версий промптов в M3 пойдёт по разным продуктам и
перестанет что-либо означать.

Сам SQL детерминирован по построению: порядок задан `md5(code || seed)`,
а не `random()`. Здесь проверяется то, что живёт в Python: квоты по языкам,
пропорция частей выборки, устойчивый порядок и предупреждение о недоборе.
"""

from __future__ import annotations

import logging

import pytest

from nutri_radar.config import ExtractSettings, Settings
from nutri_radar.extract import corpus as corpus_module
from nutri_radar.extract.corpus import (
    CorpusItem,
    _language_quotas,
    collect_stats,
    sample_for_benchmark,
    select_llm_corpus,
)


@pytest.fixture
def corpus_settings(settings: Settings) -> Settings:
    """Языки из фикстуры — en и ru; доля unknown — 70% по ADR-006."""
    return settings.model_copy(
        update={"extract": ExtractSettings(corpus_size=100, unknown_share=0.7, random_seed=42)}
    )


def _fake_fetch(available: dict[tuple[str, bool], int]):
    """Подделка запроса к БД: отдаёт столько строк, сколько есть «в базе»."""

    async def fetch(settings, *, lang, limit, unknown, seed) -> list[CorpusItem]:
        supply = available.get((lang, unknown), limit)
        count = min(limit, supply)
        stratum = "unknown" if unknown else "control"
        return [
            CorpusItem(
                code=f"{lang}-{stratum}-{index:04d}",
                ingredients_text="Sugar, Salt",
                lang=lang,
                nutriscore_grade="c",
                unknown_ingredients_n=1 if unknown else 0,
                stratum=stratum,
            )
            for index in range(count)
        ]

    return fetch


class TestКвотыПоЯзыкам:
    def test_квоты_равные_а_не_пропорциональные(self):
        """Пропорциональные воспроизвели бы перекос базы: ru в 65 раз меньше fr."""
        quotas = _language_quotas(["en", "ru", "de", "fr", "pl"], 100)

        assert set(quotas.values()) == {20}

    def test_остаток_добирается_первыми_языками(self):
        quotas = _language_quotas(["en", "ru", "de"], 10)

        assert quotas == {"en": 4, "ru": 3, "de": 3}
        assert sum(quotas.values()) == 10

    def test_пустой_список_языков_даёт_пустые_квоты(self):
        assert _language_quotas([], 100) == {}


class TestОтборКорпуса:
    async def test_пропорция_частей_выборки(self, corpus_settings: Settings, monkeypatch):
        monkeypatch.setattr(corpus_module, "_fetch_stratum", _fake_fetch({}))

        items = await select_llm_corpus(corpus_settings)
        stats = collect_stats(items)

        assert stats.total == 100
        assert stats.by_stratum["unknown"] == 70
        assert stats.by_stratum["control"] == 30
        assert stats.unknown_share == 0.7

    async def test_языки_представлены_поровну(self, corpus_settings: Settings, monkeypatch):
        monkeypatch.setattr(corpus_module, "_fetch_stratum", _fake_fetch({}))

        stats = collect_stats(await select_llm_corpus(corpus_settings))

        assert stats.by_lang == {"en": 50, "ru": 50}

    async def test_повторный_вызов_даёт_тот_же_набор(self, corpus_settings: Settings, monkeypatch):
        """Ключевое свойство: иначе версии промптов сравнивать не с чем."""
        monkeypatch.setattr(corpus_module, "_fetch_stratum", _fake_fetch({}))

        first = await select_llm_corpus(corpus_settings)
        second = await select_llm_corpus(corpus_settings)

        assert [item.code for item in first] == [item.code for item in second]

    async def test_порядок_не_зависит_от_обхода_языков(
        self, corpus_settings: Settings, monkeypatch
    ):
        monkeypatch.setattr(corpus_module, "_fetch_stratum", _fake_fetch({}))

        codes = [item.code for item in await select_llm_corpus(corpus_settings)]

        assert codes == sorted(codes)

    async def test_размер_можно_переопределить_аргументом(
        self, corpus_settings: Settings, monkeypatch
    ):
        monkeypatch.setattr(corpus_module, "_fetch_stratum", _fake_fetch({}))

        items = await select_llm_corpus(corpus_settings, size=20)

        assert len(items) == 20


class TestВыборкаДляЗамера:
    """Дефект найден на живом замере: первые 20 по коду — все англоязычные."""

    async def test_в_замер_попадают_все_языки(self, corpus_settings: Settings, monkeypatch):
        monkeypatch.setattr(corpus_module, "_fetch_stratum", _fake_fetch({}))
        items = await select_llm_corpus(corpus_settings)

        sample = sample_for_benchmark(items, 20)

        assert {item.lang for item in sample} == {"en", "ru"}

    async def test_обе_части_выборки_представлены(self, corpus_settings: Settings, monkeypatch):
        monkeypatch.setattr(corpus_module, "_fetch_stratum", _fake_fetch({}))
        items = await select_llm_corpus(corpus_settings)

        sample = sample_for_benchmark(items, 20)

        assert {item.stratum for item in sample} == {"unknown", "control"}

    async def test_первые_по_коду_дали_бы_одну_группу(self, corpus_settings: Settings, monkeypatch):
        """Тест на сам дефект: срез по порядку не представителен."""
        monkeypatch.setattr(corpus_module, "_fetch_stratum", _fake_fetch({}))
        items = await select_llm_corpus(corpus_settings)

        naive = items[:20]

        assert len({item.lang for item in naive}) == 1

    async def test_выборка_детерминирована(self, corpus_settings: Settings, monkeypatch):
        monkeypatch.setattr(corpus_module, "_fetch_stratum", _fake_fetch({}))
        items = await select_llm_corpus(corpus_settings)

        first = [item.code for item in sample_for_benchmark(items, 20)]
        second = [item.code for item in sample_for_benchmark(items, 20)]

        assert first == second

    async def test_размер_соблюдается(self, corpus_settings: Settings, monkeypatch):
        monkeypatch.setattr(corpus_module, "_fetch_stratum", _fake_fetch({}))
        items = await select_llm_corpus(corpus_settings)

        assert len(sample_for_benchmark(items, 7)) == 7

    def test_запрошено_больше_чем_есть(self):
        items = [
            CorpusItem(
                code="1",
                ingredients_text="Sugar",
                lang="en",
                nutriscore_grade=None,
                unknown_ingredients_n=0,
                stratum="control",
            )
        ]

        assert len(sample_for_benchmark(items, 20)) == 1
        assert sample_for_benchmark([], 20) == []
        assert sample_for_benchmark(items, 0) == []


class TestНедоборКвоты:
    async def test_недобор_предупреждает_а_не_падает(
        self, corpus_settings: Settings, monkeypatch, caplog
    ):
        """Русских составов в базе мало — это характеристика данных.

        Но знать о недоборе обязательно: он прямо ослабляет выводы M3
        по этим языкам.
        """
        # В базе всего 5 русских продуктов со сложным составом.
        monkeypatch.setattr(corpus_module, "_fetch_stratum", _fake_fetch({("ru", True): 5}))

        with caplog.at_level(logging.WARNING, logger="nutri_radar.extract.corpus"):
            items = await select_llm_corpus(corpus_settings)

        assert len(items) == 100 - (35 - 5)
        assert any("Квоты по языкам" in record.message for record in caplog.records)

    async def test_пустая_база_логирует_ошибку(
        self, corpus_settings: Settings, monkeypatch, caplog
    ):
        empty = {(lang, unknown): 0 for lang in ("en", "ru") for unknown in (True, False)}
        monkeypatch.setattr(corpus_module, "_fetch_stratum", _fake_fetch(empty))

        with caplog.at_level(logging.ERROR, logger="nutri_radar.extract.corpus"):
            items = await select_llm_corpus(corpus_settings)

        assert items == []
        assert any("Корпус пуст" in record.message for record in caplog.records)
