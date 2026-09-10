"""Тесты карточки продукта.

Три проверяемых свойства.

**Число разных форм сахара показывается там, где оно есть, и не выдумывается
там, где его нет.** Это главная величина проекта, и она существует только
для продуктов, прошедших через извлечение — 1187 из 147 тысяч корпуса.
Показать ноль вместо неизвестности значило бы соврать числом: читатель
прочтёт «сахара не нашли», хотя состав никто не разбирал.

**Формулировки описательные.** Ни одного оценочного ярлыка в тексте, который
проект пишет от себя. Требование брифа обязано ломать тест, а не ждать
вычитки — поэтому проверка автоматическая, против `nutri_radar.wording`.

**Фолбэк в живой API включается только при промахе по корпусу.** Иначе
проект начал бы дёргать OFF на каждый запрос, а разработчики OFF просят
этого не делать.

Ни сети, ни БД: репозитории и HTTP подставные (правило 4 брифа).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest

from nutri_radar.config import Settings
from nutri_radar.db.repositories.extraction import ExtractionRow
from nutri_radar.db.repositories.product import ProductSummary
from nutri_radar.errors import DataSourceError
from nutri_radar.retrieval import product_card
from nutri_radar.retrieval.product_card import (
    CardSource,
    ProductCard,
    card_from_corpus,
    card_from_off,
    format_card,
    load_card,
)
from nutri_radar.wording import evaluative_labels_in

CODE = "3017620425035"


def _product(**overrides: object) -> ProductSummary:
    defaults: dict[str, object] = {
        "code": CODE,
        "product_name": "Молочный шоколад",
        "brands": "Alpen Gold, Mondelez",
        "ingredients_text": "Сахар, какао тёртое, сироп глюкозы, натуральный ароматизатор",
        "ingredients_text_lang": "ru",
        "nutriscore_grade": "e",
        "nova_group": 4,
        "allergens_tags": ["en:milk"],
    }
    defaults.update(overrides)
    return ProductSummary(**defaults)  # type: ignore[arg-type]


def _extraction(**overrides: object) -> ExtractionRow:
    defaults: dict[str, object] = {
        "code": CODE,
        "distinct_sugar_forms": 3,
        "e_additives_count": 1,
        "ingredients_count": 8,
        "allergens": ["молоко"],
        "model_name": "qwen2.5:3b-instruct-q4_K_M",
        "prompt_version": "v3",
    }
    defaults.update(overrides)
    return ExtractionRow(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def patched_db(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Подменить доступ к БД. Что вернуть — задаёт сам тест."""
    state: dict[str, object] = {"product": None, "extraction": None, "codes": []}

    @asynccontextmanager
    async def fake_session(*_args: object, **_kwargs: object) -> AsyncIterator[object]:
        yield object()

    class FakeProductRepository:
        def __init__(self, _session: object) -> None: ...

        async def get_by_code(self, code: str) -> ProductSummary | None:
            state["codes"].append(code)  # type: ignore[union-attr]
            return state["product"]  # type: ignore[return-value]

    class FakeExtractionRepository:
        def __init__(self, _session: object) -> None: ...

        async def latest_by_code(self, _code: str) -> ExtractionRow | None:
            return state["extraction"]  # type: ignore[return-value]

    monkeypatch.setattr(product_card, "get_session", fake_session)
    monkeypatch.setattr(product_card, "ProductRepository", FakeProductRepository)
    monkeypatch.setattr(product_card, "ExtractionRepository", FakeExtractionRepository)
    return state


def _off_client(
    payload: dict[str, object],
    *,
    calls: list[str] | None = None,
    status: int = 200,
) -> httpx.AsyncClient:
    """Подставной Open Food Facts.

    Код ответа задаётся явно и по умолчанию 200 — но именно умолчание
    однажды и соврало: для несуществующего продукта живой OFF отдаёт **404**
    с телом «product not found», а тест мокал 200 со `status: 0`. Из-за
    расхождения фикстуры с реальностью самый частый случай — продукта нет —
    доезжал до пользователя как «источник недоступен».
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(str(request.url))
        return httpx.Response(status, content=json.dumps(payload).encode())

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TestСборкаКарточки:
    def test_формы_сахара_берутся_из_разбора(self) -> None:
        card = card_from_corpus(_product(), _extraction())

        assert card.source is CardSource.CORPUS
        assert card.distinct_sugar_forms == 3
        assert card.e_additives_count == 1
        assert card.has_extraction

    def test_без_разбора_формы_сахара_неизвестны_а_не_равны_нулю(self) -> None:
        card = card_from_corpus(_product(), None)

        # Ноль означал бы «сахара не нашли». Разницу между «не считали»
        # и «посчитали и не нашли» стирать нельзя: первое честно, второе — ложь.
        assert card.distinct_sugar_forms is None
        assert not card.has_extraction

    def test_аллергены_из_разбора_важнее_тегов_парсера(self) -> None:
        card = card_from_corpus(_product(), _extraction())
        assert card.allergens == ["молоко"]

    def test_без_разбора_остаются_теги_парсера(self) -> None:
        card = card_from_corpus(_product(), None)
        assert card.allergens == ["en:milk"]

    def test_карточка_из_живого_апи_не_знает_форм_сахара(self) -> None:
        card = card_from_off(
            product_card.OffProduct(code=CODE, product_name="Nutella", nova_group=4)
        )

        assert card.source is CardSource.OPENFOODFACTS
        assert card.distinct_sugar_forms is None
        assert card.extraction_model is None


class TestФормулировки:
    def test_в_карточке_нет_оценочных_ярлыков(self) -> None:
        card = card_from_corpus(_product(ingredients_text="Сахар, какао тёртое"), _extraction())
        text = format_card(card)

        assert evaluative_labels_in(text) == [], f"Оценочный ярлык в карточке: {text}"

    def test_состав_из_источника_под_запрет_не_попадает(self) -> None:
        """«Натуральный ароматизатор» — цитата с этикетки, а не наша оценка.

        Вырезать её значило бы искажать данные источника. Проверка касается
        только текста, который проект пишет от себя, и этот тест фиксирует
        границу явно — иначе следующий разработчик «починит» её не в ту сторону.
        """
        card = card_from_corpus(_product(), _extraction())
        text = format_card(card)

        assert "натуральный ароматизатор" in text
        assert evaluative_labels_in(text.replace(card.ingredients_text or "", "")) == []

    def test_число_форм_сахара_названо_числом(self) -> None:
        text = format_card(card_from_corpus(_product(), _extraction()))
        assert "Разных форм сахара в составе: 3" in text

    def test_отсутствие_разбора_названо_прямо(self) -> None:
        text = format_card(card_from_corpus(_product(), None))
        assert "не разбирался" in text
        assert "форм сахара: 0" not in text.lower()

    def test_источник_указан_всегда(self) -> None:
        corpus = format_card(card_from_corpus(_product(), _extraction()))
        remote = format_card(card_from_off(product_card.OffProduct(code=CODE)))

        assert "из базы проекта" in corpus
        assert "в корпус проекта не входит" in remote

    def test_модель_разбора_названа_рядом_с_числом(self) -> None:
        """Число форм сахара без имени модели — число неизвестно чьего разбора."""
        text = format_card(card_from_corpus(_product(), _extraction()))
        assert "qwen2.5:3b-instruct-q4_K_M" in text
        assert "v3" in text


class TestЗагрузка:
    async def test_продукт_из_корпуса_не_идёт_в_живой_апи(
        self, settings: Settings, patched_db: dict[str, object]
    ) -> None:
        patched_db["product"] = _product()
        patched_db["extraction"] = _extraction()
        calls: list[str] = []

        card = await load_card(
            CODE,
            settings=settings,
            client=_off_client({"status": 1, "product": {}}, calls=calls),
        )

        assert card is not None
        assert card.source is CardSource.CORPUS
        assert calls == [], "Корпус ответил — в сеть ходить незачем"

    async def test_промах_по_корпусу_уходит_в_живой_апи(
        self, settings: Settings, patched_db: dict[str, object]
    ) -> None:
        patched_db["product"] = None
        calls: list[str] = []
        payload = {"status": 1, "product": {"product_name": "Nutella", "nova_group": 4}}

        card = await load_card(CODE, settings=settings, client=_off_client(payload, calls=calls))

        assert card is not None
        assert card.source is CardSource.OPENFOODFACTS
        assert card.product_name == "Nutella"
        assert len(calls) == 1

    async def test_фолбэк_можно_выключить(
        self, settings: Settings, patched_db: dict[str, object]
    ) -> None:
        patched_db["product"] = None
        calls: list[str] = []

        card = await load_card(
            CODE,
            settings=settings,
            client=_off_client({"status": 1, "product": {}}, calls=calls),
            allow_remote=False,
        )

        assert card is None
        assert calls == []

    async def test_неизвестный_штрихкод_даёт_none(
        self, settings: Settings, patched_db: dict[str, object]
    ) -> None:
        """Форма ответа взята с живого API, а не придумана.

        OFF отвечает 404 и телом «product not found». Раньше здесь стоял
        200 со `status: 0`, и тест пропускал баг: `raise_for_status()`
        превращал 404 в отказ источника, и человек со штрихкодом,
        которого нет в базе OFF, получал «попробуйте позже».
        """
        patched_db["product"] = None

        card = await load_card(
            CODE,
            settings=settings,
            client=_off_client(
                {"code": CODE, "status": 0, "status_verbose": "product not found"},
                status=404,
            ),
        )

        assert card is None

    async def test_двухсотка_со_статусом_ноль_тоже_означает_отсутствие(
        self, settings: Settings, patched_db: dict[str, object]
    ) -> None:
        """Вторая форма того же ответа: OFF отдаёт и её."""
        patched_db["product"] = None

        card = await load_card(
            CODE, settings=settings, client=_off_client({"status": 0, "product": {}})
        )

        assert card is None

    async def test_ошибка_сервера_не_выдаётся_за_отсутствие(
        self, settings: Settings, patched_db: dict[str, object]
    ) -> None:
        """500 — это отказ источника, и путать его с 404 нельзя."""
        patched_db["product"] = None

        with pytest.raises(DataSourceError, match="500"):
            await load_card(CODE, settings=settings, client=_off_client({}, status=500))

    async def test_не_штрихкод_отсекается_до_базы(
        self, settings: Settings, patched_db: dict[str, object]
    ) -> None:
        card = await load_card("шоколад", settings=settings)

        assert card is None
        assert patched_db["codes"] == [], "В базу за названием ходить незачем"

    async def test_недоступный_апи_не_притворяется_отсутствием(
        self, settings: Settings, patched_db: dict[str, object]
    ) -> None:
        """«Не нашли» и «не смогли спросить» — разные факты.

        Вернуть `None` при отказе сети значило бы сказать пользователю,
        что продукта не существует, хотя мы просто не дозвонились.
        """
        patched_db["product"] = None

        def handler(_request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("OFF недоступен (подделка для теста)")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        with pytest.raises(DataSourceError):
            await load_card(CODE, settings=settings, client=client)


class TestРазметка:
    def test_html_снимается_из_состава(self) -> None:
        """Треть корпуса приходит с `<span class="allergen">`.

        Поймано на живом боте: карточка показывала теги пользователю как
        есть. Слово-аллерген при этом обязано остаться — теряется только
        обёртка вокруг него.
        """
        card = card_from_corpus(
            _product(ingredients_text='<span class="allergen">Oats</span> (69%), Sugar'),
            _extraction(),
        )

        assert card.ingredients_text is not None
        assert "<span" not in card.ingredients_text
        assert "</span>" not in card.ingredients_text
        assert "Oats" in card.ingredients_text
        assert "Sugar" in card.ingredients_text

    def test_html_снимается_и_у_карточки_из_живого_апи(self) -> None:
        card = card_from_off(
            product_card.OffProduct(code=CODE, ingredients_text="Sucre, <b>NOISETTES</b> 13%")
        )

        assert card.ingredients_text == "Sucre, NOISETTES 13%"


class TestЗаголовок:
    def test_продукт_без_названия_не_ломает_карточку(self) -> None:
        card = ProductCard(code=CODE, source=CardSource.CORPUS)
        assert card.title == "без названия"
        assert format_card(card).startswith(f"[{CODE}] без названия")
