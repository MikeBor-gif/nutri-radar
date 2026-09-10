"""Тесты бота: тексты, маршрутизация и приватность.

Три требования брифа проверяются здесь, а не вычиткой.

**Атрибуция ODbL и дисклеймер — в первом сообщении.** Не в подвале справки
и не в описании бота: человек, который принимает решение по составу, обязан
знать про происхождение и качество данных до решения.

**Оценочных ярлыков нет ни в одном тексте.** Проверка идёт по всем
константам модуля разом, включая те, что допишут завтра, — иначе тест
защищает только то, что помнил его автор.

**Пользовательские данные не попадают в лог.** Лог — это хранение, а бриф
запрещает хранить дольше, чем нужно для ответа.

Хендлеры вызываются напрямую с подставными объектами Telegram: сети нет
ни к Telegram, ни к Ollama, ни к базе (правило 4 брифа).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest
from aiogram.types import CallbackQuery, Chat, Message, Update, User

from nutri_radar.bot import texts
from nutri_radar.bot.handlers import ask as ask_handler
from nutri_radar.bot.handlers import barcode as barcode_handler
from nutri_radar.bot.handlers import commands as commands_handler
from nutri_radar.bot.keyboards import (
    ACTION_CARD,
    ACTION_INGREDIENTS,
    ACTION_SUGAR,
    back_keyboard,
    callback,
    card_keyboard,
    parse_callback,
)
from nutri_radar.bot.middlewares import PrivacyLoggingMiddleware, chat_fingerprint
from nutri_radar.config import Settings
from nutri_radar.errors import DatabaseError
from nutri_radar.retrieval.pipeline import AskResult
from nutri_radar.retrieval.product_card import CardSource, ProductCard
from nutri_radar.retrieval.rag import REFUSAL, RagAnswer
from nutri_radar.retrieval.search import SearchResult
from nutri_radar.wording import evaluative_labels_in

CODE = "4016463697295"


# =============================================================================
# Подставные объекты Telegram
# =============================================================================


@dataclass
class FakeMessage:
    """Сообщение, которое умеет ровно то, что использует хендлер."""

    text: str | None = None
    photo: list[Any] = field(default_factory=list)
    sent: list[str] = field(default_factory=list)
    edits: list[str] = field(default_factory=list)
    markups: list[Any] = field(default_factory=list)

    async def answer(self, text: str, **_kwargs: Any) -> FakeMessage:
        self.sent.append(text)
        # Заглушка возвращает себя же: хендлер правит её через edit_text,
        # и так весь диалог виден в одном объекте.
        return self

    async def edit_text(self, text: str, reply_markup: Any = None, **_kwargs: Any) -> None:
        self.edits.append(text)
        self.markups.append(reply_markup)

    @property
    def last(self) -> str:
        """Последнее, что увидел пользователь."""
        return (self.edits or self.sent)[-1]


@dataclass
class FakeCallbackQuery:
    """Нажатие кнопки."""

    data: str
    message: FakeMessage
    answers: list[tuple[str | None, bool]] = field(default_factory=list)

    async def answer(self, text: str | None = None, show_alert: bool = False) -> None:
        self.answers.append((text, show_alert))


def _card(**overrides: Any) -> ProductCard:
    defaults: dict[str, Any] = {
        "code": CODE,
        "source": CardSource.CORPUS,
        "product_name": "Молочный шоколад",
        "ingredients_text": "Сахар, какао тёртое",
        "distinct_sugar_forms": 3,
        "e_additives_count": 1,
        "ingredients_count": 8,
        "extraction_model": "qwen2.5:3b-instruct-q4_K_M",
        "extraction_prompt_version": "v3",
    }
    defaults.update(overrides)
    return ProductCard(**defaults)


# =============================================================================
# Тексты
# =============================================================================


class TestТребованияБрифа:
    async def test_первое_сообщение_несёт_атрибуцию_и_дисклеймер(self) -> None:
        """Прямое требование брифа: в первом сообщении, а не в подвале."""
        message = FakeMessage(text="/start")
        await commands_handler.start(message)

        assert len(message.sent) == 1
        first = message.sent[0]
        assert "Open Food Facts" in first
        assert "ODbL" in first
        assert "краудсорсинговые" in first

    def test_ни_в_одном_тексте_нет_оценочных_ярлыков(self) -> None:
        """Проверяются все константы модуля, включая будущие."""
        offenders = {
            text[:60]: evaluative_labels_in(text)
            for text in texts.all_texts()
            if evaluative_labels_in(text)
        }
        assert offenders == {}, f"Оценочные ярлыки в текстах бота: {offenders}"

    def test_текстов_проверяется_больше_одного(self) -> None:
        """Страховка от тихо сломавшегося сборщика текстов.

        Если `all_texts()` однажды начнёт возвращать пустой список,
        предыдущий тест станет зелёным и бесполезным.
        """
        assert len(texts.all_texts()) > 10

    def test_дисклеймер_объясняет_границу_продукта(self) -> None:
        assert "не медицинский советчик" in texts.START


# =============================================================================
# Маршрутизация
# =============================================================================


class TestМаршрутизация:
    async def test_цифры_уходят_в_карточку(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _load(*_args: Any, **_kwargs: Any) -> ProductCard:
            return _card()

        monkeypatch.setattr(barcode_handler, "load_card", _load)
        message = FakeMessage(text=f" {CODE} ")
        await barcode_handler.by_barcode(message, settings, client := object())  # type: ignore[arg-type]

        assert client is not None
        assert texts.SEARCHING in message.sent
        assert "Разных форм сахара в составе: 3" in message.last

    async def test_слова_уходят_в_вопрос(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _ask(question: str, **_kwargs: Any) -> AskResult:
            return AskResult(
                search=SearchResult(query=question),
                answer=RagAnswer(
                    question=question,
                    text=f"Есть такой продукт [{CODE}].",
                    sources=[CODE],
                ),
            )

        monkeypatch.setattr(ask_handler, "pipeline_ask", _ask)
        message = FakeMessage(text="какие есть конфеты без сахара")
        await ask_handler.ask(message, settings, object())  # type: ignore[arg-type]

        assert texts.THINKING in message.sent
        assert CODE in message.last

    async def test_слишком_короткий_ввод_не_идёт_в_модель(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Гнать «??» через эмбеддинг и генерацию — секунды GPU впустую."""
        called = False

        async def _ask(*_args: Any, **_kwargs: Any) -> AskResult:
            nonlocal called
            called = True
            raise AssertionError("не должно быть вызвано")

        monkeypatch.setattr(ask_handler, "pipeline_ask", _ask)
        message = FakeMessage(text="??")
        await ask_handler.ask(message, settings, object())  # type: ignore[arg-type]

        assert not called
        assert message.last == texts.NOT_A_BARCODE


class TestОтветы:
    async def test_ненайденный_штрихкод_объясняется_словами(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _none(*_args: Any, **_kwargs: Any) -> ProductCard | None:
            return None

        monkeypatch.setattr(barcode_handler, "load_card", _none)
        message = FakeMessage(text=CODE)
        await barcode_handler.by_barcode(message, settings, object())  # type: ignore[arg-type]

        assert CODE in message.last
        assert "не найден" in message.last

    async def test_недоступный_источник_не_выдаётся_за_отсутствие(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """«Не нашли» и «не смогли спросить» — разные факты.

        Сказать «продукт не найден», когда база лежит, значит подтолкнуть
        человека к выводу о продукте, которого делать нельзя.
        """

        async def _boom(*_args: Any, **_kwargs: Any) -> ProductCard:
            raise DatabaseError("база недоступна (подделка для теста)")

        monkeypatch.setattr(barcode_handler, "load_card", _boom)
        message = FakeMessage(text=CODE)
        await barcode_handler.by_barcode(message, settings, object())  # type: ignore[arg-type]

        assert message.last == texts.SOURCE_UNAVAILABLE
        assert "не найден" not in message.last

    async def test_отказ_модели_доезжает_отказом(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _ask(question: str, **_kwargs: Any) -> AskResult:
            return AskResult(
                search=SearchResult(query=question),
                answer=RagAnswer(question=question, text=REFUSAL, refused=True),
            )

        monkeypatch.setattr(ask_handler, "pipeline_ask", _ask)
        message = FakeMessage(text="чего в базе точно нет")
        await ask_handler.ask(message, settings, object())  # type: ignore[arg-type]

        assert message.last == REFUSAL
        # К отказу не приписывается предложение прислать штрихкод: ссылаться
        # не на что, и подсказка выглядела бы как намёк, что ответ есть.
        assert "Пришлите любой штрихкод" not in message.last


# =============================================================================
# Клавиатура и кнопки
# =============================================================================


class TestКнопки:
    def test_штрихкод_помещается_в_callback_data(self) -> None:
        """Предел Telegram — 64 байта. Состояние диалога нигде не хранится."""
        for action in (ACTION_CARD, ACTION_INGREDIENTS, ACTION_SUGAR):
            data = callback(action, "12345678901234")
            assert len(data.encode()) <= 64

    def test_разбор_возвращает_действие_и_код(self) -> None:
        assert parse_callback(callback(ACTION_SUGAR, CODE)) == (ACTION_SUGAR, CODE)

    def test_чужая_кнопка_не_роняет_разбор(self) -> None:
        """В чат прилетают кнопки от прошлой версии бота."""
        assert parse_callback("мусор-без-разделителя") is None
        assert parse_callback("") is None

    def test_кнопка_форм_сахара_есть_и_без_разбора(self) -> None:
        """Иначе пользователь гадает, почему у одних продуктов её нет."""
        markup = card_keyboard(CODE, has_extraction=False)
        actions = [
            parse_callback(button.callback_data or "")
            for row in markup.inline_keyboard
            for button in row
        ]
        assert (ACTION_SUGAR, CODE) in actions

    def test_возврат_ведёт_к_той_же_карточке(self) -> None:
        markup = back_keyboard(CODE)
        button = markup.inline_keyboard[0][0]
        assert parse_callback(button.callback_data or "") == (ACTION_CARD, CODE)

    async def test_нажатие_правит_то_же_сообщение(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Кнопки не должны сыпать новыми сообщениями: карточка одна."""

        async def _load(*_args: Any, **_kwargs: Any) -> ProductCard:
            return _card()

        monkeypatch.setattr(barcode_handler, "load_card", _load)
        message = FakeMessage()
        query = FakeCallbackQuery(data=callback(ACTION_INGREDIENTS, CODE), message=message)

        await barcode_handler.card_button(query, settings, object())  # type: ignore[arg-type]

        assert message.sent == []
        assert message.edits == ["Сахар, какао тёртое"]

    async def test_формы_сахара_без_разбора_объясняются(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Не «0 форм сахара», а «состав не разбирался»."""

        async def _load(*_args: Any, **_kwargs: Any) -> ProductCard:
            return _card(
                distinct_sugar_forms=None,
                e_additives_count=None,
                ingredients_count=None,
                extraction_model=None,
            )

        monkeypatch.setattr(barcode_handler, "load_card", _load)
        message = FakeMessage()
        query = FakeCallbackQuery(data=callback(ACTION_SUGAR, CODE), message=message)

        await barcode_handler.card_button(query, settings, object())  # type: ignore[arg-type]

        assert message.edits[-1] == texts.NO_EXTRACTION
        assert "0" not in message.edits[-1].split("\n")[0]


# =============================================================================
# Приватность
# =============================================================================


class TestПриватность:
    """Проверяется на настоящих объектах aiogram, а не на подделках.

    Middleware висит на уровне `update`, и туда приходит `Update`, а не
    `Message`. Подделка, притворяющаяся сообщением, прошла бы мимо этой
    разницы и подтвердила бы работу кода, который в бою деградирует
    до «kind: Update».
    """

    @staticmethod
    def _update(text: str = "привет", chat_id: int = 555444333) -> Update:
        return Update(
            update_id=1,
            message=Message(
                message_id=1,
                date=datetime(2026, 9, 9, tzinfo=UTC),
                chat=Chat(id=chat_id, type="private"),
                text=text,
            ),
        )

    def test_отпечаток_чата_не_равен_идентификатору(self) -> None:
        fingerprint = chat_fingerprint(123456789)

        assert "123456789" not in fingerprint
        assert fingerprint == chat_fingerprint(123456789), "Отпечаток должен быть устойчивым"
        assert fingerprint != chat_fingerprint(987654321)

    async def test_текст_сообщения_не_попадает_в_лог(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Лог — это хранение, а бриф запрещает хранить дольше ответа."""
        secret = "мой диагноз и что мне можно есть"

        async def handler(_event: Any, _data: dict[str, Any]) -> str:
            return "ok"

        middleware = PrivacyLoggingMiddleware()
        with caplog.at_level(logging.DEBUG):
            await middleware(handler, self._update(secret), {"event_chat": None})

        recorded = "\n".join(str(record.__dict__) for record in caplog.records)
        assert secret not in recorded
        # Длина при этом пишется: она объясняет латентность и не раскрывает
        # содержания.
        assert any(getattr(record, "chars", None) == len(secret) for record in caplog.records)

    async def test_идентификатор_чата_не_попадает_в_лог(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        chat = Chat(id=555444333, type="private")

        async def handler(_event: Any, _data: dict[str, Any]) -> str:
            return "ok"

        middleware = PrivacyLoggingMiddleware()
        with caplog.at_level(logging.DEBUG):
            await middleware(handler, self._update(), {"event_chat": chat})

        recorded = "\n".join(str(record.__dict__) for record in caplog.records)
        assert "555444333" not in recorded
        assert chat_fingerprint(555444333) in recorded

    async def test_падение_хендлера_логируется_без_содержания(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        secret = "очень личный вопрос про здоровье"

        async def handler(_event: Any, _data: dict[str, Any]) -> str:
            raise ValueError("что-то пошло не так")

        middleware = PrivacyLoggingMiddleware()
        with caplog.at_level(logging.DEBUG), pytest.raises(ValueError):
            await middleware(handler, self._update(secret), {"event_chat": None})

        recorded = "\n".join(str(record.__dict__) for record in caplog.records)
        assert secret not in recorded
        # Тип ошибки при этом виден: иначе разбор падения невозможен.
        assert "ValueError" in recorded

    async def test_команда_пишется_в_лог_целиком(self, caplog: pytest.LogCaptureFixture) -> None:
        """Множество команд известно заранее и о человеке ничего не говорит."""

        async def handler(_event: Any, _data: dict[str, Any]) -> str:
            return "ok"

        middleware = PrivacyLoggingMiddleware()
        with caplog.at_level(logging.DEBUG):
            await middleware(handler, self._update("/start"), {"event_chat": None})

        assert any(getattr(record, "kind", "") == "command:/start" for record in caplog.records)

    async def test_нажатие_кнопки_описывается_действием_без_кода(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Действие — наш формат, штрихкод — то, что смотрел человек."""
        update = Update(
            update_id=2,
            callback_query=CallbackQuery(
                id="1",
                from_user=User(id=1, is_bot=False, first_name="кто-то"),
                chat_instance="1",
                data=callback(ACTION_SUGAR, CODE),
            ),
        )

        async def handler(_event: Any, _data: dict[str, Any]) -> str:
            return "ok"

        middleware = PrivacyLoggingMiddleware()
        with caplog.at_level(logging.DEBUG):
            await middleware(handler, update, {"event_chat": None})

        recorded = "\n".join(str(record.__dict__) for record in caplog.records)
        assert any(getattr(r, "kind", "") == f"callback:{ACTION_SUGAR}" for r in caplog.records)
        assert CODE not in recorded


class TestПодсказкаПроШтрихкоды:
    """Приписка «пришлите любой штрихкод из ответа» — только когда они есть.

    Поймано на живом прогоне бота: модель ответила рассуждением, не сославшись
    ни на один штрихкод, а приписка всё равно пришла и отправила человека
    искать в тексте то, чего там нет. Условие стояло по `sources` — выдаче,
    отданной модели, — а не по `cited`, тому, что она реально процитировала.
    """

    @staticmethod
    def _answer(text: str, sources: list[str]) -> RagAnswer:
        return RagAnswer(question="вопрос", text=text, sources=sources)

    async def _ask(
        self,
        settings: Settings,
        monkeypatch: pytest.MonkeyPatch,
        answer: RagAnswer,
    ) -> FakeMessage:
        async def _pipeline(question: str, **_kwargs: Any) -> AskResult:
            return AskResult(search=SearchResult(query=question), answer=answer)

        monkeypatch.setattr(ask_handler, "pipeline_ask", _pipeline)
        message = FakeMessage(text="какой-нибудь вопрос про состав")
        await ask_handler.ask(message, settings, object())  # type: ignore[arg-type]
        return message

    async def test_есть_ссылки_есть_подсказка(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        answer = self._answer(f"Смотрите [{CODE}].", [CODE])
        message = await self._ask(settings, monkeypatch, answer)

        assert "Пришлите любой штрихкод" in message.last

    async def test_ссылок_нет_подсказки_нет(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Выдача непустая, но модель не процитировала ничего.
        answer = self._answer("Такого в предоставленных продуктах нет.", [CODE])
        message = await self._ask(settings, monkeypatch, answer)

        assert message.last == "Такого в предоставленных продуктах нет."
        assert "Пришлите любой штрихкод" not in message.last
