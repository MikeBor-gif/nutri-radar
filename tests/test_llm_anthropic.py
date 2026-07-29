"""Тесты адаптера Anthropic. В сеть тест не ходит — правило 4 брифа.

Клиент подменяется заглушкой целиком: адаптер обращается к нему ровно одним
вызовом `messages.create`, и подделать этот вызов надёжнее, чем поднимать
HTTP-мок поверх SDK. Заодно заглушка запоминает payload — а именно в payload
живут три решения, которые проще всего сломать незаметно:

* параметры сэмплирования не отправляются (поколение Sonnet 5 отвечает на них
  ошибкой 400);
* «мышление» гасится явно у моделей, где оно включено по умолчанию, — иначе
  оно делит бюджет `max_tokens` с ответом и обрезает список ингредиентов;
* `effort` не отправляется никогда, потому что у Haiku 4.5 такого параметра нет.

Развилка «ретраить или нет» проверяется отдельно и по тому же правилу, что
у Ollama (ADR-019): недоступность лечится паузой, непригодный ответ — нет.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import anthropic
import httpx
import pytest

from nutri_radar.config import AnthropicSettings
from nutri_radar.errors import ExtractionError, LLMUnavailableError
from nutri_radar.extract.schemas import ExtractionResult
from nutri_radar.llm.adapters.anthropic import AnthropicLLM, to_claude_schema

LIMIT = 2048

SONNET = "claude-sonnet-5"
HAIKU = "claude-haiku-4-5-20251001"

GOOD_CONTENT = json.dumps(
    {
        "ingredients": [{"canonical_name": "sugar", "kind": "sugar", "e_number": None}],
        "allergens": [],
        "unreadable": False,
        "model_confidence": 0.9,
    }
)

# Обрыв посреди массива — то же, что ловил ADR-019 на локальной модели.
TRUNCATED_CONTENT = '{\n  "ingredients": [\n    {"canonical_name": "almonds", "kind": "base"},\n {'


# =============================================================================
# Заглушки
# =============================================================================


@dataclass
class FakeUsage:
    input_tokens: int = 220
    output_tokens: int = 250


@dataclass
class FakeBlock:
    text: str
    type: str = "text"


@dataclass
class FakeMessage:
    content: list[FakeBlock]
    usage: FakeUsage = field(default_factory=FakeUsage)
    stop_reason: str = "end_turn"


class FakeMessages:
    """Подмена `client.messages`: запоминает payload и отдаёт заготовку."""

    def __init__(self, result: Any) -> None:
        self._result = result
        self.calls: list[dict[str, Any]] = []

    async def create(self, **payload: Any) -> Any:
        self.calls.append(payload)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class FakeClient:
    def __init__(self, result: Any) -> None:
        self.messages = FakeMessages(result)


def _with_client(result: Any, *, model: str = SONNET) -> tuple[AnthropicLLM, FakeClient]:
    """Адаптер и клиент отдельно: payload проверяется через клиент, а не через
    приватное поле адаптера."""
    settings = AnthropicSettings(api_key="test-key-not-real", model=model, max_output_tokens=LIMIT)
    client = FakeClient(result)
    return AnthropicLLM(client, settings), client  # type: ignore[arg-type]


def _adapter(result: Any, *, model: str = SONNET) -> AnthropicLLM:
    return _with_client(result, model=model)[0]


def _message(content: str, *, output_tokens: int = 250, stop_reason: str = "end_turn"):
    return FakeMessage(
        content=[FakeBlock(text=content)],
        usage=FakeUsage(output_tokens=output_tokens),
        stop_reason=stop_reason,
    )


def _http_error(status: int) -> anthropic.APIStatusError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status, request=request)
    error = anthropic.RateLimitError if status == 429 else anthropic.APIStatusError
    return error(f"ошибка {status} (подделка для теста)", response=response, body=None)


# =============================================================================
# Схема
# =============================================================================


class TestСхемаДляClaude:
    @pytest.fixture
    def schema(self) -> dict[str, Any]:
        return to_claude_schema(ExtractionResult.json_schema_for_llm())

    def test_у_объектов_запрещены_лишние_поля(self, schema: dict[str, Any]):
        """Без этого Claude отклоняет схему с ошибкой 400."""
        assert schema["additionalProperties"] is False
        assert schema["properties"]["ingredients"]["items"]["additionalProperties"] is False

    def test_числовые_ограничения_убраны(self, schema: dict[str, Any]):
        """`maximum` у `model_confidence` — ровно тот случай, что ломает запрос."""
        assert "maximum" not in schema["properties"]["model_confidence"]
        assert "minimum" not in schema["properties"]["model_confidence"]

    def test_диапазон_всё_равно_проверяется_после_разбора(self):
        """Удаление ограничений из схемы ничего не ослабляет — есть pydantic."""
        with pytest.raises(ValueError, match="model_confidence"):
            ExtractionResult.model_validate({"model_confidence": 1.5})

    def test_все_поля_объявлены_обязательными(self, schema: dict[str, Any]):
        assert set(schema["required"]) == set(schema["properties"])

    def test_вложенные_объекты_обрабатываются_рекурсивно(self):
        source = {
            "type": "object",
            "properties": {
                "outer": {
                    "type": "object",
                    "properties": {"inner": {"type": "integer", "maximum": 5, "default": 1}},
                }
            },
        }

        converted = to_claude_schema(source)
        inner = converted["properties"]["outer"]["properties"]["inner"]

        assert converted["properties"]["outer"]["additionalProperties"] is False
        assert "maximum" not in inner
        assert "default" not in inner

    def test_объект_без_свойств_не_получает_пустой_required(self):
        converted = to_claude_schema({"type": "object"})

        assert converted["additionalProperties"] is False
        assert "required" not in converted


# =============================================================================
# Payload запроса
# =============================================================================


@pytest.mark.asyncio
class TestPayload:
    async def test_температура_не_отправляется(self):
        """Поколение Sonnet 5 отвечает на любой параметр сэмплирования 400-й."""
        adapter, client = _with_client(_message(GOOD_CONTENT))

        await adapter.generate("состав", json_schema={"type": "object"})

        payload = client.messages.calls[0]
        assert "temperature" not in payload
        assert "top_p" not in payload
        assert "top_k" not in payload

    async def test_effort_не_отправляется_никогда(self):
        """У Haiku 4.5 такого параметра нет — он вернёт ошибку."""
        adapter, client = _with_client(_message(GOOD_CONTENT), model=HAIKU)

        await adapter.generate("состав", json_schema={"type": "object"})

        assert "effort" not in client.messages.calls[0]

    async def test_у_sonnet_мышление_гасится_явно(self):
        """Иначе оно делит бюджет max_tokens с ответом и обрежет список."""
        adapter, client = _with_client(_message(GOOD_CONTENT), model=SONNET)

        await adapter.generate("состав", json_schema={"type": "object"})

        assert client.messages.calls[0]["thinking"] == {"type": "disabled"}

    async def test_у_haiku_поля_thinking_нет(self):
        """Отсутствие поля уже означает «не думать»; лишнее поле — риск 400."""
        adapter, client = _with_client(_message(GOOD_CONTENT), model=HAIKU)

        await adapter.generate("состав", json_schema={"type": "object"})

        assert "thinking" not in client.messages.calls[0]

    async def test_схема_уходит_в_output_config(self):
        adapter, client = _with_client(_message(GOOD_CONTENT))

        await adapter.generate("состав", json_schema=ExtractionResult.json_schema_for_llm())

        fmt = client.messages.calls[0]["output_config"]["format"]
        assert fmt["type"] == "json_schema"
        assert fmt["schema"]["additionalProperties"] is False

    async def test_лимит_вывода_можно_переопределить_на_вызов(self):
        adapter, client = _with_client(_message(GOOD_CONTENT))

        await adapter.generate("состав", json_schema={"type": "object"}, max_output_tokens=99)

        assert client.messages.calls[0]["max_tokens"] == 99


# =============================================================================
# Ошибки: что ретраится, а что нет
# =============================================================================


@pytest.mark.asyncio
class TestОшибкиAPI:
    async def test_обрыв_соединения_это_недоступность(self):
        """Инфраструктура — её лечит пауза, поэтому ретраится."""
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        adapter = _adapter(anthropic.APIConnectionError(request=request))

        with pytest.raises(LLMUnavailableError, match="недоступна"):
            await adapter.generate("состав", json_schema={"type": "object"})

    async def test_лимит_запросов_это_недоступность(self):
        adapter = _adapter(_http_error(429))

        with pytest.raises(LLMUnavailableError):
            await adapter.generate("состав", json_schema={"type": "object"})

    async def test_ошибка_сервера_это_недоступность(self):
        adapter = _adapter(_http_error(503))

        with pytest.raises(LLMUnavailableError, match="503"):
            await adapter.generate("состав", json_schema={"type": "object"})

    async def test_отклонённый_запрос_не_ретраится(self):
        """400-я — это наш неверный запрос. Повтор даст ровно то же самое.

        И, главное, ошибка в схеме не должна выглядеть отказом сети: иначе
        прогон будет молча ретраить заведомо битый запрос до конца выборки.
        """
        adapter = _adapter(_http_error(400))

        with pytest.raises(ExtractionError, match="отклонила запрос"):
            await adapter.generate("состав", json_schema={"type": "object"})


# =============================================================================
# Разбор ответа
# =============================================================================


class TestРазборОтвета:
    def test_нормальный_ответ_разбирается(self):
        response = _adapter(None)._to_response(_message(GOOD_CONTENT), 1.0, LIMIT)

        assert response.raw_json["ingredients"][0]["canonical_name"] == "sugar"
        assert response.usage.input_tokens == 220
        assert response.truncated is False
        assert response.model_name == SONNET

    def test_отказ_классификатора_это_непригодный_ответ(self):
        """`stop_reason: refusal` приходит с кодом 200 и пустым содержимым."""
        message = FakeMessage(content=[], stop_reason="refusal")

        with pytest.raises(ExtractionError, match="refusal"):
            _adapter(None)._to_response(message, 1.0, LIMIT)

    def test_отказ_несёт_потраченные_токены(self):
        message = FakeMessage(content=[], stop_reason="refusal")

        with pytest.raises(ExtractionError) as info:
            _adapter(None)._to_response(message, 1.0, LIMIT)

        assert info.value.input_tokens == 220

    def test_обрыв_на_лимите_это_проблема_данных(self):
        """Повтор даст ту же обрезку — ретраить нечего."""
        message = _message(TRUNCATED_CONTENT, output_tokens=LIMIT, stop_reason="max_tokens")

        with pytest.raises(ExtractionError, match="лимите вывода"):
            _adapter(None)._to_response(message, 1.0, LIMIT)

    def test_обрыв_несёт_потраченные_токены(self):
        """Продукт пропущен, но генерация оплачена — она входит в стоимость."""
        message = _message(TRUNCATED_CONTENT, output_tokens=LIMIT, stop_reason="max_tokens")

        with pytest.raises(ExtractionError) as info:
            _adapter(None)._to_response(message, 1.0, LIMIT)

        assert info.value.output_tokens == LIMIT
        assert info.value.input_tokens == 220

    def test_мусор_без_лимита_это_отказ_модели(self):
        """Не-JSON при заданной схеме и без упора в лимит — похоже на сбой."""
        with pytest.raises(LLMUnavailableError, match="не-JSON"):
            _adapter(None)._to_response(_message("не json вовсе", output_tokens=17), 1.0, LIMIT)

    def test_stop_reason_max_tokens_помечает_обрезание(self):
        """JSON закрылся, но список ингредиентов может быть неполным."""
        message = _message(GOOD_CONTENT, output_tokens=100, stop_reason="max_tokens")

        response = _adapter(None)._to_response(message, 1.0, LIMIT)

        assert response.truncated is True

    def test_ответ_вплотную_к_лимиту_помечается_обрезанным(self):
        """Порог тот же, что у Ollama: 98% лимита — уже подозрительно."""
        message = _message(GOOD_CONTENT, output_tokens=LIMIT)

        response = _adapter(None)._to_response(message, 1.0, LIMIT)

        assert response.truncated is True

    def test_нетекстовые_блоки_не_попадают_в_json(self):
        """У моделей с мышлением в content приходят и другие типы блоков."""
        message = FakeMessage(
            content=[FakeBlock(text="это мысль", type="thinking"), FakeBlock(text=GOOD_CONTENT)]
        )

        response = _adapter(None)._to_response(message, 1.0, LIMIT)

        assert response.raw_json["ingredients"][0]["kind"] == "sugar"


class TestИмяМодели:
    def test_адаптер_называет_свою_модель(self):
        """Имя уходит в заголовок таблицы сравнения — оно часть результата."""
        assert _adapter(None, model=HAIKU).model_name == HAIKU
