"""Адаптер облачной модели Anthropic. Реализация порта `StructuredLLM`.

Нужен на M3 как эталон в evals: локальная 3B сравнивается с облачной, иначе
непонятно, чей предел мы измеряем — модели или промпта.

Три вещи, проверенные по актуальному контракту API, а не по памяти:

1. **Структурированный вывод — это `output_config.format` с JSON-схемой**,
   прямой аналог параметра `format` у Ollama. Порт ложится на оба провайдера
   без изменений.
2. **Схему приходится адаптировать.** Claude требует `additionalProperties:
   false` у каждого объекта и не принимает числовые ограничения (`maximum`
   у `model_confidence`). Схема, годная для Ollama, тут вернёт ошибку.
3. **Модели ведут себя по-разному, и делать вид, что они одинаковые, нельзя.**
   У поколения Sonnet 5 параметры сэмплирования отклоняются с ошибкой 400,
   а «мышление» включено по умолчанию и делит бюджет `max_tokens` с ответом —
   для извлечения состава его гасим явно. У Haiku 4.5 наоборот: параметра
   `effort` нет вовсе, а отсутствие поля `thinking` уже означает «не думать».
   У семейства Fable третий случай: мышление не выключается совсем, и явный
   `thinking: disabled` там вернёт 400 — поле не отправляется. Одним флагом
   «умеет думать» эти три случая не описываются, поэтому в коде два списка.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import anthropic

from nutri_radar.config import AnthropicSettings
from nutri_radar.errors import ConfigurationError, ExtractionError, LLMUnavailableError
from nutri_radar.llm.models import LLMResponse, TokenUsage
from nutri_radar.logging import safe_extra

logger = logging.getLogger(__name__)

# Доля от лимита вывода, начиная с которой ответ считается подозрительно
# близким к обрезанию. Тот же порог и тот же смысл, что у адаптера Ollama.
_TRUNCATION_RATIO = 0.98

# Ключевые слова JSON-схемы, которые Claude не принимает: числовые и строковые
# ограничения. `model_confidence` с `maximum: 1.0` — ровно этот случай.
# Проверку диапазона всё равно делает pydantic после разбора, так что удаление
# ничего не ослабляет.
_UNSUPPORTED_SCHEMA_KEYS = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "uniqueItems",
        "pattern",
        "default",
    }
)

# Поведение «мышления» у Claude делится на три случая, и это факт внешнего
# API, а не настройка проекта. Держать их одним списком нельзя: параметр,
# уместный в одном случае, в другом возвращает 400.
#
# 1. Гасится явно — модели ниже. Мышление у них включено по умолчанию и делит
#    бюджет `max_tokens` с ответом, поэтому для извлечения состава мы шлём
#    `thinking: disabled` (ADR-019: обрезанный JSON — та беда, которую чиним).
_THINKING_DISABLED_MODELS = (
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
)

# 2. Не выключается вовсе. У семейства Fable мышление всегда включено, и явный
#    `thinking: disabled` возвращает 400 — параметр нужно не передавать.
#    Бюджет `max_tokens` эти модели делят между мышлением и ответом, так что
#    лимит вывода им нужен с запасом; предупреждаем об этом при создании.
_ALWAYS_THINKING_MODELS = (
    "claude-fable-5",
    "claude-mythos-5",
)

# 3. Мышления нет (Haiku 4.5 и старше). Отсутствие поля `thinking` уже
#    означает «не думать», а параметра `effort` у них нет вовсе — он вернёт
#    ошибку. Отдельного списка не требуется: это поведение по умолчанию.


class AnthropicLLM:
    """Облачная модель. Реализация порта `StructuredLLM`."""

    def __init__(self, client: anthropic.AsyncAnthropic, settings: AnthropicSettings) -> None:
        self._client = client
        self._settings = settings
        logger.info(
            "Адаптер Anthropic создан",
            extra={
                "model": settings.model,
                "max_output_tokens": settings.max_output_tokens,
                "timeout_s": settings.timeout_s,
                "thinking_disabled": self._thinking_disabled,
            },
        )
        if self._always_thinking:
            # Не ошибка конфигурации, а предупреждение: модель рабочая, но
            # погасить мышление у неё нельзя, и лимит вывода делится на двоих.
            logger.warning(
                "У модели мышление не выключается — лимит вывода делится с ответом",
                extra=safe_extra(
                    model=settings.model, max_output_tokens=settings.max_output_tokens
                ),
            )

    @property
    def model_name(self) -> str:
        return self._settings.model

    @property
    def _thinking_disabled(self) -> bool:
        return self._settings.model.startswith(_THINKING_DISABLED_MODELS)

    @property
    def _always_thinking(self) -> bool:
        return self._settings.model.startswith(_ALWAYS_THINKING_MODELS)

    async def generate(
        self,
        prompt: str,
        *,
        json_schema: dict[str, Any],
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        limit = max_output_tokens or self._settings.max_output_tokens
        payload: dict[str, Any] = {
            "model": self._settings.model,
            "max_tokens": limit,
            "messages": [{"role": "user", "content": prompt}],
            # Ограничение схемой, а не просьба в промпте — то же решение, что
            # и у Ollama, и по той же причине.
            "output_config": {
                "format": {"type": "json_schema", "schema": to_claude_schema(json_schema)}
            },
        }

        # Извлечение состава — задача на аккуратность, а не на рассуждение.
        # Мышление тут только съедает бюджет `max_tokens`, деля его с ответом,
        # и рискует обрезать список ингредиентов — ровно та беда, которую
        # разбирает ADR-019. Где мышление не выключается (семейство Fable),
        # поле не отправляется вовсе: явный `disabled` там вернёт 400.
        if self._thinking_disabled:
            payload["thinking"] = {"type": "disabled"}

        # Температуру не передаём сознательно: у поколения Sonnet 5 любой
        # параметр сэмплирования возвращает 400. Детерминизм обеспечивается
        # схемой и промптом.

        started = time.perf_counter()
        try:
            message = await self._client.messages.create(**payload)
        except (anthropic.APIConnectionError, anthropic.RateLimitError) as exc:
            # Сеть и лимиты — инфраструктура, её лечит пауза.
            logger.error(
                "Anthropic недоступна",
                extra=safe_extra(model=self._settings.model, error=type(exc).__name__),
            )
            raise LLMUnavailableError(f"Anthropic недоступна: {exc}") from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                logger.error(
                    "Anthropic вернула ошибку сервера",
                    extra=safe_extra(model=self._settings.model, status=exc.status_code),
                )
                raise LLMUnavailableError(f"Anthropic вернула {exc.status_code}: {exc}") from exc
            # 400-е — это наш неверный запрос. Ретрай его не лечит, и молча
            # уходить в отказ модели нельзя: так ошибка в схеме или в наборе
            # параметров выглядела бы недоступностью сети.
            logger.error(
                "Anthropic отклонила запрос",
                extra=safe_extra(model=self._settings.model, status=exc.status_code),
            )
            raise ExtractionError(f"Anthropic отклонила запрос ({exc.status_code}): {exc}") from exc
        latency = time.perf_counter() - started

        return self._to_response(message, latency, limit)

    def _to_response(self, message: Any, latency: float, limit: int) -> LLMResponse:
        usage = TokenUsage(
            input_tokens=int(getattr(message.usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(message.usage, "output_tokens", 0) or 0),
        )
        truncated = message.stop_reason == "max_tokens" or usage.output_tokens >= int(
            limit * _TRUNCATION_RATIO
        )

        # Классификаторы безопасности могут отклонить запрос — это обычный
        # HTTP 200 со `stop_reason: refusal`, а не ошибка. Проверяем до чтения
        # содержимого: у отказа `content` пустой.
        if message.stop_reason == "refusal":
            logger.warning(
                "Anthropic отказалась отвечать",
                extra=safe_extra(model=self._settings.model, stop_reason=message.stop_reason),
            )
            raise ExtractionError(
                "Anthropic отказалась отвечать (stop_reason=refusal)",
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                latency_s=latency,
            )

        text = "".join(block.text for block in message.content if block.type == "text")

        try:
            raw_json = json.loads(text)
        except json.JSONDecodeError as exc:
            # Та же развилка, что у Ollama (ADR-019): упёрлись в лимит вывода —
            # это проблема данных и ретрай её не лечит; оборвались раньше —
            # похоже на сбой, и вот его пауза лечит.
            if truncated:
                logger.warning(
                    "Ответ обрезан лимитом вывода и не разобрался — продукт невалиден",
                    extra=safe_extra(
                        model=self._settings.model,
                        output_tokens=usage.output_tokens,
                        limit=limit,
                        head=text[:200],
                    ),
                )
                raise ExtractionError(
                    f"Anthropic оборвала JSON на лимите вывода ({usage.output_tokens} "
                    f"из {limit} токенов): состав длиннее, чем помещается в ответ",
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    latency_s=latency,
                ) from exc
            logger.error(
                "Anthropic вернула не-JSON при заданной схеме",
                extra=safe_extra(model=self._settings.model, head=text[:200]),
            )
            raise LLMUnavailableError(f"Anthropic вернула не-JSON: {exc}") from exc

        if truncated:
            logger.warning(
                "Ответ упёрся в лимит вывода — список ингредиентов может быть неполным",
                extra=safe_extra(output_tokens=usage.output_tokens, limit=limit),
            )

        logger.debug(
            "Ответ Anthropic получен",
            extra=safe_extra(
                latency_s=round(latency, 2),
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                truncated=truncated,
            ),
        )
        return LLMResponse(
            raw_json=raw_json,
            usage=usage,
            latency_s=latency,
            model_name=self._settings.model,
            truncated=truncated,
        )


def to_claude_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Привести JSON-схему к тому, что принимает Claude.

    Два отличия от схемы для Ollama, оба обязательные:

    - у каждого объекта должен стоять `additionalProperties: false`;
    - числовые и строковые ограничения не поддерживаются и возвращают ошибку.

    Ослабления валидации тут нет: диапазоны всё равно проверяет pydantic
    после разбора ответа, а схема нужна лишь чтобы ограничить генерацию.
    """

    def convert(node: Any) -> Any:
        if isinstance(node, dict):
            result = {
                key: convert(value)
                for key, value in node.items()
                if key not in _UNSUPPORTED_SCHEMA_KEYS
            }
            if result.get("type") == "object":
                result["additionalProperties"] = False
                # Все поля обязательны: модель, которой позволено пропустить
                # поле, пропустит его непредсказуемо, и сравнивать системы
                # станет сложнее. Значения по умолчанию подставит pydantic.
                if properties := result.get("properties"):
                    result["required"] = list(properties)
            return result
        if isinstance(node, list):
            return [convert(item) for item in node]
        return node

    converted = convert(schema)
    if not isinstance(converted, dict):  # pragma: no cover - схема всегда объект
        raise ConfigurationError("JSON-схема должна быть объектом")
    return converted
