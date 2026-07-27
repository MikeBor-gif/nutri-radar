"""Тесты разбора ответа Ollama. Сети нет — разбирается готовый payload.

Проверяется одно решение, но дорогое: считать ли непрочитанный ответ отказом
модели. От него зависит, ретраить продукт или пропустить, а ретрай генерации
до лимита вывода стоит минут GPU и ничего не меняет.
"""

from __future__ import annotations

import httpx
import pytest

from nutri_radar.config import OllamaSettings
from nutri_radar.errors import ExtractionError, LLMUnavailableError
from nutri_radar.llm.adapters.ollama import OllamaLLM

LIMIT = 2048

GOOD_CONTENT = (
    '{"ingredients": [{"canonical_name": "sugar", "kind": "sugar", '
    '"e_number": null}], "allergens": [], "unreadable": false, '
    '"model_confidence": 0.9}'
)

# Ровно то, что пришло на продукте 0718604977580: JSON оборван посреди массива.
TRUNCATED_CONTENT = (
    '{\n  "ingredients": [\n    { "canonical_name": "almonds", "kind": "additive" },\n'
    '    { "canonical_name": "banana", "kind": "additive" },\n    {'
)


def _adapter() -> OllamaLLM:
    return OllamaLLM(httpx.AsyncClient(base_url="http://localhost:11434"), OllamaSettings())


def _payload(content: str, *, output_tokens: int) -> dict:
    return {
        "message": {"content": content},
        "prompt_eval_count": 220,
        "eval_count": output_tokens,
    }


class TestОборванныйОтвет:
    def test_обрыв_на_лимите_это_проблема_данных(self):
        """Повтор даст ту же обрезку — ретраить нечего."""
        with pytest.raises(ExtractionError, match="лимите вывода"):
            _adapter()._to_response(_payload(TRUNCATED_CONTENT, output_tokens=LIMIT), 1.0, LIMIT)

    def test_обрыв_несёт_потраченные_токены(self):
        """Продукт пропущен, но генерация оплачена — она входит в стоимость."""
        with pytest.raises(ExtractionError) as info:
            _adapter()._to_response(_payload(TRUNCATED_CONTENT, output_tokens=LIMIT), 1.0, LIMIT)

        assert info.value.output_tokens == LIMIT
        assert info.value.input_tokens == 220

    def test_мусор_без_лимита_это_отказ_модели(self):
        """Модель ответила ерундой, не упёршись в лимит: похоже на сбой загрузки."""
        with pytest.raises(LLMUnavailableError):
            _adapter()._to_response(_payload("не json вовсе", output_tokens=17), 1.0, LIMIT)

    def test_разобранный_но_упёршийся_в_лимит_помечается_обрезанным(self):
        """JSON закрылся ровно на лимите: схему проходит, но список неполон."""
        response = _adapter()._to_response(_payload(GOOD_CONTENT, output_tokens=LIMIT), 1.0, LIMIT)

        assert response.truncated is True

    def test_обычный_ответ_обрезанным_не_считается(self):
        response = _adapter()._to_response(_payload(GOOD_CONTENT, output_tokens=250), 1.0, LIMIT)

        assert response.truncated is False
