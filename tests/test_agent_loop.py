"""Тесты цикла агента.

Проверяется не «агент отвечает», а **завершимость и честность исхода**.

**Завершимость.** Слабая модель склонна повторять один и тот же вызов.
Лимит шагов здесь не настройка удобства, а условие того, что цикл вообще
кончится, — и проверяется он числом вызовов модели, а не наблюдением
за прогоном.

**Честность исхода.** «Ответил» и «упёрся в лимит» — разные вещи, и
складывать их в одну колонку значит потерять то, ради чего лимит вводился.
Пустой ответ при действии «отвечаю» — тоже не ответ: засчитать его значило
бы записать в успех пустоту.

Сети, БД и GPU здесь нет: модель подставная (правило 4 брифа).
"""

from __future__ import annotations

import pytest

from nutri_radar.agent.loop import AgentRun, Step, run_agent
from nutri_radar.agent.tools import Tool, ToolRegistry, ToolResult
from nutri_radar.config import AgentSettings, Settings
from nutri_radar.llm.adapters import FakeLLM


@pytest.fixture
def agent_settings(settings: Settings) -> Settings:
    return settings.model_copy(update={"agent": AgentSettings(max_steps=4, max_tokens=10_000)})


def _registry(result: ToolResult | None = None) -> ToolRegistry:
    outcome = result or ToolResult(ok=True, content="[3017620425035] Шоколад")

    async def find(query: str = "") -> ToolResult:
        return outcome

    return ToolRegistry(
        [
            Tool(
                name="find",
                description="Ищет продукты",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}},
                run=find,
            )
        ]
    )


def _answer(text: str = "Готово [3017620425035].") -> dict:
    return {"thought": "хватит", "action": "final_answer", "answer": text}


def _call(query: str = "шоколад") -> dict:
    return {"thought": "поищу", "action": "find", "arguments": {"query": query}}


class TestУспешныйПуть:
    async def test_агент_доходит_до_ответа(self, agent_settings: Settings):
        llm = FakeLLM(response_factory=lambda prompt: _answer())

        run = await run_agent("вопрос", llm, _registry(), agent_settings)

        assert run.answered is True
        assert run.stop_reason == "ответ"
        assert "[3017620425035]" in run.answer

    async def test_вызов_инструмента_попадает_в_шаги(self, agent_settings: Settings):
        calls = iter([_call(), _answer()])
        llm = FakeLLM(response_factory=lambda prompt: next(calls))

        run = await run_agent("вопрос", llm, _registry(), agent_settings)

        assert run.tool_calls == 1
        assert run.steps[0].action == "find"
        assert run.steps[0].result is not None
        assert run.steps[0].result.ok is True

    async def test_наблюдение_возвращается_модели(self, agent_settings: Settings):
        """Без этого цикл не цикл: модель обязана видеть результат
        своего вызова на следующем шаге."""
        seen: list[str] = []

        def factory(prompt: str) -> dict:
            seen.append(prompt)
            return _answer() if len(seen) > 1 else _call()

        llm = FakeLLM(response_factory=factory)
        await run_agent("вопрос", llm, _registry(), agent_settings)

        assert "observation" in seen[1]
        assert "[3017620425035]" in seen[1]


class TestЛимиты:
    async def test_лимит_шагов_обрывает_зацикливание(self, agent_settings: Settings):
        """Модель, зовущая одно и то же, не должна крутиться вечно."""
        llm = FakeLLM(response_factory=lambda prompt: _call())

        run = await run_agent("вопрос", llm, _registry(), agent_settings)

        assert run.stop_reason == "лимит шагов"
        assert len(run.steps) == agent_settings.agent.max_steps
        assert llm.call_count == agent_settings.agent.max_steps

    async def test_упёрся_в_лимит_это_не_ответ(self, agent_settings: Settings):
        llm = FakeLLM(response_factory=lambda prompt: _call())

        run = await run_agent("вопрос", llm, _registry(), agent_settings)

        assert run.answered is False
        assert run.answer == ""

    async def test_повторы_считаются(self, agent_settings: Settings):
        """Признак зацикливания. Без счётчика поведение выглядит как
        «работал долго», а не как «ходил по кругу»."""
        llm = FakeLLM(response_factory=lambda prompt: _call("одно и то же"))

        run = await run_agent("вопрос", llm, _registry(), agent_settings)

        assert run.repeated_calls == agent_settings.agent.max_steps - 1

    async def test_разные_вызовы_повторами_не_считаются(self, agent_settings: Settings):
        counter = iter(range(100))
        llm = FakeLLM(response_factory=lambda prompt: _call(f"запрос {next(counter)}"))

        run = await run_agent("вопрос", llm, _registry(), agent_settings)

        assert run.repeated_calls == 0

    async def test_лимит_токенов_останавливает_цикл(self, settings: Settings):
        """Второй предохранитель: шаги бывают дешёвыми по числу
        и дорогими по длине наблюдений."""
        tight = settings.model_copy(update={"agent": AgentSettings(max_steps=50, max_tokens=1)})
        llm = FakeLLM(response_factory=lambda prompt: _call())

        run = await run_agent("вопрос", llm, _registry(), tight)

        assert run.stop_reason == "лимит токенов"
        assert len(run.steps) < 50


class TestЧестностьИсхода:
    async def test_пустой_ответ_не_считается_ответом(self, agent_settings: Settings):
        """Засчитать его значило бы записать в успех пустоту."""
        llm = FakeLLM(response_factory=lambda prompt: _answer("   "))

        run = await run_agent("вопрос", llm, _registry(), agent_settings)

        assert run.stop_reason == "пустой ответ"
        assert run.answered is False

    async def test_отказ_модели_завершает_цикл_а_не_роняет(self, agent_settings: Settings):
        """Прогон по набору вопросов не должен падать целиком из-за одного."""
        llm = FakeLLM(default_response=_answer(), fail_times=99)

        run = await run_agent("вопрос", llm, _registry(), agent_settings)

        assert run.stop_reason == "модель не ответила"
        assert run.answered is False

    async def test_ошибка_инструмента_возвращается_модели(self, agent_settings: Settings):
        """Агент, умирающий от неудачного вызова, бесполезен ровно там,
        где нужен."""
        seen: list[str] = []

        def factory(prompt: str) -> dict:
            seen.append(prompt)
            return _answer() if len(seen) > 1 else _call()

        llm = FakeLLM(response_factory=factory)
        broken = _registry(ToolResult.failure("таблица недоступна"))

        run = await run_agent("вопрос", llm, broken, agent_settings)

        assert run.failed_calls == 1
        assert "ОШИБКА" in seen[1]
        assert run.answered is True


class TestУчёт:
    async def test_токены_складываются_по_шагам(self, agent_settings: Settings):
        """Цена агентности: десять вызовов стоят в десять раз дороже
        одного, и это надо показать числом."""
        run = AgentRun(question="q")
        run.steps = [
            Step(number=1, thought="", action="find", input_tokens=100, output_tokens=10),
            Step(number=2, thought="", action="final_answer", input_tokens=200, output_tokens=20),
        ]

        assert run.input_tokens == 300
        assert run.output_tokens == 30
        assert run.total_tokens == 330

    async def test_финальный_шаг_не_считается_вызовом_инструмента(self):
        run = AgentRun(question="q")
        run.steps = [
            Step(number=1, thought="", action="find"),
            Step(number=2, thought="", action="final_answer"),
        ]

        assert run.tool_calls == 1

    async def test_модель_и_версия_промпта_записываются(self, agent_settings: Settings):
        llm = FakeLLM(response_factory=lambda prompt: _answer())

        run = await run_agent("вопрос", llm, _registry(), agent_settings)

        assert run.model_name == "fake-model"
        assert run.prompt_version == agent_settings.agent.prompt_version


class TestТрассировка:
    async def test_отсутствие_langfuse_не_роняет_агента(self, agent_settings: Settings):
        """Наблюдаемость — удобство разбора, а не часть работы: прогон,
        падающий из-за недоступного контейнера трассировки, теряет данные
        ради их же протоколирования."""
        from nutri_radar.tracing import get_tracer

        tracer = get_tracer(agent_settings)
        llm = FakeLLM(response_factory=lambda prompt: _answer())

        run = await run_agent("вопрос", llm, _registry(), agent_settings, tracer=tracer)

        assert run.answered is True
