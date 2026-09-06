"""Цикл агента: план → инструмент → наблюдение → ответ.

Написан руками, без LangChain и аналогов — правило 7 брифа. Это осознанная
часть демонстрации: весь цикл умещается в один читаемый файл, и на
собеседовании его можно объяснить построчно, а не сказать «фреймворк
как-то так делает».

**Действие приходит JSON по схеме, а не через tool-calling API.** Порт
`StructuredLLM` уже умеет генерацию по JSON-схеме и работает и с Ollama,
и с Anthropic. Tool-calling API у этих провайдеров разный, и завязка на
него означала бы две реализации цикла вместо одной. Побочная выгода: имена
инструментов лежат в `enum` схемы, поэтому вызов несуществующего
инструмента физически невозможен — самая частая ошибка слабой модели
отсекается схемой, а не проверкой после.

**Лимиты — условие завершимости, а не настройка удобства.** Слабая модель
склонна повторять один и тот же вызов; без потолка шагов цикл не кончится.
Превышение — не исключение, а честное завершение: «не уложился за N шагов»
полезнее, чем стектрейс.

**Ошибка инструмента возвращается модели.** Агент, умирающий от одного
неверного аргумента, бесполезен ровно там, где он нужен: модель должна
увидеть текст ошибки и попробовать иначе.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from nutri_radar.agent.prompts import load_prompt
from nutri_radar.agent.tools import ToolRegistry, ToolResult
from nutri_radar.config import Settings, get_settings
from nutri_radar.errors import ExtractionError, LLMUnavailableError
from nutri_radar.llm.ports import StructuredLLM
from nutri_radar.logging import safe_extra
from nutri_radar.tracing import NoOpTracer, Tracer

logger = logging.getLogger(__name__)

FINAL_ACTION = "final_answer"


@dataclass
class Step:
    """Один шаг агента: что подумал, что вызвал, что получил."""

    number: int
    thought: str
    action: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: ToolResult | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0

    @property
    def is_final(self) -> bool:
        return self.action == FINAL_ACTION


@dataclass
class AgentRun:
    """Итог работы агента по одному вопросу."""

    question: str
    answer: str = ""
    steps: list[Step] = field(default_factory=list)
    # Почему цикл кончился. Различать важно: «ответил» и «упёрся в лимит» —
    # разные исходы, и складывать их в одну колонку значит потерять то,
    # ради чего лимит и вводился.
    stop_reason: str = ""
    model_name: str = ""
    prompt_version: str = ""

    @property
    def answered(self) -> bool:
        return self.stop_reason == "ответ"

    @property
    def tool_calls(self) -> int:
        return sum(1 for step in self.steps if not step.is_final)

    @property
    def failed_calls(self) -> int:
        return sum(1 for step in self.steps if step.result is not None and not step.result.ok)

    @property
    def input_tokens(self) -> int:
        return sum(step.input_tokens for step in self.steps)

    @property
    def output_tokens(self) -> int:
        return sum(step.output_tokens for step in self.steps)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def latency_s(self) -> float:
        return sum(step.latency_s for step in self.steps)

    @property
    def repeated_calls(self) -> int:
        """Сколько раз агент повторил уже сделанный вызов.

        Признак зацикливания. Слабая модель, не поняв результат, охотно
        зовёт то же самое ещё раз — и без этого счётчика поведение выглядит
        как «работал долго», а не как «ходил по кругу».
        """
        seen: set[str] = set()
        repeats = 0
        for step in self.steps:
            if step.is_final:
                continue
            key = f"{step.action}:{json.dumps(step.arguments, sort_keys=True, ensure_ascii=False)}"
            if key in seen:
                repeats += 1
            seen.add(key)
        return repeats


def _observation(result: ToolResult) -> str:
    """Как результат инструмента выглядит для модели."""
    status = "OK" if result.ok else "ОШИБКА"
    return f"[{status}] {result.content}"


async def run_agent(
    question: str,
    llm: StructuredLLM,
    tools: ToolRegistry,
    settings: Settings | None = None,
    *,
    tracer: Tracer | None = None,
) -> AgentRun:
    """Прогнать агента по одному вопросу.

    Returns:
        Итог с перечнем шагов. Пустой ответ при `stop_reason != "ответ"` —
        нормальный исход: агент не обязан справляться.
    """
    settings = settings or get_settings()
    cfg = settings.agent
    tracer = tracer or NoOpTracer()

    prompt = load_prompt(cfg.prompt_version).render(
        question=question, tools=tools.describe(), max_steps=cfg.max_steps
    )
    schema = tools.action_schema()
    transcript = [prompt]

    run = AgentRun(
        question=question,
        model_name=llm.model_name,
        prompt_version=cfg.prompt_version,
    )

    with tracer.span("agent.run", question=question[:200], model=llm.model_name):
        for number in range(1, cfg.max_steps + 1):
            if run.total_tokens >= cfg.max_tokens:
                run.stop_reason = "лимит токенов"
                break

            started = time.perf_counter()
            try:
                response = await llm.generate("\n\n".join(transcript), json_schema=schema)
            except (LLMUnavailableError, ExtractionError) as exc:
                # Отказ модели — конец цикла, но не исключение наружу:
                # прогон по набору вопросов не должен падать целиком
                # из-за одного.
                logger.warning(
                    "Модель не ответила — цикл прерван",
                    extra=safe_extra(step=number, error=type(exc).__name__),
                )
                run.stop_reason = "модель не ответила"
                break

            payload = response.raw_json
            step = Step(
                number=number,
                thought=str(payload.get("thought", "")),
                action=str(payload.get("action", "")),
                arguments=dict(payload.get("arguments") or {}),
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                latency_s=time.perf_counter() - started,
            )

            if step.is_final:
                run.steps.append(step)
                run.answer = str(payload.get("answer", "")).strip()
                # Пустой ответ при действии «отвечаю» — это не ответ.
                # Засчитать его значило бы записать в успех пустоту.
                run.stop_reason = "ответ" if run.answer else "пустой ответ"
                break

            with tracer.span("agent.tool", tool=step.action, step=number):
                step.result = await tools.call(step.action, step.arguments)
            run.steps.append(step)

            transcript.append(
                f"Step {number}\n"
                f"thought: {step.thought}\n"
                f"action: {step.action}({json.dumps(step.arguments, ensure_ascii=False)})\n"
                f"observation: {_observation(step.result)}"
            )
            logger.info(
                "Шаг агента",
                extra=safe_extra(
                    step=number,
                    action=step.action,
                    ok=step.result.ok,
                    tokens=step.input_tokens + step.output_tokens,
                ),
            )
        else:
            # `for` дошёл до конца без `break` — лимит шагов исчерпан.
            run.stop_reason = "лимит шагов"

    logger.info(
        "Агент закончил",
        extra=safe_extra(
            stop_reason=run.stop_reason,
            steps=len(run.steps),
            tool_calls=run.tool_calls,
            failed=run.failed_calls,
            repeated=run.repeated_calls,
            tokens=run.total_tokens,
            seconds=round(run.latency_s, 1),
        ),
    )
    return run
