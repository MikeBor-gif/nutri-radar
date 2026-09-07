"""CLI слайса `agent`.

Composition root слайса: собирает инструменты, модель и трассировщик,
форматирует вывод. Логики цикла здесь нет (`ARCHITECTURE.md`).
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx
import typer

from nutri_radar.agent.evaluate import (
    AgentReport,
    format_report,
    read_questions,
    save_runs,
    score_run,
    write_report,
)
from nutri_radar.agent.loop import AgentRun, run_agent
from nutri_radar.agent.tools import ToolRegistry
from nutri_radar.agent.tools.lookup_barcode import lookup_barcode_tool
from nutri_radar.agent.tools.sql_query import sql_query_tool
from nutri_radar.agent.tools.vector_search import vector_search_tool
from nutri_radar.config import Settings, get_settings
from nutri_radar.db.session import dispose_engine
from nutri_radar.llm.adapters import OllamaLLM
from nutri_radar.tracing import get_tracer

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="agent",
    help="Агент с инструментами: поиск, SQL и штрихкоды.",
    no_args_is_help=True,
)

QUESTIONS_FILE = Path("data/agent/questions.jsonl")
REPORT_FILE = Path("reports/m6_agent.md")


def _printable(text: str) -> str:
    """Сделать текст выводимым в текущую консоль.

    Ответы агента приходят на пяти языках, а консоль Windows однобайтная:
    французское `é` роняет вывод. Диагностика не должна падать из-за
    кодировки терминала.
    """
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def _run[T](coro_factory: Callable[[], Awaitable[T]]) -> T:
    """Выполнить корутину и закрыть движок."""

    async def main() -> T:
        try:
            return await coro_factory()
        finally:
            await dispose_engine()

    return asyncio.run(main())


def build_registry(settings: Settings) -> ToolRegistry:
    """Собрать реестр инструментов.

    Порядок регистрации не важен — реестр сортирует по имени, чтобы промпт
    был одинаковым между запусками. Разный порядок инструментов в промпте
    менял бы поведение модели, и сравнивать прогоны стало бы нельзя.
    """
    return ToolRegistry(
        [
            lookup_barcode_tool(settings),
            sql_query_tool(settings),
            vector_search_tool(settings),
        ]
    )


def format_run(run: AgentRun) -> str:
    """Прогон в читаемом виде: что делал, чем кончил, сколько стоил."""
    lines = [f"Вопрос: {run.question}", ""]
    for step in run.steps:
        if step.is_final:
            lines.append(f"[{step.number}] ответ")
            continue
        status = "ok" if step.result and step.result.ok else "ОШИБКА"
        args = json.dumps(step.arguments, ensure_ascii=False)[:120]
        lines.append(f"[{step.number}] {step.action}({args}) -> {status}")
    lines += [
        "",
        f"Итог:      {run.stop_reason}",
        f"Ответ:     {run.answer or '—'}",
        f"Шагов:     {len(run.steps)} (вызовов инструментов {run.tool_calls}, "
        f"неудачных {run.failed_calls}, повторов {run.repeated_calls})",
        f"Токенов:   {run.total_tokens}",
        f"Время:     {run.latency_s:.1f} с",
    ]
    return "\n".join(lines)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Вопрос агенту."),
) -> None:
    """Задать агенту один вопрос."""
    settings = get_settings()
    tools = build_registry(settings)
    tracer = get_tracer(settings)

    async def go() -> AgentRun:
        async with httpx.AsyncClient(base_url=settings.ollama.base_url) as client:
            llm = OllamaLLM(client, settings.ollama)
            return await run_agent(question, llm, tools, settings, tracer=tracer)

    run = _run(go)
    typer.echo(_printable(format_run(run)))


@app.command()
def tools() -> None:
    """Показать инструменты так, как их видит модель.

    Отдельная команда, потому что описание инструмента — это часть промпта,
    и смотреть на неё надо глазами: модель выбирает инструмент по описанию,
    и расплывчатое описание превращается в неверный выбор.
    """
    typer.echo(_printable(build_registry(get_settings()).describe()))


@app.command()
def evaluate() -> None:
    """Прогнать агента по набору вопросов и посчитать проверяемое.

    Ни одно из чисел не проверяет, был ли ответ верным: это требует
    человека. Измеряется то, что видно из протокола — дошёл ли до ответа,
    тот ли инструмент выбрал, сослался ли на штрихкоды, сколько потратил.
    """
    settings = get_settings()
    questions = read_questions()
    tools = build_registry(settings)
    tracer = get_tracer(settings)

    async def go() -> tuple[AgentReport, list[tuple[object, AgentRun]]]:
        report = AgentReport()
        runs: list[tuple[object, AgentRun]] = []
        async with httpx.AsyncClient(base_url=settings.ollama.base_url) as client:
            llm = OllamaLLM(client, settings.ollama)
            for question in questions:
                run = await run_agent(question.question, llm, tools, settings, tracer=tracer)
                report.scores.append(score_run(question, run))
                runs.append((question, run))
                typer.echo(_printable(f"— {question.question}: {run.stop_reason}"))
        return report, runs

    report, runs = _run(go)
    save_runs(runs)  # type: ignore[arg-type]

    model = runs[0][1].model_name if runs else settings.ollama.model
    text = format_report(report, model)
    typer.echo("")
    typer.echo(_printable(text))
    typer.echo(f"\nОтчёт записан -> {write_report(text)}")
