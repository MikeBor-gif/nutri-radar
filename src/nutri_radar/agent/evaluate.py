"""Прогон агента по набору вопросов и подсчёт того, что можно проверить.

**«Агент справился» определяется проверяемым признаком, а не оценкой
качества ответа.** Спрашивать модель, справилась ли она, значит мерить
самооценку — ошибка, отвергнутая в M5. Оценивать ответы глазами тоже
нельзя: это ручная разметка, а её делает человек (правило 6).

Поэтому считается только то, что видно из протокола:

* **дошёл ли до ответа** — или упёрся в лимит, или модель отказала;
* **выбрал ли ожидаемый инструмент ПЕРВЫМ вызовом** — вопрос про штрихкод
  должен вести в `lookup_barcode`, а не в векторный поиск. Именно первым:
  «позвал где-то среди восьми шагов» засчитало бы агента, перебравшего все
  инструменты подряд. Ожидание записано в наборе вопросов до прогона;
* **сослался ли на штрихкоды** — ответ без ссылок нечем проверить;
* **сколько шагов, повторов и токенов потратил** — цена агентности.

Ни одно из этих чисел не говорит, был ли ответ *верным*. Это ограничение
названо прямо, а не спрятано: проверка верности требует человека, и
подменять её автоматикой значило бы получить красивую цифру ни о чём.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel

from nutri_radar.agent.loop import AgentRun
from nutri_radar.logging import safe_extra

logger = logging.getLogger(__name__)

QUESTIONS_FILE = Path("data/agent/questions.jsonl")
RUNS_FILE = Path("data/agent/runs.jsonl")

_BARCODE = re.compile(r"\[(\d{6,14})\]")


class Question(BaseModel):
    """Вопрос агенту вместе с ожиданием, записанным до прогона."""

    question: str
    # Какой инструмент должен был выбраться. Пусто — вопрос вне области,
    # и правильное поведение здесь противоположное: не звать инструменты,
    # а честно сказать, что данных нет.
    expects_tool: str = ""
    kind: str = ""

    @property
    def out_of_scope(self) -> bool:
        return not self.expects_tool


@dataclass
class RunScore:
    """Что видно из протокола одного прогона."""

    question: str
    kind: str
    stop_reason: str
    answered: bool
    used_expected_tool: bool
    cited: int
    steps: int
    tool_calls: int
    failed_calls: int
    repeated_calls: int
    tokens: int
    seconds: float


@dataclass
class AgentReport:
    """Свод по набору вопросов."""

    scores: list[RunScore] = field(default_factory=list)

    @property
    def answered_share(self) -> float:
        if not self.scores:
            return 0.0
        return sum(1 for item in self.scores if item.answered) / len(self.scores)

    @property
    def right_tool_share(self) -> float:
        """Доля вопросов, где ожидаемый инструмент выбран ПЕРВЫМ вызовом.

        Считается только по вопросам, для которых ожидание записано:
        у вопроса вне области правильного инструмента нет.
        """
        relevant = [item for item in self.scores if item.kind != "вне области"]
        if not relevant:
            return 0.0
        return sum(1 for item in relevant if item.used_expected_tool) / len(relevant)

    @property
    def cited_share(self) -> float:
        answered = [item for item in self.scores if item.answered]
        if not answered:
            return 0.0
        return sum(1 for item in answered if item.cited) / len(answered)

    @property
    def total_repeats(self) -> int:
        return sum(item.repeated_calls for item in self.scores)

    @property
    def mean_steps(self) -> float:
        if not self.scores:
            return 0.0
        return sum(item.steps for item in self.scores) / len(self.scores)

    @property
    def mean_tokens(self) -> float:
        if not self.scores:
            return 0.0
        return sum(item.tokens for item in self.scores) / len(self.scores)

    @property
    def mean_seconds(self) -> float:
        if not self.scores:
            return 0.0
        return sum(item.seconds for item in self.scores) / len(self.scores)


def read_questions(path: Path | None = None) -> list[Question]:
    """Прочитать набор вопросов."""
    file = path or QUESTIONS_FILE
    if not file.exists():
        raise FileNotFoundError(f"Набор вопросов не найден: {file}")
    questions = [
        Question.model_validate_json(line)
        for line in file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    logger.info("Вопросы прочитаны", extra=safe_extra(path=str(file), count=len(questions)))
    return questions


def score_run(question: Question, run: AgentRun) -> RunScore:
    """Свести прогон к проверяемым числам."""
    calls = [step.action for step in run.steps if not step.is_final]
    first = calls[0] if calls else ""
    return RunScore(
        question=question.question,
        kind=question.kind,
        stop_reason=run.stop_reason,
        answered=run.answered,
        # По первому вызову, а не по множеству использованных: «позвал
        # где-то среди восьми шагов» засчитало бы агента, перебравшего все
        # инструменты подряд, и число перестало бы отличать понимание
        # от перебора.
        #
        # У вопроса вне области ожидание обратное: правильно НЕ звать
        # инструменты вовсе. Считать это «выбрал не тот инструмент» было бы
        # наказанием за верное поведение.
        used_expected_tool=(not calls) if question.out_of_scope else first == question.expects_tool,
        cited=len(set(_BARCODE.findall(run.answer))),
        steps=len(run.steps),
        tool_calls=run.tool_calls,
        failed_calls=run.failed_calls,
        repeated_calls=run.repeated_calls,
        tokens=run.total_tokens,
        seconds=run.latency_s,
    )


def save_runs(runs: list[tuple[Question, AgentRun]], path: Path | None = None) -> Path:
    """Сохранить протоколы прогонов.

    Коммитятся: без них числа в отчёте нельзя ни проверить, ни оспорить,
    а повторить прогон стоит минут GPU. Тот же довод, что у предсказаний
    в M3 и M4.
    """
    file = path or RUNS_FILE
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("w", encoding="utf-8") as sink:
        for question, run in runs:
            sink.write(
                json.dumps(
                    {
                        "question": question.question,
                        "kind": question.kind,
                        "expects_tool": question.expects_tool,
                        "stop_reason": run.stop_reason,
                        "answer": run.answer,
                        "model": run.model_name,
                        "prompt_version": run.prompt_version,
                        "steps": [
                            {
                                "n": step.number,
                                "action": step.action,
                                "arguments": step.arguments,
                                "ok": None if step.result is None else step.result.ok,
                            }
                            for step in run.steps
                        ],
                        "tokens": run.total_tokens,
                        "seconds": round(run.latency_s, 2),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    logger.info("Протоколы прогонов сохранены", extra=safe_extra(path=str(file), runs=len(runs)))
    return file


def format_report(report: AgentReport, model: str) -> str:
    """Отчёт по прогону агента."""
    lines = [
        f"# Агент: прогон на {len(report.scores)} вопросах",
        "",
        f"Модель: `{model}`",
        "",
        "| Величина | Значение |",
        "|---|---|",
        f"| Дошёл до ответа | {report.answered_share:.0%} |",
        f"| Выбрал ожидаемый инструмент первым вызовом | {report.right_tool_share:.0%} |",
        f"| Ответов со ссылками на штрихкоды | {report.cited_share:.0%} |",
        f"| Повторных вызовов всего | {report.total_repeats} |",
        f"| Шагов в среднем | {report.mean_steps:.1f} |",
        f"| Токенов в среднем | {report.mean_tokens:.0f} |",
        f"| Секунд в среднем | {report.mean_seconds:.1f} |",
        "",
        "**Что эти числа НЕ говорят.** Ни одно из них не проверяет, был ли "
        "ответ верным. Проверка верности требует человека, и подменять её "
        "автоматикой значило бы получить красивую цифру ни о чём. Здесь "
        "измерено только то, что видно из протокола: дошёл ли агент до "
        "ответа, тот ли инструмент выбрал, сослался ли на штрихкоды и "
        "сколько это стоило.",
        "",
        "**Повторные вызовы — главный признак.** Модель, не понявшая "
        "результат, зовёт то же самое ещё раз. Без этого счётчика поведение "
        "выглядит как «работал долго», а не как «ходил по кругу».",
        "",
        "## По вопросам",
        "",
        "| Вопрос | Итог | Инструмент | Ссылок | Шагов | Повторов | Токенов |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in report.scores:
        tool = "да" if item.used_expected_tool else "нет"
        lines.append(
            f"| {item.question} | {item.stop_reason} | {tool} | {item.cited} "
            f"| {item.steps} | {item.repeated_calls} | {item.tokens} |"
        )

    by_kind: dict[str, list[RunScore]] = {}
    for item in report.scores:
        by_kind.setdefault(item.kind or "без типа", []).append(item)
    if len(by_kind) > 1:
        lines += [
            "",
            "## По типам вопросов",
            "",
            "| Тип | Вопросов | Дошёл до ответа | Верный инструмент |",
            "|---|---|---|---|",
        ]
        for kind, items in sorted(by_kind.items()):
            answered = sum(1 for entry in items if entry.answered) / len(items)
            right = sum(1 for entry in items if entry.used_expected_tool) / len(items)
            lines.append(f"| {kind} | {len(items)} | {answered:.0%} | {right:.0%} |")

    return "\n".join(lines)


def write_report(text: str, path: Path | None = None) -> Path:
    """Записать отчёт."""
    file = path or Path("reports/m6_agent.md")
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(text, encoding="utf-8")
    logger.info("Отчёт записан", extra=safe_extra(path=str(file)))
    return file
