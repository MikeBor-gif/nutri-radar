"""Метрики поиска и RAG: recall@k и подтверждённость ответа.

**recall@k, а не precision.** Вопрос пользователя звучит «найди мне продукты
с таким-то свойством», и цена пропуска здесь выше цены лишнего результата:
лишний он отбросит глазами за секунду, пропущенный не увидит никогда.
Precision считается рядом, но решение принимается по recall.

**Подтверждённость считается кодом, а не моделью.** Соблазн — попросить
модель оценить, подтверждён ли её собственный ответ источниками. Это
измеряет самооценку, а не факт: модель, склонная выдумывать, столь же
охотно подтвердит выдуманное. Здесь проверка механическая — штрихкод либо
есть в выдаче поиска, либо нет, и никакого суждения для этого не требуется.

Что именно проверяется:

* **доля ссылок, ведущих в выдачу** — сколько процитированных штрихкодов
  действительно были найдены поиском;
* **число выдуманных ссылок** — самый опасный отказ: утверждение выглядит
  подтверждённым, а проверить может только тот, кто пойдёт в базу;
* **доля ответов без единой ссылки** — формально не ложь, но и не то,
  ради чего строился RAG.

Оценивать смысловое соответствие ответа составу здесь не пытаемся: это
работа человека, и подменять её моделью значило бы получить ту же
самооценку, только окольным путём.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

from nutri_radar.logging import safe_extra
from nutri_radar.retrieval.rag import RagAnswer
from nutri_radar.retrieval.search import SearchResult

logger = logging.getLogger(__name__)

QUERIES_FILE = Path("data/retrieval/queries.jsonl")


class GoldQuery(BaseModel):
    """Эталонный запрос: вопрос и продукты, считающиеся верным ответом.

    Составляется **человеком** — правило 6 брифа. И составляется **до**
    первого прогона поиска: эталон, написанный после того, как автор увидел
    выдачу, измеряет согласие системы с самой собой.
    """

    query: str
    # Штрихкоды, которые обязаны найтись. Не «все подходящие в базе» —
    # столько человек не разметит, — а те, про которые он уверен.
    expected: list[str] = Field(default_factory=list)
    # Свободная пометка: чем этот запрос отличается от других. Нужна для
    # разбивки в отчёте — «запросы про отсутствие ингредиента» ведут себя
    # иначе, чем «запросы про наличие».
    kind: str = ""
    lang: str = ""
    author: str = ""

    @property
    def is_usable(self) -> bool:
        return bool(self.query.strip() and self.expected)


@dataclass
class RecallScore:
    """recall@k по одному запросу."""

    query: str
    kind: str
    expected: int
    found: int
    returned: int
    latency_s: float = 0.0

    @property
    def recall(self) -> float:
        return self.found / self.expected if self.expected else 0.0

    @property
    def precision(self) -> float:
        return self.found / self.returned if self.returned else 0.0


@dataclass
class GroundingScore:
    """Подтверждённость одного ответа RAG."""

    question: str
    refused: bool
    cited: int
    invented: int
    sources: int

    @property
    def grounded_share(self) -> float:
        """Доля ссылок, ведущих в выдачу поиска."""
        if not self.cited:
            return 0.0
        return (self.cited - self.invented) / self.cited

    @property
    def has_citations(self) -> bool:
        return self.cited > 0


@dataclass
class RetrievalReport:
    """Свод по всем запросам."""

    k: int
    recalls: list[RecallScore] = field(default_factory=list)
    groundings: list[GroundingScore] = field(default_factory=list)

    @property
    def mean_recall(self) -> float:
        return (
            sum(item.recall for item in self.recalls) / len(self.recalls) if self.recalls else 0.0
        )

    @property
    def mean_precision(self) -> float:
        if not self.recalls:
            return 0.0
        return sum(item.precision for item in self.recalls) / len(self.recalls)

    @property
    def median_latency(self) -> float:
        if not self.recalls:
            return 0.0
        values = sorted(item.latency_s for item in self.recalls)
        return values[len(values) // 2]

    @property
    def answered(self) -> list[GroundingScore]:
        return [item for item in self.groundings if not item.refused]

    @property
    def refusal_share(self) -> float:
        if not self.groundings:
            return 0.0
        return sum(1 for item in self.groundings if item.refused) / len(self.groundings)

    @property
    def mean_grounded(self) -> float:
        answered = self.answered
        if not answered:
            return 0.0
        return sum(item.grounded_share for item in answered) / len(answered)

    @property
    def total_invented(self) -> int:
        return sum(item.invented for item in self.groundings)

    @property
    def without_citations(self) -> int:
        return sum(1 for item in self.answered if not item.has_citations)


def read_queries(path: Path | None = None) -> list[GoldQuery]:
    """Прочитать эталонные запросы.

    Нет файла — не ошибка формата, а понятное сообщение: запросы составляет
    владелец проекта, и до этого момента считать нечего.
    """
    file = path or QUERIES_FILE
    if not file.exists():
        raise FileNotFoundError(
            f"Эталонные запросы не найдены: {file}. Их составляет человек "
            "(правило 6 брифа) командой `nutri-radar retrieval add-query`."
        )
    queries = []
    for line in file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            queries.append(GoldQuery.model_validate_json(line))
    logger.info("Эталонные запросы прочитаны", extra=safe_extra(path=str(file), count=len(queries)))
    return queries


def append_query(query: GoldQuery, path: Path | None = None) -> Path:
    """Дописать один запрос. Прогресс сохраняется сразу, как в разметке M3."""
    file = path or QUERIES_FILE
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("a", encoding="utf-8") as sink:
        sink.write(query.model_dump_json() + "\n")
    return file


def score_recall(query: GoldQuery, result: SearchResult) -> RecallScore:
    """Посчитать recall@k по одному запросу.

    Считается по множеству кодов, а не по позициям: «нашлось в первых пяти»
    — это про попадание, а не про порядок внутри пятёрки.
    """
    returned = set(result.codes)
    expected = set(query.expected)
    return RecallScore(
        query=query.query,
        kind=query.kind,
        expected=len(expected),
        found=len(expected & returned),
        returned=len(returned),
        latency_s=result.latency_s,
    )


def score_grounding(answer: RagAnswer) -> GroundingScore:
    """Посчитать подтверждённость одного ответа.

    Никакого суждения о смысле: штрихкод либо был в выдаче, либо нет.
    """
    return GroundingScore(
        question=answer.question,
        refused=answer.refused,
        cited=len(answer.cited),
        invented=len(answer.cited_outside_sources),
        sources=len(answer.sources),
    )


def format_report(report: RetrievalReport) -> str:
    """Отчёт по метрикам поиска и RAG."""
    lines = [
        f"# Поиск и RAG: метрики на {len(report.recalls)} эталонных запросах",
        "",
        f"## recall@{report.k}",
        "",
        "| Величина | Значение |",
        "|---|---|",
        f"| Средний recall@{report.k} | {report.mean_recall:.1%} |",
        f"| Средний precision@{report.k} | {report.mean_precision:.1%} |",
        f"| Медианная латентность | {report.median_latency * 1000:.0f} мс |",
        "",
        "Решение принимается по recall: цена пропуска выше цены лишнего "
        "результата — лишний пользователь отбросит глазами за секунду, "
        "пропущенный не увидит никогда.",
    ]

    by_kind: dict[str, list[RecallScore]] = {}
    for item in report.recalls:
        by_kind.setdefault(item.kind or "без типа", []).append(item)
    if len(by_kind) > 1:
        lines += [
            "",
            "### По типам запросов",
            "",
            "| Тип | Запросов | recall |",
            "|---|---|---|",
        ]
        for kind, items in sorted(by_kind.items()):
            mean = sum(entry.recall for entry in items) / len(items)
            lines.append(f"| {kind} | {len(items)} | {mean:.1%} |")

    lines += [
        "",
        "## Подтверждённость ответов",
        "",
        "| Величина | Значение |",
        "|---|---|",
        f"| Отказов «не знаю» | {report.refusal_share:.1%} |",
        f"| Доля ссылок, ведущих в выдачу | {report.mean_grounded:.1%} |",
        f"| Выдуманных штрихкодов | {report.total_invented} |",
        f"| Ответов без единой ссылки | {report.without_citations} |",
        "",
        "Подтверждённость считается **кодом, а не моделью**: штрихкод либо "
        "есть в выдаче поиска, либо нет. Просить модель оценить собственный "
        "ответ значило бы измерить её самооценку — та, что склонна выдумывать, "
        "столь же охотно подтвердит выдуманное.",
        "",
        "Выдуманный штрихкод — самый опасный из отказов: утверждение выглядит "
        "подтверждённым, а проверить может только тот, кто пойдёт в базу.",
    ]
    return "\n".join(lines)


def write_report(text: str, path: Path | None = None) -> Path:
    """Записать отчёт."""
    file = path or Path("reports/m5_retrieval.md")
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(text, encoding="utf-8")
    logger.info("Отчёт записан", extra=safe_extra(path=str(file)))
    return file


def dump_queries_template(path: Path) -> Path:
    """Записать пример файла запросов — чтобы формат был виден до заполнения."""
    examples = [
        GoldQuery(
            query="шоколад без пальмового масла",
            expected=["ЗАМЕНИТЬ_НА_ШТРИХКОД"],
            kind="отсутствие ингредиента",
            lang="ru",
            author="",
        ),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(item.model_dump(), ensure_ascii=False) for item in examples) + "\n",
        encoding="utf-8",
    )
    return path
