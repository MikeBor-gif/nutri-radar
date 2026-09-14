"""Метрики поиска и RAG: recall@k и подтверждённость ответа.

**Главная метрика — доля выдачи, обладающая запрошенным свойством,
а не recall@k.** Первая версия этого модуля считала recall против набора
из трёх «правильных» продуктов на запрос, и он вышел нулевым на всех
двадцати запросах. Дело было не в поиске: под условие «шоколад с пальмовым
маслом» подходит **2035 продуктов**, под «снеки с глутаматом» — 1952.
Просить систему вернуть в первой пятёрке именно те три, что оракул выбрал
случайно из двух тысяч, — задача с нулевым решением у любого поиска,
включая идеальный.

Правильный вопрос при тысячах верных ответов другой: **обладают ли
найденные продукты запрошенным свойством**. Он проверяется тем же точным
условием, которым набор и определялся, и отвечает ровно на то, что
интересует пользователя: «я попросил шоколад с пальмовым маслом — мне
дали шоколад с пальмовым маслом?»

`recall@k` остаётся в коде: он верен для эталона, где человек назвал
конкретные продукты, которые обязаны найтись. На оракульном наборе
он не считается — и в отчёте объясняется почему.

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
from sqlalchemy import text

from nutri_radar.config import Settings, get_settings
from nutri_radar.db.session import get_session
from nutri_radar.logging import safe_extra
from nutri_radar.retrieval.language import (
    RESOLVABLE_LANGUAGES,
    UNDETERMINED,
    detect_language,
)
from nutri_radar.retrieval.rag import RagAnswer
from nutri_radar.retrieval.search import SearchResult

logger = logging.getLogger(__name__)

QUERIES_FILE = Path("data/retrieval/queries.jsonl")
OUT_OF_DOMAIN_FILE = Path("data/retrieval/out_of_domain.jsonl")


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
    # Точное условие, определяющее «правильный ответ». Заполняется, когда
    # эталон выведен оракулом: тогда качество выдачи проверяется свойством
    # каждого найденного продукта, а не совпадением с горсткой примеров.
    # У эталона, составленного человеком, остаётся пустым.
    predicate: str = ""
    # Свободная пометка: чем этот запрос отличается от других. Нужна для
    # разбивки в отчёте — «запросы про отсутствие ингредиента» ведут себя
    # иначе, чем «запросы про наличие».
    kind: str = ""
    lang: str = ""
    author: str = ""

    @property
    def is_usable(self) -> bool:
        return bool(self.query.strip() and self.expected)


class OutOfDomainQuestion(BaseModel):
    """Вопрос, на который система обязана отказать по построению.

    **Это не эталонная разметка.** У такого вопроса нет верного ответа
    в составах продуктов — ни одного, ни тысячи, — поэтому размечать
    нечего: ожидаемое поведение одно на весь файл и записано здесь,
    в докстроке, а не в каждой строке файла.

    Отличие от `GoldQuery` принципиальное и разнесено по разным файлам
    намеренно. `MIN_SIMILARITY` ловит «в базе нет ничего похожего»:
    косинус низкий, до модели дело не доходит. Эти вопросы проходят мимо
    порога — «сколько стоит хлеб» находит хлеб с высокой близостью, —
    и отказ должен наступить по другой причине. Смешать их с эталоном
    поиска значило бы считать свойство выдачи по вопросам, у которых
    выдачи быть не должно.
    """

    question: str
    # Чем этот вопрос посторонний: цена, доставка, рецепт, погода,
    # личный совет по питанию. Для разбивки в отчёте: по одним
    # категориям система отказывает надёжнее, чем по другим.
    kind: str = ""
    lang: str = ""
    author: str = ""


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
class PropertyScore:
    """Доля выдачи, обладающая запрошенным свойством.

    Главная метрика поиска на корпусе, где верных ответов тысячи. Отвечает
    на вопрос пользователя буквально: «я попросил шоколад с пальмовым
    маслом — мне дали шоколад с пальмовым маслом?»
    """

    query: str
    kind: str
    returned: int
    matching: int
    latency_s: float = 0.0

    @property
    def precision(self) -> float:
        return self.matching / self.returned if self.returned else 0.0


@dataclass
class LanguageScore:
    """Совпал ли язык ответа с языком вопроса.

    Язык вопроса **берётся из поля `lang` эталона, а не угадывается**:
    угаданный язык вопроса добавил бы к измерению вторую ошибку,
    и разделить их потом было бы нечем.
    """

    question: str
    # Из эталона. Пустой — у запроса не проставлен язык, такие в долю
    # не входят вовсе.
    expected: str
    # Из ответа, по алфавиту. `UNDETERMINED`, если ответ слишком короткий.
    actual: str

    @property
    def is_resolvable(self) -> bool:
        """Способен ли признак по алфавиту различить язык этого вопроса."""
        return self.expected in RESOLVABLE_LANGUAGES

    @property
    def is_comparable(self) -> bool:
        """Есть ли что сравнивать: язык вопроса различим, язык ответа определён."""
        return self.is_resolvable and self.actual != UNDETERMINED

    @property
    def matched(self) -> bool:
        return self.is_comparable and self.expected == self.actual


@dataclass
class RefusalScore:
    """Отказала ли система на вопросе вне домена.

    Одна величина и никакой оценки текста: отказ либо наступил, либо нет.
    Судить о качестве формулировки отказа моделью было бы той же
    самооценкой, от которой отказались в подтверждённости.
    """

    question: str
    kind: str
    refused: bool


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
    properties: list[PropertyScore] = field(default_factory=list)
    recalls: list[RecallScore] = field(default_factory=list)
    groundings: list[GroundingScore] = field(default_factory=list)
    languages: list[LanguageScore] = field(default_factory=list)
    out_of_domain: list[RefusalScore] = field(default_factory=list)
    # Версия промпта RAG, на которой считались числа. Без неё отчёты двух
    # версий неразличимы, а сравнивать их построчно — единственный способ
    # понять, что изменила правка промпта.
    prompt_version: str = ""

    @property
    def mean_property_precision(self) -> float:
        if not self.properties:
            return 0.0
        return sum(item.precision for item in self.properties) / len(self.properties)

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
        source = self.properties or self.recalls
        if not source:
            return 0.0
        values = sorted(item.latency_s for item in source)
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
    def comparable_languages(self) -> list[LanguageScore]:
        """Ответы, про которые вообще можно сказать, на каком они языке."""
        return [item for item in self.languages if item.is_comparable]

    @property
    def language_match_share(self) -> float:
        """Доля ответов на языке вопроса — среди тех, где язык определился.

        Знаменатель — только сравнимые. Считать неопределившиеся промахами
        значило бы наказывать систему за короткий ответ; считать их
        попаданиями — прятать их. Поэтому они выведены отдельным числом.
        """
        comparable = self.comparable_languages
        if not comparable:
            return 0.0
        return sum(1 for item in comparable if item.matched) / len(comparable)

    @property
    def out_of_domain_refusal_share(self) -> float:
        """Доля честных отказов на вопросах вне домена.

        Здесь отказ — это успех, в отличие от `refusal_share` на эталонных
        запросах, где отказ означает, что система не нашла того, что есть
        в базе. Два числа с похожим смыслом и противоположным знаком —
        поэтому они считаются по разным наборам и в отчёте разведены.
        """
        if not self.out_of_domain:
            return 0.0
        return sum(1 for item in self.out_of_domain if item.refused) / len(self.out_of_domain)

    @property
    def language_undetermined(self) -> int:
        """Ответы, где язык вопроса различим, а язык ответа определить не вышло.

        Отказы и слишком короткие ответы. Считаются отдельно: промах и
        «нечего мерить» — разные события.
        """
        return sum(
            1 for item in self.languages if item.is_resolvable and item.actual == UNDETERMINED
        )

    @property
    def language_unresolvable(self) -> int:
        """Запросы на языках, которых признак по алфавиту не различает.

        Немецкий и французский: латиница, как у английского. Такие запросы
        не входят в долю ни с какой стороны — это граница метода, и она
        публикуется числом, а не умалчивается.
        """
        return sum(1 for item in self.languages if not item.is_resolvable)

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


def read_out_of_domain(path: Path | None = None) -> list[OutOfDomainQuestion]:
    """Прочитать вопросы вне домена.

    Отдельный файл, а не поле в эталоне поиска: по этим вопросам не
    считаются ни свойство выдачи, ни recall — по ним считается ровно
    одна величина, доля отказов. Держать их в одном файле значило бы
    рано или поздно посчитать по ним не ту метрику.
    """
    file = path or OUT_OF_DOMAIN_FILE
    if not file.exists():
        raise FileNotFoundError(
            f"Вопросы вне домена не найдены: {file}. Это перечень вопросов, "
            "на которые система обязана отказывать; ожидаемое поведение "
            "одно на весь файл — отказ."
        )
    questions = [
        OutOfDomainQuestion.model_validate_json(line)
        for line in file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    logger.info(
        "Вопросы вне домена прочитаны",
        extra=safe_extra(path=str(file), count=len(questions)),
    )
    return questions


def score_out_of_domain(question: OutOfDomainQuestion, answer: RagAnswer) -> RefusalScore:
    """Отказала ли система на постороннем вопросе."""
    score = RefusalScore(question=question.question, kind=question.kind, refused=answer.refused)
    logger.info(
        "Вопрос вне домена проверен",
        extra=safe_extra(
            question=question.question[:120],
            kind=question.kind or "без типа",
            refused=score.refused,
            sources=len(answer.sources),
        ),
    )
    return score


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


def score_property(query: GoldQuery, result: SearchResult, matching: set[str]) -> PropertyScore:
    """Посчитать долю выдачи, обладающую запрошенным свойством.

    Args:
        query: эталонный запрос.
        result: выдача поиска.
        matching: коды из выдачи, удовлетворяющие условию запроса. Считаются
            снаружи одним запросом в базу — проверять свойство здесь значило
            бы тащить БД в модуль метрик.
    """
    return PropertyScore(
        query=query.query,
        kind=query.kind,
        returned=len(result.codes),
        matching=len(set(result.codes) & matching),
        latency_s=result.latency_s,
    )


def score_language(query: GoldQuery, answer: RagAnswer, *, min_letters: int) -> LanguageScore:
    """Сравнить язык ответа с языком вопроса.

    Отказ не оценивается по языку: текст отказа — константа проекта,
    а не выбор модели, и засчитывать его значило бы мерить собственную
    строку. Такие записи получают `UNDETERMINED` и уходят в «не
    определено», а не в промахи.
    """
    actual = (
        UNDETERMINED if answer.refused else detect_language(answer.text, min_letters=min_letters)
    )
    score = LanguageScore(question=query.query, expected=query.lang, actual=actual)
    logger.info(
        "Язык ответа проверен",
        extra=safe_extra(
            question=query.query[:120],
            expected=score.expected or "не задан",
            actual=score.actual or "не определено",
            matched=score.matched,
            refused=answer.refused,
        ),
    )
    return score


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


def _language_section(report: RetrievalReport) -> list[str]:
    """Раздел отчёта о языке ответа.

    Отдельной функцией, а не ещё сотней строк в `format_report`: раздел
    целиком опциональный — без прогона RAG его в отчёте нет вовсе.
    """
    if not report.languages:
        return []

    comparable = report.comparable_languages
    lines = [
        "",
        "## Язык ответа",
        "",
        "| Величина | Значение |",
        "|---|---|",
        f"| Ответов на языке вопроса | {report.language_match_share:.1%} |",
        f"| Запросов в знаменателе | {len(comparable)} из {len(report.languages)} |",
        f"| Язык ответа не определился | {report.language_undetermined} |",
        f"| Язык вопроса неразличим признаком | {report.language_unresolvable} |",
        "",
        "Правило «отвечай на языке вопроса» **уже есть в промпте** — пунктом 5. "
        "Поэтому число выше измеряет не отсутствие инструкции, а то, как часто "
        "модель на 3B её игнорирует. Это разные вещи: первое чинится строкой "
        "в промпте, второе — только структурным принуждением или сменой модели.",
        "",
        "Язык определяется по алфавиту — кириллица против латиницы. На паре "
        "«русский и английский» признак не ошибается: у них не пересекаются "
        "буквы. Ценой узости, и она здесь не теоретическая: в эталоне поиска "
        "есть запросы на немецком и французском, и на них признак отвечает "
        "«английский» всегда — алфавит общий. Такие запросы выведены из доли "
        "целиком, отдельной строкой таблицы: записать их в промахи значило бы "
        "измерить ограничение прибора, а не систему. Ответы, где букв меньше "
        "порога, тоже в долю не входят — иначе метрика мерила бы длину ответа. "
        "Язык вопроса берётся из поля `lang` эталона, а не угадывается.",
    ]

    mismatched = [score for score in comparable if not score.matched]
    if mismatched:
        lines += [
            "",
            "### Ответы не на языке вопроса",
            "",
            "| Вопрос | Спросили на | Ответили на |",
            "|---|---|---|",
        ]
        lines += [
            f"| {score.question} | {score.expected} | {score.actual} |" for score in mismatched
        ]
    return lines


def _out_of_domain_section(report: RetrievalReport) -> list[str]:
    """Раздел отчёта об отказах на посторонних вопросах."""
    if not report.out_of_domain:
        return []

    lines = [
        "",
        "## Отказ на вопросах вне домена",
        "",
        "| Величина | Значение |",
        "|---|---|",
        f"| Вопросов вне домена | {len(report.out_of_domain)} |",
        f"| Честных отказов | {report.out_of_domain_refusal_share:.1%} |",
        "",
        "**Это не тот отказ, что выше.** В разделе о подтверждённости отказ "
        "означает неудачу: система не нашла того, что в базе есть. Здесь отказ "
        "— успех: вопрос про цену, доставку или личный совет по питанию лежит "
        "за границами продукта, и отвечать на него инструмент прозрачности "
        "состава не должен.",
        "",
        "Порог `MIN_SIMILARITY` такие вопросы **не ловит**. Он отсекает «в базе "
        "нет ничего похожего», а «сколько стоит хлеб» находит хлеб с высокой "
        "близостью: в базе он есть, просто вопрос не про состав. Отказ здесь "
        "обязан наступить по другой причине, и число выше показывает, "
        "наступает ли он вообще.",
    ]

    grouped: dict[str, list[RefusalScore]] = {}
    for score in report.out_of_domain:
        grouped.setdefault(score.kind or "без типа", []).append(score)
    if len(grouped) > 1:
        lines += [
            "",
            "### По категориям посторонних вопросов",
            "",
            "| Категория | Вопросов | Отказов |",
            "|---|---|---|",
        ]
        for kind, scores in sorted(grouped.items()):
            refused = sum(1 for score in scores if score.refused)
            lines.append(f"| {kind} | {len(scores)} | {refused / len(scores):.0%} |")

    answered = [score for score in report.out_of_domain if not score.refused]
    if answered:
        lines += [
            "",
            "### Вопросы, на которые система всё же ответила",
            "",
            "| Вопрос | Категория |",
            "|---|---|",
        ]
        lines += [f"| {score.question} | {score.kind or 'без типа'} |" for score in answered]
    return lines


def format_report(report: RetrievalReport) -> str:
    """Отчёт по метрикам поиска и RAG."""
    total = len(report.properties) or len(report.recalls)
    lines = [
        f"# Поиск и RAG: метрики на {total} эталонных запросах",
        "",
    ]
    if report.prompt_version:
        lines += [
            f"Версия промпта RAG: **{report.prompt_version}**. Числа привязаны "
            "к ней: смена версии делает прошлые значения несравнимыми, поэтому "
            "они пересчитываются, а не переносятся.",
            "",
        ]
    lines += [
        f"## Свойство выдачи (top-{report.k})",
        "",
        "| Величина | Значение |",
        "|---|---|",
        f"| Доля выдачи с запрошенным свойством | {report.mean_property_precision:.1%} |",
        f"| Медианная латентность | {report.median_latency * 1000:.0f} мс |",
        "",
        "**Почему не recall@k.** Под условие каждого запроса подходят тысячи "
        "продуктов: «шоколад с пальмовым маслом» — 2035, «снеки с глутаматом» — "
        "1952. Требовать, чтобы в первой пятёрке оказались именно те три, "
        "что оракул выбрал случайно из двух тысяч, — задача с нулевым решением "
        "у любого поиска, включая идеальный. Первая версия этой оценки считала "
        "именно так и дала 0,0% на всех двадцати запросах; число измеряло "
        "ошибку в методике, а не качество поиска.",
        "",
        "Правильный вопрос при тысячах верных ответов — **обладают ли найденные "
        "продукты запрошенным свойством**. Он проверяется тем же точным условием, "
        "которым определялся набор, и отвечает буквально на то, что спросил "
        "пользователь.",
    ]

    by_kind: dict[str, list[PropertyScore]] = {}
    for item in report.properties:
        by_kind.setdefault(item.kind or "без типа", []).append(item)
    if len(by_kind) > 1:
        lines += [
            "",
            "### По типам запросов",
            "",
            "| Тип | Запросов | Доля с свойством |",
            "|---|---|---|",
        ]
        for kind, items in sorted(
            by_kind.items(), key=lambda pair: -sum(e.precision for e in pair[1]) / len(pair[1])
        ):
            mean = sum(entry.precision for entry in items) / len(items)
            lines.append(f"| {kind} | {len(items)} | {mean:.1%} |")

    lines += [
        "",
        "### По запросам",
        "",
        "| Запрос | С свойством | Всего |",
        "|---|---|---|",
    ]
    for item in sorted(report.properties, key=lambda entry: -entry.precision):
        lines.append(f"| {item.query} | {item.matching} | {item.returned} |")

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

    lines += _language_section(report)
    lines += _out_of_domain_section(report)

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


async def matching_codes(
    predicate: str, codes: list[str], settings: Settings | None = None
) -> set[str]:
    """Какие из найденных продуктов удовлетворяют условию запроса.

    Условие подставляется в SQL как есть — оно приходит из файла эталона,
    который лежит в репозитории и проходит ревью вместе с кодом, а не
    из пользовательского ввода. Коды передаются параметром: они как раз
    приходят снаружи.

    Одним запросом на всю выдачу, а не по продукту: пять круговых поездок
    в базу на каждый запрос превратили бы оценку двадцати запросов в сотню.
    """
    if not codes:
        return set()

    settings = settings or get_settings()
    statement = text(f"SELECT code FROM products WHERE code = ANY(:codes) AND ({predicate})")
    async with get_session(settings.db) as session:
        rows = (await session.execute(statement, {"codes": list(codes)})).all()
    return {str(row[0]) for row in rows}
