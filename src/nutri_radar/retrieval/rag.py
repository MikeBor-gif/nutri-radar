"""Ответ на вопрос строго по найденным продуктам.

**Отказ «не знаю» — главное свойство, а не запасной путь.** Инструмент
прозрачности, который на любой вопрос выдаёт правдоподобный текст, хуже
отсутствия инструмента: он выглядит одинаково уверенно и когда данные есть,
и когда их нет. Поэтому пустая выдача поиска **не доходит до модели вообще**:
отказ формируется кодом, а не просьбой в промпте. Просить модель не выдумывать
и надеяться — это не проверяемое свойство.

**В промпт уходят только найденные продукты.** Никакой общей эрудиции модели:
если ответа нет в выданных составах, его нет.

**Ссылки на штрихкоды обязательны.** Утверждение без ссылки нечем проверить,
а проверяемость — весь смысл проекта. Доля подтверждённых ссылок считается
отдельно (`metrics.py`).

**Формулировки описательные.** Границы продукта из брифа: «в составе три
разные формы сахара», а не «вредно». Правило записано в промпт и проверяется
тестом на запрещённых словах.

**Язык ответа принуждается схемой, а не просьбой — начиная с `rag_v2`.**
Правило «отвечай на языке вопроса» в `rag_v1` было и раньше, пунктом 5.
Замер показал, что модель на 3B нарушает его в 23,5% случаев, и все
нарушения одного вида: русский вопрос, английский ответ. Поэтому в схему
ответа добавлено поле `language`, которое модель обязана заполнить **до**
`answer` — тот же приём, что вытащил извлечение на M2: схема как параметр
генерации надёжнее просьбы в тексте.

Принуждение асимметрично, и это не небрежность. Кириллица в вопросе
определяет язык однозначно — ни один латинский язык её не использует,
— и схема сужается до `enum: ["ru"]`. Вопрос на латинице может быть
английским, немецким или французским, различить их алфавитом нельзя,
и навязать ему «английский» значило бы сделать хуже, чем `rag_v1`:
там модель хотя бы имела шанс ответить по-немецки. Для таких вопросов
поле `language` **присутствует, но свободной строкой**: модель обязана
назвать язык до ответа, а какой именно — решает сама.

Первая версия этого кода поле для латиницы не добавляла вовсе, хотя
промпт его требовал, а описание обещало «свободную строку». Промпт
просил поле, которого схема не допускала, и принуждения для двух третей
языков корпуса не было ни в каком виде. Чтобы такое не повторилось,
текст правила и разрешённые значения поля возвращает **одна функция**
(`language_rule`), а требование про JSON живёт внутри этого текста,
а не отдельным абзацем промпта: разойтись им теперь негде.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from nutri_radar.config import Settings, get_settings
from nutri_radar.errors import ExtractionError, LLMUnavailableError
from nutri_radar.llm.ports import StructuredLLM
from nutri_radar.logging import safe_extra
from nutri_radar.retrieval.language import LANGUAGE_NAMES, is_cyrillic
from nutri_radar.retrieval.prompts import load_prompt
from nutri_radar.retrieval.search import SearchHit, SearchResult

logger = logging.getLogger(__name__)

# Текст отказа. Константа, а не строка в промпте: отказ формируется кодом,
# когда выдача пуста, и должен читаться одинаково независимо от того,
# дошло ли дело до модели.
REFUSAL = "Не знаю: в базе не нашлось продуктов, по которым можно ответить на этот вопрос."

# Ниже этой близости продукт считается нерелевантным. Косинусная близость
# 0,5 на bge-m3 — это уже «что-то общее по теме», а не ответ на вопрос.
# Величина подобрана грубо и намеренно: точное значение выводится из recall,
# а его считать не на чем до эталонных запросов.
MIN_SIMILARITY = 0.5

_BARCODE = re.compile(r"\[(\d{6,14})\]")


@dataclass
class RagAnswer:
    """Ответ вместе с тем, на чём он построен."""

    question: str
    text: str
    sources: list[str] = field(default_factory=list)
    refused: bool = False
    prompt_version: str = ""
    model_name: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0

    @property
    def cited(self) -> list[str]:
        """Штрихкоды, на которые модель действительно сослалась."""
        return sorted(set(_BARCODE.findall(self.text)))

    @property
    def cited_outside_sources(self) -> list[str]:
        """Ссылки на то, чего в выдаче не было.

        Самый опасный вид ошибки: модель придумала штрихкод. Выглядит
        как подтверждённое утверждение, а проверить может только тот,
        кто пойдёт в базу.
        """
        known = set(self.sources)
        return [code for code in self.cited if code not in known]


def format_products(hits: list[SearchHit]) -> str:
    """Собрать блок продуктов для промпта.

    Штрихкод идёт первым и в тех же скобках, что модель обязана
    воспроизвести: чем ближе форма ссылки в промпте к требуемой в ответе,
    тем реже модель изобретает свою.
    """
    blocks = []
    for hit in hits:
        lines = [f"[{hit.code}] {hit.product_name or 'без названия'}"]
        if hit.brands:
            lines.append(f"  бренд: {hit.brands.split(',')[0].strip()}")
        if hit.nutriscore_grade:
            lines.append(f"  оценка по базе: {hit.nutriscore_grade}")
        if hit.nova_group:
            lines.append(f"  группа переработки NOVA: {hit.nova_group}")
        if hit.ingredients_text:
            lines.append(f"  состав: {hit.ingredients_text}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def language_rule(question: str) -> tuple[str, list[str]]:
    """Правило о языке для промпта и допустимые значения поля `language`.

    Returns:
        Пара «текст правила, список разрешённых языков». Пустой список
        означает «язык вопроса неизвестен»: поле остаётся свободным.

    Note:
        Возвращается ровно то, что уходит в промпт и в схему, — одной
        функцией, чтобы они не могли разойтись. Промпт, называющий
        русский, и схема, разрешающая что угодно, дали бы принуждение
        только на бумаге.
    """
    fill_first = (
        "Fill the `language` field of your JSON reply FIRST, before you write "
        "a single word of `answer`, and then write `answer` in that language."
    )
    if not is_cyrillic(question):
        return (
            "Answer in the same language as the question: if it is in German, "
            "answer in German; if in French, answer in French; if in English, "
            f"answer in English. {fill_first}",
            [],
        )
    name = LANGUAGE_NAMES["ru"]
    return (
        f"The question is written in {name}. Your answer MUST be in {name}. {fill_first}",
        ["ru"],
    )


def answer_schema(
    allowed_languages: list[str] | None = None, *, require_language: bool = False
) -> dict[str, Any]:
    """JSON-схема ответа.

    Args:
        allowed_languages: чем ограничить поле `language`. Пустой список
            или `None` — поле остаётся свободной строкой.
        require_language: добавлять ли поле `language` вообще. Отдельный
            флаг, а не «непустой список», потому что «поле есть, значения
            любые» и «поля нет» — разные вещи, и различать их по пустоте
            списка означало бы молча отключать принуждение там, где язык
            вопроса просто не определился. Ровно так и вышло в первой
            версии: для латиницы поля не было, хотя промпт его требовал.

    Note:
        Порядок ключей здесь значим. Модель заполняет поля в том порядке,
        в котором они объявлены, и `language` перед `answer` заставляет
        её назвать язык до того, как она напишет первое слово текста.
        Поменять местами значило бы получить поле-отметку задним числом.
    """
    properties: dict[str, Any] = {}
    required: list[str] = []
    if require_language:
        language: dict[str, Any] = {"type": "string"}
        if allowed_languages:
            language["enum"] = list(allowed_languages)
        properties["language"] = language
        required.append("language")
    properties["answer"] = {"type": "string"}
    # Модель обязана перечислить использованные штрихкоды отдельным
    # полем, а не только в тексте. Расхождение между полем и текстом
    # — сигнал, что ссылки в тексте выдуманы.
    properties["sources"] = {"type": "array", "items": {"type": "string"}}
    required += ["answer", "sources"]
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def relevant_hits(
    result: SearchResult, *, min_similarity: float = MIN_SIMILARITY
) -> list[SearchHit]:
    """Отсеять то, что нашлось, но не относится к вопросу.

    Векторный поиск возвращает ближайшее всегда — даже когда ничего
    подходящего нет. Без порога «не знаю» не наступит никогда: пять
    случайных продуктов всегда найдутся.
    """
    return [hit for hit in result.hits if hit.similarity >= min_similarity]


async def answer(
    question: str,
    result: SearchResult,
    llm: StructuredLLM,
    settings: Settings | None = None,
    *,
    min_similarity: float = MIN_SIMILARITY,
) -> RagAnswer:
    """Ответить на вопрос по найденным продуктам.

    Отказ формируется кодом при пустой или нерелевантной выдаче — модель
    об этом даже не спрашивается. Просьба «не выдумывай» в промпте
    не является проверяемым свойством системы.
    """
    settings = settings or get_settings()
    version = settings.retrieval.rag_prompt_version
    hits = relevant_hits(result, min_similarity=min_similarity)

    if not hits:
        logger.info(
            "RAG отказался: релевантного не нашлось",
            extra=safe_extra(
                question=question[:120],
                prompt_version=version,
                found=len(result.hits),
                min_similarity=min_similarity,
            ),
        )
        return RagAnswer(
            question=question,
            text=REFUSAL,
            refused=True,
            prompt_version=version,
            model_name=llm.model_name,
        )

    template = load_prompt(version)
    rule, allowed = language_rule(question) if template.forces_language else ("", [])
    prompt = template.render(question=question, products=format_products(hits), language_rule=rule)
    schema = answer_schema(allowed, require_language=template.forces_language)
    if template.forces_language:
        logger.debug(
            "Язык ответа принуждается схемой",
            extra=safe_extra(
                prompt_version=version,
                allowed=allowed or "любой",
                question=question[:120],
            ),
        )

    try:
        response = await llm.generate(prompt, json_schema=schema)
    except (LLMUnavailableError, ExtractionError) as exc:
        logger.warning(
            "Модель не ответила — отказ вместо выдумки",
            extra=safe_extra(
                question=question[:120],
                prompt_version=version,
                error=type(exc).__name__,
            ),
        )
        return RagAnswer(
            question=question,
            text=REFUSAL,
            sources=[hit.code for hit in hits],
            refused=True,
            prompt_version=version,
            model_name=llm.model_name,
        )

    text = str(response.raw_json.get("answer", "")).strip()
    if not text:
        # Пустой ответ при валидной схеме — тот же отказ, только молчаливый.
        # Выдать его как ответ значило бы показать пользователю пустоту.
        return RagAnswer(
            question=question,
            text=REFUSAL,
            sources=[hit.code for hit in hits],
            refused=True,
            prompt_version=version,
            model_name=llm.model_name,
        )

    answer_obj = RagAnswer(
        question=question,
        text=text,
        sources=[hit.code for hit in hits],
        refused=False,
        prompt_version=version,
        model_name=llm.model_name,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        latency_s=response.latency_s,
    )
    invented = answer_obj.cited_outside_sources
    if invented:
        # Не исправляем и не прячем: выдуманный штрихкод — это факт о работе
        # системы, и он обязан попасть в метрику, а не быть отфильтрован.
        logger.warning(
            "Модель сослалась на штрихкоды вне выдачи",
            extra=safe_extra(question=question[:120], invented=invented),
        )
    logger.info(
        "RAG ответил",
        # Версия промпта — в КАЖДОЙ записи, а не только в шапке прогона.
        # Два прогона разных версий дают одинаковые с виду строки лога,
        # и разобрать потом, где чьё, можно только по этому полю.
        extra=safe_extra(
            question=question[:120],
            prompt_version=version,
            sources=len(answer_obj.sources),
            cited=len(answer_obj.cited),
            invented=len(invented),
            latency_s=round(answer_obj.latency_s, 2),
        ),
    )
    return answer_obj
