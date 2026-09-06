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
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from nutri_radar.config import Settings, get_settings
from nutri_radar.errors import ExtractionError, LLMUnavailableError
from nutri_radar.llm.ports import StructuredLLM
from nutri_radar.logging import safe_extra
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

    prompt = load_prompt(version).render(question=question, products=format_products(hits))
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            # Модель обязана перечислить использованные штрихкоды отдельным
            # полем, а не только в тексте. Расхождение между полем и текстом
            # — сигнал, что ссылки в тексте выдуманы.
            "sources": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["answer", "sources"],
        "additionalProperties": False,
    }

    try:
        response = await llm.generate(prompt, json_schema=schema)
    except (LLMUnavailableError, ExtractionError) as exc:
        logger.warning(
            "Модель не ответила — отказ вместо выдумки",
            extra=safe_extra(question=question[:120], error=type(exc).__name__),
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
        extra=safe_extra(
            question=question[:120],
            sources=len(answer_obj.sources),
            cited=len(answer_obj.cited),
            invented=len(invented),
            latency_s=round(answer_obj.latency_s, 2),
        ),
    )
    return answer_obj
