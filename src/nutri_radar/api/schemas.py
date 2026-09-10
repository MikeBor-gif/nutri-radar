"""Схемы запросов и ответов HTTP-API.

Отдельные модели, а не внутренние структуры слайсов наружу. Причина та же,
по которой ORM-объекты не покидают `db/`: внутренняя структура меняется
вместе с реализацией, а контракт API — нет. Отдав `SearchHit` напрямую,
проект получил бы сломанных клиентов при первом же добавлении поля
во внутренний dataclass.

**В каждом ответе есть атрибуция.** Лицензия ODbL требует указывать
источник, и ответ API — это место, где данные покидают проект.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from nutri_radar.retrieval.product_card import SOURCE_LABELS, ProductCard
from nutri_radar.retrieval.rag import RagAnswer
from nutri_radar.retrieval.search import SearchHit, SearchResult
from nutri_radar.wording import ATTRIBUTION, DISCLAIMER


class ServiceInfo(BaseModel):
    """Корневой ответ: что это и на каких данных работает."""

    name: str
    version: str
    description: str
    attribution: str = ATTRIBUTION
    disclaimer: str = DISCLAIMER


class HealthCheck(BaseModel):
    """Одна проверка готовности среды."""

    name: str
    status: str
    detail: str


class HealthResponse(BaseModel):
    """Готовность среды целиком."""

    healthy: bool
    checks: list[HealthCheck] = Field(default_factory=list)


class ProductCardResponse(BaseModel):
    """Карточка продукта."""

    code: str
    product_name: str | None = None
    brands: str | None = None
    lang: str | None = None
    ingredients_text: str | None = None

    nutriscore_grade: str | None = None
    nova_group: int | None = None

    # `None` означает «состав моделью не разбирали», и это не то же самое,
    # что 0 — «разобрали и сахара не нашли». Клиент обязан видеть разницу,
    # поэтому поле необязательное, а не заполняется нулём.
    distinct_sugar_forms: int | None = None
    e_additives_count: int | None = None
    ingredients_count: int | None = None
    allergens: list[str] = Field(default_factory=list)

    source: str
    source_label: str
    extraction_model: str | None = None
    extraction_prompt_version: str | None = None

    attribution: str = ATTRIBUTION

    @classmethod
    def from_card(cls, card: ProductCard) -> ProductCardResponse:
        return cls(
            code=card.code,
            product_name=card.product_name,
            brands=card.brands,
            lang=card.lang,
            ingredients_text=card.ingredients_text,
            nutriscore_grade=card.nutriscore_grade,
            nova_group=card.nova_group,
            distinct_sugar_forms=card.distinct_sugar_forms,
            e_additives_count=card.e_additives_count,
            ingredients_count=card.ingredients_count,
            allergens=list(card.allergens),
            source=card.source.value,
            source_label=SOURCE_LABELS[card.source],
            extraction_model=card.extraction_model,
            extraction_prompt_version=card.extraction_prompt_version,
        )


class SearchRequest(BaseModel):
    """Запрос семантического поиска."""

    query: str = Field(min_length=1, max_length=500)
    limit: int | None = Field(default=None, ge=1, le=50)

    lang: str | None = Field(default=None, max_length=8)
    category: str | None = Field(default=None, max_length=128)
    # Кортежи, а не строки через запятую: разбор строки — это логика,
    # а точка входа логики не содержит.
    grade_in: tuple[str, ...] = ()
    nova_in: tuple[int, ...] = ()


class SearchHitResponse(BaseModel):
    """Один найденный продукт."""

    code: str
    product_name: str | None = None
    brands: str | None = None
    lang: str | None = None
    nutriscore_grade: str | None = None
    nova_group: int | None = None
    ingredients_text: str | None = None
    # Близость, а не расстояние: человеку понятнее «насколько похоже»,
    # чем «насколько далеко».
    similarity: float

    @classmethod
    def from_hit(cls, hit: SearchHit) -> SearchHitResponse:
        return cls(
            code=hit.code,
            product_name=hit.product_name,
            brands=hit.brands,
            lang=hit.lang,
            nutriscore_grade=hit.nutriscore_grade,
            nova_group=hit.nova_group,
            ingredients_text=hit.ingredients_text,
            similarity=round(hit.similarity, 4),
        )


class SearchResponse(BaseModel):
    """Выдача поиска вместе с тем, как она получена."""

    query: str
    hits: list[SearchHitResponse] = Field(default_factory=list)
    filters: str
    # Признак того, что под фильтром включался итеративный обход HNSW.
    # Наружу отдаётся намеренно: он объясняет разницу в латентности между
    # одинаковыми на вид запросами.
    iterative_scan: bool = False
    latency_s: float = 0.0
    attribution: str = ATTRIBUTION

    @classmethod
    def from_result(cls, result: SearchResult) -> SearchResponse:
        return cls(
            query=result.query,
            hits=[SearchHitResponse.from_hit(hit) for hit in result.hits],
            filters=result.filters.describe(),
            iterative_scan=result.iterative_scan,
            latency_s=round(result.latency_s, 3),
        )


class AskRequest(BaseModel):
    """Вопрос о продуктах."""

    question: str = Field(min_length=1, max_length=500)
    limit: int | None = Field(default=None, ge=1, le=20)
    lang: str | None = Field(default=None, max_length=8)


class AskResponse(BaseModel):
    """Ответ, построенный строго по найденным продуктам."""

    question: str
    answer: str
    # Штрихкоды, которые получила модель. По ним ответ можно перепроверить —
    # ради этого свойства проект и существует.
    sources: list[str] = Field(default_factory=list)
    # Штрихкоды, на которые модель сослалась в тексте.
    cited: list[str] = Field(default_factory=list)
    # Ссылки на то, чего в выдаче не было. Самый опасный вид ошибки: выглядит
    # как подтверждённое утверждение. Отдаётся клиенту, а не прячется.
    invented: list[str] = Field(default_factory=list)
    refused: bool = False

    model_name: str = ""
    prompt_version: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    retrieval_latency_s: float = 0.0
    generation_latency_s: float = 0.0
    attribution: str = ATTRIBUTION

    @classmethod
    def from_answer(cls, answer: RagAnswer, *, retrieval_latency_s: float) -> AskResponse:
        return cls(
            question=answer.question,
            answer=answer.text,
            sources=list(answer.sources),
            cited=answer.cited,
            invented=answer.cited_outside_sources,
            refused=answer.refused,
            model_name=answer.model_name,
            prompt_version=answer.prompt_version,
            input_tokens=answer.input_tokens,
            output_tokens=answer.output_tokens,
            retrieval_latency_s=round(retrieval_latency_s, 3),
            generation_latency_s=round(answer.latency_s, 3),
        )


class AgentRequest(BaseModel):
    """Вопрос агенту с инструментами."""

    question: str = Field(min_length=1, max_length=500)


class AgentStep(BaseModel):
    """Один шаг агента: что он сделал и чем это кончилось."""

    number: int
    action: str
    arguments: dict[str, object] = Field(default_factory=dict)
    ok: bool | None = None
    is_final: bool = False


class AgentResponse(BaseModel):
    """Прогон агента целиком.

    Шаги отдаются наружу намеренно: агент, у которого не видно протокола, —
    это чёрный ящик, а протокол — единственное, по чему можно судить,
    решал он задачу или перебирал инструменты.
    """

    question: str
    answer: str | None = None
    stop_reason: str
    steps: list[AgentStep] = Field(default_factory=list)

    tool_calls: int = 0
    failed_calls: int = 0
    repeated_calls: int = 0
    total_tokens: int = 0
    latency_s: float = 0.0
    model_name: str = ""
    attribution: str = ATTRIBUTION
