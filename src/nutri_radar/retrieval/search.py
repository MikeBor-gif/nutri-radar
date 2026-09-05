"""Семантический поиск по векторам профилей.

**Косинусное расстояние `<=>` и явный каст к `halfvec(1024)`.** Оператор
обязан соответствовать классу операций индекса (`halfvec_cosine_ops`),
иначе Postgres **молча** не использует индекс: запрос отработает и вернёт
правильный ответ полным перебором на 146 тысячах векторов. Отказ, который
видно только по латентности, — поэтому есть `explain_search`.

**Фильтры и обрыв обхода HNSW.** Под `WHERE` индекс по умолчанию
останавливается, набрав своё число кандидатов, и если фильтр отсёк
большинство, результатов окажется меньше запрошенных. Лечится
`hnsw.iterative_scan`, но включать его всегда значит платить латентностью
там, где фильтр широкий. Правило выбора — в `needs_iterative_scan`.

**`ef_search` не меньше `LIMIT`.** Иначе список кандидатов короче
запрошенного числа результатов, и часть выдачи теряется на ровном месте.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from sqlalchemy import Row, String, bindparam, select, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.sql import Select

from nutri_radar.config import Settings, get_settings
from nutri_radar.db.models.embedding import EMBEDDING_DIM, ProductEmbedding
from nutri_radar.db.models.product import Product
from nutri_radar.db.session import get_session
from nutri_radar.logging import safe_extra

logger = logging.getLogger(__name__)

# Доля корпуса, ниже которой фильтр считается узким. Не из головы:
# на широком фильтре обход HNSW и так набирает кандидатов, а iterative_scan
# только добавляет латентность. Порог грубый намеренно — точное значение
# зависит от распределения, и подбирать его до первых замеров бессмысленно.
NARROW_FILTER_SHARE = 0.1

# Режим итеративного обхода. `relaxed_order` быстрее `strict_order` и не
# требует строгой сортировки внутри обхода: финальный `ORDER BY` всё равно
# пересортирует результат.
ITERATIVE_SCAN_MODE = "relaxed_order"


@dataclass(frozen=True)
class SearchFilters:
    """Чем можно сузить выдачу.

    Фильтры живут в SQL, а не в тексте профиля: по нутриентам и оценке ищут
    числом, а не смыслом, и подмешивать их в эмбеддинг значило бы получить
    поиск, возвращающий продукты по совпадению оценки (см. `profile.py`).
    """

    lang: str | None = None
    category: str | None = None
    grade_in: tuple[str, ...] = ()
    nova_in: tuple[int, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not (self.lang or self.category or self.grade_in or self.nova_in)

    def describe(self) -> str:
        parts = []
        if self.lang:
            parts.append(f"язык={self.lang}")
        if self.category:
            parts.append(f"категория={self.category}")
        if self.grade_in:
            parts.append(f"оценка∈{{{','.join(self.grade_in)}}}")
        if self.nova_in:
            parts.append(f"nova∈{{{','.join(map(str, self.nova_in))}}}")
        return ", ".join(parts) or "без фильтров"


@dataclass(frozen=True)
class SearchHit:
    """Один найденный продукт."""

    code: str
    product_name: str | None
    brands: str | None
    ingredients_text: str | None
    lang: str | None
    nutriscore_grade: str | None
    nova_group: int | None
    distance: float

    @property
    def similarity(self) -> float:
        """Косинусная близость. Читается человеком лучше расстояния."""
        return 1.0 - self.distance


@dataclass
class SearchResult:
    """Выдача вместе с тем, как она получена."""

    query: str
    hits: list[SearchHit] = field(default_factory=list)
    filters: SearchFilters = field(default_factory=SearchFilters)
    iterative_scan: bool = False
    latency_s: float = 0.0

    @property
    def is_empty(self) -> bool:
        return not self.hits

    @property
    def codes(self) -> list[str]:
        return [hit.code for hit in self.hits]


def needs_iterative_scan(filters: SearchFilters) -> bool:
    """Включать ли итеративный обход HNSW.

    Под `WHERE` индекс обрывает обход, набрав своё число кандидатов, и при
    узком фильтре выдача оказывается короче запрошенной. Но на широком
    фильтре итеративный обход только добавляет латентность, ничего не
    исправляя.

    Правило: включаем, когда фильтр сужает по значению, которое отсекает
    большую часть корпуса, — язык, категория или конкретные классы оценки.
    Пустой фильтр обходится без него всегда.
    """
    # Любой заданный фильтр здесь узкий по построению: язык отсекает
    # 50–99% корпуса, категория — больше, перечисление классов оценки —
    # тем более. Порог `NARROW_FILTER_SHARE` остаётся документированной
    # величиной на случай, когда появятся широкие фильтры и правило
    # придётся уточнять числом, а не перечнем полей.
    return not filters.is_empty


def build_search_statement(
    embedding: list[float],
    *,
    limit: int,
    filters: SearchFilters,
) -> Select:
    """Собрать запрос поиска.

    Каст к `halfvec(1024)` явный: без него в подготовленном запросе тип
    выводится как `vector`, класс операций индекса не совпадает, и Postgres
    молча уходит в полный перебор.
    """
    # Тип параметра объявляется явно. Без него плейсхолдер не имеет типа,
    # и компиляция с литералами (нужная для `EXPLAIN`) падает: рендерить
    # значение неизвестного типа SQLAlchemy отказывается. С `String` вектор
    # становится обычной строкой, а `CAST` превращает её в `halfvec`.
    query_vector = text(f"CAST(:query_vec AS halfvec({EMBEDDING_DIM}))").bindparams(
        bindparam("query_vec", type_=String)
    )
    distance = ProductEmbedding.embedding.cosine_distance(query_vector).label("distance")

    statement = (
        select(
            Product.code,
            Product.product_name,
            Product.brands,
            Product.ingredients_text,
            Product.ingredients_text_lang,
            Product.nutriscore_grade,
            Product.nova_group,
            distance,
        )
        .join(ProductEmbedding, ProductEmbedding.code == Product.code)
        .order_by(distance)
        .limit(limit)
    )

    if filters.lang:
        statement = statement.where(Product.ingredients_text_lang == filters.lang)
    if filters.category:
        statement = statement.where(Product.categories_tags.any(filters.category))
    if filters.grade_in:
        statement = statement.where(Product.nutriscore_grade.in_(filters.grade_in))
    if filters.nova_in:
        statement = statement.where(Product.nova_group.in_(filters.nova_in))

    return statement.params(query_vec=str(list(embedding)))


def _to_hit(row: Row) -> SearchHit:
    return SearchHit(
        code=str(row[0]),
        product_name=row[1],
        brands=row[2],
        ingredients_text=row[3],
        lang=row[4],
        nutriscore_grade=row[5],
        nova_group=row[6],
        distance=float(row[7]),
    )


async def search(
    embedding: list[float],
    *,
    query: str = "",
    limit: int | None = None,
    filters: SearchFilters | None = None,
    settings: Settings | None = None,
) -> SearchResult:
    """Найти ближайшие продукты по вектору запроса.

    Args:
        embedding: вектор запроса, посчитанный ТОЙ ЖЕ моделью, что и профили.
            Сравнивать векторы разных моделей бессмысленно — это разные
            пространства.
        query: исходный текст запроса. Нужен только для отчёта и логов.
        limit: сколько вернуть; по умолчанию из настроек.
        filters: чем сузить выдачу.
    """
    settings = settings or get_settings()
    cfg = settings.retrieval
    top_k = limit or cfg.top_k
    filters = filters or SearchFilters()

    # `ef_search` не меньше LIMIT: иначе список кандидатов короче
    # запрошенного числа результатов, и выдача теряется на ровном месте.
    ef_search = max(cfg.hnsw_ef_search, top_k)
    iterative = needs_iterative_scan(filters)

    statement = build_search_statement(embedding, limit=top_k, filters=filters)

    started = time.perf_counter()
    async with get_session(settings.db) as session:
        await session.execute(text(f"SET LOCAL hnsw.ef_search = {int(ef_search)}"))
        if iterative:
            await session.execute(text(f"SET LOCAL hnsw.iterative_scan = {ITERATIVE_SCAN_MODE}"))
        rows = (await session.execute(statement)).all()
    latency = time.perf_counter() - started

    result = SearchResult(
        query=query,
        hits=[_to_hit(row) for row in rows],
        filters=filters,
        iterative_scan=iterative,
        latency_s=latency,
    )
    logger.info(
        "Поиск выполнен",
        extra=safe_extra(
            found=len(result.hits),
            requested=top_k,
            filters=filters.describe(),
            iterative_scan=iterative,
            ef_search=ef_search,
            latency_s=round(latency, 3),
        ),
    )
    if len(result.hits) < top_k and not filters.is_empty:
        # Признак того, что обход оборвался: под фильтром это штатная
        # ситуация, но знать о ней надо — она портит recall@k.
        logger.warning(
            "Найдено меньше запрошенного под фильтром",
            extra=safe_extra(found=len(result.hits), requested=top_k),
        )
    return result


async def explain_search(
    embedding: list[float],
    *,
    limit: int | None = None,
    filters: SearchFilters | None = None,
    settings: Settings | None = None,
) -> str:
    """План запроса. Нужен, чтобы убедиться, что индекс используется.

    Несовпадение оператора с классом операций индекса — отказ, который
    не виден по результату: запрос вернёт правильный ответ полным перебором
    по 146 тысячам векторов. Проверять это на веру нельзя, поэтому план
    достаётся кодом, а не глазами в psql.

    Запрос компилируется с литералами, а не с плейсхолдерами: `EXPLAIN`
    внутри `text()` теряет привязку параметров, и сервер получает запрос
    с `$1`, но без аргументов. Для диагностики литеральный вектор
    приемлем — он не уходит в горячий путь.
    """
    settings = settings or get_settings()
    top_k = limit or settings.retrieval.top_k
    filters = filters or SearchFilters()
    statement = build_search_statement(embedding, limit=top_k, filters=filters)

    compiled = statement.compile(
        dialect=postgresql.dialect(),
        compile_kwargs={"literal_binds": True},
    )
    async with get_session(settings.db) as session:
        await session.execute(
            text(f"SET LOCAL hnsw.ef_search = {int(settings.retrieval.hnsw_ef_search)}")
        )
        if needs_iterative_scan(filters):
            await session.execute(text(f"SET LOCAL hnsw.iterative_scan = {ITERATIVE_SCAN_MODE}"))
        rows = (await session.execute(text(f"EXPLAIN {compiled}"))).all()
    return "\n".join(str(row[0]) for row in rows)


def uses_index(plan: str) -> bool:
    """Виден ли в плане обход индекса.

    Отдельная функция, потому что на это опирается тест: «поиск работает»
    и «поиск работает через индекс» — разные утверждения, и второе
    проверяется только планом.
    """
    return "Index Scan" in plan and "ix_product_embedding_hnsw" in plan


__all__ = [
    "SearchFilters",
    "SearchHit",
    "SearchResult",
    "build_search_statement",
    "explain_search",
    "needs_iterative_scan",
    "search",
    "uses_index",
]
