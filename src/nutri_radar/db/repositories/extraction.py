"""Запись и чтение результатов извлечения.

Отвечает за два требования DoD майлстоуна сразу:

1. **Повторный запуск не создаёт дублей.** Upsert по
   `(code, model_name, prompt_version)` через `ON CONFLICT`, а не «сначала
   SELECT, потом INSERT»: последнее дало бы гонку при параллельных прогонах
   и лишний запрос на каждую строку.
2. **Возобновляемость.** `extracted_codes` отвечает на вопрос «что этой
   версией промпта уже разобрано». Условие живёт в запросе, а не в состоянии
   процесса, поэтому перезапуск просто продолжает с места обрыва.

Наружу отдаются Pydantic-модели и простые типы, ORM-объекты не покидают модуль
(см. ARCHITECTURE.md, антипаттерн «ORM-объекты за пределами db/»).
"""

from __future__ import annotations

import logging
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import CursorResult, func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from nutri_radar.db.models.extraction import ProductExtraction
from nutri_radar.db.models.product import Product
from nutri_radar.logging import safe_extra

logger = logging.getLogger(__name__)

# Предел числа аргументов одного запроса в протоколе Postgres (int16). asyncpg
# сообщает о нём как «the number of query arguments cannot exceed 32767».
# Ограничение на АРГУМЕНТЫ, а не на строки, поэтому размер куска считается
# от числа колонок таблицы.
_MAX_QUERY_ARGS = 32767

# Колонки, которые не перезаписываются при конфликте: ключ связки и суррогатный id.
_CONFLICT_KEYS = ("code", "model_name", "prompt_version")


class ExtractionRow(BaseModel):
    """Одна строка `product_extraction` на границе с БД.

    Модель живёт здесь, а не в `extract/`: `db` не должна знать о конкретных
    стадиях пайплайна, иначе общий слой начнёт зависеть от слайса.
    """

    # Pydantic резервирует префикс `model_`, а колонки таблицы называются
    # `model_name` и `model_confidence`. Имена колонок важнее предупреждения.
    model_config = ConfigDict(protected_namespaces=())

    code: str
    source_lang: str | None = None

    ingredients: list[dict[str, Any]] = Field(default_factory=list)
    distinct_sugar_forms: int = 0
    e_additives_count: int = 0
    ingredients_count: int = 0
    allergens: list[str] = Field(default_factory=list)

    unreadable: bool = False
    # Значения — из `SkipReason` слайса `extract`. Здесь строка: `db` не знает
    # о стадиях пайплайна, иначе общий слой начнёт зависеть от слайса.
    skip_reason: str | None = None
    model_confidence: float | None = None

    model_name: str
    prompt_version: str

    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float | None = None


class ExtractionRepository:
    """Доступ к `product_extraction`."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert_batch(self, rows: list[ExtractionRow]) -> int:
        """Записать батч извлечений.

        Повторное извлечение той же версией промпта **обновляет** строку:
        прогон могли прервать на середине и перезапустить, и вторая попытка
        по тому же продукту должна заменить первую, а не лечь рядом.
        """
        if not rows:
            return 0

        payload = [row.model_dump() for row in rows]
        columns = len(ProductExtraction.__table__.columns)
        chunk_size = max(_MAX_QUERY_ARGS // columns, 1)

        affected = 0
        for start in range(0, len(payload), chunk_size):
            affected += await self._upsert_chunk(payload[start : start + chunk_size])

        logger.debug(
            "Батч извлечений записан",
            extra=safe_extra(rows=len(rows), affected=affected, chunk_size=chunk_size),
        )
        return affected

    async def _upsert_chunk(self, rows: list[dict[str, Any]]) -> int:
        statement = insert(ProductExtraction).values(rows)
        updatable = {
            column.name: statement.excluded[column.name]
            for column in ProductExtraction.__table__.columns
            if column.name not in _CONFLICT_KEYS and column.name not in ("id", "extracted_at")
        }
        # Время извлечения обновляется явно: иначе перезаписанная строка
        # осталась бы с датой первой попытки, и по журналу нельзя было бы
        # понять, когда результат получен на самом деле.
        updatable["extracted_at"] = func.now()

        statement = statement.on_conflict_do_update(
            index_elements=list(_CONFLICT_KEYS),
            set_=updatable,
        )
        result = await self._session.execute(statement)
        # rowcount есть у CursorResult, но статически execute объявлен как
        # Result — берём через getattr, чтобы не врать типами.
        return int(getattr(result, "rowcount", 0) or 0)

    async def latest_by_code(self, code: str) -> ExtractionRow | None:
        """Самый свежий разбор состава этого продукта. Нет разбора — `None`.

        У продукта может быть несколько извлечений: разными версиями промпта
        и разными моделями (см. докстринг модуля). Карточке нужен один,
        и берётся самый свежий по времени — не «лучший», потому что критерия
        лучшести без эталона нет, а свежесть проверяема.

        Отсутствие разбора — норма: через LLM прошло 1187 продуктов из
        147 тысяч корпуса, и у остального будут только поля Open Food Facts.
        """
        statement = (
            select(ProductExtraction)
            .where(ProductExtraction.code == code)
            .order_by(ProductExtraction.extracted_at.desc(), ProductExtraction.id.desc())
            .limit(1)
        )
        row = (await self._session.execute(statement)).scalar_one_or_none()
        if row is None:
            logger.debug("Разбора состава нет", extra=safe_extra(code=code))
            return None

        logger.debug(
            "Разбор состава прочитан",
            extra=safe_extra(
                code=code,
                model=row.model_name,
                prompt_version=row.prompt_version,
                sugar_forms=row.distinct_sugar_forms,
            ),
        )
        return ExtractionRow.model_validate(row, from_attributes=True)

    async def extracted_codes(
        self,
        codes: list[str],
        *,
        model_name: str,
        prompt_version: str,
    ) -> set[str]:
        """Какие из переданных кодов уже разобраны этой моделью и версией.

        Сердце возобновляемости. Запрос идёт по конкретному списку кодов,
        а не по всей таблице: корпус детерминирован и известен заранее,
        а таблица растёт с каждой новой версией промпта.
        """
        if not codes:
            return set()

        # Список кодов режется на куски: 3000 параметров помещаются в лимит
        # протокола, но корпус — параметр конфигурации и может вырасти.
        chunk_size = max(_MAX_QUERY_ARGS // 4, 1)
        found: set[str] = set()

        for start in range(0, len(codes), chunk_size):
            chunk = codes[start : start + chunk_size]
            statement = select(ProductExtraction.code).where(
                ProductExtraction.code.in_(chunk),
                ProductExtraction.model_name == model_name,
                ProductExtraction.prompt_version == prompt_version,
            )
            found.update((await self._session.scalars(statement)).all())

        logger.debug(
            "Проверено, что уже извлечено",
            extra=safe_extra(
                requested=len(codes),
                found=len(found),
                model=model_name,
                prompt_version=prompt_version,
            ),
        )
        return found

    async def iter_ingredients(
        self,
        *,
        model_name: str | None = None,
        prompt_version: str | None = None,
    ) -> list[tuple[list[dict[str, Any]], str | None]]:
        """Списки ингредиентов из сохранённых извлечений и язык состава.

        Нужны отчёту `extract dict unknown`: он показывает, какие имена
        словарь не закрывает, и тем самым говорит, куда его пополнять.
        Читается целиком — корпус в тысячах строк, а не в миллионах.

        Возвращается сырой JSONB, а не Pydantic-модель ингредиента: разбор
        и канонизация — дело слайса `extract`, а не слоя доступа к данным.
        """
        statement = select(ProductExtraction.ingredients, ProductExtraction.source_lang).where(
            ProductExtraction.unreadable.is_(False)
        )
        if model_name is not None:
            statement = statement.where(ProductExtraction.model_name == model_name)
        if prompt_version is not None:
            statement = statement.where(ProductExtraction.prompt_version == prompt_version)

        rows = (await self._session.execute(statement)).all()
        return [(row[0] or [], row[1]) for row in rows]

    async def iter_for_sugar_recount(
        self,
        *,
        model_name: str | None = None,
        prompt_version: str | None = None,
    ) -> list[tuple[str, list[dict[str, Any]], str | None, int, str | None]]:
        """Всё, что нужно для пересчёта числа форм сахара.

        Отдельно от `iter_ingredients`, потому что пересчёту нужны ещё код
        строки, текущее значение и ИСХОДНЫЙ ТЕКСТ состава: без кода некуда
        писать, без текущего значения нечего сравнивать, а без текста нечем
        подтвердить, что найденная форма сахара в составе действительно есть
        (ADR-035).

        Текст берётся из `products` join'ом по коду. Язык там тот же, что
        в `source_lang` извлечения, — проверено на корпусе: совпадает у всех
        1187 разборов.

        `unreadable` здесь НЕ отфильтрован, в отличие от отчёта о неизвестных
        именах: у нечитаемых составов величина тоже хранится, и оставить её
        пересчитанной наполовину значило бы держать в одной колонке числа,
        посчитанные двумя разными способами.
        """
        statement = select(
            ProductExtraction.code,
            ProductExtraction.ingredients,
            ProductExtraction.source_lang,
            ProductExtraction.distinct_sugar_forms,
            Product.ingredients_text,
        ).join(Product, Product.code == ProductExtraction.code, isouter=True)
        if model_name is not None:
            statement = statement.where(ProductExtraction.model_name == model_name)
        if prompt_version is not None:
            statement = statement.where(ProductExtraction.prompt_version == prompt_version)

        rows = (await self._session.execute(statement)).all()
        return [(row[0], row[1] or [], row[2], row[3], row[4]) for row in rows]

    async def update_sugar_forms(
        self,
        values: dict[str, int],
        *,
        model_name: str | None = None,
        prompt_version: str | None = None,
    ) -> int:
        """Переписать `distinct_sugar_forms` у перечисленных кодов.

        Одним запросом на код: строк тут тысячи, а не миллионы, и городить
        ради них батч-апдейт через VALUES значило бы усложнить на ровном месте.
        """
        if not values:
            return 0

        affected = 0
        for code, forms in values.items():
            statement = (
                update(ProductExtraction)
                .where(ProductExtraction.code == code)
                .values(distinct_sugar_forms=forms)
            )
            if model_name is not None:
                statement = statement.where(ProductExtraction.model_name == model_name)
            if prompt_version is not None:
                statement = statement.where(ProductExtraction.prompt_version == prompt_version)
            result = await self._session.execute(statement)
            # rowcount живёт на CursorResult; типовой стаб execute() отдаёт
            # Result, у которого его нет. Приведение честнее, чем ignore.
            affected += cast("CursorResult[Any]", result).rowcount or 0

        await self._session.commit()
        logger.info("Число форм сахара пересчитано", extra=safe_extra(rows=affected))
        return affected

    async def count(
        self,
        *,
        model_name: str | None = None,
        prompt_version: str | None = None,
    ) -> int:
        """Сколько строк извлечения есть для связки модель-промпт.

        Нужен приёмке: «повторный запуск не создал дублей» проверяется
        сравнением этого числа до и после второго запуска.
        """
        statement = select(func.count()).select_from(ProductExtraction)
        if model_name is not None:
            statement = statement.where(ProductExtraction.model_name == model_name)
        if prompt_version is not None:
            statement = statement.where(ProductExtraction.prompt_version == prompt_version)
        return int((await self._session.scalar(statement)) or 0)
