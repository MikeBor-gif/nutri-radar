"""Векторизация профилей корпуса и запись в pgvector.

**Возобновляемость по хешу, а не по наличию строки.** Прогон на 146 тысячах
продуктов идёт часами и почти наверняка прервётся. Проверять «есть ли строка»
недостаточно: если правило сборки профиля изменилось, строка есть, а вектор
устарел. Пропускается только то, у чего совпал хеш профиля.

**Чтение потоком, а не целиком.** 146 тысяч профилей в памяти — это гигабайты
текста без всякой нужды: батч уходит в модель и в базу, дальше не нужен.

**Сначала замер, потом прогон.** Тот же порядок, что в M2 и M4: на двух сотнях
профилей меряются секунды на продукт, экстраполируются на корпус, и только
после этого принимается решение о его размере.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field

from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert

from nutri_radar.config import Settings, get_settings
from nutri_radar.db.models.embedding import ProductEmbedding
from nutri_radar.db.models.product import Product
from nutri_radar.db.session import get_session
from nutri_radar.llm.ports import EmbeddingModel
from nutri_radar.logging import safe_extra
from nutri_radar.retrieval.profile import PROFILE_VERSION, Profile, build_profile

logger = logging.getLogger(__name__)


@dataclass
class EmbedProgress:
    """Итог прогона. Каждое число отвечает на свой вопрос при разборе."""

    total: int = 0
    embedded: int = 0
    skipped_fresh: int = 0
    skipped_unusable: int = 0
    seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def per_product(self) -> float:
        return self.seconds / self.embedded if self.embedded else 0.0

    def format(self) -> str:
        lines = [
            f"Профилей просмотрено: {self.total}",
            f"Векторизовано:        {self.embedded}",
            f"Пропущено свежих:     {self.skipped_fresh}",
            f"Пропущено негодных:   {self.skipped_unusable}",
        ]
        if self.embedded:
            lines.append(
                f"Время:                {self.seconds / 60:.1f} мин "
                f"({self.per_product:.3f} с на продукт)"
            )
        if self.errors:
            lines.append(f"Ошибок:               {len(self.errors)}")
        return "\n".join(lines)


async def count_candidates(settings: Settings) -> int:
    """Сколько продуктов вообще пригодно к векторизации."""
    statement = (
        select(func.count())
        .select_from(Product)
        .where(
            Product.ingredients_text.is_not(None),
            func.length(Product.ingredients_text) > settings.analytics.min_text_length,
        )
    )
    async with get_session(settings.db) as session:
        return int((await session.execute(statement)).scalar_one())


async def iter_profiles(
    settings: Settings,
    *,
    limit: int | None = None,
) -> AsyncIterator[Profile]:
    """Пройти корпус, отдавая профили по одному.

    Порядок по коду, а не случайный: прогон прерывается и продолжается,
    и стабильный порядок делает прогресс предсказуемым.
    """
    statement = (
        select(
            Product.code,
            Product.product_name,
            Product.brands,
            Product.categories_tags,
            Product.ingredients_text,
        )
        .where(
            Product.ingredients_text.is_not(None),
            func.length(Product.ingredients_text) > settings.analytics.min_text_length,
        )
        .order_by(Product.code)
    )
    if limit is not None:
        statement = statement.limit(limit)

    async with get_session(settings.db) as session:
        result = await session.stream(statement)
        async for row in result:
            yield build_profile(
                row[0],
                product_name=row[1],
                brands=row[2],
                categories_tags=list(row[3] or []),
                ingredients_text=row[4],
            )


async def fresh_hashes(settings: Settings, model_name: str) -> dict[str, str]:
    """Что уже векторизовано этой моделью: код → хеш профиля.

    Читается один раз в начале прогона. Запрос на каждый батч превратил бы
    прогон в тысячи круговых поездок в базу ради того, что не меняется.
    """
    statement = select(ProductEmbedding.code, ProductEmbedding.profile_hash).where(
        ProductEmbedding.model_name == model_name
    )
    async with get_session(settings.db) as session:
        rows = (await session.execute(statement)).all()
    known = {str(code): str(profile_hash) for code, profile_hash in rows}
    logger.info(
        "Прочитано уже векторизованное",
        extra=safe_extra(model=model_name, vectors=len(known)),
    )
    return known


async def save_batch(
    settings: Settings,
    model_name: str,
    profiles: Sequence[Profile],
    vectors: Sequence[Sequence[float]],
) -> int:
    """Записать батч векторов.

    Upsert по `(code, model_name)`: перевекторизация после смены правила
    сборки обязана обновлять строку, а не падать на уникальном ограничении.
    """
    if not profiles:
        return 0

    rows = [
        {
            "code": profile.code,
            "model_name": model_name,
            "profile_version": profile.version,
            "profile_hash": profile.hash,
            "embedding": list(vector),
        }
        for profile, vector in zip(profiles, vectors, strict=True)
    ]
    statement = insert(ProductEmbedding).values(rows)
    statement = statement.on_conflict_do_update(
        constraint="uq_product_embedding_code_model",
        set_={
            "profile_version": statement.excluded.profile_version,
            "profile_hash": statement.excluded.profile_hash,
            "embedding": statement.excluded.embedding,
            "created_at": func.now(),
        },
    )
    async with get_session(settings.db) as session:
        await session.execute(statement)
        await session.commit()
    return len(rows)


async def embed_corpus(
    model: EmbeddingModel,
    settings: Settings | None = None,
    *,
    limit: int | None = None,
) -> EmbedProgress:
    """Векторизовать корпус, продолжая с прерванного места.

    Args:
        model: модель эмбеддингов через порт.
        settings: настройки.
        limit: ограничить число просмотренных продуктов. Не «сколько
            векторизовать»: пропущенные свежие тоже считаются просмотренными,
            иначе повторный запуск с тем же лимитом бесконечно перебирал бы
            один и тот же хвост.
    """
    settings = settings or get_settings()
    cfg = settings.retrieval
    known = await fresh_hashes(settings, model.model_name)

    progress = EmbedProgress()
    pending: list[Profile] = []
    to_write: list[tuple[Profile, list[float]]] = []
    started = time.perf_counter()

    async def flush_model() -> None:
        """Отправить накопленный батч в модель."""
        if not pending:
            return
        vectors = await model.embed([profile.text for profile in pending])
        to_write.extend(zip(pending, vectors, strict=True))
        progress.embedded += len(pending)
        pending.clear()

    async def flush_db() -> None:
        """Записать накопленное в базу."""
        if not to_write:
            return
        await save_batch(
            settings,
            model.model_name,
            [profile for profile, _ in to_write],
            [vector for _, vector in to_write],
        )
        elapsed = time.perf_counter() - started
        logger.info(
            "Векторизация идёт",
            extra=safe_extra(
                embedded=progress.embedded,
                seen=progress.total,
                elapsed_min=round(elapsed / 60, 1),
                per_product=round(elapsed / progress.embedded, 3) if progress.embedded else 0,
            ),
        )
        to_write.clear()

    logger.info(
        "Векторизация начата",
        extra=safe_extra(model=model.model_name, profile_version=PROFILE_VERSION, limit=limit),
    )

    async for profile in iter_profiles(settings, limit=limit):
        progress.total += 1

        if not profile.is_usable:
            progress.skipped_unusable += 1
            continue
        if known.get(profile.code) == profile.hash:
            progress.skipped_fresh += 1
            continue

        pending.append(profile)
        if len(pending) >= cfg.embed_batch_size:
            await flush_model()
        if len(to_write) >= cfg.db_batch_size:
            await flush_db()

    await flush_model()
    await flush_db()

    progress.seconds = time.perf_counter() - started
    logger.info(
        "Векторизация завершена",
        extra=safe_extra(
            model=model.model_name,
            embedded=progress.embedded,
            skipped_fresh=progress.skipped_fresh,
            skipped_unusable=progress.skipped_unusable,
            minutes=round(progress.seconds / 60, 1),
        ),
    )
    return progress


async def build_index(settings: Settings | None = None) -> str:
    """Построить индекс HNSW поверх залитых векторов.

    **Отдельным шагом, а не миграцией.** На заполненной таблице граф
    получается лучше, а сборка быстрее: миграция создала бы индекс на пустой
    таблице, и он дорастал бы по одному вектору при каждой вставке.

    `maintenance_work_mem` поднимается на время сборки: дефолт Postgres —
    64 МБ, и на 146 тысячах векторов сборка превращается в часы дискового
    шуршания вместо минут работы в памяти.

    Класс операций `halfvec_cosine_ops` обязан соответствовать оператору
    в запросе (`<=>`). Несовпадение — отказ, который не виден по результату:
    запрос вернёт правильный ответ полным перебором.
    """
    settings = settings or get_settings()
    cfg = settings.retrieval
    name = "ix_product_embedding_hnsw"

    started = time.perf_counter()
    async with get_session(settings.db) as session:
        await session.execute(text(f"SET maintenance_work_mem = '{cfg.maintenance_work_mem}'"))
        await session.execute(text("SET max_parallel_maintenance_workers = 4"))
        logger.info(
            "Сборка индекса HNSW начата",
            extra=safe_extra(
                m=cfg.hnsw_m,
                ef_construction=cfg.hnsw_ef_construction,
                maintenance_work_mem=cfg.maintenance_work_mem,
            ),
        )
        await session.execute(
            text(
                f"CREATE INDEX IF NOT EXISTS {name} ON product_embedding "
                f"USING hnsw (embedding halfvec_cosine_ops) "
                f"WITH (m = {int(cfg.hnsw_m)}, "
                f"ef_construction = {int(cfg.hnsw_ef_construction)})"
            )
        )
        await session.commit()
        size = (
            await session.execute(text(f"SELECT pg_size_pretty(pg_relation_size('{name}'))"))
        ).scalar_one()

    minutes = (time.perf_counter() - started) / 60
    logger.info(
        "Индекс HNSW построен",
        extra=safe_extra(name=name, size=str(size), minutes=round(minutes, 1)),
    )
    return f"Индекс {name} построен за {minutes:.1f} мин, размер {size}."
