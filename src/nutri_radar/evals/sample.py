"""Отбор выборки для ручной разметки.

Три требования, каждое из которых меняет запрос.

**Только продукты, уже разобранные локальной моделью в M2.** Иначе предсказания
систем окажутся на разных продуктах, и сравнение развалится: нельзя сказать,
что одна система лучше другой, если они отвечали на разные вопросы.

**Равные квоты по языкам**, а не пропорциональные. Пропорциональная выборка
воспроизвела бы перекос базы (fr 151k против ru 2,3k) и оставила бы на русский
пару продуктов — а вопрос «почему русские составы отстают» именно на этой
разбивке и решается.

**Детерминированность.** Порядок задаётся хешем от кода и seed, а не `random()`:
выборка обязана воспроизводиться, потому что размеченное вручную привязано
к конкретным продуктам.

Нечитаемые для модели продукты из выборки **не исключаются**. Соблазн убрать их
велик — они портят картину, — но исключение сместило бы выборку в сторону
случаев, с которыми модель справилась, и метрика польстила бы системе.
"""

from __future__ import annotations

import logging
from collections import Counter

from sqlalchemy import func, select

from nutri_radar.config import Settings, get_settings
from nutri_radar.db.models.extraction import ProductExtraction
from nutri_radar.db.models.product import Product
from nutri_radar.db.session import get_session
from nutri_radar.evals.schemas import SampleItem
from nutri_radar.logging import safe_extra

logger = logging.getLogger(__name__)


def language_quotas(languages: list[str], total: int) -> dict[str, int]:
    """Равные квоты по языкам с добором остатка на первые языки."""
    if not languages:
        return {}
    base = total // len(languages)
    remainder = total - base * len(languages)
    return {lang: base + (1 if index < remainder else 0) for index, lang in enumerate(languages)}


async def _fetch_language(
    settings: Settings,
    *,
    lang: str,
    limit: int,
    model_name: str,
    prompt_version: str,
    seed: int,
) -> list[SampleItem]:
    """Выбрать продукты одного языка среди уже разобранных."""
    if limit <= 0:
        return []

    statement = (
        select(
            Product.code,
            Product.ingredients_text_lang,
            Product.ingredients_text,
            Product.unknown_ingredients_n,
        )
        # Join, а не подзапрос: нужен именно факт наличия разбора этой моделью
        # и этой версией промпта — на других предсказаний просто нет.
        .join(ProductExtraction, ProductExtraction.code == Product.code)
        .where(
            Product.ingredients_text.is_not(None),
            Product.ingredients_text_lang == lang,
            ProductExtraction.model_name == model_name,
            ProductExtraction.prompt_version == prompt_version,
        )
        .order_by(func.md5(Product.code + str(seed)))
        .limit(limit)
    )

    async with get_session(settings.db) as session:
        rows = (await session.execute(statement)).all()

    return [
        SampleItem(
            code=row[0],
            lang=row[1] or lang,
            ingredients_text=row[2],
            unknown_ingredients_n=row[3] or 0,
        )
        for row in rows
    ]


async def select_sample(
    settings: Settings | None = None,
    *,
    size: int | None = None,
    model_name: str | None = None,
    prompt_version: str | None = None,
) -> list[SampleItem]:
    """Отобрать выборку для разметки.

    Args:
        settings: настройки; по умолчанию из `get_settings()`.
        size: размер выборки; по умолчанию `EVALS__GOLD_SIZE`.
        model_name: модель, чьи разборы уже есть; по умолчанию из `OLLAMA__MODEL`.
        prompt_version: версия промпта; по умолчанию `EXTRACT__PROMPT_VERSION`.

    Returns:
        Список продуктов, упорядоченный по коду — чтобы разметка шла
        предсказуемо, а diff файла выборки был читаемым.
    """
    settings = settings or get_settings()
    total = size or settings.evals.gold_size
    model = model_name or settings.ollama.model
    version = prompt_version or settings.extract.prompt_version
    languages = settings.ingest.languages
    quotas = language_quotas(languages, total)

    logger.info(
        "Отбор выборки для разметки начат",
        extra=safe_extra(
            target_size=total,
            languages=languages,
            quotas=quotas,
            model=model,
            prompt_version=version,
            seed=settings.evals.random_seed,
        ),
    )

    items: list[SampleItem] = []
    for lang, quota in quotas.items():
        picked = await _fetch_language(
            settings,
            lang=lang,
            limit=quota,
            model_name=model,
            prompt_version=version,
            seed=settings.evals.random_seed,
        )
        if len(picked) < quota:
            # Недобор — это факт о данных, а не о коде. Молча взять меньше
            # значит потом объяснять расхождение чисел в отчёте.
            logger.warning(
                "Квота по языку не набрана",
                extra=safe_extra(lang=lang, quota=quota, picked=len(picked)),
            )
        items.extend(picked)

    items.sort(key=lambda item: item.code)
    by_lang = Counter(item.lang for item in items)
    logger.info(
        "Выборка отобрана",
        extra=safe_extra(total=len(items), by_lang=dict(by_lang)),
    )
    return items
