"""Сбор предсказаний четырёх систем в одном формате.

Системы отвечают на одни и те же 100 продуктов, иначе сравнивать нечего.

**Парсер OFF читается из дампа, а не из базы, и не из того поля, что кажется
очевидным.** В `products.ingredients_tags` таксономия раскрыта до предков:
на составе «Pasteurized milk, cheese cultures, salt, enzymes» там семь тегов
вместо четырёх — к молоку добавлены `en:dairy`, к закваскам `en:ferment`
и `en:microbial-culture`. Считать предков ингредиентами значит завалить
precision парсера на ровном месте и объявить победу над соломенным чучелом.
В `ingredients_original_tags` лежат ровно те четыре, что в составе, и в том же
каноническом виде. Измерено на выборке: 26,0 тега против 16,3.

**Типы парсеру OFF не приписываются из словаря.** Он их не выдаёт — только
отдельный список `additives_tags`. Подставить типы из нашего словаря значит
дать baseline способность, которой у него нет, и мерить словарь вместо
парсера. Всё, что не помечено добавкой, идёт как `base`; низкий F1 по типам
у OFF — это факт о парсере, а не о методике.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
from sqlalchemy import select

from nutri_radar.config import Settings, get_settings
from nutri_radar.db.models.extraction import ProductExtraction
from nutri_radar.db.session import get_session
from nutri_radar.errors import ExtractionError, LLMUnavailableError
from nutri_radar.evals.schemas import PredictionRecord, SampleItem
from nutri_radar.extract.preprocess import prepare_text
from nutri_radar.extract.prompts import Prompt
from nutri_radar.extract.schemas import ExtractionResult, Ingredient, IngredientKind
from nutri_radar.llm.ports import StructuredLLM
from nutri_radar.logging import safe_extra

logger = logging.getLogger(__name__)

OFF_SYSTEM = "off-parser"

# Префикс языка в тегах OFF: `en:sugar`. Снимается перед сопоставлением —
# иначе каждый тег отличался бы от нашего имени на четыре символа.
_TAG_PREFIX_SEPARATOR = ":"


def strip_tag_prefix(tag: str) -> str:
    """`en:glucose-syrup` → `glucose syrup`."""
    _, _, name = tag.partition(_TAG_PREFIX_SEPARATOR)
    return (name or tag).replace("-", " ").strip().lower()


def off_extraction(ingredient_tags: list[str], additive_tags: list[str]) -> ExtractionResult:
    """Собрать ответ парсера OFF в общем формате.

    Типы не выдумываются: добавка — если тег есть в `additives_tags`, иначе
    `base`. Парсер OFF формы сахара не считает вовсе, и приписать ему это
    умение означало бы измерить не его.
    """
    additives = {strip_tag_prefix(tag) for tag in additive_tags}
    ingredients: list[Ingredient] = []
    for tag in ingredient_tags:
        name = strip_tag_prefix(tag)
        if not name:
            continue
        is_additive = name in additives
        ingredients.append(
            Ingredient(
                canonical_name=name,
                kind=IngredientKind.ADDITIVE if is_additive else IngredientKind.BASE,
                e_number=name.upper() if is_additive and name.startswith("e") else None,
            )
        )
    return ExtractionResult(ingredients=ingredients, allergens=[], unreadable=not ingredients)


def collect_off_predictions(
    sample: list[SampleItem],
    dump_path: Path,
) -> list[PredictionRecord]:
    """Прочитать разбор парсера OFF из дампа для продуктов выборки.

    Точечный запрос по кодам, а не перезаливка базы: поле нужно только по
    эталонной выборке, и тащить его во весь корпус ради ста продуктов —
    работа, которой майлстоун не требует.
    """
    if not dump_path.exists():
        raise FileNotFoundError(
            f"Дамп не найден: {dump_path}. Он нужен для baseline парсера OFF "
            "(поле ingredients_original_tags в базу не заливалось)."
        )

    codes = [item.code for item in sample]
    logger.info(
        "Чтение разбора OFF из дампа",
        extra=safe_extra(dump=str(dump_path), codes=len(codes)),
    )

    placeholders = ", ".join("?" for _ in codes)
    rows = duckdb.sql(
        f"""
        SELECT code, ingredients_original_tags, additives_tags
        FROM read_parquet('{dump_path.as_posix()}')
        WHERE code IN ({placeholders})
        """,
        params=codes,
    ).fetchall()

    predictions = [
        PredictionRecord(
            code=str(row[0]),
            system=OFF_SYSTEM,
            extraction=off_extraction(list(row[1] or []), list(row[2] or [])),
        )
        for row in rows
    ]
    logger.info(
        "Разбор OFF собран",
        extra=safe_extra(found=len(predictions), requested=len(codes)),
    )
    return predictions


async def collect_local_predictions(
    sample: list[SampleItem],
    settings: Settings | None = None,
    *,
    model_name: str | None = None,
    prompt_version: str | None = None,
) -> list[PredictionRecord]:
    """Забрать разбор локальной модели из базы.

    Заново не прогоняем: M2 уже отработал эти продукты, и повторный прогон
    дал бы другие числа из-за разброса между запусками (ADR-018) — сравнение
    поехало бы по причине, не имеющей отношения к качеству систем.
    """
    settings = settings or get_settings()
    model = model_name or settings.ollama.model
    version = prompt_version or settings.extract.prompt_version
    codes = [item.code for item in sample]

    statement = select(ProductExtraction).where(
        ProductExtraction.code.in_(codes),
        ProductExtraction.model_name == model,
        ProductExtraction.prompt_version == version,
    )
    async with get_session(settings.db) as session:
        rows = (await session.execute(statement)).scalars().all()

    predictions = [
        PredictionRecord(
            code=row.code,
            system=model,
            extraction=ExtractionResult.model_validate(
                {
                    "ingredients": row.ingredients,
                    "allergens": row.allergens,
                    "unreadable": row.unreadable,
                    "model_confidence": float(row.model_confidence or 0.0),
                }
            ),
            input_tokens=row.input_tokens,
            output_tokens=row.output_tokens,
            latency_s=float(row.latency_s or 0.0),
        )
        for row in rows
    ]
    logger.info(
        "Разбор локальной модели собран",
        extra=safe_extra(
            model=model, prompt_version=version, found=len(predictions), requested=len(codes)
        ),
    )
    return predictions


async def collect_cloud_predictions(
    sample: list[SampleItem],
    llm: StructuredLLM,
    prompt: Prompt,
    settings: Settings | None = None,
) -> list[PredictionRecord]:
    """Прогнать выборку через облачную модель.

    Промпт передаётся тот же, что был у локальной модели: сравнивать системы
    на разных промптах — значит мерить промпт, а не систему.
    """
    settings = settings or get_settings()
    schema = ExtractionResult.json_schema_for_llm()
    predictions: list[PredictionRecord] = []

    logger.info(
        "Прогон облачной модели начат",
        extra=safe_extra(model=llm.model_name, products=len(sample)),
    )

    for index, item in enumerate(sample, start=1):
        prepared = prepare_text(item.ingredients_text, num_ctx=settings.ollama.num_ctx)
        if not prepared.is_usable:
            logger.debug(
                "Состав в модель не отправляется",
                extra=safe_extra(code=item.code, length=prepared.cleaned_length),
            )
            continue

        rendered = prompt.render(prepared.cleaned, lang=item.lang)
        try:
            response = await llm.generate(rendered, json_schema=schema)
        except (LLMUnavailableError, ExtractionError) as exc:
            # Продукт без ответа не выбрасывается: метрика засчитает его как
            # пустой ответ, и система не получит бонуса за молчание (ADR-022).
            logger.warning(
                "Облачная модель не ответила по продукту",
                extra=safe_extra(code=item.code, error=type(exc).__name__),
            )
            continue

        predictions.append(
            PredictionRecord(
                code=item.code,
                system=llm.model_name,
                extraction=ExtractionResult.model_validate(response.raw_json),
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                latency_s=response.latency_s,
            )
        )

        if index % 20 == 0:
            logger.info(
                "Прогресс облачного прогона",
                extra=safe_extra(done=index, total=len(sample), collected=len(predictions)),
            )

    logger.info(
        "Прогон облачной модели завершён",
        extra=safe_extra(model=llm.model_name, collected=len(predictions), requested=len(sample)),
    )
    return predictions
