"""Отбор корпуса для прогона через LLM.

ADR-006 в действии. Корпус берётся **не случайно**: 70% отбирается со смещением
в `unknown_ingredients_n > 0` — туда, где собственный парсер Open Food Facts
не справился. Это измеримый ответ на вопрос «зачем здесь LLM, если база уже
всё разобрала»: там, где таксономия сработала, LLM просто дублирует готовое.

Оставшиеся 30% — контрольная случайная часть. Без неё сравнение было бы
нечестным: нельзя показать преимущество только на сложных случаях и умолчать,
как система ведёт себя на лёгких.

Стратификация по языку обязательна. Фактические числа корпуса M1:
fr 151k, en 124k, de 59k, pl 5,7k, **ru 2,3k**. Пропорциональная выборка дала бы
на 3000 продуктов около 30 русских составов — и заявленная многоязычность
осталась бы недоказанной.

Выборка **детерминирована**: тот же seed даёт тот же набор кодов. Иначе
сравнение версий промптов пошло бы по разным продуктам и стало бы бессмысленным.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

from sqlalchemy import func, select

from nutri_radar.config import ExtractSettings, Settings, get_settings
from nutri_radar.db.models.product import Product
from nutri_radar.db.session import get_session
from nutri_radar.logging import safe_extra

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CorpusItem:
    """Продукт, отобранный в корпус."""

    code: str
    ingredients_text: str
    lang: str
    nutriscore_grade: str | None
    unknown_ingredients_n: int
    # Из какой части выборки пришёл: смещённой или контрольной. Нужен в M3,
    # чтобы считать метрики отдельно по обеим частям.
    stratum: str


@dataclass
class CorpusStats:
    """Состав выборки. Смотрится до прогона, который идёт часами."""

    total: int = 0
    by_lang: Counter[str] = field(default_factory=Counter)
    by_grade: Counter[str] = field(default_factory=Counter)
    by_stratum: Counter[str] = field(default_factory=Counter)

    def add(self, item: CorpusItem) -> None:
        self.total += 1
        self.by_lang[item.lang] += 1
        self.by_grade[item.nutriscore_grade or "нет метки"] += 1
        self.by_stratum[item.stratum] += 1

    @property
    def unknown_share(self) -> float:
        return self.by_stratum["unknown"] / self.total if self.total else 0.0


def _language_quotas(languages: list[str], total: int) -> dict[str, int]:
    """Равные квоты по языкам с добором остатка на первые языки.

    Равные, а не пропорциональные: пропорциональные воспроизвели бы перекос
    базы, где русских составов в 65 раз меньше французских.
    """
    if not languages:
        return {}
    base = total // len(languages)
    remainder = total - base * len(languages)
    return {lang: base + (1 if index < remainder else 0) for index, lang in enumerate(languages)}


async def _fetch_stratum(
    settings: Settings,
    *,
    lang: str,
    limit: int,
    unknown: bool,
    seed: int,
) -> list[CorpusItem]:
    """Выбрать продукты одного языка из одной части выборки.

    Порядок задаётся хешем от кода и seed, а не `random()`: так выборка
    воспроизводима между запусками и не зависит от состояния БД.
    """
    if limit <= 0:
        return []

    condition = (
        Product.unknown_ingredients_n > 0
        if unknown
        else func.coalesce(Product.unknown_ingredients_n, 0) == 0
    )
    ordering = func.md5(Product.code + str(seed))

    statement = (
        select(
            Product.code,
            Product.ingredients_text,
            Product.ingredients_text_lang,
            Product.nutriscore_grade,
            Product.unknown_ingredients_n,
        )
        .where(
            Product.ingredients_text.is_not(None),
            Product.ingredients_text_lang == lang,
            condition,
        )
        .order_by(ordering)
        .limit(limit)
    )

    async with get_session(settings.db) as session:
        rows = (await session.execute(statement)).all()

    stratum = "unknown" if unknown else "control"
    return [
        CorpusItem(
            code=row[0],
            ingredients_text=row[1],
            lang=row[2] or lang,
            nutriscore_grade=row[3],
            unknown_ingredients_n=row[4] or 0,
            stratum=stratum,
        )
        for row in rows
    ]


async def select_llm_corpus(
    settings: Settings | None = None,
    *,
    size: int | None = None,
) -> list[CorpusItem]:
    """Отобрать корпус для прогона через LLM.

    Args:
        settings: настройки; по умолчанию из `get_settings()`.
        size: размер корпуса; по умолчанию `EXTRACT__CORPUS_SIZE`.
    """
    settings = settings or get_settings()
    extract: ExtractSettings = settings.extract
    total = size or extract.corpus_size
    languages = settings.ingest.languages

    unknown_total = int(total * extract.unknown_share)
    control_total = total - unknown_total

    logger.info(
        "Отбор LLM-корпуса начат",
        extra=safe_extra(
            target_size=total,
            unknown_share=extract.unknown_share,
            unknown_target=unknown_total,
            control_target=control_total,
            languages=languages,
            seed=extract.random_seed,
        ),
    )

    items: list[CorpusItem] = []
    shortfalls: dict[str, int] = {}

    for unknown, part_total in ((True, unknown_total), (False, control_total)):
        quotas = _language_quotas(languages, part_total)
        for lang, quota in quotas.items():
            batch = await _fetch_stratum(
                settings, lang=lang, limit=quota, unknown=unknown, seed=extract.random_seed
            )
            items.extend(batch)
            if len(batch) < quota:
                key = f"{lang}/{'unknown' if unknown else 'control'}"
                shortfalls[key] = quota - len(batch)

    # Порядок фиксирован по коду: набор и его последовательность не должны
    # зависеть от порядка обхода языков.
    items.sort(key=lambda item: item.code)

    if shortfalls:
        # Недобор квоты — не ошибка, а характеристика данных. Но знать о нём
        # обязательно: он прямо влияет на силу выводов M3 по этим языкам.
        logger.warning(
            "Квоты по языкам выполнены не полностью — выводы по этим языкам будут слабее",
            extra=safe_extra(shortfalls=shortfalls),
        )

    if not items:
        logger.error("Корпус пуст: проверьте, залит ли корпус M1 и совпадают ли языки")

    return items


def collect_stats(items: list[CorpusItem]) -> CorpusStats:
    stats = CorpusStats()
    for item in items:
        stats.add(item)

    logger.info(
        "Корпус отобран",
        extra=safe_extra(
            total=stats.total,
            unknown_share=f"{stats.unknown_share:.0%}",
            by_lang=dict(stats.by_lang),
            by_stratum=dict(stats.by_stratum),
        ),
    )
    return stats


def format_stats(stats: CorpusStats) -> str:
    """Человекочитаемая сводка для CLI."""
    lines = [
        f"Отобрано продуктов: {stats.total}",
        "",
        "По частям выборки:",
        f"  unknown (парсер OFF не справился): {stats.by_stratum['unknown']}"
        f"  ({stats.unknown_share:.0%})",
        f"  control (контрольная случайная):   {stats.by_stratum['control']}",
        "",
        "По языкам состава:",
    ]
    lines.extend(f"  {lang:6s} {count:6d}" for lang, count in sorted(stats.by_lang.items()))
    lines.append("")
    lines.append("По оценке качества:")
    lines.extend(f"  {grade:10s} {count:6d}" for grade, count in sorted(stats.by_grade.items()))
    return "\n".join(lines)
