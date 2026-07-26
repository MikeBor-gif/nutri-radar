"""Фильтрация корпуса в DuckDB.

Обязательный этап раздела 6 брифа: отбор происходит **до** Postgres. В базу
уезжает отфильтрованное подмножество, а не 4,63 млн строк.

Все критерии приходят из настроек — ни одного литерала в коде (правило 5).
Единственное исключение — служебная запись `lang = 'main'`: это не параметр,
а особенность формата дампа, подтверждённая разведкой.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import duckdb

from nutri_radar.config import IngestSettings
from nutri_radar.ingest.models import MAIN_LANG_MARKER, RawProduct
from nutri_radar.ingest.sources.parquet import ParquetSource

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SelectionStats:
    """Что даёт фильтр до заливки. Нужно, чтобы понять, попадаем ли в вилку."""

    total_rows: int
    matched_rows: int
    by_language: dict[str, int]
    by_grade: dict[str, int]
    with_nutrition: int
    unknown_ingredients: int

    @property
    def match_share(self) -> float:
        return self.matched_rows / self.total_rows if self.total_rows else 0.0


def matches_corpus(product: RawProduct, settings: IngestSettings) -> bool:
    """Проходит ли продукт критерии корпуса.

    Питоновский двойник SQL-фильтра из `build_where`. Он нужен потому, что
    дельта-экспорты читаются построчно из JSONL, а не через DuckDB, и прогнать
    их тем же запросом нельзя.

    Дублирование критериев в двух формах — осознанная цена, но она опасна:
    условия могут разъехаться незаметно. Поэтому тест обязан проверять, что обе
    формы дают **одинаковый набор кодов** на одной фикстуре
    (`test_ingest_delta.py`). Меняешь одно — меняй и второе.
    """
    if not product.is_quality_ok:
        return False
    if not product.has_usable_ingredients(settings.languages, settings.min_ingredients_length):
        return False
    return bool(set(product.categories_tags) & set(settings.category_tags))


def build_where(settings: IngestSettings) -> tuple[str, list[Any]]:
    """Собрать условие отбора и параметры к нему.

    Параметризация, а не подстановка значений в текст: списки категорий и
    языков приходят из конфигурации, и склейка строк тут была бы и небезопасной,
    и ломкой на кавычках.
    """
    where = f"""
        NOT coalesce(obsolete, false)
        AND len(coalesce(data_quality_errors_tags, [])) = 0
        AND len(list_filter(
                coalesce(ingredients_text, []),
                x -> x.lang <> '{MAIN_LANG_MARKER}'
                     AND list_contains(?, x.lang)
                     AND x.text IS NOT NULL
                     AND length(trim(x.text)) >= ?
            )) > 0
        AND list_has_any(coalesce(categories_tags, []), ?)
    """
    params: list[Any] = [
        settings.languages,
        settings.min_ingredients_length,
        settings.category_tags,
    ]
    return " ".join(where.split()), params


def collect_stats(
    source: ParquetSource,
    settings: IngestSettings,
    con: duckdb.DuckDBPyConnection,
) -> SelectionStats:
    """Посчитать статистику выборки без записи в БД.

    Считается одним проходом по отфильтрованному подмножеству: отдельные
    запросы на каждый разрез означали бы столько же полных сканирований файла.
    """
    where, params = build_where(settings)

    total_row = con.execute("SELECT count(*) FROM read_parquet(?)", [source.path]).fetchone()
    total = int(total_row[0]) if total_row else 0

    logger.info("Считаем статистику выборки", extra={"total_rows": total})

    # Языки: раскрываем список составов и считаем только настоящие языки.
    lang_rows = con.execute(
        f"""
        WITH matched AS (
            SELECT ingredients_text FROM read_parquet(?) WHERE {where}
        )
        SELECT u.lang, count(*) AS n
        FROM matched, UNNEST(ingredients_text) AS t(u)
        WHERE u.lang <> '{MAIN_LANG_MARKER}' AND list_contains(?, u.lang)
        GROUP BY ALL ORDER BY n DESC
        """,
        [source.path, *params, settings.languages],
    ).fetchall()

    summary = con.execute(
        f"""
        SELECT
            count(*) AS matched,
            count(*) FILTER (WHERE NOT coalesce(no_nutrition_data, false)) AS with_nutrition,
            count(*) FILTER (WHERE coalesce(unknown_ingredients_n, 0) > 0) AS unknown_ing,
            nutriscore_grade
        FROM read_parquet(?) WHERE {where}
        GROUP BY nutriscore_grade
        """,
        [source.path, *params],
    ).fetchall()

    by_grade: dict[str, int] = {}
    matched = 0
    with_nutrition = 0
    unknown_ing = 0
    for row in summary:
        matched += int(row[0])
        with_nutrition += int(row[1])
        unknown_ing += int(row[2])
        grade = row[3] if row[3] in {"a", "b", "c", "d", "e"} else "нет метки"
        by_grade[grade] = by_grade.get(grade, 0) + int(row[0])

    stats = SelectionStats(
        total_rows=total,
        matched_rows=matched,
        by_language={row[0]: int(row[1]) for row in lang_rows},
        by_grade=dict(sorted(by_grade.items())),
        with_nutrition=with_nutrition,
        unknown_ingredients=unknown_ing,
    )
    _log_stats(stats, settings)
    return stats


def _log_stats(stats: SelectionStats, settings: IngestSettings) -> None:
    logger.info(
        "Фильтр применён",
        extra={
            "total_rows": stats.total_rows,
            "matched_rows": stats.matched_rows,
            "share": f"{stats.match_share:.2%}",
            "languages": settings.languages,
            "category_tags": len(settings.category_tags),
            "min_length": settings.min_ingredients_length,
        },
    )
    logger.info("Языки корпуса", extra={"by_language": stats.by_language})
    logger.info("Оценки качества", extra={"by_grade": stats.by_grade})
    logger.info(
        "Пригодность для следующих майлстоунов",
        extra={
            # Столько строк годится для sanity-check и обучения на числах (M4).
            "with_nutrition": stats.with_nutrition,
            # Столько строк — кандидаты в LLM-корпус: парсер OFF на них
            # не справился (ADR-006).
            "unknown_ingredients": stats.unknown_ingredients,
        },
    )

    if stats.matched_rows < settings.corpus_min_size:
        logger.warning(
            "Корпус меньше целевого: фильтр слишком узкий",
            extra={"matched": stats.matched_rows, "target_min": settings.corpus_min_size},
        )
    elif stats.matched_rows > settings.corpus_max_size:
        logger.warning(
            "Корпус больше целевого: стоит сузить список категорий",
            extra={"matched": stats.matched_rows, "target_max": settings.corpus_max_size},
        )
    else:
        logger.info(
            "Корпус в целевой вилке",
            extra={
                "matched": stats.matched_rows,
                "target": f"{settings.corpus_min_size}-{settings.corpus_max_size}",
            },
        )


def format_stats(stats: SelectionStats) -> str:
    """Человекочитаемая статистика выборки для CLI."""
    lines = [
        f"Всего строк в дампе:  {stats.total_rows}",
        f"Прошло фильтр:        {stats.matched_rows} ({stats.match_share:.2%})",
        "",
        "Языки состава:",
    ]
    lines.extend(f"  {lang:6s} {count:8d}" for lang, count in stats.by_language.items())
    lines.append("")
    lines.append("Оценка качества питания:")
    lines.extend(f"  {grade:10s} {count:8d}" for grade, count in stats.by_grade.items())
    lines.append("")
    lines.append(f"С таблицей питательности:      {stats.with_nutrition}  (пригодны для M4)")
    lines.append(
        f"Парсер OFF не справился:      {stats.unknown_ingredients}  (кандидаты в LLM-корпус M2)"
    )
    return "\n".join(lines)


def iter_selected(
    source: ParquetSource,
    settings: IngestSettings,
    con: duckdb.DuckDBPyConnection,
    *,
    limit: int | None = None,
) -> Iterator[list[RawProduct]]:
    """Отдавать отфильтрованные продукты батчами."""
    where, params = build_where(settings)
    logger.info(
        "Выборка корпуса начата",
        extra={
            "path": source.path,
            "batch_size": settings.batch_size,
            "limit": limit,
        },
    )
    yield from source.iter_products(con, where, params, limit=limit)
