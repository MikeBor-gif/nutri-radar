"""Разведка схемы дампа Open Food Facts.

Первая задача M1 по требованию брифа: имена и форму колонок не угадывать.
Модуль отвечает на вопросы, ответ на которые даёт только реальный файл, и
делает это **до** скачивания 7,7 ГБ: Parquet колоночный, а URL дампа
поддерживает Range-запросы, поэтому DuckDB через `httpfs` читает только нужные
колонки и только первые row groups.

Зачем это команда, а не разовый скрипт: у дампа есть `schema_version`, и формат
может измениться. Разведку нужно уметь повторить, а не вспоминать, что было
однажды выяснено.

Стоимость: удалённое чтение 20 тыс. строк — около 6 минут. DuckDB делает много
мелких range-запросов. После `ingest dump` те же запросы локально идут за
секунды, поэтому есть режим `--local`.

**Важное ограничение выборки.** `LIMIT` читает первые row groups файла, а они
упорядочены по штрихкоду, то есть географически кластеризованы. Поэтому
разведка достоверно отвечает на вопросы о **схеме** — какие имена полей и
нутриентов существуют, есть ли служебная запись `main` — но её распределения
по языкам и категориям **не репрезентативны** для всей базы. Оценивать размер
корпуса по ним нельзя: для этого нужен полный проход (`ingest select --dry-run`).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import duckdb
from pydantic import BaseModel, Field

from nutri_radar.config import IngestSettings, get_settings
from nutri_radar.errors import DataSourceError

logger = logging.getLogger(__name__)

# Служебное значение внутри многоязычных списков. Дублирует текст главного
# языка продукта и НЕ является языком: если считать его языком, языковая
# статистика удвоится, а дедупликация по языкам поедет.
MAIN_LANG_MARKER = "main"

# Нутриенты, которые проекту действительно нужны. Имена подтверждены разведкой,
# а не взяты из data-fields.txt (тот описывает CSV-экспорт).
REQUIRED_NUTRIENTS = (
    "energy-kcal",
    "fat",
    "saturated-fat",
    "carbohydrates",
    "sugars",
    "fiber",
    "proteins",
    "salt",
    "sodium",
    # Входит в формулу Nutri-Score. Без него sanity-check M4 не сойдётся (ADR-005).
    "fruits-vegetables-nuts-estimate-from-ingredients",
)


class LanguageCount(BaseModel):
    lang: str
    count: int


class NutrientCount(BaseModel):
    name: str
    rows: int
    with_100g: int


class ProbeReport(BaseModel):
    """Результат разведки."""

    source: str
    rows_sampled: int
    elapsed_s: float

    column_count: int
    schema_versions: list[int] = Field(default_factory=list)

    # Вопрос 1
    has_main_lang_marker: bool
    languages: list[LanguageCount] = Field(default_factory=list)

    # Вопрос 2
    nutrients: list[NutrientCount] = Field(default_factory=list)

    @property
    def real_languages(self) -> list[LanguageCount]:
        """Языки без служебной записи `main`."""
        return [item for item in self.languages if item.lang != MAIN_LANG_MARKER]

    @property
    def missing_required_nutrients(self) -> list[str]:
        found = {item.name for item in self.nutrients}
        return [name for name in REQUIRED_NUTRIENTS if name not in found]

    def coverage(self, nutrient: str) -> float:
        """Доля строк выборки, где нутриент присутствует со значением на 100 г."""
        for item in self.nutrients:
            if item.name == nutrient:
                return item.with_100g / self.rows_sampled if self.rows_sampled else 0.0
        return 0.0


def _connect(memory_limit: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    # httpfs нужен только для удалённого чтения, но грузить его всегда дешевле,
    # чем ветвиться: расширение маленькое и кэшируется.
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    con.execute(f"SET memory_limit = '{memory_limit}'")
    return con


def _resolve_source(settings: IngestSettings, *, local: bool) -> str:
    """Вернуть источник: локальный путь или URL."""
    if not local:
        return settings.dump_url

    path = Path(settings.dump_path)
    if not path.exists():
        raise DataSourceError(
            f"Локальный дамп не найден: {path}. "
            "Выполните `nutri-radar ingest dump` или запустите probe без --local."
        )
    return str(path)


def probe_schema(
    settings: IngestSettings | None = None,
    *,
    local: bool = False,
    rows: int | None = None,
) -> ProbeReport:
    """Ответить на открытые вопросы о схеме дампа.

    Args:
        settings: настройки ingestion; по умолчанию из `get_settings()`.
        local: читать скачанный дамп вместо URL.
        rows: размер выборки; по умолчанию из настроек.
    """
    settings = settings or get_settings().ingest
    sample_rows = rows or settings.probe_sample_rows
    source = _resolve_source(settings, local=local)

    logger.info(
        "Разведка схемы дампа запущена",
        extra={
            "source": source,
            "rows_sampled": sample_rows,
            "mode": "local" if local else "remote",
        },
    )
    if not local:
        logger.info(
            "Удалённое чтение: каждый запрос занимает минуты, DuckDB делает "
            "много мелких range-запросов. После `ingest dump` используйте --local."
        )

    started = time.perf_counter()
    con = _connect(settings.duckdb_memory_limit)
    try:
        column_count = _probe_columns(con, source)
        # Выборка материализуется ОДИН раз. Каждое обращение к read_parquet по
        # сети — отдельное сканирование примерно на 200 с, поэтому три запроса
        # подряд стоили бы втрое дороже. Нужные колонки берутся сразу.
        _materialize_sample(con, source, sample_rows)
        schema_versions = _probe_schema_versions(con)
        languages = _probe_languages(con)
        nutrients = _probe_nutrients(con)
    except duckdb.Error as exc:
        raise DataSourceError(
            f"Не удалось прочитать дамп ({source}): {exc}. "
            "При удалённом чтении помогает `nutri-radar ingest dump` "
            "и повторный запуск с --local."
        ) from exc
    finally:
        con.close()

    report = ProbeReport(
        source=source,
        rows_sampled=sample_rows,
        elapsed_s=round(time.perf_counter() - started, 1),
        column_count=column_count,
        schema_versions=schema_versions,
        has_main_lang_marker=any(item.lang == MAIN_LANG_MARKER for item in languages),
        languages=languages,
        nutrients=nutrients,
    )

    _log_conclusions(report, settings)
    return report


def _probe_columns(con: duckdb.DuckDBPyConnection, source: str) -> int:
    """Число колонок. Читает только футер Parquet — дешёвая операция.

    Намеренно НЕ используется `parquet_metadata()`: он тянет метаданные всех
    row groups и на удалённом файле занимает минуты.
    """
    started = time.perf_counter()
    sql = f"SELECT count(*) FROM (DESCRIBE SELECT * FROM read_parquet('{source}'))"
    logger.debug("Запрос числа колонок", extra={"sql": sql})
    count = con.execute(sql).fetchone()
    elapsed = round(time.perf_counter() - started, 1)
    logger.debug("Колонок в схеме", extra={"columns": count, "elapsed_s": elapsed})
    return int(count[0]) if count else 0


_SAMPLE_TABLE = "probe_sample"


def _materialize_sample(con: duckdb.DuckDBPyConnection, source: str, rows: int) -> None:
    """Прочитать выборку один раз и положить в локальную временную таблицу.

    Единственное место, которое реально обращается к источнику. Дальше все
    вопросы задаются локальной таблице, поэтому стоимость разведки не растёт
    с числом вопросов.
    """
    started = time.perf_counter()
    sql = f"""
        CREATE OR REPLACE TEMP TABLE {_SAMPLE_TABLE} AS
        SELECT ingredients_text, nutriments, schema_version
        FROM read_parquet('{source}')
        LIMIT {rows}
    """
    logger.debug("Материализация выборки", extra={"sql": " ".join(sql.split())})
    con.execute(sql)
    actual = con.execute(f"SELECT count(*) FROM {_SAMPLE_TABLE}").fetchone()
    logger.info(
        "Выборка прочитана",
        extra={
            "rows": int(actual[0]) if actual else 0,
            "elapsed_s": round(time.perf_counter() - started, 1),
        },
    )


def _probe_schema_versions(con: duckdb.DuckDBPyConnection) -> list[int]:
    """Версии схемы в выборке.

    LIMIT применяется к выборке до DISTINCT (в `_materialize_sample`), а не
    к результату: иначе DuckDB просканировал бы весь файл ради поиска
    уникальных значений.
    """
    sql = f"""
        SELECT DISTINCT schema_version
        FROM {_SAMPLE_TABLE}
        WHERE schema_version IS NOT NULL
        ORDER BY schema_version
    """
    result = [int(row[0]) for row in con.execute(sql).fetchall()]
    logger.debug("Версии схемы дампа", extra={"versions": result})
    return result


def _probe_languages(con: duckdb.DuckDBPyConnection) -> list[LanguageCount]:
    """Вопрос 1: какие значения `lang` встречаются в `ingredients_text`.

    `ingredients_text` — это `LIST<STRUCT(lang, text)>`, поэтому нужен UNNEST,
    а не обращение к колонке.
    """
    sql = f"""
        SELECT u.lang, count(*) AS n
        FROM {_SAMPLE_TABLE}, UNNEST(ingredients_text) AS t(u)
        GROUP BY ALL
        ORDER BY n DESC
    """
    result = [LanguageCount(lang=row[0], count=row[1]) for row in con.execute(sql).fetchall()]
    logger.debug("Языки состава получены", extra={"distinct": len(result)})
    return result


def _probe_nutrients(con: duckdb.DuckDBPyConnection) -> list[NutrientCount]:
    """Вопрос 2: точные значения `name` внутри `nutriments` и их покрытие.

    `nutriments` — список структур, а не набор колонок: плоского `sugars_100g`
    в Parquet не существует.
    """
    sql = f"""
        SELECT u.name, count(*) AS rows_with, count(u['100g']) AS with_100g
        FROM {_SAMPLE_TABLE}, UNNEST(nutriments) AS t(u)
        GROUP BY ALL
        ORDER BY rows_with DESC
    """
    result = [
        NutrientCount(name=row[0], rows=row[1], with_100g=row[2])
        for row in con.execute(sql).fetchall()
    ]
    logger.debug("Нутриенты получены", extra={"distinct": len(result)})
    return result


def _log_conclusions(report: ProbeReport, settings: IngestSettings) -> None:
    """Выводы по итогам разведки. Именно они нужны при проектировании схемы."""
    if report.has_main_lang_marker:
        logger.info(
            "Служебная запись lang='main' ПРИСУТСТВУЕТ — исключать из подсчёта языков",
            extra={"real_languages": len(report.real_languages)},
        )
    else:
        logger.info("Служебной записи lang='main' нет")

    missing = report.missing_required_nutrients
    if missing:
        logger.warning(
            "В выборке не найдены нужные нутриенты — проверьте имена",
            extra={"missing": missing},
        )
    else:
        logger.info("Все нужные нутриенты присутствуют", extra={"count": len(REQUIRED_NUTRIENTS)})

    fvn = "fruits-vegetables-nuts-estimate-from-ingredients"
    coverage = report.coverage(fvn)
    logger.info(
        "Покрытие нутриента формулы Nutri-Score",
        extra={"nutrient": fvn, "coverage": f"{coverage:.1%}"},
    )
    if coverage < 0.5:
        logger.warning(
            "Покрытие ниже половины: sanity-check M4 придётся считать на "
            "подмножестве, долю исключённых строк указать в отчёте (ADR-005)"
        )

    # Языки проекта против фактически встречающихся в выборке.
    # Отсутствие языка здесь НЕ означает, что его мало в базе: выборка берётся
    # с начала файла, а он упорядочен по штрихкоду и потому кластеризован
    # географически. Размер корпуса оценивает только `ingest select --dry-run`.
    available = {item.lang for item in report.real_languages}
    unseen = [lang for lang in settings.languages if lang not in available]
    if unseen:
        logger.info(
            "Языки из настроек не встретились в выборке. Это ожидаемо: выборка "
            "берётся с начала файла и географически смещена, а не случайна",
            extra={"unseen": unseen, "sample_rows": report.rows_sampled},
        )

    if len(report.schema_versions) > 1:
        logger.warning(
            "В дампе несколько версий схемы — записи могут различаться по составу полей",
            extra={"versions": report.schema_versions},
        )


def format_report(report: ProbeReport, *, top_languages: int = 12, top_nutrients: int = 25) -> str:
    """Человекочитаемый отчёт для CLI."""
    lines = [
        f"Источник:        {report.source}",
        f"Выборка:         {report.rows_sampled} строк, {report.elapsed_s} с",
        f"Колонок в схеме: {report.column_count}",
        f"schema_version:  {report.schema_versions or '—'}",
        "",
        f"Служебная запись lang='main': {'ЕСТЬ' if report.has_main_lang_marker else 'нет'}",
    ]
    if report.has_main_lang_marker:
        lines.append("  → исключать из подсчёта языков, иначе статистика удвоится")

    lines.append("")
    lines.append(
        "ВНИМАНИЕ: выборка взята с начала файла (упорядочен по штрихкоду) и "
        "географически смещена.\nРаспределения ниже НЕ репрезентативны — размер "
        "корпуса оценивает `ingest select --dry-run`."
    )
    lines.append("")
    lines.append(f"Языки состава (топ {top_languages}, без 'main'):")
    for language in report.real_languages[:top_languages]:
        lines.append(f"  {language.lang:6s} {language.count:8d}")

    lines.append("")
    lines.append(f"Нутриенты (топ {top_nutrients}):")
    for nutrient in report.nutrients[:top_nutrients]:
        share = nutrient.with_100g / report.rows_sampled if report.rows_sampled else 0.0
        mark = " *" if nutrient.name in REQUIRED_NUTRIENTS else ""
        lines.append(f"  {nutrient.name:52s} {nutrient.with_100g:7d}  {share:6.1%}{mark}")
    lines.append("  * — нужен проекту")

    missing = report.missing_required_nutrients
    if missing:
        lines.append("")
        lines.append(f"НЕ НАЙДЕНЫ нужные нутриенты: {', '.join(missing)}")

    return "\n".join(lines)


def report_as_dict(report: ProbeReport) -> dict[str, Any]:
    return report.model_dump()
