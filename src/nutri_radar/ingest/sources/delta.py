"""Адаптер дельта-экспортов Open Food Facts.

Второй источник, и он **не является тем же ридером с другим путём** (ADR-004).
Формат принципиально другой:

* полный дамп — Parquet, `ingredients_text` = `LIST<STRUCT(lang, text)>`,
  `nutriments` = список структур;
* дельты — gzip-JSONL в MongoDB-форме: состав лежит ПЛОСКИМИ ключами
  `ingredients_text_en`, `ingredients_text_ru`, нутриенты — плоским объектом
  с ключами вида `sugars_100g`.

Вся разница форматов заканчивается в этом модуле: наружу уходит тот же
`RawProduct`, что и от Parquet-адаптера.

**Ограничение, которое нельзя замалчивать.** Дельты не отслеживают удаление
продуктов — это ограничение самого Open Food Facts. Полностью консистентное
состояние даёт только перезалив полного дампа. Хранение дельт — 14 дней:
если последний прогон был раньше, часть истории потеряна безвозвратно.
"""

from __future__ import annotations

import gzip
import json
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError

from nutri_radar.config import IngestSettings
from nutri_radar.errors import DataSourceError
from nutri_radar.ingest.models import RawProduct

logger = logging.getLogger(__name__)

# Открытое ограничение самого OFF: дельты хранятся 14 дней.
DELTA_RETENTION_DAYS = 14

# Имена файлов индекса содержат UNIX-таймстампы: `1690000000.json.gz`
# или диапазон `1690000000_1690086400.json.gz`.
_TIMESTAMP_RE = re.compile(r"(\d{9,11})")

# Суффикс плоских ключей нутриентов в MongoDB-форме: `sugars_100g` → `sugars`.
_NUTRIENT_SUFFIX = "_100g"

# Префиксы многоязычных полей. В дельтах язык — часть имени ключа.
_MULTILANG_PREFIXES = {
    "ingredients_text": "ingredients_text",
    "product_name": "product_name",
    "generic_name": "generic_name",
}


@dataclass(frozen=True)
class DeltaFile:
    """Один файл дельта-экспорта."""

    name: str
    timestamp: int

    def url(self, index_url: str) -> str:
        base = index_url.rsplit("/", 1)[0]
        return f"{base}/{self.name}"


def parse_index(content: str) -> list[DeltaFile]:
    """Разобрать `index.txt` и упорядочить файлы по времени.

    Порядок важен: дельты применяются последовательно, иначе более старая
    запись перезапишет более свежую.
    """
    files: list[DeltaFile] = []
    for line in content.splitlines():
        name = line.strip()
        if not name or not name.endswith(".gz"):
            continue
        match = _TIMESTAMP_RE.search(name)
        if not match:
            logger.warning("Имя файла дельты без таймстампа, пропущено", extra={"file_name": name})
            continue
        files.append(DeltaFile(name=name, timestamp=int(match.group(1))))

    files.sort(key=lambda item: item.timestamp)
    logger.debug("Индекс дельт разобран", extra={"files": len(files)})
    return files


def select_files(files: list[DeltaFile], watermark: int | None) -> list[DeltaFile]:
    """Отобрать файлы новее водяного знака последнего успешного прогона."""
    if watermark is None:
        logger.info("Водяного знака нет — применяем все доступные дельты")
        return files
    fresh = [item for item in files if item.timestamp > watermark]
    logger.info(
        "Отобраны дельты новее последнего прогона",
        extra={"watermark": watermark, "available": len(files), "selected": len(fresh)},
    )
    return fresh


def check_retention_gap(files: list[DeltaFile], watermark: int | None) -> bool:
    """Не потеряна ли часть истории.

    Возвращает `True`, если разрыв слишком велик: самый старый доступный файл
    новее водяного знака, значит между ними были дельты, которых уже нет.
    """
    if watermark is None or not files:
        return False
    oldest = files[0].timestamp
    if oldest > watermark:
        logger.warning(
            "Разрыв в истории дельт: часть обновлений уже удалена из индекса "
            f"(хранение {DELTA_RETENTION_DAYS} дней). Нужен полный перезалив: "
            "`ingest dump` и `ingest select`",
            extra={"watermark": watermark, "oldest_available": oldest},
        )
        return True
    return False


def _collect_multilang(record: dict[str, Any], prefix: str) -> dict[str, str]:
    """Собрать плоские ключи `<prefix>_<lang>` в словарь по языкам.

    Здесь и живёт главное отличие формата: в Parquet язык — поле структуры,
    в дельтах — часть имени ключа.
    """
    result: dict[str, str] = {}
    for key, value in record.items():
        if not key.startswith(f"{prefix}_") or not isinstance(value, str):
            continue
        lang = key[len(prefix) + 1 :]
        # Отсекаем служебные суффиксы вроде `_debug_tags`, оставляя только
        # двух- и трёхбуквенные коды языков.
        if not lang.isalpha() or not 2 <= len(lang) <= 3:
            continue
        if value.strip():
            result[lang] = value.strip()
    return result


def _collect_nutriments(record: dict[str, Any]) -> dict[str, float]:
    """Развернуть плоский объект нутриентов в словарь `имя -> значение`."""
    raw = record.get("nutriments")
    if not isinstance(raw, dict):
        return {}

    result: dict[str, float] = {}
    for key, value in raw.items():
        if not key.endswith(_NUTRIENT_SUFFIX) or value is None:
            continue
        name = key[: -len(_NUTRIENT_SUFFIX)]
        try:
            result[name] = float(value)
        except (TypeError, ValueError):
            logger.debug("Нечисловое значение нутриента пропущено", extra={"key": key})
    return result


def to_raw_product(record: dict[str, Any]) -> RawProduct | None:
    """Привести запись дельты к `RawProduct`.

    Возвращает `None`, если запись не проходит валидацию: битый штрихкод не
    повод ронять применение всего файла.
    """
    payload: dict[str, Any] = {
        "code": record.get("code"),
        "lang": record.get("lang"),
        "brands": record.get("brands"),
        "categories_tags": record.get("categories_tags") or [],
        "food_groups_tags": record.get("food_groups_tags") or [],
        "countries_tags": record.get("countries_tags") or [],
        "labels_tags": record.get("labels_tags") or [],
        "nutriscore_grade": record.get("nutriscore_grade"),
        "nutriscore_score": record.get("nutriscore_score"),
        "nova_group": record.get("nova_group"),
        "nutriments": _collect_nutriments(record),
        "nutrition_data_per": record.get("nutrition_data_per"),
        "no_nutrition_data": bool(record.get("no_nutrition_data") or False),
        "ingredients_tags": record.get("ingredients_tags") or [],
        "ingredients_original_tags": record.get("ingredients_original_tags") or [],
        "additives_tags": record.get("additives_tags") or [],
        "allergens_tags": record.get("allergens_tags") or [],
        "traces_tags": record.get("traces_tags") or [],
        "ingredients_analysis_tags": record.get("ingredients_analysis_tags") or [],
        "ingredients_n": record.get("ingredients_n"),
        "known_ingredients_n": record.get("known_ingredients_n"),
        "unknown_ingredients_n": record.get("unknown_ingredients_n"),
        "additives_n": record.get("additives_n"),
        "with_sweeteners": record.get("with_sweeteners"),
        "with_non_nutritive_sweeteners": record.get("with_non_nutritive_sweeteners"),
        "obsolete": bool(record.get("obsolete") or False),
        "completeness": record.get("completeness"),
        "data_quality_errors": record.get("data_quality_errors_tags") or [],
        "unique_scans_n": record.get("unique_scans_n"),
        "popularity_key": record.get("popularity_key"),
        "rev": record.get("rev"),
        "last_modified_t": record.get("last_modified_t"),
        "created_t": record.get("created_t"),
        "schema_version": record.get("schema_version"),
        "source": "delta",
    }

    for field, prefix in _MULTILANG_PREFIXES.items():
        payload[field] = _collect_multilang(record, prefix)

    ingredients = record.get("ingredients")
    payload["ingredients_json"] = (
        json.dumps(ingredients, ensure_ascii=False) if ingredients is not None else None
    )

    try:
        return RawProduct.model_validate(payload)
    except ValidationError as exc:
        logger.debug(
            "Запись дельты не прошла валидацию",
            extra={"code": record.get("code"), "error": exc.errors()[0]["msg"]},
        )
        return None


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    """Читать gzip-JSONL построчно, не разворачивая файл в память."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning(
                    "Битая строка в дельте пропущена",
                    extra={"file": path.name, "line": number},
                )


def fetch_index(client: httpx.Client, settings: IngestSettings) -> list[DeltaFile]:
    """Скачать и разобрать индекс дельт."""
    logger.info("Читаем индекс дельт", extra={"url": settings.delta_index_url})
    try:
        response = client.get(settings.delta_index_url)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise DataSourceError(
            f"Индекс дельт недоступен ({settings.delta_index_url}): {exc}"
        ) from exc
    return parse_index(response.text)


def download_delta(client: httpx.Client, file: DeltaFile, settings: IngestSettings) -> Path:
    """Скачать один файл дельты во временный каталог данных."""
    target = Path(settings.data_dir) / "delta" / file.name
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and target.stat().st_size > 0:
        logger.debug("Файл дельты уже скачан", extra={"file_name": file.name})
        return target

    url = file.url(settings.delta_index_url)
    response = client.get(url, follow_redirects=True)
    response.raise_for_status()
    target.write_bytes(response.content)
    logger.debug(
        "Файл дельты скачан",
        extra={"file_name": file.name, "size_kb": round(len(response.content) / 1024, 1)},
    )
    return target
