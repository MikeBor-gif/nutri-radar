"""Контракт эталонного набора.

Два решения, от которых зависит весь майлстоун.

**Эталон хранится файлом в репозитории, а не в БД.** Это следствие требования
брифа: гейт evals обязан работать в CI без GPU. Живи эталон в базе, CI
пришлось бы её поднимать, а прогон — повторять. Файл же коммитится рядом
с предсказаниями, и гейт превращается в чистую функцию от двух файлов.

**Разметка человека хранится тем же типом `ExtractionResult`, что и ответ
модели.** Соблазн завести отдельный «человеческий» тип велик, но тогда число
разных форм сахара считалось бы двумя разными кусками кода — и метрика
сравнивала бы не системы, а реализации подсчёта. Эталон и предсказание
обязаны проходить через одни и те же свойства.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from nutri_radar.extract.schemas import ExtractionResult
from nutri_radar.logging import safe_extra

logger = logging.getLogger(__name__)

GOLD_DIR = Path("data/evals")
GOLD_FILE = GOLD_DIR / "extraction_gold.jsonl"
SAMPLE_FILE = GOLD_DIR / "sample.jsonl"
PREDICTIONS_DIR = GOLD_DIR / "predictions"


class GoldRecord(BaseModel):
    """Один размеченный человеком продукт."""

    code: str
    lang: str
    # Исходный текст хранится вместе с разметкой: без него нельзя перепроверить
    # спорную метку, а состав продукта в базе со временем меняется — дельты
    # M1 его обновляют.
    ingredients_text: str

    # Тот же тип, что у ответа модели. Именно это гарантирует, что
    # `distinct_sugar_forms` у эталона и у предсказания считается одним кодом.
    extraction: ExtractionResult

    annotator: str
    annotated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # Показывались ли разметчику предсказания модели. По умолчанию нет:
    # увидев их, человек правит вывод модели вместо того, чтобы размечать
    # независимо, и эталон незаметно становится копией предсказания.
    # Флаг хранится в записи, чтобы смещение было видно в отчёте, а не
    # растворилось в устном «вроде смотрел».
    assisted: bool = False

    @property
    def distinct_sugar_forms(self) -> int:
        return self.extraction.distinct_sugar_forms


def write_jsonl(path: Path, records: Iterable[BaseModel]) -> int:
    """Записать записи в JSONL. Возвращает число строк.

    Файл переписывается целиком: он маленький (сотни строк) и лежит в git,
    поэтому дописывание в конец только усложнило бы жизнь при правках
    отдельной записи.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [record.model_dump_json() for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    logger.debug("JSONL записан", extra=safe_extra(path=str(path), records=len(lines)))
    return len(lines)


def read_jsonl[T: BaseModel](path: Path, model: type[T]) -> list[T]:
    """Прочитать записи из JSONL.

    Пустые строки пропускаются, строка с битым JSON называет свой номер:
    файл правится руками, и «ошибка где-то в файле» — бесполезное сообщение.
    """
    if not path.exists():
        logger.debug("JSONL не найден", extra=safe_extra(path=str(path)))
        return []

    records: list[T] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            records.append(model.model_validate_json(stripped))
        except ValueError as exc:
            raise ValueError(f"{path}, строка {number}: {exc}") from exc

    logger.debug("JSONL прочитан", extra=safe_extra(path=str(path), records=len(records)))
    return records


def iter_gold(path: Path | None = None) -> Iterator[GoldRecord]:
    """Пройти по эталонным записям."""
    yield from read_jsonl(path or GOLD_FILE, GoldRecord)
