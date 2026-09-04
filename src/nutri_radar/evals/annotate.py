"""Ручная разметка эталона. Метки ставит человек — правило 6 брифа.

Модуль сознательно не умеет размечать сам и не подсказывает ответ. Его работа —
показать состав, принять решение человека и не потерять его.

**Слепота к предсказаниям — не удобство, а метод.** Стоит показать разметчику
ответ модели, и он перестаёт размечать: он начинает его править. Итог выглядит
как эталон, но по сути это отредактированный вывод модели, и F1 такой системы
завышен просто потому, что её же ответ и был отправной точкой. Режим с
подсказкой оставлен (иногда нужно понять, почему модель ошиблась), но он
включается явным флагом и **записывается в каждую запись**, чтобы смещение
было видно в отчёте, а не осталось в памяти разметчика.

**Прогресс сохраняется после каждого продукта.** Сто составов руками — это
часы; прерывание не должно стоить работы.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from nutri_radar.evals.schemas import (
    GOLD_FILE,
    SAMPLE_FILE,
    GoldRecord,
    SampleItem,
    read_jsonl,
    write_jsonl,
)
from nutri_radar.extract.schemas import ExtractionResult, Ingredient, IngredientKind
from nutri_radar.logging import safe_extra

logger = logging.getLogger(__name__)

# Что вводит разметчик, чтобы объявить состав нечитаемым, и что — чтобы выйти.
UNREADABLE_TOKEN = "?"
QUIT_TOKEN = "q"


def pending_items(
    sample: list[SampleItem], done: list[GoldRecord], lang: str | None = None
) -> list[SampleItem]:
    """Что осталось разметить.

    Сравнение по коду, а не по позиции: выборка могла быть пересортирована,
    а размеченное — нет.

    Args:
        sample: вся выборка.
        done: уже размеченное.
        lang: ограничить одним языком. Выборка перемешана, поэтому без фильтра
            «размечу двадцать русских» превращается в двадцать случайных.
    """
    annotated = {record.code for record in done}
    pending = [item for item in sample if item.code not in annotated]
    if lang is None:
        return pending
    return [item for item in pending if item.lang == lang]


def parse_ingredients(line: str) -> list[Ingredient]:
    """Разобрать строку разметчика в список ингредиентов.

    Формат — `имя:тип`, через запятую: `sugar:sugar, water:base, e322:additive`.
    Тип можно опустить, тогда берётся `base`: большинство ингредиентов — основа,
    и заставлять писать это сто раз значит напрашиваться на опечатки в важных
    типах.

    Raises:
        ValueError: тип не из перечисления. Молча подставить `base` нельзя —
            это тихо испортило бы эталон.
    """
    ingredients: list[Ingredient] = []
    for chunk in line.split(","):
        item = chunk.strip()
        if not item:
            continue
        name, _, raw_kind = item.partition(":")
        name = name.strip().lower()
        if not name:
            continue
        kind_text = raw_kind.strip().lower() or IngredientKind.BASE.value
        try:
            kind = IngredientKind(kind_text)
        except ValueError as exc:
            allowed = ", ".join(k.value for k in IngredientKind)
            raise ValueError(
                f"неизвестный тип «{kind_text}» у «{name}». Допустимо: {allowed}"
            ) from exc
        ingredients.append(Ingredient(canonical_name=name, kind=kind))
    return ingredients


def build_record(
    item: SampleItem,
    ingredients: list[Ingredient],
    *,
    allergens: list[str],
    unreadable: bool,
    annotator: str,
    assisted: bool,
) -> GoldRecord:
    """Собрать эталонную запись.

    `model_confidence` у человека всегда 1.0: поле существует ради ответа
    модели, и притворяться, что разметчик в чём-то не уверен, незачем —
    сомнение выражается флагом `unreadable`.
    """
    return GoldRecord(
        code=item.code,
        lang=item.lang,
        ingredients_text=item.ingredients_text,
        extraction=ExtractionResult(
            ingredients=ingredients,
            allergens=allergens,
            unreadable=unreadable,
            model_confidence=1.0,
        ),
        annotator=annotator,
        annotated_at=datetime.now(UTC),
        assisted=assisted,
    )


def annotate_session(
    *,
    annotator: str,
    ask: Callable[[str], str],
    show: Callable[[str], None],
    sample_path: Path | None = None,
    gold_path: Path | None = None,
    assisted: bool = False,
    limit: int | None = None,
    lang: str | None = None,
) -> int:
    """Провести сессию разметки. Возвращает число новых записей.

    Ввод-вывод передаётся аргументами, а не берётся из `input`/`print`:
    так сессию можно проверить тестом, не поднимая терминал.

    Args:
        annotator: кто размечает. Попадает в каждую запись.
        ask: задать вопрос и получить ответ.
        show: показать текст разметчику.
        sample_path: файл выборки; по умолчанию `data/evals/sample.jsonl`.
        gold_path: файл эталона; по умолчанию `data/evals/extraction_gold.jsonl`.
        assisted: показывались ли предсказания модели. Пишется в запись.
        limit: сколько продуктов разметить за сессию.
        lang: размечать только продукты одного языка.
    """
    sample_file = sample_path or SAMPLE_FILE
    gold_file = gold_path or GOLD_FILE

    sample = read_jsonl(sample_file, SampleItem)
    if not sample:
        raise FileNotFoundError(
            f"Выборка не найдена: {sample_file}. Сначала соберите её командой `evals sample`."
        )

    if lang is not None:
        # Опечатку в коде языка ловим здесь, а не пустой сессией: «размечено 0»
        # выглядит как «всё уже сделано» и молча съедает заход разметчика.
        available = sorted({item.lang for item in sample})
        if lang not in available:
            raise ValueError(f"Языка {lang!r} в выборке нет. Доступны: {', '.join(available)}.")

    done = read_jsonl(gold_file, GoldRecord)
    pending = pending_items(sample, done, lang=lang)
    logger.info(
        "Сессия разметки начата",
        extra=safe_extra(
            annotator=annotator,
            total=len(sample),
            already_done=len(done),
            pending=len(pending),
            assisted=assisted,
            lang=lang or "все",
        ),
    )

    if assisted:
        # Предупреждение видит человек, а не только лог: смещение он вносит
        # своими руками, и знать о нём должен в момент, когда это происходит.
        show(
            "ВНИМАНИЕ: режим с подсказкой. Разметка будет помечена assisted=true "
            "и учтена в отчёте отдельно — она слабее слепой."
        )

    added = 0
    for index, item in enumerate(pending, start=1):
        if limit is not None and added >= limit:
            break

        show(f"\n[{index}/{len(pending)}] {item.code} ({item.lang})")
        show(item.ingredients_text)
        show(
            f"Формат: имя:тип через запятую. Типы: "
            f"{', '.join(k.value for k in IngredientKind)}. "
            f"«{UNREADABLE_TOKEN}» — состав нечитаем, «{QUIT_TOKEN}» — выйти."
        )

        # Выход — обычный сценарий, а не ошибка, поэтому он возвращается
        # значением, а не исключением: сто составов руками почти наверняка
        # размечаются в несколько заходов.
        answer = _ask_ingredients(ask, show)
        if answer is None:
            show("Сессия прервана. Размеченное сохранено.")
            break
        ingredients, unreadable = answer

        allergens = _ask_allergens(ask)
        if allergens is None:
            show("Сессия прервана. Размеченное сохранено.")
            break

        record = build_record(
            item,
            ingredients,
            allergens=allergens,
            unreadable=unreadable,
            annotator=annotator,
            assisted=assisted,
        )
        done.append(record)
        # Пишем после каждого продукта: сессия идёт часами, и прерывание
        # не должно стоить работы.
        write_jsonl(gold_file, done)
        added += 1
        logger.debug(
            "Продукт размечен",
            extra=safe_extra(
                code=item.code,
                ingredients=len(ingredients),
                sugar_forms=record.distinct_sugar_forms,
                unreadable=unreadable,
            ),
        )

    logger.info(
        "Сессия разметки завершена",
        extra=safe_extra(added=added, total_annotated=len(done), remaining=len(pending) - added),
    )
    return added


def _ask_ingredients(
    ask: Callable[[str], str], show: Callable[[str], None]
) -> tuple[list[Ingredient], bool] | None:
    """Спросить ингредиенты, пока ответ не станет разбираемым.

    Returns:
        Пара «ингредиенты, нечитаем» либо `None`, если разметчик вышел.
    """
    while True:
        answer = ask("Ингредиенты: ").strip()
        if answer.lower() == QUIT_TOKEN:
            return None
        if answer == UNREADABLE_TOKEN:
            return [], True
        try:
            return parse_ingredients(answer), False
        except ValueError as exc:
            # Не принимаем и не «чиним» — просим переввести. Тихая подстановка
            # типа испортила бы эталон незаметно для разметчика.
            show(f"Не понял: {exc}")


def _ask_allergens(ask: Callable[[str], str]) -> list[str] | None:
    """Спросить аллергены. Пустой ответ — их нет, `None` — разметчик вышел."""
    answer = ask("Аллергены (через запятую, пусто — нет): ").strip()
    if answer.lower() == QUIT_TOKEN:
        return None
    return [part.strip().lower() for part in answer.split(",") if part.strip()]
