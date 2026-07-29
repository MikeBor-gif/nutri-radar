"""Отчёт приёмки M3: таблица сравнения систем и материал для ответов на M2.

Модуль сводит вместе то, что уже посчитано метриками, и добавляет ровно те
описательные числа, которых метрикам не хватает для двух вопросов, оставшихся
от M2. Ни одного вывода он не делает — выводы пишет человек в `DECISIONS.md`.
Задача отчёта в другом: разложить числа так, чтобы ответ стал механическим,
а не результатом впечатления.

**Вопрос 1: 43,9% составов без единой формы сахара — предел модели или
свойство данных?** Различается сравнением двух долей на одних и тех же
продуктах. Если человек нашёл сахар там, где система вернула ноль, — это
предел системы. Если и человек нашёл ноль — корпус просто такой. Столбец
«вернула ноль вместо N» и есть прямой ответ.

**Вопрос 2: русские составы дают 1,28 формы против 2,42 у немецких — язык
данных или слабость модели?** Различается тем же способом: если тот же перекос
виден в эталоне, дело в том, как написаны составы; если только у системы —
дело в системе. Поэтому таблица по языкам показывает эталон и системы рядом.

**Каждое число идёт с размером выборки.** На 20 продуктах доверительный
интервал шире многих наблюдаемых разниц, и число без знаменателя вводит
в заблуждение сильнее, чем его отсутствие. По той же причине разницы меньше
измеренного в ADR-018 разброса между прогонами отчёт помечает как неотличимые
от шума.
"""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from nutri_radar.config import Settings, get_settings
from nutri_radar.evals.metrics import (
    ComparisonResult,
    format_by_language,
    format_comparison,
    score_system,
    sugar_forms,
)
from nutri_radar.evals.schemas import GoldRecord, PredictionRecord, read_jsonl
from nutri_radar.extract.normalize import AliasIndex
from nutri_radar.extract.schemas import ExtractionResult
from nutri_radar.logging import safe_extra

logger = logging.getLogger(__name__)

REPORTS_DIR = Path("reports")

# Как в отчёте называется сам эталон. Он стоит в тех же таблицах, что системы,
# потому что вопрос 2 решается именно сравнением с ним, а не между системами.
GOLD_LABEL = "эталон (человек)"


@dataclass
class SugarProfile:
    """Описательная статистика по формам сахара в разрезе одного языка.

    Считается тем же `sugar_forms`, что и метрики: собственный подсчёт здесь
    означал бы, что описательная часть отчёта и метрика меряют разное, и
    расхождение между ними списывалось бы на «ну там по-другому считается».
    """

    products: int = 0
    zero: int = 0
    counts: list[int] = field(default_factory=list)

    def add(self, count: int) -> None:
        self.products += 1
        self.counts.append(count)
        if count == 0:
            self.zero += 1

    @property
    def mean(self) -> float:
        return statistics.mean(self.counts) if self.counts else 0.0

    @property
    def zero_share(self) -> float:
        return self.zero / self.products if self.products else 0.0


def profile_gold(gold: list[GoldRecord], index: AliasIndex) -> dict[str, SugarProfile]:
    """Формы сахара в эталоне по языкам."""
    by_lang: dict[str, SugarProfile] = defaultdict(SugarProfile)
    for record in gold:
        by_lang[record.lang].add(sugar_forms(record.extraction, index, lang=record.lang))
    return dict(by_lang)


def profile_system(
    gold: list[GoldRecord],
    predictions: list[PredictionRecord],
    index: AliasIndex,
) -> dict[str, SugarProfile]:
    """Формы сахара у одной системы по языкам.

    Продукт без ответа входит как ноль — по тому же правилу, что и в метриках
    (ADR-022). Считать среднее только по отвеченным значило бы, что система,
    промолчавшая на трудных составах, выглядит аккуратнее ответившей на всех.
    """
    by_code = {record.code: record for record in predictions}
    by_lang: dict[str, SugarProfile] = defaultdict(SugarProfile)
    for record in gold:
        prediction = by_code.get(record.code)
        extraction = prediction.extraction if prediction is not None else ExtractionResult()
        by_lang[record.lang].add(sugar_forms(extraction, index, lang=record.lang))
    return dict(by_lang)


def read_all_predictions(directory: Path) -> dict[str, list[PredictionRecord]]:
    """Прочитать предсказания всех систем из каталога.

    Ключ — имя системы из самих записей, а не из имени файла: имя системы
    уходит в заголовки таблиц, и брать его из файла значило бы, что
    переименование файла молча переименует систему в отчёте.
    """
    by_system: dict[str, list[PredictionRecord]] = {}
    if not directory.exists():
        logger.warning("Каталог предсказаний не найден", extra=safe_extra(path=str(directory)))
        return by_system

    for file in sorted(directory.glob("*.jsonl")):
        records = read_jsonl(file, PredictionRecord)
        if not records:
            logger.warning("Пустой файл предсказаний", extra=safe_extra(path=str(file)))
            continue
        by_system[records[0].system] = records

    logger.info(
        "Предсказания прочитаны",
        extra=safe_extra(systems=len(by_system), path=str(directory)),
    )
    return by_system


def format_zero_sugar(
    gold_profile: dict[str, SugarProfile],
    system_profiles: dict[str, dict[str, SugarProfile]],
    results: dict[str, ComparisonResult],
) -> str:
    """Материал для вопроса 1: ноль форм сахара — чей предел.

    Столбец «вернула ноль вместо N» отвечает на вопрос прямо: это те продукты,
    где человек сахар нашёл, а система нет. Доля нулей у самого эталона
    отвечает на вторую половину — сколько нулей объясняется корпусом.
    """
    gold_products = sum(profile.products for profile in gold_profile.values())
    gold_zero = sum(profile.zero for profile in gold_profile.values())
    gold_share = gold_zero / gold_products if gold_products else 0.0

    lines = [
        "### Вопрос 1: составы без единой формы сахара",
        "",
        f"Эталон, размеченный человеком: **{gold_zero} из {gold_products}** "
        f"составов без единой формы сахара ({gold_share:.1%}).",
        "",
        "| Система | Нулей | Доля нулей | Вернула ноль вместо N | Пропустила весь сахар |",
        "|---|---|---|---|---|",
    ]

    for system in sorted(system_profiles):
        profiles = system_profiles[system]
        products = sum(profile.products for profile in profiles.values())
        zero = sum(profile.zero for profile in profiles.values())
        missed = results[system].overall.sugar.missed_all
        share = f"{zero / products:.1%}" if products else "—"
        missed_share = f"{missed / products:.1%}" if products else "—"
        lines.append(f"| {system} | {zero} | {share} | {missed} | {missed_share} |")

    lines += [
        "",
        "**Как читать.** «Вернула ноль вместо N» — продукты, где человек нашёл "
        "формы сахара, а система не нашла ни одной. Это предел системы. "
        "Разница между её долей нулей и долей нулей эталона — то, что "
        "объясняется корпусом, а не системой.",
    ]
    return "\n".join(lines)


def format_language_gap(
    gold_profile: dict[str, SugarProfile],
    system_profiles: dict[str, dict[str, SugarProfile]],
    settings: Settings,
) -> str:
    """Материал для вопроса 2: перекос между языками — данные или система.

    Эталон стоит первой строкой каждого языка намеренно: сравнивать системы
    между собой здесь бессмысленно, вопрос ставится к разнице «система против
    человека на одном и том же языке».
    """
    threshold = settings.evals.significant_diff_share
    langs = sorted(gold_profile)
    lines = [
        "### Вопрос 2: разница между языками",
        "",
        "| Язык | Продуктов | Источник | Форм сахара в среднем | Составов без сахара |",
        "|---|---|---|---|---|",
    ]

    for lang in langs:
        profile = gold_profile[lang]
        lines.append(
            f"| {lang} | {profile.products} | {GOLD_LABEL} "
            f"| {profile.mean:.2f} | {profile.zero_share:.0%} |"
        )
        for system in sorted(system_profiles):
            system_lang = system_profiles[system].get(lang)
            if system_lang is None:
                continue
            lines.append(
                f"| {lang} | {system_lang.products} | {system} "
                f"| {system_lang.mean:.2f} | {system_lang.zero_share:.0%} |"
            )

    lines += [
        "",
        f"**Как читать.** Если перекос между языками виден уже в строке "
        f"«{GOLD_LABEL}» — так написаны составы, и это свойство данных. "
        "Если он появляется только у системы — это её слабость на языке.",
        "",
        f"**Порог различимости.** ADR-018 измерил разброс между двумя "
        f"идентичными прогонами локальной модели: {threshold:.0%}. Разницу "
        f"меньше этой по одному прогону значимой называть нельзя.",
    ]
    return "\n".join(lines)


def format_sample_note(gold: list[GoldRecord], settings: Settings) -> str:
    """Честный размер выборки и предупреждение о смещённых записях."""
    by_lang: dict[str, int] = defaultdict(int)
    for record in gold:
        by_lang[record.lang] += 1

    assisted = sum(1 for record in gold if record.assisted)
    planned = settings.evals.gold_size

    lines = [
        f"Эталон: **{len(gold)} продуктов** из запланированных {planned} "
        f"({', '.join(f'{lang}: {count}' for lang, count in sorted(by_lang.items()))}).",
    ]
    if len(gold) < planned:
        lines.append(
            f"Разметка не закончена — метрики посчитаны на подмножестве в "
            f"{len(gold)} продуктов, и это указано здесь намеренно, а не "
            "выяснится при разборе."
        )
    if assisted:
        lines.append(
            f"**Из них с подсказкой модели: {assisted}.** Такая разметка слабее "
            "слепой: человек правил вывод модели, а не размечал независимо. "
            "Метрики систем на этих продуктах завышены."
        )
    return "\n".join(lines)


def build_report(
    gold: list[GoldRecord],
    predictions_by_system: dict[str, list[PredictionRecord]],
    index: AliasIndex,
    settings: Settings | None = None,
) -> str:
    """Собрать отчёт приёмки целиком.

    Args:
        gold: эталонные записи. Пустой список — не ошибка: разметку ведёт
            человек, и отчёт до её конца честно говорит, что считать нечего.
        predictions_by_system: ответы систем.
        index: словарь алиасов.
        settings: настройки; по умолчанию берутся глобальные.

    Returns:
        Отчёт в markdown. В README уезжает таблица сравнения из него.
    """
    settings = settings or get_settings()
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    header = ["# Сравнение систем извлечения состава", "", f"Дата: {stamp}", ""]

    if not gold:
        logger.info(
            "Отчёт собран без эталона", extra=safe_extra(systems=len(predictions_by_system))
        )
        return "\n".join(
            [
                *header,
                "Эталон пуст — разметка ещё не сделана, считать нечего.",
                "",
                "Метки ставит человек (правило 6 брифа). Разметка запускается "
                "командой `nutri-radar evals annotate --annotator <имя>`.",
            ]
        )

    if not predictions_by_system:
        logger.info("Отчёт собран без предсказаний", extra=safe_extra(gold=len(gold)))
        return "\n".join(
            [
                *header,
                "Нет ни одного файла предсказаний — сравнивать не с чем.",
                "",
                "Предсказания собираются командой `nutri-radar evals predict`.",
            ]
        )

    results = {
        system: score_system(gold, predictions, index)
        for system, predictions in predictions_by_system.items()
    }
    ordered = [results[system] for system in sorted(results)]
    system_profiles = {
        system: profile_system(gold, predictions, index)
        for system, predictions in predictions_by_system.items()
    }

    parts = [
        *header,
        format_sample_note(gold, settings),
        "",
        "## Сравнение систем",
        "",
        format_comparison(ordered),
        "",
        "Столбец «Нет ответа» — продукты, по которым система не ответила вовсе. "
        "Они входят в метрику как пустой ответ, а не выбрасываются: иначе "
        "молчание на трудных составах выглядело бы как безошибочность.",
        "",
        "## Разбивка по языкам",
        "",
    ]
    for result in ordered:
        parts += [format_by_language(result), ""]

    parts += [
        "## Вопросы, оставленные M2",
        "",
        format_zero_sugar(profile_gold(gold, index), system_profiles, results),
        "",
        format_language_gap(profile_gold(gold, index), system_profiles, settings),
    ]

    logger.info(
        "Отчёт собран",
        extra=safe_extra(gold=len(gold), systems=len(results)),
    )
    return "\n".join(parts)


def write_report(text: str, path: Path | None = None) -> Path:
    """Сохранить отчёт в `reports/`.

    Каталог в `.gitignore`: отчёт пересобирается из закоммиченных эталона
    и предсказаний одной командой, и держать в истории обе его версии —
    файл и данные, из которых он получен, — значит заводить два источника
    правды об одних числах.
    """
    file = path or REPORTS_DIR / f"evals_{datetime.now(UTC).strftime('%Y-%m-%d')}.md"
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(text + "\n", encoding="utf-8")
    logger.info("Отчёт записан", extra=safe_extra(path=str(file)))
    return file
