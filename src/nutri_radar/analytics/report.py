"""Отчёт M4: таблица сравнения подходов и графики.

Задача отчёта — сделать ответ на вопрос майлстоуна механическим, а не
делом впечатления. Вопрос звучит так: нужна ли здесь языковая модель.
Ответ читается из двух колонок рядом — обгон базлайна и стоимость.

**Две таблицы, а не одна.** Полный тест и общая подвыборка — разные
множества, и класть их в одну таблицу значит приглашать читателя сравнить
несравнимое. TF-IDF на 26 388 продуктах и LLM на 1000 — это два числа,
между которыми нельзя ставить знак сравнения.

**Обгон базлайна вместо голой accuracy.** При 44% большинства accuracy
почти не отличает работающую модель от вырожденной. Подход с обгоном
около нуля не работает, какой бы ни была его точность.

**Стоимость в одной единице.** Секунды CPU, GPU-часы и токены не
складываются, поэтому всё приводится к «секунд на 1000 продуктов»
на этом железе. Токены остаются отдельной колонкой — они единственное,
что переносится на другое железо и в облако.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

# Бэкенд без дисплея выбирается ДО импорта pyplot: в CI дисплея нет,
# и дефолтный интерактивный бэкенд там падает на импорте.
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from nutri_radar.analytics.features import (
    SET_COMMON,
    SET_FULL,
    Score,
    load_scores,
)
from nutri_radar.logging import safe_extra

logger = logging.getLogger(__name__)

REPORTS_DIR = Path("reports")


def format_comparison(scores: list[Score], title: str, note: str) -> str:
    """Таблица сравнения подходов на одном множестве."""
    if not scores:
        return f"### {title}\n\nРезультатов нет."

    lines = [
        f"### {title}",
        "",
        note,
        "",
        "| Подход | Продуктов | Accuracy | Обгон базлайна | macro-F1 "
        "| Обучение, с | Инференс, с/1000 | Токенов |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for score in sorted(scores, key=lambda s: -s.accuracy):
        tokens = score.input_tokens + score.output_tokens
        lines.append(
            f"| {score.system} | {score.products} | {score.accuracy:.1%} "
            f"| {score.lift:+.1f} | {score.macro_f1:.3f} "
            f"| {score.fit_seconds:.0f} | {score.seconds_per_1000:.1f} "
            f"| {tokens if tokens else '—'} |"
        )

    baseline = scores[0].baseline
    lines += [
        "",
        f"Baseline большинства класса на этом множестве — {baseline:.1%}. "
        "«Обгон» считается от него: подход с обгоном около нуля не работает, "
        "какой бы ни была его accuracy.",
    ]
    return "\n".join(lines)


def format_by_language(scores: list[Score], title: str) -> str:
    """Точность по языкам. Корпус смещён, и общее число это скрывает."""
    scored = [s for s in scores if s.by_lang]
    if not scored:
        return ""

    langs = sorted({lang for score in scored for lang in score.by_lang})
    lines = [
        f"### {title}",
        "",
        "| Подход | " + " | ".join(langs) + " |",
        "|---" * (len(langs) + 1) + "|",
    ]
    for score in sorted(scored, key=lambda s: -s.accuracy):
        cells = []
        for lang in langs:
            entry = score.by_lang.get(lang)
            cells.append(f"{entry[1]:.1%} ({entry[0]})" if entry else "—")
        lines.append(f"| {score.system} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "В скобках — сколько продуктов этого языка попало в множество. "
        "Корпус смещён (fr 48%, ru 0,2%), и на языках с горсткой продуктов "
        "доверительный интервал шире любой наблюдаемой разницы.",
    ]
    return "\n".join(lines)


def build_report(target: str, root: Path | None = None) -> str:
    """Собрать отчёт целиком из сохранённых результатов подходов."""
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    full = load_scores(target, SET_FULL, root)
    common = load_scores(target, SET_COMMON, root)

    parts = [
        f"# Предсказание `{target}` по тексту состава",
        "",
        f"Дата: {stamp}",
        "",
        "Признаки собраны **только из текста состава**. Ни одного нутриента: "
        "`nutriscore_grade` вычисляется по ним, и подмешать их значило бы "
        "заново вывести формулу вместо предсказания по составу.",
        "",
        format_comparison(
            full,
            "На полном тесте",
            "Здесь только подходы, которым полный тест по карману. "
            "LLM zero-shot в этой таблице нет намеренно: бриф запрещает "
            "гонять через модель больше нескольких тысяч продуктов.",
        ),
        "",
        format_comparison(
            common,
            "На общей подвыборке",
            "Одно и то же множество продуктов для всех подходов, "
            "воспроизводимое по seed. Только эти числа можно сравнивать "
            "между собой напрямую.",
        ),
        "",
        format_by_language(common, "По языкам (общая подвыборка)"),
    ]

    logger.info(
        "Отчёт собран",
        extra=safe_extra(target=target, full=len(full), common=len(common)),
    )
    return "\n".join(part for part in parts if part)


def plot_accuracy_vs_cost(scores: list[Score], target: str, path: Path | None = None) -> Path:
    """График «точность против стоимости».

    Главная картинка майлстоуна: на ней ответ виден без чтения таблицы.
    Ось стоимости логарифмическая — подходы различаются на порядки, и на
    линейной шкале дешёвые схлопнулись бы в одну точку у нуля.
    """
    file = path or REPORTS_DIR / f"m4_{target}_accuracy_vs_cost.png"
    file.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(figsize=(8, 5))
    for score in scores:
        # Ноль на логарифмической оси не рисуется. Подход, чей инференс
        # быстрее миллисекунды на продукт, ставится на границу шкалы.
        cost = max(score.seconds_per_1000, 0.01)
        axes.scatter(cost, score.accuracy * 100, s=90)
        axes.annotate(
            score.system,
            (cost, score.accuracy * 100),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=9,
        )

    if scores:
        axes.axhline(
            scores[0].baseline * 100,
            linestyle="--",
            linewidth=1,
            label=f"базлайн большинства класса ({scores[0].baseline:.1%})",
        )
        axes.legend(loc="lower right", fontsize=9)

    axes.set_xscale("log")
    axes.set_xlabel("Стоимость инференса, секунд на 1000 продуктов (логарифм)")
    axes.set_ylabel("Accuracy, %")
    axes.set_title(f"Точность против стоимости: {target}")
    axes.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(file, dpi=140)
    plt.close(figure)

    logger.info("График сохранён", extra=safe_extra(path=str(file)))
    return file


def plot_confusion(score: Score, target: str, path: Path | None = None) -> Path:
    """Матрица ошибок одного подхода.

    Нужна там, где accuracy молчит: она показывает, путает ли модель
    соседние классы (c и d) или дальние (a и e). Первое — модель работает
    и ошибается по краям, второе — не работает вовсе.
    """
    safe = score.system.replace(":", "_").replace("/", "_").replace(" ", "_")
    file = path or REPORTS_DIR / f"m4_{target}_confusion_{safe}.png"
    file.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(figsize=(6, 5))
    image = axes.imshow(score.matrix, cmap="Blues")
    axes.set_xticks(range(len(score.labels)), score.labels)
    axes.set_yticks(range(len(score.labels)), score.labels)
    axes.set_xlabel("Предсказано")
    axes.set_ylabel("Истина")
    axes.set_title(f"{score.system}: {target}")

    for i, row in enumerate(score.matrix):
        for j, value in enumerate(row):
            axes.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > max(map(max, score.matrix)) / 2 else "black",
            )

    figure.colorbar(image, ax=axes)
    figure.tight_layout()
    figure.savefig(file, dpi=140)
    plt.close(figure)

    logger.info("Матрица ошибок сохранена", extra=safe_extra(path=str(file)))
    return file


def write_report(text: str, target: str, path: Path | None = None) -> Path:
    """Записать отчёт на диск."""
    file = path or REPORTS_DIR / f"m4_{target}.md"
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(text, encoding="utf-8")
    logger.info("Отчёт записан", extra=safe_extra(path=str(file)))
    return file
