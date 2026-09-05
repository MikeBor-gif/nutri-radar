"""Предсказание `nova_group` по тексту состава.

Вторая задача майлстоуна. Машинерия та же, что у `nutriscore_grade`, —
и это не лень, а требование: подходы обязаны сравниваться одним и тем же
кодом, иначе разница их чисел включит ещё и разницу реализаций.

**Чем эта задача отличается от оценки Nutri-Score.** NOVA описывает степень
переработки, и она гораздо ближе к тексту состава: длинный список с
эмульгаторами, ароматизаторами и модифицированными крахмалами — это
определение группы 4, а не косвенный её признак. Nutri-Score же считается
по нутриентам, которых в тексте нет вовсе. Поэтому ожидание обратное:
здесь модель должна работать заметно лучше, и если это не так — вопрос
не к данным, а к признакам.

**Класс 2 исключается, и это записано до прогона.** Он встречается 25 раз
на 138 254 продукта. Объединять его с классом 1 неверно по существу:
NOVA 1 — необработанная еда, NOVA 2 — кулинарные ингредиенты (масло, соль,
сахар), которые не едят сами по себе. Склеить их ради красивой цифры
значит испортить разметку. Исключение видно в отчёте, а не спрятано.
"""

from __future__ import annotations

import logging

import pandas as pd

from nutri_radar.analytics.dataset import drop_rare_classes
from nutri_radar.config import Settings, get_settings
from nutri_radar.logging import safe_extra

logger = logging.getLogger(__name__)

TARGET = "nova_group"


def prepare_nova(
    frame: pd.DataFrame,
    settings: Settings | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Подготовить набор под задачу NOVA.

    Returns:
        Пара «набор, исключённые классы». Второй элемент уходит в отчёт:
        исключение обязано быть видимым числом, а не умолчанием.
    """
    settings = settings or get_settings()
    prepared, dropped = drop_rare_classes(frame, TARGET, settings)
    logger.info(
        "Набор NOVA подготовлен",
        extra=safe_extra(rows=len(prepared), dropped=dropped or "ничего"),
    )
    return prepared, dropped


def format_dropped(dropped: dict[str, int], total: int) -> str:
    """Строка для отчёта об исключённых классах."""
    if not dropped:
        return ""
    parts = ", ".join(f"«{label}» — {count}" for label, count in sorted(dropped.items()))
    share = sum(dropped.values()) / total if total else 0.0
    return (
        f"**Исключены классы:** {parts} (всего {sum(dropped.values())} продуктов, "
        f"{share:.3%} набора). Решение принято до прогона: на классе с таким "
        "числом примеров модель ничему не научится, а macro-F1 усреднится "
        "по величине, про которую нельзя сказать ничего. Объединение с соседним "
        "классом отвергнуто: NOVA 1 — необработанная еда, NOVA 2 — кулинарные "
        "ингредиенты, это разные вещи."
    )
