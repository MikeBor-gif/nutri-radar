"""Внутренняя модель источника данных.

Ключевое решение ADR-004: полный дамп и дельта-экспорты имеют **разные схемы**.

* Parquet: `ingredients_text` — `LIST<STRUCT(lang, text)>`, `nutriments` —
  список структур с полем `100g`.
* Дельты: JSONL в MongoDB-форме, состав лежит плоскими ключами
  `ingredients_text_en`, `ingredients_text_ru`, нутриенты — плоским объектом.

Оба адаптера сводят данные к `RawProduct`. Ниже слоя нормализации разница
форматов не видна — иначе ветвление по формату протечёт в бизнес-логику.

Здесь же живёт логика пригодности состава (`has_usable_ingredients`): она
проверяется тестами без DuckDB и без БД, потому что это правило предметной
области, а не деталь чтения файла.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

SourceKind = Literal["parquet", "delta"]

# Служебная запись внутри многоязычных полей Parquet. Дублирует текст главного
# языка продукта и языком НЕ является: подтверждено разведкой — 19 180 записей
# 'main' на 20 000 строк. Если считать её языком, статистика удвоится.
MAIN_LANG_MARKER = "main"

# Штрихкоды в OFF — цифры; продукты без штрихкода получают внутренний номер
# с префиксом 200. Записи с пустым или нецифровым кодом бесполезны: по ним
# нельзя ни сослаться, ни дедуплицировать.
_BARCODE_RE = re.compile(r"^\d{4,20}$")

# Допустимые значения оценки. Всё остальное (`unknown`, `not-applicable`) —
# служебное и приводится к None, чтобы не попасть в обучение как класс.
_VALID_GRADES = frozenset({"a", "b", "c", "d", "e"})


class RawProduct(BaseModel):
    """Продукт в форме, единой для всех источников."""

    # Модель создаётся из внешних данных сомнительного качества, поэтому
    # неизвестные поля игнорируем, а не падаем: схема дампа имеет пять версий
    # (999, 1001-1004) и состав полей у записей различается.
    model_config = ConfigDict(extra="ignore", frozen=True)

    code: str

    # --- языки и тексты -----------------------------------------------------
    lang: str | None = None
    # Язык → текст. Словарь, а не список структур: так обе формы источников
    # сходятся к одному виду, а доступ по языку становится тривиальным.
    ingredients_text: dict[str, str] = Field(default_factory=dict)
    product_name: dict[str, str] = Field(default_factory=dict)
    generic_name: dict[str, str] = Field(default_factory=dict)
    brands: str | None = None

    # --- классификация ------------------------------------------------------
    categories_tags: list[str] = Field(default_factory=list)
    food_groups_tags: list[str] = Field(default_factory=list)
    countries_tags: list[str] = Field(default_factory=list)
    labels_tags: list[str] = Field(default_factory=list)

    # --- готовые метки для M4 ----------------------------------------------
    nutriscore_grade: str | None = None
    nutriscore_score: int | None = None
    nova_group: int | None = None

    # --- нутриенты ----------------------------------------------------------
    # Имя нутриента → значение на 100 г. Плоский словарь: список структур
    # Parquet и плоский объект дельт сходятся именно здесь.
    nutriments: dict[str, float] = Field(default_factory=dict)
    nutrition_data_per: str | None = None
    no_nutrition_data: bool = False

    # --- baseline собственного парсера OFF (ADR-006) ------------------------
    # Забираем целиком: это то, с чем в M3 будет соревноваться наша LLM,
    # и критерий отбора LLM-корпуса (unknown_ingredients_n > 0).
    ingredients_tags: list[str] = Field(default_factory=list)
    ingredients_original_tags: list[str] = Field(default_factory=list)
    additives_tags: list[str] = Field(default_factory=list)
    allergens_tags: list[str] = Field(default_factory=list)
    traces_tags: list[str] = Field(default_factory=list)
    ingredients_analysis_tags: list[str] = Field(default_factory=list)
    ingredients_n: int | None = None
    known_ingredients_n: int | None = None
    unknown_ingredients_n: int | None = None
    additives_n: int | None = None
    ingredients_json: str | None = None
    with_sweeteners: int | None = None
    with_non_nutritive_sweeteners: int | None = None

    # --- качество и служебное ----------------------------------------------
    obsolete: bool = False
    completeness: float | None = None
    data_quality_errors: list[str] = Field(default_factory=list)
    unique_scans_n: int | None = None
    popularity_key: int | None = None
    rev: int | None = None
    last_modified_t: int | None = None
    created_t: int | None = None
    schema_version: int | None = None

    source: SourceKind

    # ------------------------------------------------------------------ #
    # Валидация
    # ------------------------------------------------------------------ #

    @field_validator("code", mode="before")
    @classmethod
    def _normalize_code(cls, value: object) -> str:
        if value is None:
            raise ValueError("Штрихкод отсутствует")
        code = str(value).strip()
        if not _BARCODE_RE.match(code):
            raise ValueError(f"Штрихкод {code!r} не похож на код продукта: ожидаются 4-20 цифр")
        return code

    @field_validator(
        "ingredients_text",
        "product_name",
        "generic_name",
        mode="after",
    )
    @classmethod
    def _drop_main_marker_and_blanks(cls, value: dict[str, str]) -> dict[str, str]:
        """Убрать служебный `main` и пустые тексты.

        Чистка делается здесь, а не в адаптерах: иначе каждый источник
        повторял бы её по-своему, и правило разъехалось бы.
        """
        return {
            lang: text.strip()
            for lang, text in value.items()
            if lang != MAIN_LANG_MARKER and text and text.strip()
        }

    @field_validator("nutriments", mode="before")
    @classmethod
    def _drop_missing_nutrients(cls, value: object) -> object:
        """Отсутствующий нутриент не должен превращаться в ноль.

        Ноль — осмысленное значение («сахара 0 г»), поэтому `None` из источника
        выбрасывается из словаря, а не подменяется нулём: иначе в M4 пропуски
        уедут в обучение как настоящие измерения.

        Режим `before` обязателен: источники присылают `None` для отсутствующих
        нутриентов, и на приведении к `float` запись упала бы ещё до валидатора.
        """
        if not isinstance(value, dict):
            return value
        return {name: v for name, v in value.items() if v is not None}

    @field_validator("nutriscore_grade", mode="before")
    @classmethod
    def _normalize_grade(cls, value: object) -> str | None:
        """`nutriscore_grade` принимает не только `a`-`e`.

        В базе встречаются `unknown` и `not-applicable`. Приводим их к `None`
        здесь, чтобы слой аналитики не фильтровал служебные строки вручную и
        не обучался на них как на классах.
        """
        if value is None:
            return None
        normalized = str(value).strip().lower()
        return normalized if normalized in _VALID_GRADES else None

    # ------------------------------------------------------------------ #
    # Предметные правила
    # ------------------------------------------------------------------ #

    @property
    def languages(self) -> list[str]:
        """Языки, на которых есть непустой текст состава."""
        return sorted(self.ingredients_text)

    def usable_ingredients_languages(self, languages: list[str], min_length: int) -> list[str]:
        """Языки из списка, где состав достаточно длинный, чтобы быть полезным."""
        wanted = set(languages)
        return sorted(
            lang
            for lang, text in self.ingredients_text.items()
            if lang in wanted and len(text) >= min_length
        )

    def has_usable_ingredients(self, languages: list[str], min_length: int) -> bool:
        """Годится ли продукт для корпуса по критерию состава."""
        return bool(self.usable_ingredients_languages(languages, min_length))

    def pick_ingredients_text(
        self, languages: list[str], min_length: int
    ) -> tuple[str, str] | None:
        """Выбрать текст состава для нормализованной таблицы.

        Приоритет: главный язык продукта, затем порядок из настроек. Возвращает
        пару (язык, текст) — язык обязателен, иначе непонятно, что именно
        отдали в LLM.
        """
        usable = self.usable_ingredients_languages(languages, min_length)
        if not usable:
            return None

        if self.lang in usable:
            return self.lang, self.ingredients_text[self.lang]
        for lang in languages:
            if lang in usable:
                return lang, self.ingredients_text[lang]
        first = usable[0]
        return first, self.ingredients_text[first]

    def pick_name(self, languages: list[str]) -> str | None:
        """Название на главном языке, иначе по порядку настроек, иначе любое."""
        if self.lang and self.lang in self.product_name:
            return self.product_name[self.lang]
        for lang in languages:
            if lang in self.product_name:
                return self.product_name[lang]
        return next(iter(self.product_name.values()), None)

    @property
    def has_nutrition_data(self) -> bool:
        """Есть ли пригодная таблица питательности.

        Флаг источника `no_nutrition_data` не всегда выставлен, поэтому
        проверяем ещё и фактическое наличие значений.
        """
        return not self.no_nutrition_data and bool(self.nutriments)

    @property
    def is_quality_ok(self) -> bool:
        """Прошёл ли продукт отсечки качества, не зависящие от настроек."""
        return not self.obsolete and not self.data_quality_errors

    def to_raw_json(self) -> dict[str, Any]:
        """Снимок для `products_raw`.

        Сырое не перезаписываем деструктивно (раздел 7 брифа), поэтому в базу
        уезжает полный набор отобранных полей, а не только нормализованные.
        """
        return self.model_dump(mode="json")
