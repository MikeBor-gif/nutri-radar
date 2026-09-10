"""Карточка продукта по штрихкоду: то, что показывают человеку.

**Почему карточка берётся из локальной БД, а не из живого API.** Главная
величина проекта — **число разных форм сахара** — в Open Food Facts
отсутствует: её считает наша модель на M2, и лежит она в `product_extraction`.
Собирать карточку из API значило бы выбросить ровно то, ради чего проект
существует, и показать пересказ чужих полей.

**Фолбэк в живой API нужен, потому что корпус мал по построению.** В базе
147 тысяч продуктов из 4,63 млн строк дампа — большинство реальных сканов
в него не попадает. Продукт вне корпуса отдаётся из Open Food Facts с явной
пометкой источника: у него не будет ни форм сахара, ни числа добавок,
и притворяться, что данные равноценны, нельзя.

**Формулировки описательные.** «В составе 4 разные формы сахара», а не
«много сахара»; «оценка по базе — C», а не «плохой продукт». Проверяется
тестом против `nutri_radar.wording`, а не вычиткой.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum

import httpx

from nutri_radar.config import Settings, get_settings
from nutri_radar.db.repositories.extraction import ExtractionRepository, ExtractionRow
from nutri_radar.db.repositories.product import ProductRepository, ProductSummary
from nutri_radar.db.session import get_session
from nutri_radar.logging import safe_extra
from nutri_radar.openfoodfacts import OffProduct, fetch_product, normalize_barcode
from nutri_radar.retrieval.profile import strip_markup

logger = logging.getLogger(__name__)


class CardSource(StrEnum):
    """Откуда взяты данные карточки. Показывается пользователю всегда.

    Данные из корпуса разобраны моделью и сопоставимы между продуктами;
    данные из живого API — свежий срез чужих полей без нашего разбора.
    Это разные по надёжности вещи, и молча смешивать их нельзя.
    """

    CORPUS = "corpus"
    OPENFOODFACTS = "openfoodfacts"


SOURCE_LABELS = {
    CardSource.CORPUS: "из базы проекта (состав разобран моделью)",
    CardSource.OPENFOODFACTS: "из Open Food Facts, в корпус проекта не входит",
}


@dataclass(frozen=True)
class ProductCard:
    """Продукт в том виде, в каком его показывают человеку."""

    code: str
    source: CardSource

    product_name: str | None = None
    brands: str | None = None
    lang: str | None = None
    ingredients_text: str | None = None

    nutriscore_grade: str | None = None
    nova_group: int | None = None

    # Есть только у продуктов, прошедших через извлечение. `None` означает
    # «не считали», и это не то же самое, что 0 — «сахара не нашли».
    # Показывать ноль вместо неизвестности значило бы соврать числом.
    distinct_sugar_forms: int | None = None
    e_additives_count: int | None = None
    ingredients_count: int | None = None
    allergens: list[str] = field(default_factory=list)

    # Чем разобран состав. Уезжает в карточку, потому что число форм сахара
    # без имени модели — это число неизвестно чьего разбора.
    extraction_model: str | None = None
    extraction_prompt_version: str | None = None

    @property
    def has_extraction(self) -> bool:
        return self.distinct_sugar_forms is not None

    @property
    def title(self) -> str:
        return self.product_name or "без названия"


def card_from_corpus(
    product: ProductSummary, extraction: ExtractionRow | None = None
) -> ProductCard:
    """Собрать карточку из данных корпуса. Чистая функция."""
    return ProductCard(
        code=product.code,
        source=CardSource.CORPUS,
        product_name=product.product_name,
        brands=product.brands,
        lang=product.ingredients_text_lang or product.lang,
        # Разметка снимается: треть корпуса приходит с
        # `<span class="allergen">`, и показывать её человеку — мусор.
        # Само слово-аллерген при этом остаётся, теряется только обёртка.
        ingredients_text=strip_markup(product.ingredients_text) or None,
        nutriscore_grade=product.nutriscore_grade,
        nova_group=product.nova_group,
        distinct_sugar_forms=extraction.distinct_sugar_forms if extraction else None,
        e_additives_count=extraction.e_additives_count if extraction else None,
        ingredients_count=extraction.ingredients_count if extraction else None,
        # Аллергены из разбора, если он есть; иначе — теги парсера OFF.
        # Порядок именно такой: наш разбор нормализован, теги — нет.
        allergens=list(extraction.allergens) if extraction else list(product.allergens_tags),
        extraction_model=extraction.model_name if extraction else None,
        extraction_prompt_version=extraction.prompt_version if extraction else None,
    )


def card_from_off(product: OffProduct) -> ProductCard:
    """Собрать карточку из ответа живого API. Чистая функция.

    Ни форм сахара, ни числа добавок здесь быть не может: состав не
    разбирался. Поля остаются `None`, и карточка честно это показывает.
    """
    return ProductCard(
        code=product.code,
        source=CardSource.OPENFOODFACTS,
        product_name=product.product_name,
        brands=product.brands,
        # Живой API отдаёт ту же разметку, что и дамп.
        ingredients_text=strip_markup(product.ingredients_text) or None,
        nutriscore_grade=product.nutriscore_grade,
        nova_group=product.nova_group,
    )


def format_card(card: ProductCard) -> str:
    """Карточка в виде текста. Чистая функция.

    Формулировки описательные: число, единица измерения и источник.
    Ни одного оценочного суждения — см. `nutri_radar.wording`.
    """
    lines = [f"[{card.code}] {card.title}"]
    if card.brands:
        lines.append(f"Бренд: {card.brands.split(',')[0].strip()}")

    if card.nutriscore_grade:
        lines.append(f"Оценка Nutri-Score по базе: {card.nutriscore_grade.upper()}")
    if card.nova_group:
        lines.append(f"Группа переработки NOVA: {card.nova_group}")

    if card.has_extraction:
        # Ключевая величина проекта. Формулировка описательная: сколько
        # РАЗНЫХ названий сахара встретилось в составе, а не «много ли».
        lines.append(f"Разных форм сахара в составе: {card.distinct_sugar_forms}")
        if card.e_additives_count is not None:
            lines.append(f"Добавок с E-номером: {card.e_additives_count}")
        if card.ingredients_count:
            lines.append(f"Всего ингредиентов: {card.ingredients_count}")
    else:
        lines.append("Состав моделью не разбирался: форм сахара и добавок нет.")

    if card.allergens:
        lines.append(f"Аллергены по данным источника: {', '.join(card.allergens)}")
    if card.ingredients_text:
        lines.append(f"Состав: {card.ingredients_text}")

    lines.append(f"Источник: {SOURCE_LABELS[card.source]}")
    if card.extraction_model:
        lines.append(
            f"Состав разобран моделью {card.extraction_model} "
            f"(промпт {card.extraction_prompt_version})"
        )
    return "\n".join(lines)


async def load_card(
    barcode: str,
    *,
    settings: Settings | None = None,
    client: httpx.AsyncClient | None = None,
    allow_remote: bool = True,
) -> ProductCard | None:
    """Собрать карточку по штрихкоду: сначала корпус, потом живой API.

    Args:
        barcode: ввод пользователя. Не похож на штрихкод — `None`.
        settings: настройки.
        client: клиент HTTP для запроса к OFF. Передаётся тестом, чтобы
            вызов не уходил в сеть (правило 4 брифа).
        allow_remote: разрешён ли фолбэк в живой API. Выключается там, где
            поход в сеть неуместен, — например при массовой обработке.

    Returns:
        Карточка или `None`, если штрихкод не найден нигде.

    Raises:
        DataSourceError: живой API недоступен. Наверх, а не в `None`:
            «не нашли» и «не смогли спросить» — разные факты.
    """
    settings = settings or get_settings()
    code = normalize_barcode(barcode)
    if code is None:
        logger.debug("Ввод не похож на штрихкод", extra=safe_extra(chars=len(str(barcode))))
        return None

    async with get_session(settings.db) as session:
        product = await ProductRepository(session).get_by_code(code)
        extraction = await ExtractionRepository(session).latest_by_code(code) if product else None

    if product is not None:
        card = card_from_corpus(product, extraction)
        logger.info(
            "Карточка собрана",
            extra=safe_extra(
                code=code,
                source=card.source.value,
                has_extraction=card.has_extraction,
                sugar_forms=card.distinct_sugar_forms,
            ),
        )
        return card

    if not allow_remote:
        logger.info(
            "Продукта нет в корпусе, фолбэк выключен",
            extra=safe_extra(code=code),
        )
        return None

    logger.debug("Продукта нет в корпусе, идём в живой API", extra=safe_extra(code=code))
    remote = await fetch_product(code, timeout_s=settings.agent.tool_timeout_s, client=client)
    if remote is None:
        logger.info("Штрихкод не найден нигде", extra=safe_extra(code=code))
        return None

    card = card_from_off(remote)
    logger.info(
        "Карточка собрана",
        extra=safe_extra(code=code, source=card.source.value, has_extraction=False),
    )
    return card
