"""Штрихкод строкой: карточка продукта и кнопки под ней."""

from __future__ import annotations

import asyncio
import logging

import httpx
from aiogram import F, Router
from aiogram.types import CallbackQuery, InlineKeyboardMarkup, Message

from nutri_radar.bot import keyboards
from nutri_radar.bot.errors import log_failure, user_message
from nutri_radar.bot.texts import (
    NO_EXTRACTION,
    NO_INGREDIENTS,
    NOT_FOUND,
    SEARCHING,
)
from nutri_radar.config import Settings
from nutri_radar.errors import NutriRadarError
from nutri_radar.logging import safe_extra
from nutri_radar.openfoodfacts import normalize_barcode
from nutri_radar.retrieval.pipeline import search_by_text
from nutri_radar.retrieval.product_card import ProductCard, format_card, load_card

logger = logging.getLogger(__name__)

router = Router(name="barcode")

# Сколько похожих продуктов показывать по кнопке. Пять — столько же,
# сколько отдаёт поиск по умолчанию: отдельная величина здесь означала бы
# ещё одну настройку без причины.
SIMILAR_LIMIT = 5


async def _load(code: str, settings: Settings, client: httpx.AsyncClient) -> ProductCard | None:
    return await asyncio.wait_for(
        load_card(code, settings=settings, client=client),
        timeout=settings.bot.request_timeout_s,
    )


def _card_markup(card: ProductCard) -> InlineKeyboardMarkup:
    return keyboards.card_keyboard(card.code, has_extraction=card.has_extraction)


@router.message(F.text.regexp(r"^\s*\d{6,14}\s*$"))
async def by_barcode(message: Message, settings: Settings, http_client: httpx.AsyncClient) -> None:
    """Сообщение из одних цифр — это штрихкод, а не вопрос."""
    code = normalize_barcode(message.text or "")
    if code is None:  # pragma: no cover — фильтр уже проверил формат
        return

    placeholder = await message.answer(SEARCHING)
    try:
        card = await _load(code, settings, http_client)
    except (NutriRadarError, TimeoutError) as exc:
        log_failure(exc, where="by_barcode")
        await placeholder.edit_text(user_message(exc))
        return

    if card is None:
        await placeholder.edit_text(NOT_FOUND.format(code=code))
        return

    logger.info(
        "Карточка отправлена",
        extra=safe_extra(source=card.source.value, has_extraction=card.has_extraction),
    )
    # Правим сообщение-заглушку, а не шлём новое: у пользователя остаётся
    # одна карточка, а не лента из «Ищу…» и результата.
    await placeholder.edit_text(format_card(card), reply_markup=_card_markup(card))


@router.callback_query(F.data)
async def card_button(
    query: CallbackQuery, settings: Settings, http_client: httpx.AsyncClient
) -> None:
    """Нажатие кнопки под карточкой.

    Всё состояние лежит в самой кнопке: диалог нигде не хранится, потому
    что пользовательские данные не живут дольше ответа (бриф).
    """
    parsed = keyboards.parse_callback(query.data or "")
    if parsed is None or query.message is None:
        # Кнопка от прошлой версии бота или сообщение уже недоступно.
        await query.answer()
        return

    action, code = parsed
    try:
        card = await _load(code, settings, http_client)
    except (NutriRadarError, TimeoutError) as exc:
        log_failure(exc, where=f"card_button:{action}")
        await query.answer(user_message(exc).splitlines()[0], show_alert=True)
        return

    if card is None:
        await query.answer(NOT_FOUND.format(code=code).splitlines()[0], show_alert=True)
        return

    match action:
        case keyboards.ACTION_CARD:
            text, markup = format_card(card), _card_markup(card)
        case keyboards.ACTION_INGREDIENTS:
            text = card.ingredients_text or NO_INGREDIENTS
            markup = keyboards.back_keyboard(code)
        case keyboards.ACTION_SUGAR:
            text = _sugar_text(card)
            markup = keyboards.back_keyboard(code)
        case keyboards.ACTION_SIMILAR:
            text = await _similar_text(card, settings, http_client)
            markup = keyboards.back_keyboard(code)
        case _:
            await query.answer()
            return

    logger.debug("Кнопка обработана", extra=safe_extra(action=action))
    await query.message.edit_text(text, reply_markup=markup)  # type: ignore[union-attr]
    await query.answer()


def _sugar_text(card: ProductCard) -> str:
    """Развёрнутый ответ про формы сахара.

    Формулировка описательная: сколько РАЗНЫХ названий сахара встретилось
    в составе. Ни «много», ни «мало» — это была бы оценка, которую проект
    выносить не берётся.
    """
    if not card.has_extraction:
        return NO_EXTRACTION

    lines = [
        f"Разных форм сахара в составе: {card.distinct_sugar_forms}",
        "",
        "Это число разных названий сахара в одном составе — величина, "
        "которой в Open Food Facts нет: её считает модель проекта "
        "по нормализованным именам ингредиентов. Один и тот же сироп, "
        "названный дважды, считается одной формой.",
    ]
    if card.e_additives_count is not None:
        lines.append(f"\nДобавок с E-номером: {card.e_additives_count}")
    if card.ingredients_count:
        lines.append(f"Всего ингредиентов: {card.ingredients_count}")
    if card.extraction_model:
        lines.append(
            f"\nСостав разобран моделью {card.extraction_model} "
            f"(промпт {card.extraction_prompt_version})."
        )
    return "\n".join(lines)


async def _similar_text(card: ProductCard, settings: Settings, client: httpx.AsyncClient) -> str:
    """Похожие продукты по смыслу состава."""
    query = card.ingredients_text or card.product_name
    if not query:
        return "Не по чему искать похожие: ни состава, ни названия у продукта нет."

    result = await asyncio.wait_for(
        search_by_text(query, client=client, settings=settings, limit=SIMILAR_LIMIT + 1),
        timeout=settings.bot.request_timeout_s,
    )
    # Сам продукт из выдачи убираем: он ближе всех к себе, и показывать его
    # как «похожий» бессмысленно.
    hits = [hit for hit in result.hits if hit.code != card.code][:SIMILAR_LIMIT]
    if not hits:
        return "Похожих продуктов в базе не нашлось."

    lines = ["Похожие по составу продукты:", ""]
    lines += [
        f"[{hit.code}] {hit.product_name or 'без названия'}"
        + (f" — оценка по базе {hit.nutriscore_grade.upper()}" if hit.nutriscore_grade else "")
        for hit in hits
    ]
    lines.append("\nПришлите любой из штрихкодов, чтобы увидеть карточку.")
    return "\n".join(lines)
