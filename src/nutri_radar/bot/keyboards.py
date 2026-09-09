"""Инлайн-клавиатура карточки продукта.

**Одно сообщение вместо ленты.** Кнопки правят то же самое сообщение через
`edit_message`, а не досылают новые. Причина не в красоте: пользователь,
нажавший три кнопки, иначе получает четыре сообщения и теряет карточку
в прокрутке — а карточка и есть ответ.

**Штрихкод в `callback_data`.** Состояние диалога нигде не хранится, и это
осознанно: пользовательские данные не живут дольше, чем нужно для ответа
(бриф). Всё, что нужно для обработки нажатия, лежит в самой кнопке.
Ограничение Telegram — 64 байта на `callback_data`; штрихкод занимает
до 14 символов, префикс — ещё несколько, запас многократный.
"""

from __future__ import annotations

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from nutri_radar.bot.texts import (
    BUTTON_BACK,
    BUTTON_INGREDIENTS,
    BUTTON_SIMILAR,
    BUTTON_SUGAR,
)

# Разделитель префикса и штрихкода. Двоеточие не встречается в штрихкодах,
# поэтому разбор однозначен.
SEPARATOR = ":"

ACTION_CARD = "card"
ACTION_INGREDIENTS = "ingredients"
ACTION_SUGAR = "sugar"
ACTION_SIMILAR = "similar"


def callback(action: str, code: str) -> str:
    """Собрать `callback_data` кнопки."""
    return f"{action}{SEPARATOR}{code}"


def parse_callback(data: str) -> tuple[str, str] | None:
    """Разобрать `callback_data`. Не наш формат — `None`.

    Возвращает `None`, а не бросает: в чат могут прилететь кнопки
    от прошлой версии бота, и падать из-за них незачем.
    """
    action, separator, code = data.partition(SEPARATOR)
    if not separator or not code:
        return None
    return action, code


def card_keyboard(code: str, *, has_extraction: bool) -> InlineKeyboardMarkup:
    """Кнопки под карточкой продукта.

    Кнопка форм сахара показывается всегда, в том числе когда разбора нет:
    нажатие объяснит, почему числа нет. Спрятать кнопку значило бы оставить
    пользователя гадать, почему у одних продуктов она есть, а у других нет.
    """
    buttons = [
        [
            InlineKeyboardButton(
                text=BUTTON_INGREDIENTS, callback_data=callback(ACTION_INGREDIENTS, code)
            ),
            InlineKeyboardButton(
                text=BUTTON_SUGAR if has_extraction else f"{BUTTON_SUGAR} ?",
                callback_data=callback(ACTION_SUGAR, code),
            ),
        ],
        [InlineKeyboardButton(text=BUTTON_SIMILAR, callback_data=callback(ACTION_SIMILAR, code))],
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def back_keyboard(code: str) -> InlineKeyboardMarkup:
    """Единственная кнопка возврата к карточке."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=BUTTON_BACK, callback_data=callback(ACTION_CARD, code))]
        ]
    )
