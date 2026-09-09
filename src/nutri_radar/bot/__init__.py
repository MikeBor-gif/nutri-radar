"""Точка входа Telegram: бот на aiogram 3 (M7).

Логики здесь нет — только разбор сообщения, вызов слайса и форматирование
ответа (`ARCHITECTURE.md`).
"""

from nutri_radar.bot.app import create_dispatcher, main, run_bot

__all__ = ["create_dispatcher", "main", "run_bot"]
