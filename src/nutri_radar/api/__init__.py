"""Точка входа HTTP: FastAPI поверх готовых слайсов (M7).

Логики здесь нет — только разбор запроса, вызов слайса и форматирование
ответа (`ARCHITECTURE.md`).
"""

from nutri_radar.api.app import create_app

__all__ = ["create_app"]
