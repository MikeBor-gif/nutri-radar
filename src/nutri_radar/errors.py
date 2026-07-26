"""Доменные исключения проекта.

Правило: наружу из слоя не протекают исключения внешних библиотек. Отказ
Postgres превращается в `DatabaseError`, отказ Ollama — в `LLMUnavailableError`.
Вызывающий код ловит доменное исключение и не знает, какой драйвер под ним.
"""

from __future__ import annotations


class NutriRadarError(Exception):
    """Корень иерархии. Позволяет поймать любую ошибку проекта одним except."""


class ConfigurationError(NutriRadarError):
    """Конфигурация невалидна или обязательная переменная не задана.

    Поднимается вместо того, чтобы наружу протекал `ValidationError` pydantic:
    сообщение должно объяснять, какой ключ и в каком формате нужен.
    """


class DatabaseError(NutriRadarError):
    """Отказ на стороне БД: соединение, запрос, миграция."""


class LLMUnavailableError(NutriRadarError):
    """Языковая модель недоступна или вернула отказ.

    Отдельно от `ExtractionError`: недоступность модели — инфраструктурная
    проблема, её лечит ретрай; невалидный разбор — проблема данных или промпта.
    """


class ExtractionError(NutriRadarError):
    """Модель ответила, но результат не проходит валидацию схемы."""


class DataSourceError(NutriRadarError):
    """Проблема с источником данных: дамп, дельта-экспорт, живой API OFF."""
