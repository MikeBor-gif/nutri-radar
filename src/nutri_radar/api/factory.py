"""Готовое приложение для uvicorn.

`uvicorn` с `--reload` требует строку импорта, а не объект: перезагрузка
переимпортирует модуль в новом процессе. Поэтому здесь один модуль
с единственным именем `app`, а не вызов `create_app()` по месту.
"""

from nutri_radar.api.app import create_app
from nutri_radar.config import get_settings

app = create_app(get_settings())
