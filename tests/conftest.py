"""Общие фикстуры.

Принципы (правило 4 брифа):

* тест не ходит в сеть — Ollama подменяется через `httpx.MockTransport`;
* настройки собираются из явного словаря, а не из `.env` разработчика, иначе
  результат теста зависит от машины;
* тесты, которым нужен поднятый Postgres, помечены маркером `integration`
  и по умолчанию не отбираются.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx
import pytest

from nutri_radar.config import (
    AnthropicSettings,
    AppSettings,
    DatabaseSettings,
    IngestSettings,
    LLMSettings,
    OllamaSettings,
    Settings,
)

# Значения фикстур намеренно отличаются от значений по умолчанию: если код
# случайно возьмёт настройки из окружения вместо фикстуры, тест это заметит.
TEST_DB_NAME = "nutri_radar_test"
TEST_MODEL = "test-model:3b"
TEST_EMBEDDING_MODEL = "test-embeddings"


# Ключи, за которыми стоят настоящие секреты. Переменная окружения
# в pydantic-settings приоритетнее файла `.env`, поэтому пустое значение
# здесь перекрывает то, что лежит у разработчика на диске.
_SECRET_KEYS = (
    "BOT__TOKEN",
    "ANTHROPIC__API_KEY",
    "LANGFUSE__SECRET_KEY",
)

# Пароль к базе стоит особняком: юнит-тестам он не нужен и затирается вместе
# с остальными, а интеграционным нужен по устройству — без верного пароля
# до контейнера не дойти, и фикстура `migrated_database` падает с
# `InvalidPasswordError`. Именно так и случилось: ключ добавили в общий
# список, CI позеленел на юнит-тестах и покраснел на интеграционных.
_DB_PASSWORD_KEY = "DB__PASSWORD"


def keys_to_blank(*, is_integration: bool) -> tuple[str, ...]:
    """Какие переменные окружения затирать для теста.

    Вынесено из фикстуры отдельной функцией, чтобы правило проверялось
    тестом без поднятой базы: сама фикстура автоюзная, и «что она сделала»
    изнутри обычного теста не видно.
    """
    if is_integration:
        return _SECRET_KEYS
    return (*_SECRET_KEYS, _DB_PASSWORD_KEY)


@pytest.fixture(autouse=True)
def _no_real_secrets(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ни один тест не должен видеть настоящий секрет разработчика.

    Не паранойя, а разбор случившегося. Тест `test_secret_values` строил
    `Settings()` без явной группы `bot`, та дочитывала настоящий `.env`,
    и когда там появился токен бота, тест упал — напечатав токен в diff,
    то есть в вывод pytest открытым текстом. Пока ключи были пустыми,
    дефект был невидим.

    Причина общая: любая группа настроек, не заданная в тесте явно,
    берётся с машины, на которой тест запущен. Фикстура закрывает это
    для всех тестов сразу, а не только для того, который уже обжёгся.

    Исключение одно — `DB__PASSWORD` у тестов с маркером `integration`:
    см. комментарий у `_DB_PASSWORD_KEY`. Токен бота, ключ Anthropic
    и ключ Langfuse затираются и у них тоже: настоящие секреты
    интеграционным тестам не нужны.
    """
    is_integration = request.node.get_closest_marker("integration") is not None
    for key in keys_to_blank(is_integration=is_integration):
        monkeypatch.setenv(key, "")


@pytest.fixture
def settings() -> Settings:
    """Настройки, собранные явно. От `.env` и окружения не зависят."""
    return Settings(
        app=AppSettings(log_level="DEBUG", environment="ci", json_logs=False),
        db=DatabaseSettings(
            host="localhost",
            port=5435,
            user="nutri",
            password="test_password_value",
            name=TEST_DB_NAME,
            pool_size=2,
            echo_sql=False,
            # Короткий таймаут: тесты проверяют отказ недоступной базы,
            # и ждать дефолтные 10 секунд на каждый тест незачем.
            connect_timeout_s=0.5,
        ),
        ollama=OllamaSettings(
            base_url="http://ollama.invalid:11434",
            model=TEST_MODEL,
            embedding_model=TEST_EMBEDDING_MODEL,
            num_ctx=4096,
            max_concurrency=1,
        ),
        anthropic=AnthropicSettings(api_key=None),
        ingest=IngestSettings(languages=["en", "ru"], batch_size=10),
        llm=LLMSettings(provider="ollama"),
    )


def _ollama_tags_response(models: list[str]) -> httpx.Response:
    payload = {"models": [{"name": name} for name in models]}
    return httpx.Response(200, content=json.dumps(payload).encode())


@pytest.fixture
def fake_ollama() -> Iterator[httpx.AsyncClient]:
    """Клиент Ollama, отвечающий обеими настроенными моделями.

    `MockTransport` подменяет транспорт целиком, поэтому никакой сетевой
    активности не происходит — это надёжнее патча приватных атрибутов.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags", f"Неожиданный запрос: {request.url}"
        return _ollama_tags_response([f"{TEST_MODEL}", f"{TEST_EMBEDDING_MODEL}:latest"])

    # base_url обязателен: health-check запрашивает относительный путь
    # /api/tags, и без базы httpx бросит ValueError вместо запроса.
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://ollama.invalid:11434"
    )
    yield client


@pytest.fixture
def fake_ollama_without_models() -> Iterator[httpx.AsyncClient]:
    """Ollama отвечает, но нужных моделей в ней нет."""

    def handler(request: httpx.Request) -> httpx.Response:
        return _ollama_tags_response(["совсем-другая-модель:latest"])

    # base_url обязателен: health-check запрашивает относительный путь
    # /api/tags, и без базы httpx бросит ValueError вместо запроса.
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://ollama.invalid:11434"
    )
    yield client


@pytest.fixture
def unavailable_ollama() -> Iterator[httpx.AsyncClient]:
    """Ollama недоступна: транспорт бросает ошибку соединения."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Ollama недоступна (подделка для теста)")

    # base_url обязателен: health-check запрашивает относительный путь
    # /api/tags, и без базы httpx бросит ValueError вместо запроса.
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://ollama.invalid:11434"
    )
    yield client


# =============================================================================
# Интеграционные фикстуры. Требуют поднятого Postgres.
# =============================================================================

# Суффикс обязателен: без него тест мог бы снести рабочую базу разработчика.
_REQUIRED_TEST_DB_SUFFIX = "_test"


async def _run_maintenance(settings: Settings, statements: list[str]) -> None:
    """Выполнить DDL уровня кластера на служебной базе `postgres`.

    CREATE/DROP DATABASE не работают внутри транзакции, а asyncpg по умолчанию
    работает в автокоммите — поэтому используем его напрямую, минуя SQLAlchemy.
    """
    import asyncpg

    conn = await asyncpg.connect(
        host=settings.db.host,
        port=settings.db.port,
        user=settings.db.user,
        password=settings.db.password.get_secret_value(),
        database="postgres",
    )
    try:
        for statement in statements:
            await conn.execute(statement)
    finally:
        await conn.close()


@pytest.fixture
def integration_settings() -> Settings:
    """Настройки для интеграционных тестов.

    В отличие от герметичной фикстуры `settings`, здесь берутся РЕАЛЬНЫЕ
    параметры подключения из `.env`: без верного пароля и порта до контейнера
    не дойти. Подменяется только имя базы — на него навешена защита по суффиксу.
    """
    from nutri_radar.config import get_settings

    # Кэш сбрасываем намеренно. `get_settings` кэширован на процесс, и в общем
    # прогоне (`pytest -m ""`) настройки мог собрать любой юнит-тест — с уже
    # затёртым `DB__PASSWORD`. Тогда интеграционные получили бы пустой пароль
    # не из своего окружения, а из чужого кэша, и падение зависело бы
    # от порядка тестов.
    get_settings.cache_clear()

    real = get_settings()
    return real.model_copy(
        update={
            "db": DatabaseSettings(
                **{
                    **real.db.model_dump(),
                    "name": f"{real.db.name}{_REQUIRED_TEST_DB_SUFFIX}",
                    "connect_timeout_s": 5.0,
                }
            )
        }
    )


@pytest.fixture
def migrated_database(integration_settings: Settings) -> Iterator[str]:
    """Создать отдельную базу, накатить миграции, отдать DSN, затем удалить.

    Мигрируем НЕ рабочую базу: имя обязано заканчиваться на `_test`. Без этой
    проверки `pytest -m integration`, запущенный в неверном окружении, снёс бы
    данные разработчика.
    """
    import asyncio

    from alembic.config import Config

    from alembic import command
    from nutri_radar.config import get_settings

    settings = integration_settings
    db_name = settings.db.name
    working_db_name = get_settings().db.name

    # Две независимые проверки. Первая — на случай, если суффикс перестанут
    # добавлять; вторая — главная: база под тесты обязана отличаться от рабочей,
    # потому что дальше идёт DROP DATABASE.
    if not db_name.endswith(_REQUIRED_TEST_DB_SUFFIX):
        pytest.fail(
            f"Имя тестовой базы {db_name!r} не заканчивается на "
            f"{_REQUIRED_TEST_DB_SUFFIX!r}. Интеграционные тесты не запускаются "
            "на рабочей базе."
        )
    if db_name == working_db_name:
        pytest.fail(
            f"Тестовая и рабочая база совпадают ({db_name!r}). Фикстура удаляет "
            "базу целиком — на рабочей это уничтожит данные."
        )

    # Пересоздаём с нуля: остатки предыдущего прогона сделали бы тест
    # зависимым от истории запусков.
    asyncio.run(
        _run_maintenance(
            settings,
            [f'DROP DATABASE IF EXISTS "{db_name}"', f'CREATE DATABASE "{db_name}"'],
        )
    )

    alembic_cfg = Config("alembic.ini")
    # URL передаём явно — env.py уважает заданный и не подменяет его конфигом.
    alembic_cfg.set_main_option("sqlalchemy.url", settings.db.dsn)
    command.upgrade(alembic_cfg, "head")

    try:
        yield settings.db.dsn
    finally:
        from nutri_radar.db.session import dispose_engine

        # Пул приложения держит соединение, а DROP DATABASE его не переживёт:
        # Postgres ответит ObjectInUseError. Сначала закрываем свой пул...
        asyncio.run(dispose_engine())
        asyncio.run(
            _run_maintenance(
                settings,
                [
                    # ...затем отсекаем всё, что могло остаться (например
                    # соединение самого Alembic), и только потом удаляем базу.
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    f"WHERE datname = '{db_name}' AND pid <> pg_backend_pid()",
                    f'DROP DATABASE IF EXISTS "{db_name}"',
                ],
            )
        )
