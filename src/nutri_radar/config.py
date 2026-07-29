"""Конфигурация проекта. Единственный источник параметров.

Правило 5 брифа: ни одной магической константы в коде. Порог, размер батча,
имя модели, размер контекста, лимиты — всё здесь и переопределяется через `.env`.

Вложенные группы читаются с префиксами: `DB__HOST`, `OLLAMA__NUM_CTX` и так далее
(разделитель — двойное подчёркивание). Плоские ключи приложения — без префикса:
`LOG_LEVEL`, `ENVIRONMENT`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from nutri_radar.errors import ConfigurationError

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
Environment = Literal["local", "ci", "production"]
LLMProvider = Literal["ollama", "anthropic"]

_ENV_FILE = ".env"


class AppSettings(BaseSettings):
    """Общие параметры приложения."""

    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore")

    log_level: LogLevel = "DEBUG"
    environment: Environment = "local"
    json_logs: bool = False


class DatabaseSettings(BaseSettings):
    """Подключение к Postgres."""

    model_config = SettingsConfigDict(env_prefix="DB__", env_file=_ENV_FILE, extra="ignore")

    host: str = "localhost"
    port: int = 5432
    user: str = "nutri"
    # Пароль намеренно не является подстрокой имени пользователя или БД:
    # фильтр логирования вычищает значение секрета из любого текста, и пароль
    # "nutri" затирал бы также user и name в сообщениях (см. logging.py).
    password: SecretStr = SecretStr("local_dev_password")
    name: str = "nutri_radar"
    pool_size: int = 5
    # Дефолт asyncpg — 60 секунд. Столько ждать отказа недоступной базы нельзя:
    # health-check в compose и тесты повисли бы на минуту вместо быстрого FAIL.
    connect_timeout_s: float = 10.0
    # SQL-эхо отдельным флагом, а не через LOG_LEVEL: на DEBUG оно забивает
    # вывод целиком и отладка самого пайплайна становится невозможной.
    echo_sql: bool = False

    @property
    def dsn(self) -> str:
        """DSN для asyncpg. Содержит пароль — в логи не передаётся."""
        password = self.password.get_secret_value()
        return f"postgresql+asyncpg://{self.user}:{password}@{self.host}:{self.port}/{self.name}"

    @property
    def safe_dsn(self) -> str:
        """DSN без пароля — для логов и сообщений об ошибках."""
        return f"postgresql+asyncpg://{self.user}:***@{self.host}:{self.port}/{self.name}"


class OllamaSettings(BaseSettings):
    """Локальные модели через Ollama.

    Ограничение железа: RTX 3060 Laptop, 6 ГБ VRAM. Отсюда модель 3B в q4
    и последовательная загрузка моделей вместо параллельной.
    """

    model_config = SettingsConfigDict(env_prefix="OLLAMA__", env_file=_ENV_FILE, extra="ignore")

    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:3b-instruct-q4_K_M"
    embedding_model: str = "bge-m3"
    embedding_dim: int = 1024

    # Критичный параметр. Дефолт Ollama — 4096, и он подбирается динамически
    # по доступной VRAM, то есть МОЛЧА обрезает вход. Задаём явно.
    num_ctx: int = 8192

    temperature: float = 0.0
    max_output_tokens: int = 2048
    # Модели грузятся по очереди: 6 ГБ VRAM не переживут одновременную
    # загрузку модели извлечения и модели эмбеддингов.
    keep_alive: str = "5m"
    timeout_s: float = 120.0
    # Единица — не осторожность, а измерение. A/B на одних и тех же 40 продуктах:
    # с параллелизмом 2 прогон занял 838 с, с параллелизмом 1 — 817 с. Ускорения
    # нет: на 6 ГБ VRAM модель занимает GPU целиком, и второй запрос ждёт
    # очереди, зато удваивает пиковое потребление памяти. Параметр остаётся
    # в конфиге — на другом железе выигрыш может появиться (ADR-017).
    max_concurrency: int = 1

    @field_validator("num_ctx")
    @classmethod
    def _validate_num_ctx(cls, value: int) -> int:
        if value < 2048:
            raise ValueError(
                f"num_ctx={value} слишком мал: составы с длинными списками "
                "ингредиентов будут обрезаны. Минимум 2048."
            )
        return value

    @field_validator("max_concurrency")
    @classmethod
    def _validate_max_concurrency(cls, value: int) -> int:
        if value < 1:
            raise ValueError(f"max_concurrency={value} должен быть >= 1")
        return value


class AnthropicSettings(BaseSettings):
    """Облачная модель: агент (M6) и эталон в evals (M3).

    На M0 ключ не обязателен — health-check сообщит о его отсутствии как WARN.
    """

    model_config = SettingsConfigDict(env_prefix="ANTHROPIC__", env_file=_ENV_FILE, extra="ignore")

    api_key: SecretStr | None = None
    model: str = "claude-sonnet-5"
    # Дешёвая модель для массовых прогонов вроде zero-shot в M4.
    cheap_model: str = "claude-haiku-4-5-20251001"
    max_output_tokens: int = 2048
    timeout_s: float = 60.0

    @field_validator("api_key", mode="before")
    @classmethod
    def _empty_key_is_none(cls, value: object) -> object:
        """Пустая строка — это отсутствие ключа, а не ключ.

        В `.env.example` ключ объявлен пустым (`ANTHROPIC__API_KEY=`), и без
        этой нормализации получался бы `SecretStr("")`, который health-check
        считал бы заданным ключом.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def is_configured(self) -> bool:
        return self.api_key is not None


class IngestSettings(BaseSettings):
    """Параметры сбора корпуса из дампа Open Food Facts.

    Языки и категории зафиксированы в ADR-002. Целевые размеры корпусов —
    в DESCRIPTION.md; точные числа уточняются по факту первого прогона.
    """

    model_config = SettingsConfigDict(env_prefix="INGEST__", env_file=_ENV_FILE, extra="ignore")

    # Parquet-дамп лежит только на HuggingFace: на static.openfoodfacts.org
    # его нет (см. RESEARCH.md, раздел 1).
    dump_url: str = (
        "https://huggingface.co/datasets/openfoodfacts/product-database/resolve/main/food.parquet"
    )
    delta_index_url: str = "https://static.openfoodfacts.org/data/delta/index.txt"
    data_dir: Path = Path("data")

    languages: list[str] = Field(default_factory=lambda: ["en", "ru", "de", "fr", "pl"])
    # Только КОНКРЕТНЫЕ теги, без зонтичных `en:snacks`, `en:beverages`,
    # `en:dairies`, `en:breakfasts`. Таксономия OFF иерархическая: правильно
    # категоризованный продукт несёт и подтег, и зонтичный. Поэтому отказ от
    # зонтичных убирает не категории, а плохо категоризованные записи —
    # те, у которых есть только верхний уровень (ADR-014).
    category_tags: list[str] = Field(
        default_factory=lambda: [
            "en:sweet-snacks",
            "en:salty-snacks",
            "en:biscuits-and-cakes",
            "en:chocolates",
            "en:confectioneries",
            "en:sweetened-beverages",
            "en:yogurts",
            "en:cheeses",
            "en:breakfast-cereals",
        ]
    )

    # Требовать, чтобы продукт хоть раз сканировали в приложении. Отсекает
    # заброшенные тестовые записи. Вносит смещение в сторону популярных
    # продуктов — это указано в README и ADR-014.
    require_scanned: bool = True

    # Составы короче этого не несут информации: "-", "n/a", пустые скобки.
    min_ingredients_length: int = 10
    batch_size: int = 1000

    # Размер выборки для `ingest probe`. Удалённое чтение 20 тыс. строк занимает
    # около 200 с: DuckDB делает много мелких range-запросов. Больше не нужно —
    # состав нутриентов и языков на этом объёме уже стабилен.
    probe_sample_rows: int = 20000

    # Без лимита DuckDB на файле 7,7 ГБ может съесть всю память и быть убитым ОС.
    duckdb_memory_limit: str = "4GB"

    # Доля отброшенных записей выше этой — признак сломанного фильтра
    # или адаптера, а не плохих данных.
    max_skip_share: float = 0.05

    # Целевая вилка аналитического корпуса (раздел 6 брифа). Выход за неё —
    # повод править список категорий, а не молча заливать что получилось.
    corpus_min_size: int = 50_000
    corpus_max_size: int = 150_000

    # Скачивание дампа. Файл 7,7 ГБ, поэтому докачка и ретраи обязательны.
    download_chunk_size: int = 8 * 1024 * 1024
    download_max_retries: int = 5
    download_timeout_s: float = 300.0

    @field_validator("languages")
    @classmethod
    def _validate_languages(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("languages пуст: без языков фильтр выборки отберёт нулевой корпус")
        return value

    @property
    def dump_path(self) -> Path:
        return self.data_dir / "food.parquet"


class ExtractSettings(BaseSettings):
    """Параметры извлечения структуры состава (M2)."""

    model_config = SettingsConfigDict(env_prefix="EXTRACT__", env_file=_ENV_FILE, extra="ignore")

    # Версия промпта для прогона. Уезжает в product_extraction рядом
    # с результатом: без неё сравнение версий в M3 невозможно.
    prompt_version: str = "v3"

    # Целевой размер LLM-корпуса. Бриф просил 3-5 тысяч, но замер показал
    # 20,4 с на продукт вместо ожидавшихся 2,5-10,5: 3000 продуктов — это
    # 12-17 часов, втрое выше порога max_run_hours. Сужено до 1200 по факту
    # измерения, как и предписывает таблица рисков плана M2. Квоты по языкам
    # сохраняются (~240 на язык), для метрик M3 этого достаточно (ADR-017).
    corpus_size: int = 1200

    # Доля корпуса со смещением в unknown_ingredients_n > 0 (ADR-006):
    # туда, где парсер OFF не справился. Остальное — контрольная случайная
    # часть, без неё нельзя честно показать поведение на лёгких случаях.
    unknown_share: float = 0.7

    # Детерминированность выборки. Повторный запуск обязан дать ТУ ЖЕ
    # выборку, иначе сравнение версий промптов пойдёт по разным продуктам.
    random_seed: int = 42

    # Доля невалидных ответов, выше которой запускать полный прогон нельзя.
    max_invalid_share: float = 0.1

    # Порог экстраполяции замера: дольше — повод сузить корпус, а не ждать.
    max_run_hours: float = 6.0

    # Размер выборки для замера. Раздел 3a брифа: замер на 20 продуктах перед
    # полным прогоном. Выборка берётся с начала отобранного корпуса, поэтому
    # все версии промпта меряются на ОДНИХ И ТЕХ ЖЕ продуктах.
    benchmark_size: int = 20

    # Сколько неизвестных имён показывает `extract dict unknown`.
    unknown_report_top: int = 50

    # Размер батча прогона. Результаты уходят в БД после каждого батча,
    # поэтому Ctrl+C стоит не больше одного батча работы. Больше батч —
    # реже транзакции, но дороже обрыв.
    batch_size: int = 50

    # Ретраи ТОЛЬКО на недоступности модели (LLMUnavailableError). Невалидный
    # разбор не ретраится: при temperature=0 повтор даст тот же ответ, и прогон
    # встанет на месте.
    max_retries: int = 3
    retry_backoff_s: float = 2.0

    # Столько отказов модели подряд означают, что Ollama упала, а не что
    # попался трудный состав. Продолжать бессмысленно: прогон останавливается,
    # уже записанное сохраняется, перезапуск продолжит с этого места.
    max_consecutive_failures: int = 10

    @field_validator("unknown_share")
    @classmethod
    def _validate_share(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"unknown_share={value} должна быть в диапазоне 0..1")
        return value

    @field_validator("corpus_size")
    @classmethod
    def _validate_corpus_size(cls, value: int) -> int:
        if value < 1:
            raise ValueError(f"corpus_size={value} должен быть >= 1")
        return value

    @field_validator("batch_size")
    @classmethod
    def _validate_batch_size(cls, value: int) -> int:
        if value < 1:
            raise ValueError(f"batch_size={value} должен быть >= 1")
        return value

    @field_validator("max_retries")
    @classmethod
    def _validate_max_retries(cls, value: int) -> int:
        # Ноль означал бы «ни одной попытки», а не «без ретраев»: первая
        # попытка — это тоже попытка.
        if value < 1:
            raise ValueError(
                f"max_retries={value} должен быть >= 1 (первая попытка тоже считается)"
            )
        return value


class LLMSettings(BaseSettings):
    """Выбор провайдера. По нему composition root подставляет адаптер."""

    model_config = SettingsConfigDict(env_prefix="LLM__", env_file=_ENV_FILE, extra="ignore")

    provider: LLMProvider = "ollama"


class EvalsSettings(BaseSettings):
    """Оценка качества: эталон, метрики, гейт."""

    model_config = SettingsConfigDict(env_prefix="EVALS__", env_file=_ENV_FILE, extra="ignore")

    # Сколько продуктов размечает человек. Согласовано отдельно: по 20 на каждый
    # из пяти языков. Меньше 20 на язык не даёт судить о разнице между языками —
    # а это один из двух вопросов, ради которых майлстоун и существует.
    gold_size: int = 100

    # Свой seed, а не общий с `extract`: смена seed отбора корпуса не должна
    # переставлять эталонную выборку, иначе размеченное перестанет совпадать
    # с тем, что размечали.
    random_seed: int = 20260728

    # Насколько может просесть F1, прежде чем гейт уронит сборку. В пунктах.
    # Значение из раздела «Ограничения» спецификации.
    max_f1_drop: float = 3.0

    # Относительная разница, ниже которой числа одного прогона считать
    # неразличимыми. Не выдумано: ADR-018 измерил разброс между двумя
    # идентичными прогонами локальной модели при temperature=0 — 1,95 против
    # 1,57 формы сахара на тех же 40 продуктах. Отчёт обязан отмечать
    # разницы меньше этой, иначе шум прогона выдаётся за разницу систем.
    significant_diff_share: float = 0.20

    @field_validator("gold_size")
    @classmethod
    def _validate_gold_size(cls, value: int) -> int:
        if value <= 0:
            raise ValueError(f"gold_size={value} должен быть > 0")
        return value


class Settings(BaseSettings):
    """Корневые настройки. Получать только через `get_settings()`."""

    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore")

    app: AppSettings = Field(default_factory=AppSettings)
    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)
    ingest: IngestSettings = Field(default_factory=IngestSettings)
    extract: ExtractSettings = Field(default_factory=ExtractSettings)
    evals: EvalsSettings = Field(default_factory=EvalsSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)

    def secret_values(self) -> frozenset[str]:
        """Значения, которые фильтр логирования должен вычищать.

        Собирается здесь, а не в logging.py: конфиг знает, что является
        секретом, а логирование не должно этого угадывать.
        """
        values = {self.db.password.get_secret_value()}
        if self.anthropic.api_key is not None:
            values.add(self.anthropic.api_key.get_secret_value())
        return frozenset(v for v in values if v)

    def describe(self) -> dict[str, object]:
        """Безопасный для логов снимок настроек. Секретов не содержит."""
        return {
            "environment": self.app.environment,
            "log_level": self.app.log_level,
            "db": self.db.safe_dsn,
            "db_pool_size": self.db.pool_size,
            "llm_provider": self.llm.provider,
            "ollama_model": self.ollama.model,
            "ollama_num_ctx": self.ollama.num_ctx,
            "ollama_max_concurrency": self.ollama.max_concurrency,
            "embedding_model": self.ollama.embedding_model,
            "anthropic_key": "задан" if self.anthropic.is_configured else "не задан",
            "anthropic_model": self.anthropic.model,
            "ingest_languages": self.ingest.languages,
            "ingest_categories_count": len(self.ingest.category_tags),
            "ingest_batch_size": self.ingest.batch_size,
            "prompt_version": self.extract.prompt_version,
            "corpus_size": self.extract.corpus_size,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Собрать настройки один раз на процесс.

    Ошибку валидации превращаем в `ConfigurationError`: наружу не должен
    протекать `ValidationError` pydantic с его многострочным форматом —
    вызывающему нужно понять, какой ключ поправить в `.env`.
    """
    try:
        return Settings()
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        raise ConfigurationError(
            f"Конфигурация невалидна ({problems}). "
            f"Проверьте {_ENV_FILE} — эталонный список ключей в .env.example."
        ) from exc
