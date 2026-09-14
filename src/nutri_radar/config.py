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

    # Сколько ждать очереди к моделям, прежде чем записать это в лог как
    # аномалию. Не таймаут и не отказ: длинная генерация законно держит GPU
    # десятки секунд. Но ожидание в минуту — признак того, что запросов
    # больше, чем железо переваривает, и знать об этом надо до жалоб
    # на «сервис тормозит» (см. llm/runtime.py).
    lock_warn_after_s: float = 30.0

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


class AnalyticsSettings(BaseSettings):
    """Аналитика M4: предсказание оценки качества по тексту состава."""

    model_config = SettingsConfigDict(env_prefix="ANALYTICS__", env_file=_ENV_FILE, extra="ignore")

    # Свой seed, а не общий с `extract` и `evals`: смена seed отбора корпуса
    # не должна переставлять train/test, иначе числа двух прогонов посчитаны
    # на разных сплитах и сравнивать их нельзя.
    random_seed: int = 20260905

    # Доля теста. 20% от 131 тысячи — 26 тысяч продуктов, с запасом хватает
    # даже для редкого класса «b» (4896 всего → около 980 в тесте).
    test_size: float = 0.2

    # Минимальная длина состава. Строки короче — это «-», «н/д» и мусор,
    # на котором учиться нечему, а в метрики они шум добавляют.
    min_text_length: int = 20

    # Сколько продуктов уходит в общую подвыборку, на которой меряются все
    # три подхода. Бриф запрещает гонять через LLM больше нескольких тысяч,
    # и сравнивать подходы можно только на одном и том же множестве.
    llm_subset_size: int = 1000

    # Сколько продуктов идёт в замер скорости перед полным прогоном.
    # Тот же порядок, что в M2: сначала число, потом решение о корпусе.
    benchmark_size: int = 200

    # Сколько продуктов идёт в обучение на эмбеддингах. Не «сколько есть»:
    # замер 2026-09-05 дал 0,099 с на продукт, то есть весь корпус — 219 минут
    # GPU. Тест векторизуется целиком (иначе сравнение уедет на разные
    # множества), а train режется. Стоимость этого решения — часть результата
    # майлстоуна, а не деталь запуска, поэтому TF-IDF считается ещё раз
    # на тех же 25 тысячах: иначе разницу подходов не отличить от разницы
    # в размере обучающей выборки.
    embed_train_size: int = 25_000

    # Класс, у которого меньше стольких примеров, из задачи исключается.
    # Не «на всякий случай»: `nova_group=2` встречается 25 раз на 138 254.
    # На таком классе модель ничему научиться не может, в тесте его окажется
    # около пяти штук, и macro-F1 просядет на пятую часть из-за величины,
    # про которую нельзя сказать вообще ничего. Порог задан числом и записан
    # в отчёт — исключение обязано быть видимым, а не молчаливым.
    min_class_products: int = 100

    # Версия промпта zero-shot. Параметр, а не константа: сравнение версий —
    # то же требование брифа, что и в M2.
    grade_prompt_version: str = "grade_v1"
    nova_prompt_version: str = "nova_v1"

    # --- TF-IDF ----------------------------------------------------------
    #
    # Символьные n-граммы, а не слова. Корпус многоязычный: fr 48%, en 33%,
    # de 17%. Словарная токенизация развела бы `sucre`, `Zucker` и `сахар`
    # по трём независимым признакам, хотя это одно вещество, и модель учила
    # бы каждый язык с нуля. `char_wb` держит n-граммы внутри слов, поэтому
    # ловит общие корни (`gluco`, `lecith`, `-ose`) через границы языков.
    tfidf_analyzer: str = "char_wb"
    tfidf_ngram_min: int = 3
    tfidf_ngram_max: int = 5
    # Признак, встретившийся в двух документах на 131 тысячу, — это опечатка,
    # а не сигнал. Порог обрезает хвост, который иначе раздувает словарь
    # до миллионов колонок и замедляет обучение без выигрыша в точности.
    tfidf_min_df: int = 5
    tfidf_max_features: int = 200_000

    # --- Логистическая регрессия -----------------------------------------
    #
    # Классы несбалансированы (44% против 3,7%), и без балансировки модель
    # выродится в «всегда e»: это её локальный оптимум по accuracy.
    class_weight: str | None = "balanced"
    logreg_c: float = 1.0
    logreg_max_iter: int = 400

    @field_validator("test_size")
    @classmethod
    def _validate_test_size(cls, value: float) -> float:
        if not 0.0 < value < 1.0:
            raise ValueError(f"test_size={value} должен быть в интервале (0, 1)")
        return value

    @field_validator("llm_subset_size", "benchmark_size", "min_text_length")
    @classmethod
    def _validate_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError(f"значение={value} должно быть > 0")
        return value


class RetrievalSettings(BaseSettings):
    """Поиск и RAG: векторизация профилей, pgvector, ответы по найденному."""

    model_config = SettingsConfigDict(env_prefix="RETRIEVAL__", env_file=_ENV_FILE, extra="ignore")

    # Сколько профилей уходит в Ollama одним запросом. Батч упирается
    # в num_ctx суммарно, а профиль длиннее голого состава — отсюда меньше,
    # чем в M4.
    embed_batch_size: int = 24
    # Сколько строк пишется в БД одной транзакцией. Не то же, что батч
    # модели: запись дешевле генерации, и дробить её так же мелко значит
    # платить за круги в базу.
    db_batch_size: int = 500
    # Сколько профилей идёт в замер перед полным прогоном.
    benchmark_size: int = 200

    # --- HNSW ------------------------------------------------------------
    #
    # Значения по умолчанию из документации pgvector. Трогать их стоит
    # только после того, как измерен recall: `ef_search` крутится первым,
    # `m` — последним, и только если recall упёрся в потолок.
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64
    # Query-time. Должен быть не меньше LIMIT. 100 — точка с хорошим
    # соотношением recall и латентности по опубликованным замерам.
    hnsw_ef_search: int = 100
    # Память на сборку индекса. Дефолт Postgres (64 МБ) превращает сборку
    # HNSW на 146 тысячах векторов в часы дискового шуршания.
    maintenance_work_mem: str = "2GB"

    # Сколько продуктов возвращает поиск по умолчанию.
    top_k: int = 5

    # Версия промпта RAG. Параметр, а не константа: сравнение версий —
    # то же требование брифа, что в M2 и M4.
    rag_prompt_version: str = "rag_v1"

    # --- Метрика соблюдения языка ---------------------------------------
    #
    # Минимум букв в ответе, ниже которого язык не определяется. Признак
    # «кириллица против латиницы» надёжен на связном тексте и ненадёжен
    # на коротком: ответ из штрихкодов и латинских названий брендов
    # выглядит английским, даже когда написан по-русски. Короткие ответы
    # помечаются «не определено» и считаются отдельно, а не записываются
    # в промахи — иначе метрика меряла бы длину ответа.
    language_min_letters: int = 20

    # Сколько вопросов идёт в замер цены переключения моделей. Три:
    # замер сравнивает два режима на одних и тех же вопросах, и каждый
    # вопрос в режиме с чередованием стоит двух загрузок весов. Больше —
    # это минуты ожидания ради второго знака после запятой.
    switch_benchmark_size: int = 3

    @field_validator(
        "embed_batch_size",
        "db_batch_size",
        "benchmark_size",
        "top_k",
        "language_min_letters",
        "switch_benchmark_size",
    )
    @classmethod
    def _validate_positive_retrieval(cls, value: int) -> int:
        if value <= 0:
            raise ValueError(f"значение={value} должно быть > 0")
        return value


class AgentSettings(BaseSettings):
    """Агент с инструментами: лимиты и промпт."""

    model_config = SettingsConfigDict(env_prefix="AGENT__", env_file=_ENV_FILE, extra="ignore")

    prompt_version: str = "agent_v1"

    # Потолок шагов — условие завершимости, а не удобство. Слабая модель
    # склонна повторять один и тот же вызов, и без потолка цикл не кончится.
    # Восемь: хватает на «найди → уточни → ответь» с запасом на одну ошибку,
    # и не даёт разогнаться зацикливанию.
    max_steps: int = 8

    # Потолок токенов на весь диалог. Второй предохранитель: шаги могут
    # быть дешёвыми по числу, но дорогими по длине наблюдений — выдача
    # поиска на пять продуктов это тысячи токенов.
    max_tokens: int = 20_000

    # Таймаут одного инструмента. Живой API OFF отвечает не мгновенно,
    # и висящий вызов останавливает весь прогон по набору вопросов.
    tool_timeout_s: float = 20.0

    # Сколько строк максимум возвращает `sql_query`. Не защита от вреда —
    # защита от того, что модель получит десять тысяч строк и утонет
    # в них вместе с контекстом.
    sql_max_rows: int = 20

    @field_validator("max_steps", "max_tokens", "sql_max_rows")
    @classmethod
    def _validate_positive_agent(cls, value: int) -> int:
        if value <= 0:
            raise ValueError(f"значение={value} должно быть > 0")
        return value


class LangfuseSettings(BaseSettings):
    """Наблюдаемость через Langfuse (M6).

    Живёт в отдельном compose-профиле: шесть контейнеров ради трассировки
    не должны подниматься при обычной работе. Не настроен — трассировка
    деградирует до логов, и это штатный режим, а не отказ.
    """

    model_config = SettingsConfigDict(env_prefix="LANGFUSE__", env_file=_ENV_FILE, extra="ignore")

    public_key: str = ""
    secret_key: SecretStr = SecretStr("")
    host: str = "http://localhost:3000"
    timeout_s: float = 10.0

    @property
    def is_configured(self) -> bool:
        return bool(self.public_key and self.secret_key.get_secret_value())


class ApiSettings(BaseSettings):
    """HTTP-API на FastAPI (M7).

    Точка входа, а не слой логики: здесь только то, что относится к самому
    HTTP — адрес, происхождения для CORS и таймауты ответов.
    """

    model_config = SettingsConfigDict(env_prefix="API__", env_file=_ENV_FILE, extra="ignore")

    # 0.0.0.0, а не localhost: в контейнере слушать только loopback значит
    # быть недоступным снаружи, и порт из compose никуда не ведёт.
    host: str = "0.0.0.0"
    port: int = 8000

    # Пусто — CORS не включается вовсе. Веб-фронтенда у проекта нет
    # (сознательное ограничение брифа), поэтому разрешать происхождения
    # «на всякий случай» нечему: это была бы дыра без потребителя.
    cors_origins: tuple[str, ...] = ()

    # Потолок на обычный запрос: поиск отвечает за десятки миллисекунд,
    # RAG — за единицы секунд. Всё, что дольше, — это сломанная Ollama,
    # и клиенту лучше получить honest-ошибку, чем висеть.
    request_timeout_s: float = 60.0

    # Отдельный потолок для агента: на `qwen2.5:3b` цикл доходил до восьми
    # шагов и минут работы (M6, ADR-030). Общий таймаут пришлось бы задирать
    # до агентского, и тогда он перестал бы ловить зависший поиск.
    agent_timeout_s: float = 300.0

    @field_validator("port")
    @classmethod
    def _validate_port(cls, value: int) -> int:
        if not 1 <= value <= 65535:
            raise ValueError(f"port={value} вне диапазона 1..65535")
        return value

    @field_validator("request_timeout_s", "agent_timeout_s")
    @classmethod
    def _validate_positive_api(cls, value: float) -> float:
        if value <= 0:
            raise ValueError(f"значение={value} должно быть > 0")
        return value


class BotSettings(BaseSettings):
    """Telegram-бот на aiogram 3 (M7).

    Токена может не быть: в CI его нет, и на чужой машине тоже. Отсутствие
    токена — не ошибка конфигурации, а невозможность запустить именно бота,
    поэтому проверка живёт в точке входа, а не в валидаторе.
    """

    model_config = SettingsConfigDict(env_prefix="BOT__", env_file=_ENV_FILE, extra="ignore")

    token: SecretStr = SecretStr("")

    # Потолок на фото со штрихкодом. Телефон присылает несколько мегабайт,
    # и декодировать их целиком незачем — Telegram отдаёт несколько
    # размеров, берём подходящий. Величина ограничивает память, а не
    # качество распознавания.
    max_photo_bytes: int = 5 * 1024 * 1024

    # Сколько ждать ответ конвейера, прежде чем сказать пользователю, что
    # не уложились. Меньше, чем у API: человек в чате ждёт хуже, чем
    # скрипт, и минутная пауза читается как «бот сломался».
    request_timeout_s: float = 120.0

    @property
    def is_configured(self) -> bool:
        return bool(self.token.get_secret_value())

    @field_validator("max_photo_bytes")
    @classmethod
    def _validate_max_photo_bytes(cls, value: int) -> int:
        if value <= 0:
            raise ValueError(f"max_photo_bytes={value} должен быть > 0")
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
    analytics: AnalyticsSettings = Field(default_factory=AnalyticsSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)
    bot: BotSettings = Field(default_factory=BotSettings)

    def secret_values(self) -> frozenset[str]:
        """Значения, которые фильтр логирования должен вычищать.

        Собирается здесь, а не в logging.py: конфиг знает, что является
        секретом, а логирование не должно этого угадывать.
        """
        values = {self.db.password.get_secret_value()}
        if self.langfuse.secret_key.get_secret_value():
            values.add(self.langfuse.secret_key.get_secret_value())
        if self.anthropic.api_key is not None:
            values.add(self.anthropic.api_key.get_secret_value())
        # Токен бота даёт полный контроль над ботом, и в DEBUG-логах aiogram
        # он встречается в URL запросов к API Telegram. Без этой строки он
        # утёк бы в первый же подробный лог.
        if self.bot.token.get_secret_value():
            values.add(self.bot.token.get_secret_value())
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
            "api_bind": f"{self.api.host}:{self.api.port}",
            "bot_token": "задан" if self.bot.is_configured else "не задан",
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
