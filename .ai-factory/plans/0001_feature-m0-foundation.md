# План реализации: M0 — фундамент

Ветка: `feature/m0-foundation`
Создан: 2026-07-26

## Настройки

- Тесты: да — правило 4 брифа делает их обязательными, отказ невозможен
- Логирование: подробное (DEBUG по умолчанию, уровень через `LOG_LEVEL`)
- Документация: да — обязательная сверка документации на завершении; README является витриной проекта
- Milestone из roadmap: `none` — `.ai-factory/ROADMAP.md` в проекте нет. Майлстоуны
  живут в `PROJECT_BRIEF_FOOD.md` (раздел 9); при желании их можно оформить
  через `/aif-roadmap`

Настройки не запрашивались интерактивно: значения однозначно выводятся из брифа
(правило 4 — тесты, требование `LOG_LEVEL` — логирование, README-витрина — документация).

## Цель майлстоуна и критерий готовности

**Цель:** рабочий фундамент, на который встают все последующие майлстоуны.
Кода предметной области здесь нет — только каркас, конфигурация, БД, миграции,
тесты и CI.

**DoD из брифа:** `docker compose up` поднимает всё, `pytest` и CI зелёные.

**Уточнённый DoD** (проверяется задачей 11 прогоном с чистого тома):

- `docker compose up -d` → сервис `db` переходит в `healthy`
- `alembic upgrade head` проходит; `downgrade base` откатывает начисто; повторный
  `upgrade head` снова проходит
- `nutri-radar health` — ни одного `FAIL`, код возврата 0
- `uv run pytest` зелёный **без поднятого докера** (интеграционные тесты отобраны
  маркером), `uv run pytest -m integration` зелёный при поднятом compose
- `ruff check` и `mypy src` чисты
- CI зелёный на ветке

## Контекст исследования

Источник: `.ai-factory/RESEARCH.md`

**Что влияет именно на M0:**

- `pgvector/pgvector:pg16` берётся готовым образом — своё расширение собирать не
  нужно. В CI обязателен тот же образ, иначе `CREATE EXTENSION vector` упадёт
- Расширение `vector` ставится **первой миграцией**, хотя используется только на
  M5: переставлять расширения посреди проекта дороже
- Системный Python — 3.14.2, проект целится в 3.12. Закрепляется через uv,
  конфликта нет
- Langfuse в базовый compose **не входит** (ADR-007): 6 контейнеров и 16 ГиБ по
  рекомендации вендора превратили бы DoD «`docker compose up` поднимает всё»
  в тяжёлую и медленную проверку. На M0 — только заготовка в профиле `tracing`
- Гейт evals в CI появляется на M7 (ADR-008), на M0 CI гоняет линт, типы и тесты

**Открытые вопросы, которые M0 не блокируют:**

- Библиотека логирования (structlog против python-json-logger) — решается
  задачей 1, фиксируется в ADR-010
- Декодирование штрихкода с фото — развилка M7, требует согласования зависимости

## План коммитов

- **Коммит 1** (задачи 1–3): `chore: каркас проекта, конфигурация и логирование`
- **Коммит 2** (задачи 4–6): `feat: Postgres+pgvector в compose, слой db и первая миграция`
- **Коммит 3** (задачи 7–8): `feat: health-check и инфраструктура тестов`
- **Коммит 4** (задачи 9–10): `ci: линт, типы и тесты в GitHub Actions` + `docs: README с атрибуцией ODbL`
- **Коммит 5** (задача 11): `docs: зафиксировать решения M0 в DECISIONS.md`

## Задачи

### Фаза 1: каркас проекта

- [ ] **Задача 1:** Настроить `pyproject.toml` и окружение uv — src-layout, Python 3.12,
      зависимости M0, настройки ruff / pytest / mypy, точка входа CLI.
      Файлы: `pyproject.toml`, `.python-version`
- [ ] **Задача 2:** Скелет пакета — `errors.py` (корневой `NutriRadarError`),
      `logging.py` (`setup_logging`, уровень из конфига), `tracing.py`
      (`Protocol Tracer` + `NoOpTracer`), корень Typer-CLI. Слайсы не создаём —
      они приходят со своими майлстоунами. *(зависит от 1)*
      Файлы: `src/nutri_radar/{__init__,errors,logging,tracing,cli}.py`

### Фаза 2: конфигурация

- [ ] **Задача 3:** `config.py` на pydantic-settings и `.env.example` — вложенные
      настройки (App, Database, Ollama, Anthropic, Ingest, LLM), `SecretStr` для
      секретов, `num_ctx=8192` как параметр, а не литерал, валидаторы,
      `ConfigurationError` вместо протечки pydantic. *(зависит от 2)*
      Файлы: `src/nutri_radar/config.py`, `.env.example`

<!-- Контрольная точка коммита: задачи 1-3 -->

### Фаза 3: инфраструктура данных

- [ ] **Задача 4:** `docker-compose.yml` с `pgvector/pgvector:pg16` (healthcheck
      через `pg_isready`), сервис `app` с `depends_on: service_healthy`,
      многостадийный `Dockerfile`, `.dockerignore`. Langfuse — только заготовка
      в профиле `tracing`, выключенном по умолчанию. *(зависит от 3)*
      Файлы: `docker-compose.yml`, `Dockerfile`, `.dockerignore`
- [ ] **Задача 5:** Слой `db` — `DeclarativeBase` с `naming_convention`,
      async-движок, `async_sessionmaker(expire_on_commit=False)`,
      контекстный менеджер сессии. Репозиториев пока нет. *(зависит от 3)*
      Файлы: `src/nutri_radar/db/{__init__,base,session}.py`
- [ ] **Задача 6:** Alembic по async-шаблону + первая миграция руками:
      `CREATE EXTENSION vector` и таблица `runs` (журнал прогонов всех стадий)
      с индексом по `(stage, started_at DESC)`. DSN берётся из конфига, не из
      `alembic.ini`. `downgrade` реализован полностью. *(зависит от 4, 5)*
      Файлы: `alembic.ini`, `alembic/env.py`, `alembic/versions/0001_*.py`,
      `src/nutri_radar/db/models/run.py`

<!-- Контрольная точка коммита: задачи 4-6 -->

### Фаза 4: health-check и тесты

- [ ] **Задача 7:** Команда `nutri-radar health` — пять проверок: соединение с
      Postgres, наличие расширения `vector`, совпадение ревизии Alembic с head,
      доступность Ollama с нужными моделями, факт наличия ключа Anthropic.
      Деградации дают `WARN` и код 0, отказы — `FAIL` и код 1. Флаг `--json`.
      *(зависит от 6)*
      Файлы: `src/nutri_radar/health.py`, `src/nutri_radar/cli.py`
- [ ] **Задача 8:** Тесты — `conftest.py` (настройки из словаря, тестовая БД с
      защитой от запуска на рабочей, мок Ollama через `httpx.MockTransport`,
      маркер `integration`), тесты конфига, health-check, логирования и
      **тест актуальности `.env.example`** против полей `Settings`. *(зависит от 7)*
      Файлы: `tests/conftest.py`, `tests/test_{config,health,logging}.py`

<!-- Контрольная точка коммита: задачи 7-8 -->

### Фаза 5: CI и витрина

- [ ] **Задача 9:** CI на GitHub Actions — `uv sync --frozen` → `ruff check` →
      `ruff format --check` → `mypy src` → `alembic upgrade head` → `pytest`
      с сервисным контейнером `pgvector/pgvector:pg16`. Кэш uv, `concurrency`
      с отменой устаревших прогонов. *(зависит от 8)*
      Файлы: `.github/workflows/ci.yml`
- [ ] **Задача 10:** `README.md` — описание проекта, чем не дублирует OFF,
      **атрибуция ODbL**, дисклеймер о границах продукта (не медицинский
      советчик, данные краудсорсинговые), быстрый старт, заготовки таблиц метрик
      под M3/M4/M5.
      Файлы: `README.md`

<!-- Контрольная точка коммита: задачи 9-10 -->

- [ ] **Задача 11:** Приёмка DoD — прогон с чистого тома (`down -v` → `up` →
      миграции → health → цикл `downgrade`/`upgrade` → тесты → линт), замер
      времени старта. Дописать в `DECISIONS.md` ADR-010 (логирование),
      ADR-011 (Typer), ADR-012 (готовый образ pgvector) и фактические версии.
      *(зависит от 1-10)*
      Файлы: `DECISIONS.md`

<!-- Контрольная точка коммита: задача 11 -->

## Границы майлстоуна

Сознательно **не входит** в M0, чтобы слайс остался вертикальным и тонким:

- модули `ingest/`, `extract/`, `evals/`, `analytics/`, `retrieval/`, `agent/` —
  каждый приходит со своим майлстоуном вместе со своими таблицами и тестами
- таблицы `products_raw`, `products`, `product_extraction`, `ingredients_dict`,
  `product_embeddings` — их схема проектируется на M1/M2, когда известна форма данных
- `llm/` с адаптерами Ollama и Anthropic — M2
- FastAPI, Telegram-бот, MCP-сервер — M7
- сам Langfuse (не заготовка профиля) — M6
- гейт evals в CI — M7
- скачивание дампа — M1. На M0 в `data/` ничего не появляется

## Риски

| Риск | Признак | Что делать |
|---|---|---|
| `postgres:16` вместо pgvector-образа в CI | `CREATE EXTENSION vector` падает на шаге миграций | образ `pgvector/pgvector:pg16` и локально, и в CI — проверяется задачей 9 |
| CRLF из Windows ломает shell-шаги в Actions | шаг CI падает с `bad interpreter` или `\r` в выводе | `.gitattributes` с `eol=lf` уже в репозитории; задача 9 проверяет это явно |
| `pytest` красный без докера | DoD «pytest зелёный» невыполним на чистой машине | маркер `integration`, по умолчанию не отбирается — задача 8 |
| `expire_on_commit=True` по умолчанию | `MissingGreenlet` при обращении к атрибуту после commit | явный `expire_on_commit=False` — задача 5 |
| Безымянные constraint'ы от Alembic | миграцию с удалением ограничения не написать | `naming_convention` в `Base` до первой миграции — задача 5 |
| `.env.example` расходится с `Settings` | новый разработчик не может поднять проект | тест сверки ключей — задача 8 |
| DSN с паролем в `alembic.ini` попадает в git | секрет в истории репозитория | DSN только из конфига — задача 6 |
