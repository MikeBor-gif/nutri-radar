# AGENTS.md

> Карта проекта для AI-агентов и новых разработчиков. Описывает только то, что
> реально существует. Обновляется при значимых изменениях структуры.
>
> **Текущее состояние: M0–M6 выполнены, три из них частично. Остался M7.**
> В базе 147 873 продукта, 131 939 с составом и оценкой Nutri-Score,
> 146 350 векторов `bge-m3` в pgvector. Работают все слайсы: `ingest`,
> `extract`, `evals`, `analytics`, `retrieval`, `agent`. 603 теста.
>
> Три майлстоуна закрыты **частично**, и причина у каждого записана:
> M3 — эталон извлечения размечен моделью, а не человеком (ADR-026);
> M5 — эталонные запросы поиска составлены оракулом на SQL, а не человеком,
> и recall@5 заменён на долю выдачи со свойством (ADR-029);
> M6 — агент прогнан на локальной 3B вместо Sonnet 5, выбор инструмента
> работает на 100%, завершение цикла — на 20% (ADR-030).
>
> **Общая причина двух из трёх — отсутствие `ANTHROPIC__API_KEY`.**
> Появление ключа разблокирует M6, облачную часть M3 и гибрид из ADR-009.
>
> Остались точки входа `api`, `bot`, `mcp_server` — это M7.

## О проекте

Nutri Radar разбирает состав пищевых продуктов на данных Open Food Facts:
нормализует грязные многоязычные списки ингредиентов, находит скрытые формы
сахара и добавки, предсказывает оценку качества по тексту состава и отвечает на
вопросы по базе со ссылками на конкретные штрихкоды. Проект портфолийный —
измеримость и защищаемость решений важнее объёма функциональности.

## Стек

- **Язык:** Python 3.12 (пакеты и venv через uv)
- **Массовые данные:** DuckDB (читает Parquet напрямую)
- **База данных:** PostgreSQL 16 + pgvector
- **ORM / миграции:** SQLAlchemy 2.x (async) + Alembic
- **Валидация и конфиг:** Pydantic v2, pydantic-settings
- **API / бот / MCP:** FastAPI + uvicorn, aiogram 3, MCP Python SDK
- **ML:** pandas, scikit-learn, matplotlib
- **LLM:** Ollama (`qwen2.5:3b-instruct-q4_K_M`, генерация по JSON-схеме) +
  Anthropic Claude (агент и эталон в evals)
- **Эмбеддинги:** `bge-m3` локально, 1024 измерения, мультиязычная
- **Инфра:** Docker Compose, GitHub Actions, Langfuse (отдельный профиль)

## Структура проекта

`✓` — существует, `—` — появится со своим майлстоуном. Детали и правила
зависимостей — в `.ai-factory/ARCHITECTURE.md`.

```text
nutri-radar/
├── README.md                  ✓ витрина: метрики, быстрый старт, атрибуция ODbL
├── CLAUDE.md                  ✓ свод правил работы над проектом
├── DECISIONS.md               ✓ журнал архитектурных решений (ADR-001..013)
├── AGENTS.md                  ✓ этот файл
├── docker-compose.yml         ✓ Postgres+pgvector; Langfuse в профиле tracing
├── Dockerfile                 ✓ многостадийный, непривилегированный пользователь
├── .env.example               ✓ 34 ключа конфигурации, сверяется тестом
├── pyproject.toml             ✓ зависимости, ruff / pytest / mypy
├── uv.lock                    ✓ закреплённые версии (CI ставит --frozen)
├── .github/workflows/ci.yml   ✓ линт, типы, миграции, тесты, health-check
├── alembic/
│   ├── env.py                 ✓ DSN из конфига, не из alembic.ini
│   └── versions/
│       └── 20260726_0001_...  ✓ расширение vector + таблица runs
├── src/nutri_radar/
│   ├── __init__.py            ✓ __version__
│   ├── config.py              ✓ pydantic-settings — единственный источник параметров
│   ├── errors.py              ✓ NutriRadarError и доменные исключения
│   ├── logging.py             ✓ JSON и человекочитаемый формат, фильтр секретов
│   ├── tracing.py             ✓ порт Tracer + NoOpTracer
│   ├── wording.py             ✓ обязательная атрибуция и запрещённые ярлыки (M7)
│   ├── openfoodfacts.py       ✓ единственный модуль, ходящий в живой API OFF (M7)
│   ├── serve.py               ✓ команды serve api | bot | mcp (M7)
│   ├── health.py              ✓ пять проверок готовности среды
│   ├── cli.py                 ✓ корень Typer: version, health
│   ├── db/                    ✓ ОБЩЕЕ: Base, async-движок, сессии
│   │   ├── base.py            ✓ DeclarativeBase с naming_convention
│   │   ├── session.py         ✓ движок, expire_on_commit=False
│   │   ├── models/run.py      ✓ таблица runs — журнал прогонов
│   │   ├── models/product.py  ✓ products_raw и products
│   │   └── repositories/      ✓ идемпотентный upsert по code с учётом rev
│   ├── llm/                   ✓ ОБЩЕЕ: порты и адаптеры моделей (M2)
│   │   ├── ports.py          ✓   StructuredLLM и EmbeddingModel
│   │   ├── adapters/         ✓   ollama, anthropic, fake
│   │   └── runtime.py        ✓   очередь к моделям: одна в VRAM за раз (M7)
│   ├── ingest/                ✓ СЛАЙС: дамп, DuckDB-выборка, дельты
│   │   ├── probe.py           ✓   разведка схемы дампа без скачивания
│   │   ├── download.py        ✓   идемпотентное скачивание с докачкой
│   │   ├── models.py          ✓   RawProduct — контракт обоих источников
│   │   ├── select.py          ✓   фильтр корпуса в DuckDB и его двойник
│   │   ├── load.py            ✓   батчевая заливка, дельты, журнал прогонов
│   │   └── sources/           ✓   адаптеры parquet и delta
│   ├── extract/               — СЛАЙС: схемы, промпты, формы сахара (M2)
│   ├── evals/                 — СЛАЙС: разметка, метрики, отчёт, гейт для CI (M3)
│   ├── analytics/             ✓   СЛАЙС: classic ML на метках из базы (M4)
│   │   ├── dataset.py        ✓   выгрузка из БД, стратифицированный сплит
│   │   ├── features.py       ✓   TF-IDF, метрики, хранение результатов
│   │   ├── embeddings.py     ✓   векторизация: замер, кэш, возобновляемость
│   │   ├── prompts/          ✓   промпты zero-shot, версионируемые
│   │   ├── tasks/            ✓   sanity_check, grade_from_text, nova
│   │   └── report.py         ✓   таблицы сравнения и графики
│   ├── retrieval/             ✓   СЛАЙС: эмбеддинги, pgvector, RAG (M5)
│   │   ├── profile.py        ✓   сборка текста профиля продукта
│   │   ├── embed.py          ✓   векторизация корпуса и индекс HNSW
│   │   ├── search.py         ✓   поиск с фильтрами, проверка плана
│   │   ├── rag.py            ✓   ответ по найденному, отказ «не знаю»
│   │   ├── metrics.py        ✓   свойство выдачи, подтверждённость
│   │   ├── pipeline.py       ✓   единая сборка конвейера для всех точек входа (M7)
│   │   └── product_card.py   ✓   карточка по штрихкоду: корпус, фолбэк в OFF (M7)
│   ├── agent/                 ✓   СЛАЙС: инструменты и цикл агента (M6)
│   │   ├── loop.py           ✓   цикл руками, без фреймворков
│   │   ├── tools/            ✓   sql_query, vector_search, lookup_barcode
│   │   ├── evaluate.py       ✓   прогон по вопросам, проверяемые признаки
│   │   └── registry.py       ✓   сборка реестра инструментов для всех потребителей
│   ├── api/                   ✓ ТОЧКА ВХОДА: FastAPI (M7)
│   │   ├── app.py            ✓   сборка, lifespan, request_id, CORS
│   │   ├── errors.py         ✓   доменные ошибки в коды HTTP
│   │   ├── schemas.py        ✓   контракт API, отдельный от внутренних структур
│   │   ├── factory.py        ✓   готовое приложение для uvicorn --reload
│   │   └── routes/           ✓   products, search, ask, agent
│   ├── bot/                   ✓ ТОЧКА ВХОДА: Telegram, aiogram 3 (M7)
│   │   ├── app.py            ✓   диспетчер, long polling, зависимости
│   │   ├── texts.py          ✓   всё, что бот говорит от себя; /start с ODbL
│   │   ├── middlewares.py    ✓   приватность: хэш чата, длина текста, не текст
│   │   ├── keyboards.py      ✓   инлайн-кнопки, состояние внутри callback_data
│   │   ├── barcode_image.py  ✓   чтение штрихкода с фото через zxing-cpp
│   │   ├── errors.py         ✓   доменные ошибки в сообщения пользователю
│   │   └── handlers/         ✓   commands, barcode, photo, ask
│   └── mcp_server/            ✓ ТОЧКА ВХОДА: MCP поверх реестра агента (M7)
│       └── server.py         ✓   три инструмента наружу по stdio
├── tests/
│   ├── conftest.py            ✓ герметичные настройки, мок Ollama, тестовая БД
│   ├── test_config.py         ✓ секреты, валидаторы, сверка .env.example
│   ├── test_health.py         ✓ WARN против FAIL, изоляция проверок
│   └── test_logging.py        ✓ вычищение секретов, форматтеры
├── notebooks/                 — только разведка, не продакшн-код
├── reports/                   — графики и отчёты evals
└── data/                      — gitignored, кроме data/evals/ и data/dictionaries/
```

Ключевое правило зависимостей: точки входа → слайсы → общее. Слайсы **не зависят
друг от друга** (исключения: `agent` → `retrieval`, `evals` → `extract.schemas`).
Стадии пайплайна связаны таблицами Postgres, а не импортами.

## Ключевые точки входа

| Файл | Назначение |
|---|---|
| `src/nutri_radar/config.py` | все параметры проекта; начинать чтение кода отсюда |
| `src/nutri_radar/cli.py` | точка входа `nutri-radar`; слайсы регистрируют команды здесь |
| `src/nutri_radar/health.py` | проверка готовности среды: БД, vector, миграции, Ollama, ключ |
| `src/nutri_radar/db/session.py` | движок и сессии; `expire_on_commit=False` обязателен |
| `alembic/env.py` | DSN берётся из конфига; уважает URL, переданный явно |
| `pyproject.toml` | зависимости, настройки ruff / pytest / mypy |
| `docker-compose.yml` | Postgres+pgvector; Langfuse в профиле `tracing` |
| `.env.example` | полный список ключей; сверяется тестом с полями `Settings` |
| `alembic/versions/` | история схемы БД |
| `src/nutri_radar/extract/schemas.py` | *(M2)* Pydantic-схема выхода LLM = JSON-схема генерации |
| `src/nutri_radar/llm/ports.py` | *(M2)* порты внешних моделей; точка подмены провайдера |
| `data/evals/extraction_gold.jsonl` | *(M3)* эталонная разметка; **создаётся только человеком** — сейчас нарушено, см. ADR-026 |
| `src/nutri_radar/analytics/dataset.py` | *(M4)* выгрузка без нутриентов: защита от утечки структурная, а не дисциплинарная |
| `src/nutri_radar/analytics/report.py` | *(M4)* таблица «точность против стоимости» — ответ майлстоуна |
| `src/nutri_radar/agent/tools/sql_query.py` | *(M6)* **единственное место, где SQL от модели идёт в базу.** Защита структурная: read-only, один SELECT, белый список |
| `src/nutri_radar/agent/tools/lookup_barcode.py` | *(M6)* инструмент агента поверх `openfoodfacts.py` |
| `src/nutri_radar/agent/loop.py` | *(M6)* цикл агента руками — предмет демонстрации по правилу 7 |
| `src/nutri_radar/openfoodfacts.py` | *(M7)* **единственный модуль, который ходит в живой API OFF.** Потребителей два: инструмент агента и карточка вне корпуса (ADR-031) |
| `src/nutri_radar/llm/runtime.py` | *(M7)* **очередь к моделям: одна в VRAM за раз.** Без неё два параллельных запроса выгружают модели друг у друга |
| `src/nutri_radar/retrieval/pipeline.py` | *(M7)* единственная сборка конвейера; на неё переведён и CLI |
| `src/nutri_radar/wording.py` | *(M7)* обязательная атрибуция ODbL и список запрещённых оценочных ярлыков; проверяется тестами |
| `src/nutri_radar/bot/middlewares.py` | *(M7)* приватность логов: хэш чата и длина текста, но не текст и не `user_id` |

## Документация

| Документ | Путь | Описание |
|---|---|---|
| README | `README.md` | витрина проекта: задача, метрики, графики, атрибуция ODbL |
| Исходный бриф | `PROJECT_BRIEF_FOOD.md` | требования заказчика проекта; источник правил |
| Журнал решений | `DECISIONS.md` | ADR: контекст → решение → альтернативы → последствия |

## Контекстные файлы для AI

| Файл | Назначение |
|---|---|
| `AGENTS.md` | этот файл: карта структуры проекта и точек входа |
| `CLAUDE.md` | компактный свод правил работы: как вести себя в этом проекте |
| `.ai-factory/DESCRIPTION.md` | полная спецификация: стек, корпус, метрики, границы продукта |
| `.ai-factory/ARCHITECTURE.md` | паттерн, раскладка каталогов, правила зависимостей, примеры кода |
| `.ai-factory/RESEARCH.md` | проверка выполнимости: расхождения с брифом и их разбор |
| `.ai-factory/references/openfoodfacts-parquet-fields.md` | **колонки дампа OFF**; единственный достоверный источник имён |
| `.ai-factory/rules/base.md` | конвенции кода: именование, ошибки, логи, async, тесты |
| `.ai-factory/config.yaml` | настройки AI Factory: языки, пути, git, режим верификации |

## Правила для агентов

- **Имена колонок дампа брать только из
  `.ai-factory/references/openfoodfacts-parquet-fields.md`.** Файл OFF
  `data-fields.txt` описывает CSV-экспорт, и половина имён оттуда в Parquet
  не существует.
- **Останавливаться после каждого milestone** и ждать ревью владельца проекта.
- **Не генерировать эталонную разметку для evals** — метки ставит только человек.
- **Не добавлять зависимости** без согласования. LangChain и аналогов в проекте нет.
- **Тест не ходит в сеть.** Ollama, Anthropic, HTTP OFF — только моки.
- **Правила брифа и `.ai-factory/rules/base.md` важнее советов внешних скиллов.**
  Если скилл советует Redis или фреймворк — игнорировать.
- **Разбивать shell-команды на отдельные шаги**, не склеивать через `&&`:
  - ❌ неверно: `git checkout main && git pull`
  - ✅ верно: сначала `git checkout main`, затем `git pull origin main`
- **Порт БД на машине разработчика — 5435**, а не 5432: последний занят службой
  PostgreSQL 18, а 5433 и 5434 — контейнерами других проектов. Симптом коллизии
  неочевиден, подробности в ADR-013.
- **`DB__HOST` и `OLLAMA__BASE_URL` в compose заданы жёстко**, не подстановкой
  из `.env`: изнутри контейнера `localhost` означает сам контейнер.
- Пароли не должны быть подстроками имён пользователя или БД — фильтр
  логирования вычищает секрет буквально и затрёт их тоже (ADR-010).
- Тесты, требующие БД, помечаются маркером `integration` и по умолчанию не
  отбираются: `uv run pytest` обязан быть зелёным без докера.
