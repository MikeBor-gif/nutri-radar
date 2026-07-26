# Архитектура: Structured Modules (вертикальные слайсы по стадиям пайплайна)

## Обзор

Nutri Radar — не веб-приложение, а **пайплайн обработки данных с четырьмя точками
входа**. Данные текут строго в одну сторону: дамп → выборка → нормализация →
извлечение → аналитика → поиск → агент. Поверх этого пайплайна навешаны CLI,
FastAPI, Telegram-бот и MCP-сервер, и все четыре не содержат логики — только
вызывают слои ниже и форматируют ответ.

Поэтому модуль здесь — это **стадия пайплайна**, а не сущность БД. `ingest`,
`extract`, `evals`, `analytics`, `retrieval`, `agent` — каждый инкапсулирует свою
стадию: свои Pydantic-модели, свою логику, свой доступ к данным, свой CLI.
Соседние стадии общаются через таблицы Postgres и через явные публичные функции,
а не залезая друг другу внутрь.

Выбран паттерн Structured Modules в варианте вертикальных слайсов. Ключевая
адаптация под специфику проекта: **все внешние системы спрятаны за портами**
(`Protocol`). Ollama, Anthropic, HTTP Open Food Facts, Langfuse — за интерфейсами,
у каждого есть fake-реализация для тестов. Это не академизм: правило 4 брифа
требует, чтобы тесты не ходили в сеть, а правило про переключение провайдеров
через `.env` требует подмены реализации без правки кода извлечения.

## Обоснование выбора

- **Тип проекта:** пайплайн обработки данных + ML/LLM + 4 точки входы (CLI, API, бот, MCP)
- **Стек:** Python 3.12, SQLAlchemy 2.x async, Pydantic v2, DuckDB, Postgres+pgvector
- **Размер команды:** 1 человек
- **Ключевой фактор:** структура репозитория **уже зафиксирована брифом** по
  стадиям пайплайна. Задача архитектуры — не придумать раскладку, а задать правила
  зависимостей внутри неё и вынести внешние системы за порты.

Почему не альтернативы:

- **Layered Architecture** формально подходит под «команда 1 человек», но это
  паттерн для CRUD. Здесь нет `routes → controllers → services → repositories`:
  большая часть работы — офлайн-прогоны через CLI, а HTTP появляется только в M7.
  Плоские `services/` и `repositories/` смешали бы ingestion, extraction и RAG
  в одну кашу.
- **Explicit Architecture** дала бы жёсткие границы, но для одного разработчика с
  шестью стадиями это формализм: четыре концентрических слоя внутри каждой стадии
  утроят объём кода без выигрыша. Ports уже взяты — как раз то, что здесь ценно.
  Путь миграции остаётся открытым (`models/` → `Domain/`, `adapters/` →
  `Infrastructure/`), если проект вырастет.
- **Microservices** — очевидное overkill: один разработчик, один деплой.

## Структура каталогов

Расширяет структуру из брифа. Добавлены два общих модуля — они помечены и
обоснованы ниже.

```text
src/nutri_radar/
├── config.py                 # pydantic-settings: ЕДИНСТВЕННЫЙ источник параметров
├── errors.py                 # корневой NutriRadarError и доменные исключения
├── tracing.py                # (+) порт трассировки + no-op реализация по умолчанию
│
├── llm/                      # (+) ОБЩИЙ МОДУЛЬ: доступ к языковым моделям
│   ├── ports.py              #     Protocol StructuredLLM, EmbeddingModel
│   ├── models.py             #     LLMResponse, TokenUsage — общие типы
│   └── adapters/
│       ├── ollama.py         #     локальная модель, генерация по JSON-схеме
│       ├── anthropic.py      #     облачная модель (агент, эталон в evals)
│       └── fake.py           #     детерминированная реализация для тестов
│
├── db/                       # ОБЩИЙ МОДУЛЬ: доступ к Postgres
│   ├── base.py               #     DeclarativeBase
│   ├── session.py            #     async engine, sessionmaker
│   ├── models/               #     таблицы SQLAlchemy
│   │   ├── product.py        #       products_raw, products
│   │   ├── extraction.py     #       product_extraction
│   │   ├── ingredient.py     #       ingredients_dict
│   │   ├── embedding.py      #       product_embeddings
│   │   └── run.py            #       runs — журнал прогонов
│   └── repositories/         #     запросы; наружу отдают Pydantic, не ORM-объекты
│
├── ingest/                   # ── СЛАЙС: стадия ingestion ──
│   ├── models.py             #     RawProduct — внутренний контракт источника
│   ├── sources/              #     два адаптера (ADR-004)
│   │   ├── parquet.py        #       полный дамп: LIST<STRUCT> через DuckDB
│   │   └── delta.py          #       дельты: JSONL в MongoDB-форме
│   ├── download.py           #     идемпотентное скачивание с HuggingFace
│   ├── select.py             #     фильтрация корпуса в DuckDB
│   ├── load.py               #     батчевая заливка в Postgres
│   └── cli.py                #     ingest dump | select | delta
│
├── extract/                  # ── СЛАЙС: извлечение структуры ──
│   ├── schemas.py            #     Pydantic-схема выхода = JSON-схема для LLM
│   ├── prompts/              #     версионированные промпты: v1.md, v2.md …
│   ├── normalize.py          #     словарь алиасов, канонизация ингредиентов
│   ├── sugar.py              #     подсчёт разных форм сахара — ключевая фича
│   ├── runner.py             #     батчинг, возобновляемость, учёт токенов
│   └── cli.py
│
├── evals/                    # ── СЛАЙС: оценка качества ──
│   ├── annotate.py           #     CLI ручной разметки (метки ставит ЧЕЛОВЕК)
│   ├── metrics.py            #     precision / recall / F1, accuracy
│   ├── compare.py            #     OFF-парсер vs локальная vs облачная (ADR-006)
│   ├── gate.py               #     пересчёт метрик из предсказаний для CI (ADR-008)
│   └── cli.py
│
├── analytics/                # ── СЛАЙС: classic ML ──
│   ├── features.py           #     TF-IDF, эмбеддинги, числовые фичи
│   ├── tasks/                #     sanity_check.py, grade_from_text.py, nova.py
│   ├── report.py             #     графики в reports/
│   └── cli.py
│
├── retrieval/                # ── СЛАЙС: поиск и RAG ──
│   ├── profile.py            #     сборка текста профиля продукта
│   ├── embed.py              #     bge-m3 через порт EmbeddingModel
│   ├── search.py             #     pgvector + фильтры
│   ├── rag.py                #     ответ строго по найденному, иначе «не знаю»
│   └── cli.py
│
├── agent/                    # ── СЛАЙС: агент ──
│   ├── tools/                #     sql_query, vector_search, lookup_barcode
│   ├── loop.py               #     план → инструмент → проверка, лимиты шагов и стоимости
│   └── cli.py
│
├── api/                      # ── ТОЧКА ВХОДА: FastAPI ──
├── bot/                      # ── ТОЧКА ВХОДА: aiogram 3 ──
└── mcp_server/               # ── ТОЧКА ВХОДА: MCP ──

tests/
├── data/                     # маленькие синтетические Parquet и JSONL
├── test_ingest/ …            # структура зеркалит слайсы
```

**Два добавления к структуре брифа и почему они нужны:**

1. **`llm/`** — адаптер языковых моделей нужен четырём потребителям сразу:
   `extract` (локальная модель), `evals` (эталон), `analytics` (zero-shot),
   `agent` (облачная). Положить его внутрь `extract/` означало бы, что `evals` и
   `agent` импортируют внутренности `extract` — прямое нарушение изоляции модулей.
   Бриф требует «один тонкий адаптер», и общий модуль — единственное место, где он
   не создаёт перекрёстных зависимостей.
2. **`tracing.py`** — Langfuse включается только на M6 и живёт в отдельном
   compose-профиле (ADR-007). Порт с no-op реализацией по умолчанию позволяет
   писать вызовы трассировки с самого начала, не завязываясь на поднятый Langfuse.

## Правила зависимостей

Три уровня. Стрелки только вниз.

```text
┌──────────────────────────────────────────────────────────┐
│  ТОЧКИ ВХОДА:  api/   bot/   mcp_server/   *_cli         │
│  без логики: разобрать вход → вызвать слайс → отформатировать
└────────────────────────┬─────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────┐
│  СЛАЙСЫ:  ingest  extract  evals  analytics  retrieval   │
│           agent                                          │
│  друг от друга НЕ зависят (кроме agent → retrieval)      │
└────────────────────────┬─────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────┐
│  ОБЩЕЕ:  db/   llm/   config.py   errors.py   tracing.py │
└──────────────────────────────────────────────────────────┘
```

Разрешено:

- ✅ точка входа → слайс → общее
- ✅ любой модуль → `config`, `errors`, `tracing`
- ✅ слайс → `db` и `llm` **через порты**, а не через конкретные адаптеры
- ✅ `agent` → `retrieval` — единственная разрешённая зависимость между слайсами:
  `vector_search` буквально обёртка над поиском, дублировать его бессмысленно
- ✅ `evals` → `extract.schemas` и `extract.prompts` — публичная часть `extract`,
  без неё нечего оценивать

Запрещено:

- ❌ `db` или `llm` импортируют что-либо из слайса — общее не знает о конкретных стадиях
- ❌ слайс импортирует другой слайс, кроме двух исключений выше. Нужны общие
  данные — читай из Postgres, стадии связаны таблицами, а не импортами
- ❌ `config` импортирует что-либо изнутри проекта
- ❌ точка входа обращается к `db` напрямую, минуя слайс
- ❌ ORM-объекты SQLAlchemy утекают наружу из `db/repositories/` — на границе
  возвращаются Pydantic-модели, иначе ленивая загрузка выстрелит в отвязанной сессии
- ❌ слайс импортирует конкретный адаптер (`llm.adapters.ollama`) вместо порта.
  Конкретику выбирает composition root по `.env`
- ❌ циклы между модулями в любом виде

## Как модули общаются

- **Между стадиями пайплайна — через Postgres.** `ingest` пишет `products`,
  `extract` читает `products` и пишет `product_extraction`. Прямых вызовов нет.
  Это даёт возобновляемость бесплатно: перезапуск смотрит, что уже в таблице.
- **Прогресс и метаданные прогонов — через таблицу `runs`.** Каждая стадия
  открывает запись прогона и обновляет её. Отсюда же берутся числа для отчётов:
  время, токены, обработано/пропущено.
- **Внешние системы — только через порты** из `llm/ports.py` и `tracing.py`.
- **Конкретные реализации подставляются в composition root** — по одной точке
  сборки на каждую точку входа (CLI, API, бот, MCP), выбор по `.env`.
- **Внутренний контракт источников данных — `ingest/models.py:RawProduct`.**
  Parquet-адаптер и JSONL-адаптер оба возвращают его; ниже разница не видна.

## Ключевые принципы

1. **Модуль = стадия пайплайна.** Границы совпадают с потоком данных, поэтому
   «где живёт эта логика» — вопрос без вариантов. Новая функциональность попадает
   в существующий слайс или заводит новый, но не растекается по трём.
2. **Внешние системы за портами.** Ollama, Anthropic, HTTP OFF, Langfuse — за
   `Protocol`, с fake-реализациями в тестах. Правило «тест не ходит в сеть»
   выполняется структурно, а не силой воли.
3. **Точки входа тонкие.** `api/`, `bot/`, `mcp_server/`, CLI разбирают вход,
   вызывают слайс, форматируют ответ. Если появилась ветка `if` по домену —
   она уехала не туда.
4. **Логика в моделях, не в раннерах.** `runner.py` управляет батчами, повторами
   и учётом токенов. Что считается формой сахара, какой ингредиент к какому типу
   относится, когда состав считается нечитаемым — это в `schemas.py`, `sugar.py`,
   `normalize.py` и покрывается тестами без всякой LLM.
5. **Никаких магических констант.** Порог, размер батча, `num_ctx`, лимиты
   агента — только `config.py`. Число в вызове функции — повод для правки.
6. **Общее остаётся маленьким.** `db/`, `llm/`, три файла в корне. Если в общее
   потянуло доменную логику — значит, границу слайса определили неверно.
7. **Возобновляемость по умолчанию.** Любой массовый прогон переживает Ctrl+C:
   состояние в БД, перезапуск продолжает с необработанных.

## Примеры кода

### Порт и адаптеры для языковой модели

Порт живёт в общем модуле, адаптеры рядом, слайс не знает о конкретных.

```python
# src/nutri_radar/llm/ports.py
from typing import Any, Protocol

from pydantic import BaseModel


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int


class LLMResponse(BaseModel):
    raw_json: dict[str, Any]
    usage: TokenUsage
    latency_s: float
    model_name: str


class StructuredLLM(Protocol):
    """Генерация, ограниченная JSON-схемой.

    Реализации: OllamaLLM (локальная), AnthropicLLM (облако), FakeLLM (тесты).
    """

    async def generate(
        self,
        *,
        prompt: str,
        json_schema: dict[str, Any],
        max_output_tokens: int,
    ) -> LLMResponse: ...
```

```python
# src/nutri_radar/llm/adapters/ollama.py
import time
from typing import Any

import httpx

from nutri_radar.config import OllamaSettings
from nutri_radar.errors import LLMUnavailableError
from nutri_radar.llm.ports import LLMResponse, TokenUsage


class OllamaLLM:
    """Локальная модель. num_ctx задаётся ЯВНО: дефолт Ollama 4096 и он
    подбирается динамически по VRAM, то есть молча обрезает вход."""

    def __init__(self, client: httpx.AsyncClient, settings: OllamaSettings) -> None:
        self._client = client
        self._settings = settings

    async def generate(
        self,
        *,
        prompt: str,
        json_schema: dict[str, Any],
        max_output_tokens: int,
    ) -> LLMResponse:
        started = time.perf_counter()
        try:
            response = await self._client.post(
                "/api/chat",
                json={
                    "model": self._settings.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "format": json_schema,        # ограничение генерации схемой
                    "stream": False,
                    "options": {
                        "num_ctx": self._settings.num_ctx,      # из конфига, не литерал
                        "num_predict": max_output_tokens,
                        "temperature": self._settings.temperature,
                    },
                    "keep_alive": self._settings.keep_alive,    # 6 ГБ VRAM: модели по очереди
                },
                timeout=self._settings.timeout_s,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            # внешняя ошибка не протекает наружу из слоя llm
            raise LLMUnavailableError(f"Ollama недоступна: {exc}") from exc

        payload = response.json()
        return LLMResponse(
            raw_json=payload["message"]["content"],
            usage=TokenUsage(
                input_tokens=payload.get("prompt_eval_count", 0),
                output_tokens=payload.get("eval_count", 0),
            ),
            latency_s=time.perf_counter() - started,
            model_name=self._settings.model,
        )
```

### Слайс зависит от порта, конкретику подставляет composition root

```python
# src/nutri_radar/extract/runner.py
from nutri_radar.extract.schemas import ExtractionResult
from nutri_radar.llm.ports import StructuredLLM          # ✅ порт
from nutri_radar.tracing import Tracer


class ExtractionRunner:
    def __init__(self, llm: StructuredLLM, tracer: Tracer) -> None:
        self._llm = llm          # ❌ НЕ OllamaLLM внутри — иначе провайдер не сменить
        self._tracer = tracer

    async def extract_one(self, product_id: str, ingredients_text: str) -> ExtractionResult:
        with self._tracer.span("extract_one", product_id=product_id):
            response = await self._llm.generate(
                prompt=self._prompt.render(ingredients_text),
                json_schema=ExtractionResult.model_json_schema(),
                max_output_tokens=self._settings.max_output_tokens,
            )
        # схема гарантирует форму, но не смысл — валидируем всё равно
        return ExtractionResult.model_validate(response.raw_json)
```

```python
# src/nutri_radar/extract/cli.py — composition root
def build_llm(settings: Settings, client: httpx.AsyncClient) -> StructuredLLM:
    """Единственное место, где выбирается конкретный провайдер."""
    match settings.llm.provider:
        case "ollama":
            return OllamaLLM(client, settings.ollama)
        case "anthropic":
            return AnthropicLLM(client, settings.anthropic)
        case unknown:
            raise ConfigurationError(f"Неизвестный провайдер LLM: {unknown}")
```

### Логика в модели, а не в раннере

Подсчёт разных форм сахара — ключевая фича проекта. Она тестируется без LLM.

```python
# src/nutri_radar/extract/schemas.py
from enum import StrEnum

from pydantic import BaseModel, Field


class IngredientKind(StrEnum):
    BASE = "base"
    SUGAR = "sugar"
    FAT = "fat"
    ADDITIVE = "additive"
    FLAVOURING = "flavouring"
    PRESERVATIVE = "preservative"
    SWEETENER = "sweetener"


class Ingredient(BaseModel):
    canonical_name: str
    kind: IngredientKind
    e_number: str | None = None


class ExtractionResult(BaseModel):
    """Выход LLM. Схема этого класса передаётся в параметр format."""

    ingredients: list[Ingredient]
    allergens: list[str] = Field(default_factory=list)
    unreadable: bool = Field(
        default=False,
        description="Состав нечитаем или обрезан — запись не идёт в аналитику",
    )
    model_confidence: float = Field(ge=0.0, le=1.0)

    @property
    def distinct_sugar_forms(self) -> int:
        """Сколько РАЗНЫХ названий сахара в составе.

        Величина, которой нет в Open Food Facts, и самая наглядная часть демо.
        Считается по каноническим именам, поэтому «сироп глюкозы» дважды в одном
        составе — это одна форма, а глюкозный сироп и мальтодекстрин — две.
        """
        return len({i.canonical_name for i in self.ingredients if i.kind is IngredientKind.SUGAR})

    @property
    def e_additives_count(self) -> int:
        return sum(1 for i in self.ingredients if i.e_number is not None)
```

### Репозиторий отдаёт Pydantic, не ORM-объекты

```python
# src/nutri_radar/db/repositories/product.py
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nutri_radar.db.models.product import ProductRow
from nutri_radar.db.schemas import Product          # Pydantic, не ORM


class ProductRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def iter_pending_extraction(self, *, prompt_version: str, limit: int) -> list[Product]:
        """Продукты без извлечения этой версией промпта.

        Возобновляемость прогона: условие живёт в запросе, а не в
        состоянии процесса, поэтому перезапуск просто продолжает с места.
        """
        stmt = (
            select(ProductRow)
            .outerjoin(ProductRow.extractions)
            .where(ProductRow.ingredients_text.is_not(None))
            .where(ProductRow.extraction_version_absent(prompt_version))
            .limit(limit)
        )
        rows = (await self._session.scalars(stmt)).all()
        # ✅ границу пересекают Pydantic-модели: ленивая загрузка не выстрелит
        return [Product.model_validate(row, from_attributes=True) for row in rows]
```

## Антипаттерны

- ❌ **Слайс импортирует слайс.** `analytics` тянет `from nutri_radar.extract.runner
  import ExtractionRunner`, чтобы «пересчитать на ходу». Стадии связаны таблицами:
  читай `product_extraction`. Иначе аналитика начнёт незаметно жечь GPU.
- ❌ **Конкретный адаптер внутри слайса.** `OllamaLLM()` прямо в `runner.py` —
  провайдер больше не переключить через `.env`, тесты полезут в сеть.
- ❌ **Логика в раннере вместо модели.** Подсчёт форм сахара внутри цикла батчинга:
  чтобы это протестировать, придётся поднимать LLM. Держи в `schemas.py` / `sugar.py`.
- ❌ **ORM-объекты за пределами `db/`.** Отдали `ProductRow` наружу — получили
  `MissingGreenlet` на первом же обращении к атрибуту вне сессии.
- ❌ **Толстая точка входа.** Хендлер бота, который сам ходит в pgvector и считает
  сахара. Точка входа вызывает слайс и форматирует ответ, не больше.
- ❌ **Магическая константа.** `num_ctx=8192` литералом в вызове. Это ключевой
  параметр качества извлечения — его место в `config.py`, чтобы менять и логировать.
- ❌ **Блокирующий вызов в корутине.** `duckdb.execute(...)` или `model.fit(...)`
  прямо в `async def`: событийный цикл встанет вместе со всем API. Тяжёлый CPU —
  в синхронных функциях, запускаемых отдельно.
- ❌ **Неограниченный веер запросов к Ollama.** `asyncio.gather` по всему батчу на
  6 ГБ VRAM. Параллелизм — из конфига, через семафор.
- ❌ **Молчаливое проглатывание битой записи.** `except Exception: pass` в цикле
  прогона. Помечай, считай, пиши счётчик в `runs` — иначе доля пропусков
  неизвестна, а она входит в обязательные метрики проекта.
