"""Тесты инструментов агента.

Главное здесь — **защита `sql_query`**. Это единственное место проекта,
где текст, сочинённый моделью, выполняется базой. Просить модель в промпте
«не пиши DELETE» защитой не является: промпт — пожелание, инъекция —
выполнение. Поэтому каждая проверка покрыта отдельным тестом, и тесты
написаны как список того, что должно быть отбито, а не как проверка
счастливого пути.

Второе — **`lookup_barcode` не ходит в сеть из теста** (правило 4 брифа).
Транспорт подставной; заодно проверяется, что «продукт не найден» —
нормальный ответ, а не ошибка: пометить его неудачей значило бы толкнуть
модель на повтор в надежде, что во второй раз получится.
"""

from __future__ import annotations

import httpx
import pytest

from nutri_radar.agent.tools import Tool, ToolRegistry, ToolResult
from nutri_radar.agent.tools.lookup_barcode import (
    FIELDS,
    USER_AGENT,
    format_product,
    run_lookup_barcode,
)
from nutri_radar.agent.tools.sql_query import ALLOWED_TABLES, validate_sql
from nutri_radar.agent.tools.vector_search import format_hits
from nutri_radar.config import AgentSettings, Settings
from nutri_radar.retrieval.search import SearchHit

MAX_ROWS = 20


@pytest.fixture
def agent_settings(settings: Settings) -> Settings:
    return settings.model_copy(update={"agent": AgentSettings(sql_max_rows=MAX_ROWS)})


class TestЗащитаSQL:
    def test_корректный_select_проходит(self):
        assert validate_sql("SELECT code FROM products LIMIT 5", MAX_ROWS) is None

    @pytest.mark.parametrize(
        "query",
        [
            "DELETE FROM products WHERE 1=1",
            "UPDATE products SET code = '1'",
            "DROP TABLE products",
            "INSERT INTO products (code) VALUES ('1')",
            "TRUNCATE products",
        ],
    )
    def test_запись_отбивается(self, query: str):
        """Read-only транзакция — последняя линия, но до неё дело
        доходить не должно: модель обязана получить внятный отказ."""
        assert "только SELECT" in (validate_sql(query, MAX_ROWS) or "")

    def test_две_команды_в_строке_отбиваются(self):
        """Классическая инъекция: безобидный SELECT плюс разрушительное
        продолжение через точку с запятой."""
        error = validate_sql("SELECT 1 FROM products LIMIT 1; DROP TABLE products", MAX_ROWS)

        assert "Точка с запятой" in (error or "")

    def test_завершающая_точка_с_запятой_разрешена(self):
        """Она не делает две команды — а отбивать её значило бы отвергать
        синтаксически привычный запрос без всякой пользы."""
        assert validate_sql("SELECT code FROM products LIMIT 5;", MAX_ROWS) is None

    def test_таблица_вне_белого_списка_отбивается(self):
        error = validate_sql("SELECT * FROM alembic_version LIMIT 1", MAX_ROWS)

        assert "alembic_version" in (error or "")

    def test_join_с_чужой_таблицей_отбивается(self):
        """Белый список проверяет все источники строк, а не только первый
        после FROM: иначе через JOIN можно дотянуться куда угодно."""
        error = validate_sql(
            "SELECT p.code FROM products p JOIN pg_catalog.pg_tables t ON true LIMIT 1",
            MAX_ROWS,
        )

        assert "pg_tables" in (error or "")

    def test_все_разрешённые_таблицы_проходят(self):
        for table in ALLOWED_TABLES:
            assert validate_sql(f"SELECT * FROM {table} LIMIT 1", MAX_ROWS) is None

    def test_отсутствие_limit_отбивается(self):
        """Не защита от вреда, а защита от того, что модель получит
        десять тысяч строк и утонет в них вместе с контекстом."""
        error = validate_sql("SELECT code FROM products", MAX_ROWS)

        assert "LIMIT" in (error or "")

    def test_пустой_запрос_отбивается(self):
        assert validate_sql("   ", MAX_ROWS) == "Пустой запрос."

    def test_отказ_объясняет_что_делать(self):
        """Модель должна суметь исправиться, а не просто узнать об отказе."""
        error = validate_sql("SELECT code FROM products", MAX_ROWS) or ""

        assert str(MAX_ROWS) in error

    def test_регистр_не_обходит_защиту(self):
        assert "только SELECT" in (validate_sql("delete from products", MAX_ROWS) or "")


class TestШтрихкод:
    async def test_валидация_до_сети(self, agent_settings: Settings):
        """Модель охотно подставляет название вместо кода. Ходить с ним
        в API — ждать таймаут ради заведомого отказа."""
        result = await run_lookup_barcode("шоколад", agent_settings)

        assert result.ok is False
        assert "не похож на штрихкод" in result.content

    async def test_продукт_возвращается(self, agent_settings: Settings):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "status": 1,
                    "product": {
                        "product_name": "Молочный шоколад",
                        "brands": "Alpen Gold",
                        "nutriscore_grade": "e",
                        "ingredients_text": "Сахар, какао тёртое",
                    },
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await run_lookup_barcode("3017620425035", agent_settings, client=client)

        assert result.ok is True
        assert "[3017620425035]" in result.content
        assert "какао тёртое" in result.content

    async def test_не_найдено_это_нормальный_ответ(self, agent_settings: Settings):
        """Пометить его ошибкой значило бы толкнуть модель на повтор."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"status": 0})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await run_lookup_barcode("9999999999999", agent_settings, client=client)

        assert result.ok is True
        assert result.meta["found"] is False
        assert "не найден" in result.content

    async def test_недоступность_api_это_ошибка(self, agent_settings: Settings):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("нет связи")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await run_lookup_barcode("3017620425035", agent_settings, client=client)

        assert result.ok is False
        assert "недоступен" in result.content

    async def test_user_agent_и_поля_уходят_в_запрос(self, agent_settings: Settings):
        """User-Agent с контактом — требование OFF, а не вежливость.
        Поля перечнем: полный ответ по продукту весит десятки килобайт."""
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["ua"] = request.headers.get("user-agent")
            captured["fields"] = request.url.params.get("fields")
            return httpx.Response(200, json={"status": 0})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await run_lookup_barcode("3017620425035", agent_settings, client=client)

        assert captured["ua"] == USER_AGENT
        assert captured["fields"] == FIELDS

    def test_штрихкод_подаётся_в_форме_для_цитирования(self):
        """Та же форма, что требуется в ответе: чем ближе, тем реже модель
        изобретает свою."""
        assert format_product("123456", {}).startswith("[123456]")


class TestВыдачаПоиска:
    def test_штрихкоды_в_квадратных_скобках(self):
        hit = SearchHit(
            code="3017620425035",
            product_name="Шоколад",
            brands=None,
            ingredients_text="Сахар",
            lang="ru",
            nutriscore_grade="e",
            nova_group=4,
            distance=0.2,
        )

        assert "[3017620425035]" in format_hits([hit])

    def test_пустая_выдача_называется_словами(self):
        assert format_hits([]) == "Ничего не найдено."


class TestРеестр:
    async def test_неизвестный_инструмент_это_сообщение_модели(self):
        """А не исключение: модель должна узнать, что доступно, и выбрать
        заново."""
        registry = ToolRegistry()

        result = await registry.call("нет-такого", {})

        assert result.ok is False
        assert "нет" in result.content.lower()

    async def test_неверные_аргументы_объясняются(self):
        async def echo(text: str) -> ToolResult:
            return ToolResult(ok=True, content=text)

        registry = ToolRegistry(
            [
                Tool(
                    name="echo",
                    description="Повторяет",
                    parameters={"type": "object", "properties": {"text": {"type": "string"}}},
                    run=echo,
                )
            ]
        )

        result = await registry.call("echo", {"wrong": "x"})

        assert result.ok is False
        assert "text" in result.content

    async def test_отказ_инструмента_не_роняет_цикл(self):
        async def broken() -> ToolResult:
            raise RuntimeError("внутренняя поломка")

        registry = ToolRegistry(
            [
                Tool(
                    name="broken",
                    description="Ломается",
                    parameters={"type": "object", "properties": {}},
                    run=broken,
                )
            ]
        )

        result = await registry.call("broken", {})

        assert result.ok is False
        assert "внутренняя поломка" in result.content

    def test_имя_инструмента_уникально(self):
        tool = Tool(
            name="dup",
            description="",
            parameters={"type": "object", "properties": {}},
            run=lambda: None,  # type: ignore[arg-type,return-value]
        )

        with pytest.raises(ValueError, match="уже зарегистрирован"):
            ToolRegistry([tool, tool])

    def test_схема_действия_запрещает_несуществующий_инструмент(self):
        """Самая частая ошибка слабой модели отсекается схемой,
        а не проверкой после."""
        registry = ToolRegistry(
            [
                Tool(
                    name="alpha",
                    description="",
                    parameters={"type": "object", "properties": {}},
                    run=lambda: None,  # type: ignore[arg-type,return-value]
                )
            ]
        )

        schema = registry.action_schema()

        assert schema["properties"]["action"]["enum"] == ["alpha", "final_answer"]

    def test_порядок_инструментов_в_промпте_стабилен(self):
        """Разный порядок менял бы поведение модели между запусками,
        и сравнивать прогоны стало бы нельзя."""

        def make(name: str) -> Tool:
            return Tool(
                name=name,
                description="",
                parameters={"type": "object", "properties": {}},
                run=lambda: None,  # type: ignore[arg-type,return-value]
            )

        first = ToolRegistry([make("b"), make("a")]).describe()
        second = ToolRegistry([make("a"), make("b")]).describe()

        assert first == second
