"""Проверка работоспособности среды.

Отвечает не на вопрос «контейнеры запустились», а на вопрос «средой можно
пользоваться». Пять проверок:

1. соединение с Postgres;
2. расширение `vector` действительно установлено, а не просто доступно;
3. ревизия Alembic совпадает с head;
4. Ollama отвечает и содержит настроенные модели;
5. ключ Anthropic задан.

Деградации (3, 4, 5) дают `WARN` и не роняют команду: на M0 ни Ollama, ни ключ
ещё не нужны, но узнать об их отсутствии лучше сразу. Отказы (1, 2) дают `FAIL`.

Каждая проверка изолирована: падение одной не отменяет остальные, иначе после
первой ошибки непонятно, что ещё сломано.
"""

from __future__ import annotations

import asyncio
import logging
import time
from enum import StrEnum
from pathlib import Path

import httpx
from pydantic import BaseModel, Field
from sqlalchemy import text

from nutri_radar.config import Settings, get_settings
from nutri_radar.db.session import dispose_engine, get_session

logger = logging.getLogger(__name__)


class CheckStatus(StrEnum):
    OK = "OK"
    WARN = "WARN"
    FAIL = "FAIL"


class CheckResult(BaseModel):
    """Результат одной проверки."""

    name: str
    status: CheckStatus
    detail: str
    elapsed_s: float = 0.0


class HealthReport(BaseModel):
    """Итог всех проверок."""

    checks: list[CheckResult] = Field(default_factory=list)

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status is CheckStatus.FAIL]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status is CheckStatus.WARN]

    @property
    def is_healthy(self) -> bool:
        """Здоровье определяется отсутствием FAIL. WARN допустим."""
        return not self.failures

    @property
    def exit_code(self) -> int:
        """0 при отсутствии FAIL — годится и для healthcheck в compose."""
        return 0 if self.is_healthy else 1


async def _check_postgres(settings: Settings) -> CheckResult:
    started = time.perf_counter()
    name = "Postgres: соединение"
    try:
        async with get_session(settings.db) as session:
            version = (await session.execute(text("SELECT version()"))).scalar_one()
        detail = str(version).split(",")[0]
        logger.debug("Postgres ответил", extra={"version": detail})
        status = CheckStatus.OK
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        logger.error(
            "Postgres недоступен",
            extra={"dsn": settings.db.safe_dsn, "error": detail},
            exc_info=True,
        )
        status = CheckStatus.FAIL
    return CheckResult(
        name=name, status=status, detail=detail, elapsed_s=round(time.perf_counter() - started, 3)
    )


async def _check_pgvector(settings: Settings) -> CheckResult:
    started = time.perf_counter()
    name = "Postgres: расширение vector"
    try:
        async with get_session(settings.db) as session:
            # Именно pg_extension, а не pg_available_extensions: нужно, чтобы
            # расширение было УСТАНОВЛЕНО, а не просто доступно к установке.
            installed = (
                await session.execute(
                    text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                )
            ).scalar_one_or_none()
        if installed is None:
            detail = "расширение не установлено — применены ли миграции?"
            status = CheckStatus.FAIL
            logger.error("Расширение vector отсутствует")
        else:
            detail = f"версия {installed}"
            status = CheckStatus.OK
            logger.debug("Расширение vector установлено", extra={"version": installed})
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        status = CheckStatus.FAIL
        logger.error("Не удалось проверить расширение vector", exc_info=True)
    return CheckResult(
        name=name, status=status, detail=detail, elapsed_s=round(time.perf_counter() - started, 3)
    )


def _head_revision() -> str | None:
    """Ревизия head из файлов миграций, без обращения к БД."""
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    ini_path = Path("alembic.ini")
    if not ini_path.exists():
        return None
    script = ScriptDirectory.from_config(Config(str(ini_path)))
    return script.get_current_head()


async def _check_migrations(settings: Settings) -> CheckResult:
    started = time.perf_counter()
    name = "Alembic: ревизия"
    try:
        head = _head_revision()
        async with get_session(settings.db) as session:
            current = (
                await session.execute(text("SELECT version_num FROM alembic_version"))
            ).scalar_one_or_none()

        if current is None:
            detail = "миграции не применялись — выполните alembic upgrade head"
            status = CheckStatus.WARN
            logger.warning("Таблица alembic_version пуста")
        elif head is not None and current != head:
            detail = f"применена {current}, доступна {head} — выполните alembic upgrade head"
            status = CheckStatus.WARN
            logger.warning("Ревизия отстаёт от head", extra={"current": current, "head": head})
        else:
            detail = f"{current} (head)"
            status = CheckStatus.OK
            logger.debug("Ревизия совпадает с head", extra={"revision": current})
    except Exception as exc:
        # Отсутствие таблицы alembic_version — это ещё не отказ БД: миграции
        # просто не накатывали. Поэтому WARN, а не FAIL.
        detail = f"не удалось определить ревизию ({type(exc).__name__})"
        status = CheckStatus.WARN
        logger.warning("Не удалось прочитать ревизию Alembic", exc_info=True)
    return CheckResult(
        name=name, status=status, detail=detail, elapsed_s=round(time.perf_counter() - started, 3)
    )


async def _check_ollama(settings: Settings, client: httpx.AsyncClient | None = None) -> CheckResult:
    started = time.perf_counter()
    name = "Ollama: модели"
    owns_client = client is None
    client = client or httpx.AsyncClient(base_url=settings.ollama.base_url, timeout=5.0)
    try:
        response = await client.get("/api/tags")
        response.raise_for_status()
        available = {model["name"] for model in response.json().get("models", [])}
        logger.debug("Ollama ответила", extra={"models_count": len(available)})

        # Ollama возвращает имена с тегом (`bge-m3:latest`), а в конфиге тег
        # может быть опущен — сравниваем по префиксу до двоеточия.
        def present(wanted: str) -> bool:
            return any(m == wanted or m.split(":")[0] == wanted.split(":")[0] for m in available)

        missing = [
            m for m in (settings.ollama.model, settings.ollama.embedding_model) if not present(m)
        ]
        if missing:
            detail = f"нет моделей: {', '.join(missing)} — выполните ollama pull"
            status = CheckStatus.WARN
            logger.warning("В Ollama отсутствуют модели", extra={"missing": missing})
        else:
            detail = f"обе модели на месте ({len(available)} всего)"
            status = CheckStatus.OK
    except Exception as exc:
        # На M0 Ollama ещё не нужна — деградация, а не отказ.
        detail = f"недоступна ({type(exc).__name__}) — на M0 не критично"
        status = CheckStatus.WARN
        logger.warning(
            "Ollama недоступна",
            extra={"base_url": settings.ollama.base_url, "error": str(exc)},
        )
    finally:
        if owns_client:
            await client.aclose()
    return CheckResult(
        name=name, status=status, detail=detail, elapsed_s=round(time.perf_counter() - started, 3)
    )


def _check_anthropic(settings: Settings) -> CheckResult:
    """Только факт наличия ключа. Обращаться к API не будем — за это платят."""
    name = "Anthropic: ключ"
    if settings.anthropic.is_configured:
        logger.debug("Ключ Anthropic задан", extra={"model": settings.anthropic.model})
        return CheckResult(
            name=name, status=CheckStatus.OK, detail=f"задан, модель {settings.anthropic.model}"
        )
    logger.warning("Ключ Anthropic не задан — агент и evals будут недоступны")
    return CheckResult(
        name=name,
        status=CheckStatus.WARN,
        detail="не задан — понадобится на M3 (эталон evals) и M6 (агент)",
    )


async def check_health(
    settings: Settings | None = None,
    *,
    http_client: httpx.AsyncClient | None = None,
) -> HealthReport:
    """Выполнить все проверки и вернуть отчёт.

    Args:
        settings: настройки; по умолчанию берутся из `get_settings()`.
        http_client: клиент для Ollama. Подменяется в тестах, чтобы они
            не ходили в сеть (правило 4 брифа).
    """
    settings = settings or get_settings()
    logger.info(
        "Health-check запущен",
        extra={"checks": ["postgres", "pgvector", "alembic", "ollama", "anthropic"]},
    )

    report = HealthReport()
    # Проверки БД идут последовательно: они делят один пул соединений, и при
    # недоступной базе параллельный запуск лишь размножит одинаковые таймауты.
    report.checks.append(await _check_postgres(settings))
    report.checks.append(await _check_pgvector(settings))
    report.checks.append(await _check_migrations(settings))
    report.checks.append(await _check_ollama(settings, http_client))
    report.checks.append(_check_anthropic(settings))

    logger.info(
        "Health-check завершён",
        extra={
            "ok": sum(1 for c in report.checks if c.status is CheckStatus.OK),
            "warn": len(report.warnings),
            "fail": len(report.failures),
        },
    )
    return report


def run_health_check(
    settings: Settings | None = None,
    *,
    http_client: httpx.AsyncClient | None = None,
) -> HealthReport:
    """Синхронная обёртка для CLI. Закрывает пул соединений после проверки."""

    async def _run() -> HealthReport:
        try:
            return await check_health(settings, http_client=http_client)
        finally:
            await dispose_engine()

    return asyncio.run(_run())
