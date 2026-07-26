# Многостадийная сборка: зависимости ставятся в builder, в runtime уезжает
# только готовое виртуальное окружение. Образ не тащит uv, компиляторы и кэш.

# =============================================================================
# Стадия 1: зависимости
# =============================================================================
FROM python:3.12-slim AS builder

# uv копируется готовым бинарём из официального образа — быстрее и
# воспроизводимее, чем установка через pip.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Сначала только манифесты: слой с зависимостями переиспользуется, пока
# они не менялись, и правка кода не вызывает переустановку пакетов.
COPY pyproject.toml uv.lock README.md ./

# --frozen: собираем строго по uv.lock, без обновления версий.
# --no-dev: pytest, ruff и mypy в рантайме не нужны.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Теперь код и установка самого пакета
COPY src ./src
COPY alembic.ini ./
COPY alembic ./alembic
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# =============================================================================
# Стадия 2: рантайм
# =============================================================================
FROM python:3.12-slim AS runtime

# Непривилегированный пользователь: процесс в контейнере не должен быть root
RUN groupadd --system --gid 1001 nutri \
    && useradd --system --uid 1001 --gid nutri --create-home nutri

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Сообщения проекта на русском: без UTF-8 вывод в docker logs побьётся
    PYTHONIOENCODING=utf-8

WORKDIR /app

COPY --from=builder --chown=nutri:nutri /app/.venv /app/.venv
COPY --from=builder --chown=nutri:nutri /app/src /app/src
COPY --from=builder --chown=nutri:nutri /app/alembic /app/alembic
COPY --from=builder --chown=nutri:nutri /app/alembic.ini /app/alembic.ini

# Каталоги для смонтированных томов создаём заранее с нужным владельцем,
# иначе процесс под nutri не сможет в них писать.
RUN mkdir -p /app/data /app/reports && chown -R nutri:nutri /app/data /app/reports

USER nutri

ENTRYPOINT ["nutri-radar"]
CMD ["health"]
