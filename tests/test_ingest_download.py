"""Тесты скачивания дампа.

В сеть не ходим (правило 4 брифа): транспорт подменяется `httpx.MockTransport`.
Проверяется именно то, ради чего писалась докачка: файл 7,7 ГБ, и обрыв
посреди загрузки — обычное дело, а не исключительная ситуация.
"""

from __future__ import annotations

import httpx
import pytest

from nutri_radar.config import IngestSettings
from nutri_radar.errors import DataSourceError
from nutri_radar.ingest.download import download_dump, read_dump_version

CONTENT = b"0123456789" * 100  # 1000 байт
# Только ASCII: значения HTTP-заголовков кириллицу не переживают.
ETAG = '"test-etag-value"'


@pytest.fixture
def settings(tmp_path) -> IngestSettings:
    return IngestSettings(
        data_dir=tmp_path,
        dump_url="https://example.test/food.parquet",
        download_chunk_size=64,
        download_max_retries=2,
    )


def make_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def full_handler(request: httpx.Request) -> httpx.Response:
    """Источник, поддерживающий Range."""
    if request.method == "HEAD":
        return httpx.Response(200, headers={"content-length": str(len(CONTENT)), "etag": ETAG})

    range_header = request.headers.get("Range")
    if range_header:
        start = int(range_header.removeprefix("bytes=").split("-")[0])
        return httpx.Response(206, content=CONTENT[start:])
    return httpx.Response(200, content=CONTENT)


class TestFreshDownload:
    def test_файл_скачивается_целиком(self, settings):
        result = download_dump(settings, client=make_client(full_handler))

        assert result.skipped is False
        assert result.size == len(CONTENT)
        assert result.path.read_bytes() == CONTENT

    def test_частичный_файл_не_остаётся(self, settings):
        download_dump(settings, client=make_client(full_handler))

        part = settings.dump_path.with_name(settings.dump_path.name + ".part")
        assert not part.exists()

    def test_версия_дампа_сохраняется(self, settings):
        result = download_dump(settings, client=make_client(full_handler))

        assert str(len(CONTENT)) in result.dump_version
        assert read_dump_version(settings) == result.dump_version


class TestIdempotency:
    def test_готовый_файл_не_перекачивается(self, settings):
        download_dump(settings, client=make_client(full_handler))

        calls: list[str] = []

        def counting_handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.method)
            return full_handler(request)

        result = download_dump(settings, client=make_client(counting_handler))

        assert result.skipped is True
        assert "GET" not in calls, "тело файла не должно запрашиваться повторно"

    def test_force_перекачивает_заново(self, settings):
        download_dump(settings, client=make_client(full_handler))

        result = download_dump(settings, force=True, client=make_client(full_handler))

        assert result.skipped is False

    def test_файл_неверного_размера_перекачивается(self, settings):
        settings.dump_path.parent.mkdir(parents=True, exist_ok=True)
        settings.dump_path.write_bytes("обрывок".encode())

        result = download_dump(settings, client=make_client(full_handler))

        assert result.skipped is False
        assert result.path.read_bytes() == CONTENT


class TestResume:
    def test_докачка_продолжается_с_нужного_смещения(self, settings):
        part = settings.dump_path.with_name(settings.dump_path.name + ".part")
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(CONTENT[:400])

        ranges: list[str | None] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                ranges.append(request.headers.get("Range"))
            return full_handler(request)

        result = download_dump(settings, client=make_client(handler))

        assert ranges == ["bytes=400-"], "докачка должна начаться с 400-го байта"
        assert result.path.read_bytes() == CONTENT

    def test_игнорирование_range_источником_не_портит_файл(self, settings):
        """Если сервер отдал 200 вместо 206, дописывать в конец нельзя."""
        part = settings.dump_path.with_name(settings.dump_path.name + ".part")
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(CONTENT[:400])

        def ignores_range(request: httpx.Request) -> httpx.Response:
            if request.method == "HEAD":
                return httpx.Response(200, headers={"content-length": str(len(CONTENT))})
            return httpx.Response(200, content=CONTENT)

        result = download_dump(settings, client=make_client(ignores_range))

        assert result.path.read_bytes() == CONTENT, "файл не должен быть склейкой"


class TestFailures:
    def test_расхождение_размера_даёт_ошибку_и_чистит_part(self, settings):
        def truncating(request: httpx.Request) -> httpx.Response:
            if request.method == "HEAD":
                return httpx.Response(200, headers={"content-length": str(len(CONTENT))})
            return httpx.Response(200, content=CONTENT[:100])

        with pytest.raises(DataSourceError, match="не совпадает"):
            download_dump(settings, client=make_client(truncating))

        part = settings.dump_path.with_name(settings.dump_path.name + ".part")
        assert not part.exists(), "битый частичный файл должен быть удалён"
        assert not settings.dump_path.exists(), "готовый файл не должен появиться"

    def test_источник_без_размера_даёт_ошибку(self, settings):
        def no_size(request: httpx.Request) -> httpx.Response:
            if request.method == "HEAD":
                return httpx.Response(200, headers={})
            return httpx.Response(200, content=CONTENT)

        with pytest.raises(DataSourceError, match="размер"):
            download_dump(settings, client=make_client(no_size))

    def test_ретраи_исчерпываются_и_дают_ошибку(self, settings, monkeypatch):
        monkeypatch.setattr("nutri_radar.ingest.download.time.sleep", lambda _: None)
        attempts: list[int] = []

        def failing(request: httpx.Request) -> httpx.Response:
            if request.method == "HEAD":
                return httpx.Response(200, headers={"content-length": str(len(CONTENT))})
            attempts.append(1)
            raise httpx.ConnectError("обрыв связи (подделка для теста)")

        with pytest.raises(DataSourceError, match="попыток"):
            download_dump(settings, client=make_client(failing))

        assert len(attempts) == settings.download_max_retries + 1


class TestVersion:
    def test_без_скачивания_версия_неизвестна(self, settings):
        assert read_dump_version(settings) is None
