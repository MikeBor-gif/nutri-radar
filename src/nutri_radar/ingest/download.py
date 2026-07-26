"""Скачивание Parquet-дампа Open Food Facts.

Требование брифа: скрипт идемпотентный. Файл 7,7 ГБ, поэтому докачка и
устойчивость к обрывам — не удобство, а необходимость.

Три особенности источника, проверенные вручную:

* `Accept-Ranges: bytes` — докачка возможна;
* HuggingFace отдаёт `302` на подписанный CDN-URL с параметром `Expires`.
  Подпись может истечь посреди докачки файла такого размера, поэтому редирект
  **переполучается на каждой попытке**, а не сохраняется между ними;
* размер известен заранее (`content-length` / `X-Linked-Size`), значит его можно
  сверить после скачивания и не принять обрыв за успех.

Атомарность: пишем в `<файл>.part` и переименовываем только после сверки
размера. Иначе оборванная загрузка оставила бы файл, который следующий запуск
принял бы за готовый.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from nutri_radar.config import IngestSettings, get_settings
from nutri_radar.errors import DataSourceError

logger = logging.getLogger(__name__)

_PART_SUFFIX = ".part"
_META_SUFFIX = ".meta.json"


@dataclass(frozen=True)
class RemoteInfo:
    """Что источник сообщает о файле до скачивания."""

    size: int
    etag: str | None


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    size: int
    skipped: bool
    elapsed_s: float
    etag: str | None

    @property
    def dump_version(self) -> str:
        """Версия дампа для записи в `products_raw` (раздел 7 брифа)."""
        return f"size={self.size};etag={self.etag or '—'}"


def _head_remote(client: httpx.Client, url: str) -> RemoteInfo:
    """Узнать размер и ETag, следуя редиректу.

    HuggingFace сообщает размер в `X-Linked-Size` уже на шаге редиректа, но
    надёжнее взять `content-length` с финального адреса.
    """
    response = client.head(url, follow_redirects=True)
    response.raise_for_status()

    size_header = response.headers.get("content-length") or response.headers.get("x-linked-size")
    if not size_header:
        raise DataSourceError(
            f"Источник не сообщил размер файла ({url}). "
            "Без размера невозможно проверить целостность скачивания."
        )

    info = RemoteInfo(size=int(size_header), etag=response.headers.get("etag"))
    logger.debug(
        "Метаданные источника получены",
        extra={
            "size_bytes": info.size,
            "size_gb": round(info.size / 1024**3, 2),
            "etag": info.etag,
            "accept_ranges": response.headers.get("accept-ranges"),
        },
    )
    return info


def _write_meta(target: Path, info: RemoteInfo) -> None:
    """Сохранить версию дампа рядом с файлом."""
    meta_path = target.with_name(target.name + _META_SUFFIX)
    meta_path.write_text(
        json.dumps(
            {
                "size": info.size,
                "etag": info.etag,
                "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.debug("Версия дампа записана", extra={"meta_path": str(meta_path)})


def read_dump_version(settings: IngestSettings | None = None) -> str | None:
    """Прочитать версию скачанного дампа. Нужна слою заливки."""
    settings = settings or get_settings().ingest
    meta_path = Path(settings.dump_path).with_name(Path(settings.dump_path).name + _META_SUFFIX)
    if not meta_path.exists():
        return None
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    return f"size={data.get('size')};etag={data.get('etag') or '—'}"


def _log_progress(
    *,
    downloaded_total: int,
    downloaded_now: int,
    total: int,
    started: float,
    last_logged: int,
) -> int:
    """Сообщать о прогрессе не чаще, чем раз в 5%.

    Без этого 7,7 ГБ выглядят как зависание, а с логом на каждый чанк вывод
    превращается в кашу.

    Скорость считается по байтам, скачанным **в текущей попытке**
    (`downloaded_now`): при докачке деление общего объёма на время попытки
    завысило бы скорость и занизило остаток.
    """
    step = max(total // 20, 1)
    if downloaded_total - last_logged < step and downloaded_total < total:
        return last_logged

    elapsed = max(time.perf_counter() - started, 1e-6)
    speed = downloaded_now / elapsed
    remaining = (total - downloaded_total) / speed if speed > 0 else 0.0
    logger.info(
        "Скачивание дампа",
        extra={
            "progress": f"{downloaded_total / total:.0%}",
            "downloaded_gb": round(downloaded_total / 1024**3, 2),
            "total_gb": round(total / 1024**3, 2),
            "speed_mb_s": round(speed / 1024**2, 1),
            "eta_min": round(remaining / 60, 1),
        },
    )
    return downloaded_total


def download_dump(
    settings: IngestSettings | None = None,
    *,
    force: bool = False,
    client: httpx.Client | None = None,
) -> DownloadResult:
    """Скачать дамп идемпотентно.

    Args:
        settings: настройки ingestion; по умолчанию из `get_settings()`.
        force: перекачать заново, игнорируя существующий файл.
        client: HTTP-клиент; подменяется в тестах, чтобы не ходить в сеть.
    """
    settings = settings or get_settings().ingest
    target = Path(settings.dump_path)
    part = target.with_name(target.name + _PART_SUFFIX)
    target.parent.mkdir(parents=True, exist_ok=True)

    owns_client = client is None
    client = client or httpx.Client(timeout=settings.download_timeout_s, follow_redirects=True)
    started = time.perf_counter()

    try:
        info = _head_remote(client, settings.dump_url)

        # Идемпотентность: готовый файл нужного размера не перекачиваем.
        if target.exists() and not force:
            actual = target.stat().st_size
            if actual == info.size:
                logger.info(
                    "Дамп уже на месте, скачивание пропущено",
                    extra={"path": str(target), "size_gb": round(actual / 1024**3, 2)},
                )
                _write_meta(target, info)
                return DownloadResult(
                    path=target,
                    size=actual,
                    skipped=True,
                    elapsed_s=round(time.perf_counter() - started, 1),
                    etag=info.etag,
                )
            logger.warning(
                "Размер существующего файла не совпадает с источником — перекачиваем",
                extra={"local_size": actual, "remote_size": info.size},
            )
            target.unlink()

        if force and part.exists():
            part.unlink()

        _download_with_resume(client, settings, part, info)

        actual = part.stat().st_size
        if actual != info.size:
            part.unlink(missing_ok=True)
            raise DataSourceError(
                f"Размер скачанного файла не совпадает: получено {actual}, "
                f"ожидалось {info.size}. Частичный файл удалён, повторите загрузку."
            )

        # Переименование только после сверки: до этого момента готового файла нет.
        part.replace(target)
        _write_meta(target, info)

        elapsed = round(time.perf_counter() - started, 1)
        logger.info(
            "Дамп скачан",
            extra={
                "path": str(target),
                "size_gb": round(actual / 1024**3, 2),
                "elapsed_min": round(elapsed / 60, 1),
                "avg_speed_mb_s": round(actual / max(elapsed, 1e-6) / 1024**2, 1),
            },
        )
        return DownloadResult(
            path=target, size=actual, skipped=False, elapsed_s=elapsed, etag=info.etag
        )
    finally:
        if owns_client:
            client.close()


def _download_with_resume(
    client: httpx.Client,
    settings: IngestSettings,
    part: Path,
    info: RemoteInfo,
) -> None:
    """Качать в `.part`, продолжая с текущего смещения, с ретраями."""
    attempt = 0
    while True:
        offset = part.stat().st_size if part.exists() else 0
        if offset >= info.size:
            return

        if offset:
            logger.info(
                "Продолжаем докачку",
                extra={
                    "offset_gb": round(offset / 1024**3, 2),
                    "remaining_gb": round((info.size - offset) / 1024**3, 2),
                },
            )

        try:
            _stream_range(client, settings, part, info, offset)
            return
        except (httpx.HTTPError, OSError) as exc:
            attempt += 1
            if attempt > settings.download_max_retries:
                raise DataSourceError(
                    f"Скачивание не удалось после {settings.download_max_retries} "
                    f"попыток на смещении {offset}: {type(exc).__name__}: {exc}"
                ) from exc
            delay = min(2**attempt, 60)
            logger.warning(
                "Обрыв скачивания, повтор",
                extra={
                    "attempt": attempt,
                    "max_retries": settings.download_max_retries,
                    "offset": offset,
                    "delay_s": delay,
                    "error": type(exc).__name__,
                },
            )
            time.sleep(delay)


def _stream_range(
    client: httpx.Client,
    settings: IngestSettings,
    part: Path,
    info: RemoteInfo,
    offset: int,
) -> None:
    """Один проход скачивания начиная с `offset`.

    Запрос идёт на исходный URL, а не на сохранённый подписанный: HuggingFace
    подписывает CDN-ссылку с истечением, и переполучение редиректа — обязательное
    условие устойчивой докачки большого файла.
    """
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    started = time.perf_counter()
    last_logged = offset

    with client.stream("GET", settings.dump_url, headers=headers) as response:
        response.raise_for_status()

        if offset and response.status_code != httpx.codes.PARTIAL_CONTENT:
            # Сервер проигнорировал Range и отдал файл целиком: дописывать
            # в конец нельзя, иначе получится склейка. Начинаем заново.
            logger.warning(
                "Источник не поддержал Range на этой попытке — начинаем файл заново",
                extra={"status": response.status_code, "offset": offset},
            )
            part.unlink(missing_ok=True)
            offset = 0
            last_logged = 0

        mode = "ab" if offset else "wb"
        downloaded_total = offset
        downloaded_now = 0
        with part.open(mode) as handle:
            for chunk in response.iter_bytes(settings.download_chunk_size):
                handle.write(chunk)
                downloaded_total += len(chunk)
                downloaded_now += len(chunk)
                last_logged = _log_progress(
                    downloaded_total=downloaded_total,
                    downloaded_now=downloaded_now,
                    total=info.size,
                    started=started,
                    last_logged=last_logged,
                )
