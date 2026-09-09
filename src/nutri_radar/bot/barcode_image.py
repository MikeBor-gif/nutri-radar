"""Чтение штрихкода с фотографии.

**Почему `zxing-cpp`, а не `pyzbar`.** Развилка закрыта 2026-09-06 (ADR-009):
`pyzbar` тянет нативную библиотеку `zbar`, которую на Windows нужно ставить
отдельно системными средствами. `zxing-cpp` везёт свой код в колесе, и
`uv sync` ставит его без внешних шагов. Для проекта, который должен
подниматься на чужой машине одной командой, это решающее различие.

**Файл живёт только в памяти.** На диск не попадает ни во временный каталог,
ни в кэш: пользовательские данные не хранятся дольше, чем нужно для ответа
(бриф). Фотография этикетки — это в том числе место и время покупки,
и обращаться с ней надо соответственно.

**Чистая функция.** Декодирование отделено от Telegram намеренно: так его
можно проверить тестом на сгенерированной картинке, не поднимая бота
и не ходя в сеть.
"""

from __future__ import annotations

import io
import logging
import time

import zxingcpp
from PIL import Image, UnidentifiedImageError

from nutri_radar.logging import safe_extra
from nutri_radar.openfoodfacts import normalize_barcode

logger = logging.getLogger(__name__)

# Линейные форматы, которыми маркируют еду: EAN-13 и EAN-8 в Европе,
# UPC-A и UPC-E в США. QR-коды и матричные форматы не ищем — на упаковке
# они означают ссылку на промоакцию, а не идентификатор продукта, и их
# распознавание только добавило бы ложных срабатываний.
#
# Перечень кортежем, а не через `|`: в zxing-cpp 3.1 объединение форматов
# оператором объявлено устаревшим, а `filterwarnings = ["error"]` в конфиге
# pytest превращает предупреждение в упавший тест. Это ровно то, ради чего
# такая настройка и стоит.
FORMATS = (
    zxingcpp.BarcodeFormat.EAN13,
    zxingcpp.BarcodeFormat.EAN8,
    zxingcpp.BarcodeFormat.UPCA,
    zxingcpp.BarcodeFormat.UPCE,
)


def decode_barcode(data: bytes) -> str | None:
    """Прочитать штрихкод с изображения. Не прочитался — `None`.

    Args:
        data: содержимое файла изображения (JPEG или PNG от Telegram).

    Returns:
        Штрихкод цифрами или `None`, если код не найден либо найденное
        не похоже на штрихкод.
    """
    started = time.perf_counter()
    try:
        with Image.open(io.BytesIO(data)) as image:
            # Приведение к RGB обязательно: Telegram присылает и PNG
            # с альфа-каналом, и чёрно-белые снимки, а декодер ждёт
            # предсказуемое число каналов.
            results = zxingcpp.read_barcodes(image.convert("RGB"), formats=FORMATS)
    except (UnidentifiedImageError, OSError) as exc:
        # Файл не является изображением или повреждён. Это ввод
        # пользователя, а не поломка сервиса.
        logger.info(
            "Файл не удалось открыть как изображение",
            extra=safe_extra(bytes=len(data), error=type(exc).__name__),
        )
        return None

    elapsed = time.perf_counter() - started
    for result in results:
        code = normalize_barcode(result.text)
        if code is not None:
            logger.info(
                "Штрихкод прочитан с фото",
                extra=safe_extra(
                    bytes=len(data),
                    latency_s=round(elapsed, 3),
                    barcode_format=str(result.format),
                ),
            )
            return code

    logger.info(
        "Штрихкод на фото не найден",
        extra=safe_extra(
            bytes=len(data),
            latency_s=round(elapsed, 3),
            candidates=len(results),
        ),
    )
    return None
