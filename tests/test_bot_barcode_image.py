"""Тесты чтения штрихкода с фотографии.

Картинка генерируется тут же тем же `zxing-cpp`, а не лежит файлом
в `tests/data/`: так тест проверяет полный оборот «код → изображение →
код», а не совпадение с однажды записанным артефактом. Файл в репозитории
вдобавок пришлось бы объяснять — откуда он и почему ему можно верить.

Сети здесь нет по построению: декодер работает с байтами в памяти.
"""

from __future__ import annotations

import io

import pytest
import zxingcpp
from PIL import Image

from nutri_radar.bot.barcode_image import decode_barcode

CODE = "4016463697295"


def _barcode_png(code: str = CODE, *, scale: int = 4) -> bytes:
    """Сгенерировать PNG со штрихкодом EAN-13."""
    barcode = zxingcpp.create_barcode(code, zxingcpp.BarcodeFormat.EAN13)
    image = zxingcpp.write_barcode_to_image(barcode, scale=scale)
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def _plain_png(color: str = "white", size: tuple[int, int] = (200, 200)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


class TestРаспознавание:
    def test_код_читается_с_картинки(self) -> None:
        assert decode_barcode(_barcode_png()) == CODE

    def test_jpeg_тоже_читается(self) -> None:
        """Telegram присылает фотографии в JPEG, а не в PNG."""
        barcode = zxingcpp.create_barcode(CODE, zxingcpp.BarcodeFormat.EAN13)
        image = zxingcpp.write_barcode_to_image(barcode, scale=4)
        buffer = io.BytesIO()
        Image.fromarray(image).convert("RGB").save(buffer, format="JPEG", quality=92)

        assert decode_barcode(buffer.getvalue()) == CODE

    def test_картинка_с_альфа_каналом_не_ломает_декодер(self) -> None:
        """Приведение к RGB обязательно: снимки приходят разными."""
        barcode = zxingcpp.create_barcode(CODE, zxingcpp.BarcodeFormat.EAN13)
        image = zxingcpp.write_barcode_to_image(barcode, scale=4)
        buffer = io.BytesIO()
        Image.fromarray(image).convert("RGBA").save(buffer, format="PNG")

        assert decode_barcode(buffer.getvalue()) == CODE


class TestОтказы:
    def test_пустая_картинка_даёт_none(self) -> None:
        assert decode_barcode(_plain_png()) is None

    def test_не_изображение_не_роняет_бота(self) -> None:
        """Это ввод пользователя, а не поломка сервиса."""
        assert decode_barcode("это точно не картинка".encode()) is None

    def test_пустые_байты_дают_none(self) -> None:
        assert decode_barcode(b"") is None


class TestГраницыФорматов:
    def test_qr_код_не_принимается_за_штрихкод(self) -> None:
        """На упаковке QR — это ссылка на промоакцию, а не идентификатор.

        Принимать его за штрихкод значило бы искать в базе строку вида
        `https://…` и показывать пользователю «не найдено» вместо просьбы
        сфотографировать сам штрихкод.
        """
        barcode = zxingcpp.create_barcode(
            "https://example.invalid/promo", zxingcpp.BarcodeFormat.QRCode
        )
        image = zxingcpp.write_barcode_to_image(barcode, scale=4)
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="PNG")

        assert decode_barcode(buffer.getvalue()) is None

    @pytest.mark.parametrize("code", ["96385074", "4016463697295"])
    def test_разные_длины_кода_читаются(self, code: str) -> None:
        """EAN-8 и EAN-13 — оба живут на упаковке еды."""
        fmt = zxingcpp.BarcodeFormat.EAN8 if len(code) == 8 else zxingcpp.BarcodeFormat.EAN13
        barcode = zxingcpp.create_barcode(code, fmt)
        image = zxingcpp.write_barcode_to_image(barcode, scale=4)
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="PNG")

        assert decode_barcode(buffer.getvalue()) == code
