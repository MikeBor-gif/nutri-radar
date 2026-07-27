"""Тесты предобработки состава.

Главный случай взят из **реальных данных корпуса M1**, а не придуман:
в `products` встречается HTML прямо внутри состава, и без очистки модель
возвращает аллерген вместе с тегом — это было проверено на живой модели.
"""

from __future__ import annotations

from nutri_radar.extract.preprocess import (
    PreprocessStats,
    max_text_length,
    prepare_text,
)

# Дословный фрагмент из корпуса M1 (см. план M2, раздел «Контекст исследования»).
REAL_MARKUP_SAMPLE = (
    'Hergestellt aus pasteurisierter <span class="allergen">Milch</span> in Bayern.'
)

NUM_CTX = 4096


class TestСнятиеРазметки:
    def test_реальный_пример_из_корпуса_очищается(self):
        prepared = prepare_text(REAL_MARKUP_SAMPLE, num_ctx=NUM_CTX)

        assert "<span" not in prepared.cleaned
        assert "</span>" not in prepared.cleaned
        assert "class=" not in prepared.cleaned
        # Само слово остаётся — вычищается разметка, а не содержимое.
        assert "Milch" in prepared.cleaned
        assert prepared.had_markup

    def test_html_сущности_раскрываются(self):
        prepared = prepare_text("Sugar &amp; Salt, Cocoa &gt; 30%", num_ctx=NUM_CTX)

        assert "&amp;" not in prepared.cleaned
        assert "Sugar & Salt" in prepared.cleaned
        assert prepared.had_markup

    def test_пробелы_схлопываются(self):
        prepared = prepare_text("Sugar,\n\n  Salt,\tWater", num_ctx=NUM_CTX)

        assert prepared.cleaned == "Sugar, Salt, Water"

    def test_краевой_мусор_снимается(self):
        """`_` и `*` в OFF помечают органику и сноски."""
        prepared = prepare_text("  *Sugar, Salt_  ", num_ctx=NUM_CTX)

        assert prepared.cleaned == "Sugar, Salt"

    def test_чистый_текст_не_помечается_разметкой(self):
        prepared = prepare_text("Sugar, Salt, Water", num_ctx=NUM_CTX)

        assert not prepared.had_markup
        assert prepared.cleaned == "Sugar, Salt, Water"


class TestСкобкиНеТрогаем:
    def test_вложенность_сохраняется(self):
        """Скобки несут смысл: это состав составного ингредиента."""
        text = "Flour (Wheat Flour, Calcium Carbonate), Sugar"
        prepared = prepare_text(text, num_ctx=NUM_CTX)

        assert prepared.cleaned == text


class TestДлинаИКонтекст:
    def test_слишком_длинный_состав_не_годится_для_отправки(self):
        """Ollama режет вход молча — такой текст лучше не отправлять вовсе."""
        limit = max_text_length(NUM_CTX)
        prepared = prepare_text("a, " * limit, num_ctx=NUM_CTX)

        assert prepared.too_long
        assert not prepared.is_usable

    def test_предел_зависит_от_размера_контекста(self):
        """Предел считается от num_ctx, а не задан константой."""
        assert max_text_length(8192) == 2 * max_text_length(4096)

    def test_пустой_текст_не_годится(self):
        for value in (None, "", "   ", "***"):
            assert not prepare_text(value, num_ctx=NUM_CTX).is_usable

    def test_обычный_состав_годится(self):
        assert prepare_text(REAL_MARKUP_SAMPLE, num_ctx=NUM_CTX).is_usable


class TestСтатистика:
    def test_считает_разметку_длину_и_пустые(self):
        stats = PreprocessStats()
        stats.add(prepare_text(REAL_MARKUP_SAMPLE, num_ctx=NUM_CTX))
        stats.add(prepare_text("Sugar, Salt", num_ctx=NUM_CTX))
        stats.add(prepare_text("   ", num_ctx=NUM_CTX))
        stats.add(prepare_text("a, " * max_text_length(NUM_CTX), num_ctx=NUM_CTX))

        assert stats.total == 4
        assert stats.with_markup == 1
        assert stats.empty == 1
        assert stats.too_long == 1
