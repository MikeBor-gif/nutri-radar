"""Тесты сборки профиля.

Профиль определяет, что вообще может найтись. Ошибка здесь не роняет прогон
и не проявляется в метриках как ошибка — она просто делает поиск хуже,
и списать это будет не на что. Поэтому проверяется не «функция работает»,
а конкретные решения, каждое из которых можно оспорить.

**Детерминированность и хеш.** Вектор считается один раз и живёт в базе
месяцами. Если сборка профиля недетерминирована, половина корпуса окажется
векторизована по одному тексту, половина по другому, и заметить это будет
нечем. Хеш — единственный способ отличить свежий вектор от устаревшего,
и он обязан меняться при любой правке текста.

**Отсутствие утечки.** Оценка и нутриенты в текст не попадают. Попади они —
поиск начал бы возвращать продукты по совпадению оценки, а не состава.

БД и сети здесь нет.
"""

from __future__ import annotations

import pytest

from nutri_radar.retrieval.profile import (
    PROFILE_VERSION,
    Profile,
    build_profile,
    clean_category,
    first_brand,
    strip_markup,
)


def _profile(**overrides: object) -> Profile:
    defaults: dict[str, object] = {
        "product_name": "Молочный шоколад смородина",
        "brands": "Alpen Gold",
        "categories_tags": ["en:snacks", "en:sweet-snacks", "en:chocolates"],
        "ingredients_text": "Сахар, патока крахмальная, какао тёртое, масло какао",
    }
    defaults.update(overrides)
    return build_profile("4600000000001", **defaults)  # type: ignore[arg-type]


class TestКатегории:
    def test_тег_таксономии_становится_текстом(self):
        """Эмбеддер работает с текстом; `en:` и дефисы он разберёт
        как пунктуацию неизвестного назначения."""
        assert clean_category("en:sweet-snacks") == "sweet snacks"

    def test_префикс_любого_языка_снимается(self):
        assert clean_category("fr:produits-laitiers") == "produits laitiers"

    def test_пустой_тег_не_роняет_сборку(self):
        assert clean_category("") == ""

    def test_берутся_последние_категории_а_не_первые(self):
        """Таксономия иерархическая: первые теги — самые общие
        (`plant-based-foods`), последние конкретнее (`crisps`)."""
        profile = _profile(
            categories_tags=[
                "en:plant-based-foods-and-beverages",
                "en:plant-based-foods",
                "en:snacks",
                "en:salty-snacks",
                "en:crisps",
            ]
        )

        assert "crisps" in profile.text
        assert "plant based foods and beverages" not in profile.text


class TestРазметка:
    def test_теги_снимаются_содержимое_остаётся(self):
        """Аллерген как слово в составе полезен, разметка вокруг него нет."""
        cleaned = strip_markup('Farine de <span class="allergen">blé</span> 27%')

        assert cleaned == "Farine de blé 27%"

    def test_битый_html_не_ломает_очистку(self):
        """Поля краудсорсинговые, рассчитывать на валидный HTML нельзя."""
        assert strip_markup("битый </span> тег и <b>жирный") == "битый тег и жирный"

    def test_разметка_не_попадает_в_профиль(self):
        """52 213 продуктов из 146 350 приходят с этими тегами: общая
        подстрока у трети базы сближала бы векторы без всякого сходства."""
        profile = _profile(ingredients_text='Сахар, <span class="allergen">молоко</span>, соль')

        assert "<span" not in profile.text
        assert "allergen" not in profile.text
        assert "молоко" in profile.text

    def test_разметка_в_названии_тоже_снимается(self):
        assert "<b>" not in _profile(product_name="<b>Шоколад</b>").text

    def test_текст_без_разметки_не_меняется(self):
        assert strip_markup("сахар, вода") == "сахар, вода"


class TestБренд:
    def test_берётся_первый_из_перечисления(self):
        """Повторы одного названия смещают вектор к бренду и от состава."""
        assert first_brand("Lay's,Lay's Chips,Lay's Chips Cream&Dill") == "Lay's"

    def test_отсутствие_бренда_не_даёт_заглушки(self):
        """Строка «бренд неизвестен» у трети корпуса создала бы общую
        подстроку там, где общего нет."""
        profile = _profile(brands=None)

        assert "неизвест" not in profile.text.lower()

    def test_бренд_совпавший_с_названием_не_дублируется(self):
        profile = _profile(product_name="Alpen Gold", brands="Alpen Gold")

        assert profile.text.lower().count("alpen gold") == 1


class TestСоставПрофиля:
    def test_все_четыре_части_на_месте(self):
        profile = _profile()

        assert "Молочный шоколад" in profile.text
        assert "Alpen Gold" in profile.text
        assert "chocolates" in profile.text
        assert "патока крахмальная" in profile.text

    def test_нутриентов_и_оценки_в_профиле_нет(self):
        """Попади они в текст, поиск возвращал бы продукты по совпадению
        оценки, а не состава — та же утечка, что отсекалась в M4."""
        profile = _profile()

        for leak in ("nutriscore", "sugars_100g", "nova", "энергетическая"):
            assert leak not in profile.text.lower()

    def test_штрихкод_в_текст_не_попадает(self):
        """Цифры без смысла, съедающие место в контексте."""
        assert "4600000000001" not in _profile().text

    def test_пустые_части_выбрасываются(self):
        profile = _profile(brands=None, categories_tags=None)

        assert profile.text.count("\n") == 1

    def test_длинный_состав_обрезается(self):
        """Защита не от нормы, а от записи, куда сложили всю этикетку
        вместе с адресом производителя."""
        profile = _profile(ingredients_text="сахар, " * 1000)

        assert len(profile.text) < 3000

    def test_переводы_строк_внутри_состава_схлопываются(self):
        """В базе состав часто лежит с переносами: они дали бы лишние
        границы там, где их нет."""
        profile = _profile(ingredients_text="сахар,\n\n  вода,\tсоль")

        assert "сахар, вода, соль" in profile.text


class TestХешИВерсия:
    def test_один_и_тот_же_вход_даёт_один_хеш(self):
        assert _profile().hash == _profile().hash

    def test_правка_текста_меняет_хеш(self):
        """Единственный способ отличить свежий вектор от устаревшего."""
        assert _profile().hash != _profile(ingredients_text="вода").hash

    def test_порядок_частей_влияет_на_хеш(self):
        """Он влияет и на вектор, поэтому меняться без смены версии
        не должен — тест это фиксирует."""
        first = _profile(product_name="А", ingredients_text="Б")
        second = _profile(product_name="Б", ingredients_text="А")

        assert first.hash != second.hash

    def test_версия_проставляется(self):
        assert _profile().version == PROFILE_VERSION

    def test_длина_хеша_фиксирована(self):
        """Колонка в БД — varchar(32); длиннее не влезет."""
        assert len(_profile().hash) == 32


class TestПригодность:
    def test_пустой_профиль_негоден(self):
        """Пустой текст даёт вектор, равноудалённый от всего: такой продукт
        всплывал бы в выдаче на любой запрос."""
        profile = build_profile(
            "1", product_name=None, brands=None, categories_tags=None, ingredients_text=None
        )

        assert profile.is_usable is False

    def test_профиль_из_одного_состава_годен(self):
        """Название есть не у всех: 146 350 продуктов с составом против
        145 075 с названием."""
        profile = build_profile(
            "1",
            product_name=None,
            brands=None,
            categories_tags=None,
            ingredients_text="сахар, вода",
        )

        assert profile.is_usable is True

    @pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
    def test_пробельный_состав_негоден(self, blank: str):
        profile = build_profile(
            "1",
            product_name=None,
            brands=None,
            categories_tags=None,
            ingredients_text=blank,
        )

        assert profile.is_usable is False
