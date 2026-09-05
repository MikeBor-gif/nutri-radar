"""Сборка текста профиля продукта — того, что уходит в эмбеддинг.

Профиль определяет, что вообще может найтись. Ошибка здесь не роняет прогон
и не видна в метриках как ошибка: она просто делает поиск хуже, и списать
это будет не на что.

**Что входит: название, бренд, категории, состав.**

**Чего нет и почему:**

* *Нутриенты.* По ним ищут числом («сахара меньше 5 г»), а не смыслом.
  Строка «sugars 42.3» в тексте профиля не сделает продукт ближе к запросу
  про сахар — она добавит шум из цифр, одинаковый у половины корпуса.
  Фильтровать по нутриентам — работа SQL, а не эмбеддинга.
* *Оценка Nutri-Score и группа NOVA.* Попади они в текст, поиск начал бы
  возвращать продукты по совпадению оценки, а не состава: у всех продуктов
  с «e» появилась бы общая подстрока. Это ровно тот вид утечки, который
  в M4 отсекался структурно.
* *Штрихкод.* Цифры без смысла, съедающие место в контексте.

**Категории чистятся от таксономии.** В базе они лежат тегами вида
`en:sweet-snacks` — это машинные идентификаторы, и эмбеддер разберёт их как
странные строки с двоеточиями. `en:` снимается, дефисы становятся пробелами:
`sweet snacks`. Так тег превращается в текст, который модель понимает.

**Бренд берётся один.** В базе он часто перечислен вариантами:
`Lay's,Lay's Chips,Lay's Chips Cream&Dill,Lay's Chips Cream&Dill 215g`.
Четыре повтора одного слова смещают вектор в сторону бренда и от состава.

**Версия и хеш.** Правило сборки будет меняться, и вектор, посчитанный
по старому правилу, внешне неотличим от свежего. Версия читается человеком,
хеш — кодом: по нему прогон понимает, что перевекторизовать, не перечитывая
тексты.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass

from nutri_radar.logging import safe_extra

logger = logging.getLogger(__name__)

# Версия правила сборки. Меняется при любой правке того, что попадает
# в текст: без этого старые и новые векторы окажутся в одной таблице
# неразличимыми, а метрики — посчитанными на смеси.
PROFILE_VERSION = "v1"

# Длина хеша. Полный sha256 не нужен: он лежит в колонке рядом с каждым
# вектором, а 32 символа дают запас против совпадений на 146 тысячах строк
# с колоссальным перекрытием.
_HASH_LEN = 32

# Сколько категорий берём. Таксономия OFF иерархическая, и у продукта
# их бывает под десяток от `plant-based-foods` до `salted-crisps`.
# Первые — самые общие и потому наименее полезные; последние конкретнее.
_MAX_CATEGORIES = 4

# Потолок длины состава в символах. У bge-m3 контекст 8192 токена, и обычный
# состав в него влезает с запасом. Потолок защищает не от нормы, а от записи,
# в которую краудсорсинг сложил всю этикетку целиком вместе с адресом
# производителя: такой текст размывает вектор и замедляет прогон.
_MAX_INGREDIENTS_CHARS = 2000

_TAXONOMY_PREFIX = re.compile(r"^[a-z]{2}:")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Profile:
    """Текст профиля вместе с тем, чем он получен."""

    code: str
    text: str
    version: str = PROFILE_VERSION

    @property
    def hash(self) -> str:
        """Отпечаток текста. По нему прогон различает свежее и устаревшее."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:_HASH_LEN]

    @property
    def is_usable(self) -> bool:
        """Годится ли профиль для векторизации.

        Пустой текст в косинусном расстоянии даёт вектор, равноудалённый
        от всего, и такой продукт всплывал бы в выдаче на любой запрос.
        """
        return bool(self.text.strip())


def clean_category(tag: str) -> str:
    """Превратить тег таксономии в читаемый текст.

    `en:sweet-snacks` → `sweet snacks`. Эмбеддер работает с текстом,
    а двоеточие и дефисы он разберёт как пунктуацию неизвестного назначения.
    """
    without_prefix = _TAXONOMY_PREFIX.sub("", str(tag or "").strip().lower())
    return without_prefix.replace("-", " ").strip()


def first_brand(brands: str | None) -> str:
    """Первый бренд из перечисления.

    В базе бренд часто записан вариантами одного и того же названия.
    Повторы смещают вектор в сторону бренда и от состава — ради которого
    поиск и делается.
    """
    if not brands:
        return ""
    return str(brands).split(",")[0].strip()


def build_profile(
    code: str,
    *,
    product_name: str | None,
    brands: str | None,
    categories_tags: list[str] | None,
    ingredients_text: str | None,
) -> Profile:
    """Собрать текст профиля одного продукта.

    Порядок частей фиксирован: название, бренд, категории, состав. Он влияет
    на вектор, поэтому меняться без смены `PROFILE_VERSION` не должен.

    Пустые части выбрасываются, а не подставляются заглушками: строка
    «бренд: неизвестен» у трети корпуса создала бы общую подстроку там,
    где общего нет.
    """
    parts: list[str] = []

    name = _WHITESPACE.sub(" ", str(product_name or "").strip())
    if name:
        parts.append(name)

    brand = first_brand(brands)
    if brand and brand.lower() != name.lower():
        parts.append(brand)

    categories = [clean_category(tag) for tag in (categories_tags or [])]
    categories = [category for category in categories if category][-_MAX_CATEGORIES:]
    if categories:
        parts.append(", ".join(categories))

    ingredients = _WHITESPACE.sub(" ", str(ingredients_text or "").strip())
    if len(ingredients) > _MAX_INGREDIENTS_CHARS:
        logger.debug(
            "Состав обрезан по потолку длины",
            extra=safe_extra(code=code, length=len(ingredients), limit=_MAX_INGREDIENTS_CHARS),
        )
        ingredients = ingredients[:_MAX_INGREDIENTS_CHARS]
    if ingredients:
        parts.append(ingredients)

    return Profile(code=code, text="\n".join(parts))
