"""Канонизация имён ингредиентов через словарь алиасов.

Зачем это нужно, если модель уже возвращает `canonical_name`: она возвращает
его **как получилось**. На живой проверке одна и та же сущность приходила как
`glucose-fructose syrup`, `glucose fructose syrup` и `glucose syrup`, а на
русском составе — как `сахар`. Без сведения к одному имени ключевая величина
проекта — число РАЗНЫХ форм сахара — считается неверно: три написания одного
сиропа дают три формы вместо одной.

Два числа, которые отсюда берутся:

* **число разных форм сахара** после канонизации — то, ради чего проект;
* **доля неизвестных имён** — наш прямой аналог `unknown_ingredients_n` у
  Open Food Facts. Она честно показывает предел словаря и говорит, куда его
  пополнять: `top_unknown()` возвращает самые частые незакрытые имена.

**Словарь наполняет человек, а не модель** (правило 6 брифа). Здесь только
механика: загрузка, сопоставление и подсчёт. Ни одной строки словаря код
не придумывает — незакрытое имя остаётся неизвестным и попадает в счётчик.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from nutri_radar.config import Settings, get_settings
from nutri_radar.db.repositories.ingredient import AliasRow, IngredientAliasRepository
from nutri_radar.db.session import get_session
from nutri_radar.errors import DataSourceError
from nutri_radar.extract.schemas import Ingredient, IngredientKind
from nutri_radar.logging import safe_extra

logger = logging.getLogger(__name__)

# Каталог словарей. Файлы ведёт человек — см. data/dictionaries/README.md.
DICTIONARIES_DIR = Path("data") / "dictionaries"
SUGAR_SEED_FILE = "sugar_forms.jsonl"

# Язык-заглушка для алиасов, одинаковых во всех языках (E-номера, латинские
# написания). Отдельное значение, а не пустая строка: в БД колонка NOT NULL.
ANY_LANG = "xx"

_PUNCTUATION_RE = re.compile(r"[^\w\s-]", flags=re.UNICODE)
_WHITESPACE_RE = re.compile(r"[\s_-]+")


def normalize_key(name: str) -> str:
    """Привести имя к виду, по которому идёт поиск в словаре.

    Схлопывает то, что различается только оформлением: регистр, дефисы,
    двойные пробелы, скобочные хвосты. Диакритика снимается через NFKD —
    иначе немецкий `Süßmolkenpulver` и `Sussmolkenpulver` окажутся разными
    ключами, хотя в базе встречаются оба написания.
    """
    lowered = str(name or "").strip().lower()
    # NFKD раскладывает букву и диакритику, затем комбинирующие знаки
    # выбрасываются. Для кириллицы это безопасно: там их нет.
    decomposed = unicodedata.normalize("NFKD", lowered)
    without_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    cleaned = _PUNCTUATION_RE.sub(" ", without_marks)
    return _WHITESPACE_RE.sub(" ", cleaned).strip()


@dataclass(frozen=True)
class AliasEntry:
    """Одна запись словаря: алиас на языке ведёт к каноническому имени."""

    alias: str
    lang: str
    canonical_name: str
    kind: IngredientKind | None = None


class AliasIndex:
    """Индекс алиасов для быстрого сопоставления.

    Поиск идёт в два шага: сначала по паре (язык, алиас), потом по алиасу
    без учёта языка. Второй шаг обязателен — модель по требованию промпта
    отвечает английскими именами даже на немецкий состав, и привязка алиаса
    к языку исходного текста промахивалась бы на каждом продукте.
    """

    def __init__(self, entries: Iterable[AliasEntry] = ()) -> None:
        self._by_lang: dict[tuple[str, str], AliasEntry] = {}
        self._by_alias: dict[str, AliasEntry] = {}
        self._conflicts: list[tuple[str, str, str]] = []
        for entry in entries:
            self.add(entry)

    def add(self, entry: AliasEntry) -> None:
        key = normalize_key(entry.alias)
        if not key:
            return

        self._by_lang[(entry.lang, key)] = entry
        existing = self._by_alias.get(key)
        if existing is not None and existing.canonical_name != entry.canonical_name:
            # Один алиас ведёт к двум каноническим именам. Это ошибка словаря,
            # а не данных: разрешить её может только человек, поэтому здесь
            # конфликт фиксируется, а не «решается» выбором наугад.
            self._conflicts.append((key, existing.canonical_name, entry.canonical_name))
        else:
            self._by_alias[key] = entry

    def lookup(self, name: str, *, lang: str | None = None) -> AliasEntry | None:
        key = normalize_key(name)
        if not key:
            return None
        if lang:
            entry = self._by_lang.get((lang, key))
            if entry is not None:
                return entry
        return self._by_alias.get(key)

    @property
    def size(self) -> int:
        return len(self._by_lang)

    @property
    def conflicts(self) -> list[tuple[str, str, str]]:
        return list(self._conflicts)

    def log_summary(self) -> None:
        logger.info(
            "Словарь алиасов загружен",
            extra=safe_extra(
                aliases=self.size,
                canonical_names=len({entry.canonical_name for entry in self._by_alias.values()}),
                conflicts=len(self._conflicts),
            ),
        )
        if self._conflicts:
            logger.warning(
                "В словаре есть алиасы с двумя каноническими именами — нужна правка руками",
                extra=safe_extra(conflicts=self._conflicts[:10]),
            )


def iter_seed_entries(path: Path) -> Iterator[AliasEntry]:
    """Прочитать seed-файл словаря.

    Формат — JSONL, по одной канонической сущности на строку:

        {"canonical_name": "sugar", "kind": "sugar",
         "aliases": {"en": ["sugar"], "ru": ["сахар"]}}

    JSONL, а не YAML: YAML потребовал бы зависимости, а правило 7 брифа
    запрещает добавлять их без согласования. Построчный формат к тому же
    даёт читаемый diff, когда человек дописывает алиасы.

    Raises:
        DataSourceError: файла нет или строка не разбирается. Молча
            продолжать нельзя: пустой словарь даст «все имена неизвестны»,
            и это выглядело бы как плохое качество извлечения.
    """
    if not path.exists():
        raise DataSourceError(
            f"Seed словаря не найден: {path}. Файл ведётся руками — см. "
            f"{DICTIONARIES_DIR / 'README.md'}"
        )

    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        # Пустые строки и комментарии: файл читают и правят люди.
        if not stripped or stripped.startswith("//"):
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise DataSourceError(
                f"{path}, строка {number}: не разбирается как JSON ({exc})"
            ) from exc

        canonical = str(record.get("canonical_name", "")).strip().lower()
        if not canonical:
            raise DataSourceError(f"{path}, строка {number}: пустое canonical_name")

        raw_kind = record.get("kind")
        kind = IngredientKind(raw_kind) if raw_kind else None

        aliases: dict[str, list[str]] = record.get("aliases") or {}
        # Каноническое имя — тоже алиас самого себя, иначе оно не найдётся.
        yield AliasEntry(alias=canonical, lang=ANY_LANG, canonical_name=canonical, kind=kind)
        for lang, values in aliases.items():
            for alias in values:
                yield AliasEntry(
                    alias=str(alias), lang=str(lang), canonical_name=canonical, kind=kind
                )


def load_seed_index(
    directory: Path | None = None,
    *,
    files: Iterable[str] = (SUGAR_SEED_FILE,),
) -> AliasIndex:
    """Собрать индекс из seed-файлов каталога словарей."""
    base = directory or DICTIONARIES_DIR
    index = AliasIndex()
    for name in files:
        for entry in iter_seed_entries(base / name):
            index.add(entry)
    index.log_summary()
    return index


async def sync_seed_to_db(
    directory: Path | None = None,
    *,
    files: Iterable[str] = (SUGAR_SEED_FILE,),
    settings: Settings | None = None,
) -> int:
    """Залить seed-файлы в `ingredients_dict`.

    Направление одностороннее — файл источник истины, база его копия. Обратной
    синхронизации нет намеренно: иначе правки разъехались бы между двумя
    местами, и было бы непонятно, какая версия словаря верна.
    """
    settings = settings or get_settings()
    base = directory or DICTIONARIES_DIR
    rows = [
        AliasRow(
            alias=entry.alias,
            lang=entry.lang,
            canonical_name=entry.canonical_name,
            kind=entry.kind.value if entry.kind else None,
        )
        for name in files
        for entry in iter_seed_entries(base / name)
    ]

    async with get_session(settings.db) as session:
        return await IngredientAliasRepository(session).upsert_batch(rows)


async def load_db_index(settings: Settings | None = None) -> AliasIndex:
    """Собрать индекс из таблицы `ingredients_dict`.

    Используется прогонами, которые уже работают с БД: читать файл повторно
    в каждом процессе незачем, а база гарантирует, что все потребители видят
    одну и ту же версию словаря.
    """
    settings = settings or get_settings()
    async with get_session(settings.db) as session:
        rows = await IngredientAliasRepository(session).all_aliases()

    index = AliasIndex(
        AliasEntry(
            alias=row.alias,
            lang=row.lang,
            canonical_name=row.canonical_name,
            kind=IngredientKind(row.kind) if row.kind else None,
        )
        for row in rows
    )
    index.log_summary()
    return index


@dataclass(frozen=True)
class NormalizedIngredient:
    """Ингредиент после канонизации."""

    name: str
    kind: IngredientKind
    e_number: str | None = None
    # False означает, что имени нет в словаре. Такие имена и составляют наш
    # аналог unknown_ingredients_n.
    known: bool = False
    # Что вернула модель до канонизации — нужно, чтобы понять, почему имя
    # не нашлось, и дописать алиас.
    original: str = ""


@dataclass
class NormalizationStats:
    """Сводка канонизации. Отсюда берётся доля неизвестных имён."""

    total: int = 0
    known: int = 0
    unknown_names: Counter[str] = field(default_factory=Counter)
    products: int = 0

    @property
    def unknown(self) -> int:
        return self.total - self.known

    @property
    def unknown_share(self) -> float:
        return self.unknown / self.total if self.total else 0.0

    @property
    def distinct_unknown(self) -> int:
        return len(self.unknown_names)

    def add(self, ingredients: Iterable[NormalizedIngredient]) -> None:
        self.products += 1
        for ingredient in ingredients:
            self.total += 1
            if ingredient.known:
                self.known += 1
            else:
                self.unknown_names[ingredient.name] += 1

    def top_unknown(self, limit: int) -> list[tuple[str, int]]:
        """Самые частые неизвестные имена — куда пополнять словарь.

        Порядок детерминирован: при равной частоте имена идут по алфавиту,
        иначе один и тот же прогон давал бы разные подсказки.
        """
        return sorted(self.unknown_names.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]

    def log_summary(self) -> None:
        logger.info(
            "Канонизация завершена",
            extra=safe_extra(
                products=self.products,
                ingredients=self.total,
                known=self.known,
                unknown=self.unknown,
                unknown_share=f"{self.unknown_share:.1%}",
                distinct_unknown=self.distinct_unknown,
            ),
        )


def normalize_ingredient(
    ingredient: Ingredient,
    index: AliasIndex,
    *,
    lang: str | None = None,
) -> NormalizedIngredient:
    """Свести один ингредиент к каноническому имени.

    Тип из словаря **побеждает** тип от модели: живая проверка показала, что
    на коротком промпте модель классифицирует почти всё как `flavouring`,
    и доверять её типу там, где человек уже указал верный, нельзя.
    Имени нет в словаре — оставляем как есть и помечаем неизвестным.
    """
    entry = index.lookup(ingredient.canonical_name, lang=lang)
    if entry is None:
        return NormalizedIngredient(
            name=normalize_key(ingredient.canonical_name) or ingredient.canonical_name,
            kind=ingredient.kind,
            e_number=ingredient.e_number,
            known=False,
            original=ingredient.canonical_name,
        )

    return NormalizedIngredient(
        name=entry.canonical_name,
        kind=entry.kind or ingredient.kind,
        e_number=ingredient.e_number,
        known=True,
        original=ingredient.canonical_name,
    )


def normalize_ingredients(
    ingredients: Iterable[Ingredient],
    index: AliasIndex,
    *,
    lang: str | None = None,
) -> list[NormalizedIngredient]:
    return [normalize_ingredient(item, index, lang=lang) for item in ingredients]


def distinct_sugar_forms(ingredients: Iterable[NormalizedIngredient]) -> int:
    """Число РАЗНЫХ форм сахара после канонизации.

    Ключевая величина проекта. Считается по каноническим именам, поэтому
    `glucose-fructose syrup` и `glucose fructose syrup` — это одна форма,
    а сахар и мальтодекстрин — две.
    """
    return len({item.name for item in ingredients if item.kind is IngredientKind.SUGAR})


def sugar_forms_by_dictionary(
    ingredients: Iterable[Ingredient],
    index: AliasIndex,
    *,
    lang: str | None = None,
) -> set[str]:
    """Канонические формы сахара, подтверждённые СЛОВАРЁМ.

    Отличие от `distinct_sugar_forms` принципиальное: там тип берётся
    у ингредиента (а значит, в конечном счёте у модели, если словарь имя
    не знает), здесь — только у словаря. Имени нет в словаре — это не форма
    сахара, как бы её ни назвала модель.

    Так сделано не из вкуса, а по измерению: на живых данных
    `qwen2.5:3b-instruct-q4_K_M` проставляла `kind=sugar` овсу, соли, молоку
    и списку аллергенов подряд, и ключевая величина проекта оказалась
    завышенной. Словарь детерминирован и проверяем глазами, вердикт модели —
    нет. Разбор — ADR-035.

    Цена решения названа прямо: форма сахара, которой нет в словаре,
    не посчитается. Куда словарь пополнять, показывает отчёт
    `extract dict unknown` — по данным, а не на глаз.
    """
    found: set[str] = set()
    for ingredient in ingredients:
        entry = index.lookup(ingredient.canonical_name, lang=lang)
        if entry is not None and entry.kind is IngredientKind.SUGAR:
            found.add(entry.canonical_name)
    return found


def distinct_sugar_forms_by_dictionary(
    ingredients: Iterable[Ingredient],
    index: AliasIndex,
    *,
    lang: str | None = None,
) -> int:
    """Сколько РАЗНЫХ форм сахара подтверждает словарь."""
    return len(sugar_forms_by_dictionary(ingredients, index, lang=lang))


def format_unknown_report(stats: NormalizationStats, limit: int) -> str:
    """Человекочитаемый список того, куда пополнять словарь."""
    lines = [
        f"Ингредиентов всего:  {stats.total}",
        f"Известных словарю:   {stats.known} ({1 - stats.unknown_share:.1%})",
        f"Неизвестных:         {stats.unknown} ({stats.unknown_share:.1%})",
        f"Разных неизвестных:  {stats.distinct_unknown}",
        "",
        f"Топ-{limit} неизвестных имён:",
    ]
    lines.extend(f"  {count:6d}  {name}" for name, count in stats.top_unknown(limit))
    return "\n".join(lines)
