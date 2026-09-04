"""Выгрузка обучающего набора и стратифицированный сплит.

**Почему выгрузка отделена от сборки фич.** Загрузка данных и превращение
текста в признаки — два разных уровня, и смешивать их внутри `features.py`
значит получить модуль, который нельзя протестировать без базы. Здесь
только ввод-вывод и сплит; ни одного признака этот модуль не считает.

**Почему из базы читаются только пять колонок.** `nutriscore_grade`
вычисляется по нутриентам. Если в датафрейм попадёт хоть один нутриент,
задача выродится в пересчёт формулы, и точность 0.99 будет означать, что
модель заново вывела Nutri-Score, а не научилась читать состав. Защита
здесь структурная, а не дисциплинарная: колонок с нутриентами в выборке
просто нет, и добавить их случайно нельзя.

**Почему сплит стратифицированный и с фиксированным seed.** Классы
несбалансированы (44% приходится на «e», 3,7% на «b»), и случайный сплит
дал бы тесты с плавающей долей редких классов — числа двух прогонов
перестали бы быть сравнимыми. Seed берётся из настроек и живёт отдельно
от seed-ов `extract` и `evals`: смена одного не должна переставлять другие.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd
from sklearn.model_selection import train_test_split
from sqlalchemy import func, select

from nutri_radar.config import Settings, get_settings
from nutri_radar.db.models.product import Product
from nutri_radar.db.session import get_session
from nutri_radar.logging import safe_extra

logger = logging.getLogger(__name__)

DATASET_DIR = Path("data/analytics")
DATASET_FILE = DATASET_DIR / "dataset.parquet"

# Колонки, которые слайс вообще имеет право видеть. Список полный и закрытый:
# всё, чего здесь нет, в модель попасть не может.
COLUMNS = ("code", "ingredients_text", "lang", "nutriscore_grade", "nova_group")

# Метки, которые умеет предсказывать слайс. Строкой, потому что уходит
# в имена файлов и в CLI.
TARGETS = ("nutriscore_grade", "nova_group")


def _validate_target(target: str) -> None:
    if target not in TARGETS:
        raise ValueError(f"Неизвестная метка {target!r}. Доступны: {', '.join(TARGETS)}.")


async def fetch_dataset(settings: Settings | None = None) -> pd.DataFrame:
    """Выгрузить из базы всё, на чём можно учиться.

    Берутся продукты, у которых есть текст состава и хотя бы одна из двух
    меток. Строки короче `min_text_length` отбрасываются: «-», «н/д»
    и пустые скобки не несут информации, но шум в метрики добавляют.

    Returns:
        Датафрейм с колонками `COLUMNS`. Метка может быть `None` —
        отбор под конкретную задачу делает `prepare`.
    """
    settings = settings or get_settings()
    min_length = settings.analytics.min_text_length

    statement = (
        select(
            Product.code,
            Product.ingredients_text,
            Product.ingredients_text_lang,
            Product.nutriscore_grade,
            Product.nova_group,
        )
        .where(
            Product.ingredients_text.is_not(None),
            func.length(Product.ingredients_text) > min_length,
            Product.nutriscore_grade.is_not(None) | Product.nova_group.is_not(None),
        )
        .order_by(Product.code)
    )

    logger.info(
        "Выгрузка обучающего набора начата",
        extra=safe_extra(min_text_length=min_length),
    )
    async with get_session(settings.db) as session:
        rows = (await session.execute(statement)).all()

    frame = pd.DataFrame(
        [
            {
                "code": row[0],
                "ingredients_text": row[1],
                "lang": row[2] or "unknown",
                "nutriscore_grade": row[3],
                "nova_group": row[4],
            }
            for row in rows
        ],
        columns=list(COLUMNS),
    )
    logger.info(
        "Обучающий набор выгружен",
        extra=safe_extra(
            rows=len(frame),
            with_grade=int(frame["nutriscore_grade"].notna().sum()),
            with_nova=int(frame["nova_group"].notna().sum()),
        ),
    )
    return frame


def save_dataset(frame: pd.DataFrame, path: Path | None = None) -> Path:
    """Сохранить выгрузку на диск.

    Кэш, а не артефакт репозитория: 131 тысяча строк выгружается минуты,
    и повторять это на каждый эксперимент незачем. В git не уезжает —
    каталог в `.gitignore`, а воспроизводимость обеспечивает seed, а не файл.
    """
    file = path or DATASET_FILE
    file.parent.mkdir(parents=True, exist_ok=True)
    # Parquet пишется DuckDB, а не pandas: pandas требует pyarrow, которого
    # в проекте нет, а DuckDB уже есть — им читается дамп OFF в M1. Тащить
    # второй движок parquet ради одного вызова незачем (правило 7).
    duckdb.sql("COPY frame TO ? (FORMAT PARQUET)", params=[str(file)])
    logger.info("Набор сохранён", extra=safe_extra(path=str(file), rows=len(frame)))
    return file


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    """Прочитать выгрузку с диска."""
    file = path or DATASET_FILE
    if not file.exists():
        raise FileNotFoundError(
            f"Набор не найден: {file}. Сначала выгрузите его командой `analytics dataset`."
        )
    frame = duckdb.sql("SELECT * FROM read_parquet(?)", params=[str(file)]).df()
    logger.debug("Набор прочитан", extra=safe_extra(path=str(file), rows=len(frame)))
    return frame


def prepare(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    """Оставить строки, пригодные для одной задачи.

    Метка приводится к строке: `nova_group` в базе целое, а `train_test_split`
    и метрики одинаково работают с обоими типами — но смешивать типы между
    двумя задачами значит получить два разных формата отчёта.
    """
    _validate_target(target)
    prepared = frame[frame[target].notna()].copy()
    # Целые метки сначала в int, потом в строку. Без этого `nova_group`
    # превращается в «4.0»: пропуски делают колонку float, и класс получает
    # имя, зависящее от того, как pandas хранил NaN, а не от самих данных.
    if pd.api.types.is_numeric_dtype(prepared[target]):
        prepared[target] = prepared[target].astype("int64")
    prepared[target] = prepared[target].astype(str)
    logger.debug(
        "Набор подготовлен под задачу",
        extra=safe_extra(target=target, rows=len(prepared), dropped=len(frame) - len(prepared)),
    )
    return prepared


def split(
    frame: pd.DataFrame,
    target: str,
    settings: Settings | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Разделить на train и test со стратификацией по метке.

    Стратификация обязательна: доля класса «b» — 3,7%, и на случайном сплите
    она гуляла бы между прогонами, делая числа несравнимыми.

    Классы, у которых меньше двух примеров, стратифицировать нельзя — такой
    класс физически не может попасть и в train, и в test. Они отбрасываются
    с предупреждением в лог, а не роняют прогон: `nova_group=2` встречается
    25 раз на 138 тысяч, и падать из-за него значило бы не считать nova вовсе.

    Returns:
        Пара «train, test».
    """
    _validate_target(target)
    settings = settings or get_settings()

    counts = frame[target].value_counts()
    rare = counts[counts < 2].index.tolist()
    if rare:
        logger.warning(
            "Классы отброшены: меньше двух примеров, стратифицировать нечего",
            extra=safe_extra(target=target, classes=rare),
        )
        frame = frame[~frame[target].isin(rare)]

    train, test = train_test_split(
        frame,
        test_size=settings.analytics.test_size,
        random_state=settings.analytics.random_seed,
        stratify=frame[target],
    )
    logger.info(
        "Сплит выполнен",
        extra=safe_extra(
            target=target,
            train=len(train),
            test=len(test),
            seed=settings.analytics.random_seed,
            classes=len(counts) - len(rare),
        ),
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def common_subset(
    test: pd.DataFrame,
    target: str,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Общая подвыборка теста, на которой меряются все три подхода.

    Существует ровно из-за одного ограничения: бриф запрещает гонять через
    LLM больше нескольких тысяч продуктов, а TF-IDF учится на всех. Сравнить
    точность TF-IDF на полном тесте с точностью LLM на тысяче — значит
    сравнить две разные задачи и выдать это за сравнение подходов.

    Подвыборка стратифицирована и воспроизводима по seed, поэтому её состав
    одинаков для всех трёх прогонов, в каком бы порядке их ни запускали.
    """
    _validate_target(target)
    settings = settings or get_settings()
    size = settings.analytics.llm_subset_size

    if len(test) <= size:
        logger.info(
            "Тест меньше подвыборки — берётся целиком",
            extra=safe_extra(test=len(test), requested=size),
        )
        return test.reset_index(drop=True)

    counts = test[target].value_counts()
    rare = counts[counts < 2].index.tolist()
    pool = test[~test[target].isin(rare)] if rare else test

    subset, _ = train_test_split(
        pool,
        train_size=size,
        random_state=settings.analytics.random_seed,
        stratify=pool[target],
    )
    logger.info(
        "Общая подвыборка собрана",
        extra=safe_extra(target=target, size=len(subset), seed=settings.analytics.random_seed),
    )
    return subset.reset_index(drop=True)


def majority_baseline(frame: pd.DataFrame, target: str) -> tuple[str, float]:
    """Самый частый класс и его доля.

    Число, без которого accuracy невозможно прочитать: 45% выглядит как
    работающая модель ровно до тех пор, пока не выяснится, что «всегда e»
    даёт 44,3%.
    """
    _validate_target(target)
    if frame.empty:
        return "", 0.0
    counts = frame[target].value_counts()
    return str(counts.index[0]), float(counts.iloc[0] / len(frame))
