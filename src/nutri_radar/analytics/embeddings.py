"""Векторизация корпуса моделью эмбеддингов: замер, прогон, кэш.

**Сначала замер, потом прогон.** Тот же порядок, что в M2 (ADR-017):
на 200 продуктах меряются секунды на продукт, экстраполируются на корпус,
и только после этого принимается решение о его размере. Запускать
векторизацию 131 тысячи текстов «и посмотрим» — это способ узнать через
три часа, что оно не помещалось.

**Прогон возобновляемый.** Векторы кэшируются на диск пачками, и перезапуск
продолжает с невекторизованных. Ноутбук засыпает, GPU занимают другие
процессы, прогон рвётся — терять из-за этого час работы незачем.

**Кэш привязан к имени модели.** Векторы разных моделей несравнимы, и файл
без имени модели — это набор чисел неизвестно чего. Имя уезжает и в путь,
и внутрь файла.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nutri_radar.llm.ports import EmbeddingModel
from nutri_radar.logging import safe_extra

logger = logging.getLogger(__name__)

VECTORS_DIR = Path("data/analytics/vectors")

# Сколько текстов уходит в Ollama одним запросом. Не из головы: батч
# упирается в num_ctx суммарно, и составы бывают длинными. 32 держит
# запрос в разумном размере и убирает большую часть round-trip-ов.
BATCH_SIZE = 32

# Через сколько батчей кэш сбрасывается на диск. Компромисс: чаще —
# лишний ввод-вывод, реже — больше потерь при обрыве.
FLUSH_EVERY = 20


@dataclass
class EmbeddingBenchmark:
    """Результат замера скорости перед полным прогоном."""

    model: str
    products: int
    seconds: float

    @property
    def per_product(self) -> float:
        return self.seconds / self.products if self.products else 0.0

    def extrapolate(self, corpus_size: int) -> float:
        """Сколько минут займёт корпус такого размера."""
        return self.per_product * corpus_size / 60

    def format(self, corpus_size: int) -> str:
        return (
            f"Замер эмбеддингов: {self.products} продуктов за {self.seconds:.1f} с "
            f"({self.per_product:.3f} с на продукт).\n"
            f"Экстраполяция на {corpus_size} продуктов: "
            f"{self.extrapolate(corpus_size):.0f} минут."
        )


def vectors_path(model: str, root: Path | None = None) -> Path:
    """Файл кэша векторов. Имя модели — часть пути, а не примечание."""
    safe = model.replace(":", "_").replace("/", "_")
    return (root or VECTORS_DIR) / f"{safe}.npz"


def load_vectors(model: str, root: Path | None = None) -> dict[str, np.ndarray]:
    """Прочитать кэш. Нет файла — пустой словарь, а не ошибка."""
    path = vectors_path(model, root)
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=False)
    codes = data["codes"]
    vectors = data["vectors"]
    logger.info(
        "Кэш векторов прочитан",
        extra=safe_extra(path=str(path), vectors=len(codes), model=model),
    )
    return {str(code): vectors[i] for i, code in enumerate(codes)}


def save_vectors(cache: dict[str, np.ndarray], model: str, root: Path | None = None) -> Path:
    """Сохранить кэш целиком."""
    path = vectors_path(model, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    codes = np.array(list(cache.keys()))
    vectors = np.vstack(list(cache.values())) if cache else np.empty((0, 0), dtype=np.float32)
    np.savez_compressed(path, codes=codes, vectors=vectors)
    logger.debug("Кэш векторов сохранён", extra=safe_extra(path=str(path), vectors=len(codes)))
    return path


async def benchmark(
    model: EmbeddingModel,
    texts: Sequence[str],
) -> EmbeddingBenchmark:
    """Замерить скорость векторизации на небольшой пачке.

    Отдельная функция, а не флаг у полного прогона: замер обязан быть
    дешёвым действием, которое запускают до того, как решать что-либо.
    """
    started = time.perf_counter()
    for offset in range(0, len(texts), BATCH_SIZE):
        await model.embed(texts[offset : offset + BATCH_SIZE])
    seconds = time.perf_counter() - started

    result = EmbeddingBenchmark(model=model.model_name, products=len(texts), seconds=seconds)
    logger.info(
        "Замер эмбеддингов выполнен",
        extra=safe_extra(
            model=result.model,
            products=result.products,
            seconds=round(seconds, 1),
            per_product=round(result.per_product, 4),
        ),
    )
    return result


async def embed_frame(
    frame: pd.DataFrame,
    model: EmbeddingModel,
    *,
    text_column: str = "ingredients_text",
    code_column: str = "code",
    root: Path | None = None,
) -> np.ndarray:
    """Векторизовать корпус, продолжая с прерванного места.

    Returns:
        Матрица векторов в порядке строк `frame`.
    """
    cache = load_vectors(model.model_name, root)
    pending = frame[~frame[code_column].isin(cache.keys())]
    logger.info(
        "Векторизация начата",
        extra=safe_extra(
            model=model.model_name,
            total=len(frame),
            cached=len(frame) - len(pending),
            pending=len(pending),
        ),
    )

    started = time.perf_counter()
    for batch_index, offset in enumerate(range(0, len(pending), BATCH_SIZE), start=1):
        chunk = pending.iloc[offset : offset + BATCH_SIZE]
        vectors = await model.embed(chunk[text_column].tolist())
        for code, vector in zip(chunk[code_column], vectors, strict=True):
            cache[str(code)] = np.asarray(vector, dtype=np.float32)

        if batch_index % FLUSH_EVERY == 0:
            save_vectors(cache, model.model_name, root)
            done = min(offset + BATCH_SIZE, len(pending))
            elapsed = time.perf_counter() - started
            logger.info(
                "Векторизация идёт",
                extra=safe_extra(
                    done=done,
                    pending=len(pending),
                    elapsed_min=round(elapsed / 60, 1),
                    eta_min=round(elapsed / done * (len(pending) - done) / 60, 1),
                ),
            )

    if len(pending):
        save_vectors(cache, model.model_name, root)

    logger.info(
        "Векторизация завершена",
        extra=safe_extra(
            model=model.model_name,
            vectors=len(frame),
            minutes=round((time.perf_counter() - started) / 60, 1),
        ),
    )
    return np.vstack([cache[str(code)] for code in frame[code_column]])
