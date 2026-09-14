"""Замер стоимости переключения моделей на 6 ГБ VRAM.

**Это замер, а не оптимизация.** Число, которое он даёт, не предлагается
уменьшать: оно и есть цена того, что `qwen2.5:3b` и `bge-m3` не помещаются
в видеопамять одновременно. Прятать его незачем — ограничение железа
записано в брифе, и честнее показать, во что оно обходится, чем оставить
читателя гадать.

**Откуда взялся вопрос.** В живом прогоне бота два вопроса подряд заняли
15,6 с и 88 с. Разница не в сложности вопроса: первый пришёл, когда модель
генерации уже была загружена, а второй потребовал выгрузить `bge-m3`,
загрузить `qwen2.5:3b`, а перед этим — наоборот. Догадка правдоподобная,
но догадка; здесь она проверяется.

**Как измеряется.** Одни и те же вопросы прогоняются дважды.

* *Подряд* — сначала векторизуются все вопросы разом (одна загрузка
  `bge-m3`), потом все ответы генерируются разом (одна загрузка
  `qwen2.5:3b`). Ровно одно переключение на весь прогон.
* *С чередованием* — каждый вопрос проходит полный путь «вектор → поиск →
  ответ» целиком, как в боте. Два переключения на каждый вопрос.

Разница медиан — цена двух загрузок весов. Медиана, а не среднее: одна
аномально долгая генерация не должна определять число, которое пойдёт
в README.

**Оба режима меряются по одним правилам, и это оказалось не бесплатно.**
Первая версия замера запускала таймер только вокруг поиска и генерации,
а векторизацию и переключение на модель генерации режим «подряд» платил
вне таймера. Режим с чередованием платил их внутри. Разница режимов
переставала быть разницей в переключениях: в неё попадала работа,
которую один режим просто не засекал, и цена переключения выходила
завышенной примерно на четверть. Теперь общая работа режима «подряд» —
векторизация плюс переключение — засекается отдельным счётчиком
и раскладывается по вопросам поровну: платится она один раз на весь
батч, и приписывать её целиком первому вопросу было бы так же неверно,
как не считать вовсе.

**Прогрев обязателен.** Первое обращение к модели после старта Ollama
читает веса с диска, а не из кеша страниц, и стоит заметно дороже
последующих. Замер без прогрева измерил бы состояние дискового кеша.
Тот же порядок, что в `extract benchmark`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import httpx

from nutri_radar.config import Settings, get_settings
from nutri_radar.llm.factory import build_llm
from nutri_radar.llm.runtime import get_runtime
from nutri_radar.logging import safe_extra
from nutri_radar.retrieval.pipeline import ask, embed_texts
from nutri_radar.retrieval.rag import answer as rag_answer
from nutri_radar.retrieval.search import search as search_vectors

logger = logging.getLogger(__name__)

BATCHED = "подряд"
ALTERNATING = "с чередованием"


@dataclass
class SwitchRun:
    """Один режим прогона."""

    mode: str
    latencies: list[float] = field(default_factory=list)
    # Работа, которую режим платит один раз на весь прогон, а не на каждый
    # вопрос: векторизация всех вопросов разом и переключение на модель
    # генерации. У режима с чередованием она равна нулю — там всё
    # per-question и уже сидит в `latencies`.
    shared_s: float = 0.0
    # Переключения и загрузки берутся из очереди моделей, а не считаются
    # здесь заново: очередь — единственное место, которое знает правду
    # о том, что реально произошло с VRAM.
    switches: int = 0
    loads: int = 0

    @property
    def per_question(self) -> list[float]:
        """Время на вопрос с учётом разовой работы режима.

        Разовая работа делится поровну: она платится один раз на батч,
        и приписать её целиком первому вопросу было бы так же неверно,
        как не считать вовсе.
        """
        if not self.latencies:
            return []
        share = self.shared_s / len(self.latencies)
        return [value + share for value in self.latencies]

    @property
    def median(self) -> float:
        values = sorted(self.per_question)
        if not values:
            return 0.0
        return values[len(values) // 2]

    @property
    def total(self) -> float:
        return sum(self.latencies) + self.shared_s


@dataclass
class SwitchBenchmark:
    """Свод замера: два режима и разница между ними."""

    model: str
    embedding_model: str
    batched: SwitchRun
    alternating: SwitchRun

    @property
    def questions(self) -> int:
        return len(self.batched.latencies)

    @property
    def difference(self) -> float:
        """Насколько медленнее вопрос в режиме с чередованием."""
        return self.alternating.median - self.batched.median

    @property
    def cost_per_switch(self) -> float:
        """Цена одного переключения.

        Делится на число переключений, случившихся сверх режима «подряд»,
        а не на число вопросов: вопрос в режиме с чередованием стоит
        двух переключений, и приписать всю разницу одному значило бы
        завысить цену вдвое.
        """
        extra = self.alternating.switches - self.batched.switches
        if extra <= 0:
            return 0.0
        return (self.alternating.total - self.batched.total) / extra

    def format(self) -> str:
        """Отчёт по замеру."""
        return "\n".join(
            [
                "# Цена переключения моделей на 6 ГБ VRAM",
                "",
                f"Вопросов: {self.questions}. Модель генерации: `{self.model}`, "
                f"модель эмбеддингов: `{self.embedding_model}`.",
                "",
                "| Режим | Медиана на вопрос | Всего | Разовая работа | Переключений |",
                "|---|---|---|---|---|",
                f"| Подряд | {self.batched.median:.1f} с | {self.batched.total:.1f} с "
                f"| {self.batched.shared_s:.1f} с | {self.batched.switches} |",
                f"| С чередованием | {self.alternating.median:.1f} с "
                f"| {self.alternating.total:.1f} с | {self.alternating.shared_s:.1f} с "
                f"| {self.alternating.switches} |",
                "",
                "| Величина | Значение |",
                "|---|---|",
                f"| Разница медиан | {self.difference:.1f} с |",
                f"| Цена одного переключения | {self.cost_per_switch:.1f} с |",
                "",
                "**Это замер, а не оптимизация.** Число не предлагается "
                "уменьшать: оно и есть цена того, что модель генерации и модель "
                "эмбеддингов не помещаются в 6 ГБ видеопамяти одновременно. "
                "Ограничение железа записано в брифе, и честнее показать, во что "
                "оно обходится, чем оставить читателя гадать, почему один вопрос "
                "отвечается за пятнадцать секунд, а следующий за полторы минуты.",
                "",
                "**Режим «подряд» в боте недостижим.** Он требует знать все "
                "вопросы заранее, чтобы векторизовать их одним заходом. Бот "
                "получает вопросы по одному, и каждый проходит полный путь — "
                "то есть живёт в нижней строке таблицы. Верхняя строка нужна "
                "как база отсчёта, а не как цель.",
                "",
                "**Почему медиана, а не среднее.** Одна аномально долгая "
                "генерация не должна определять число, которое пойдёт в README "
                "как характеристика железа.",
                "",
                "**Столбец «разовая работа» — про честность сравнения.** Режим "
                "«подряд» векторизует все вопросы одним заходом и переключается "
                "на модель генерации один раз; это его работа, и она входит "
                "в замер, разложенная по вопросам поровну. Первая версия этого "
                "замера её не засекала, и цена переключения выходила завышенной "
                "примерно на четверть: в разницу режимов попадало то, на что "
                "один из них просто не посмотрел на часы.",
            ]
        )


async def _answer_batched(
    questions: list[str],
    *,
    client: httpx.AsyncClient,
    settings: Settings,
    top_k: int,
) -> SwitchRun:
    """Прогон без чередования: все векторы, потом все ответы."""
    runtime = get_runtime(settings)
    before_switches, before_loads = runtime.switches, runtime.loads
    run = SwitchRun(mode=BATCHED)

    # Векторизация и переключение на модель генерации — внутри замера.
    # Режим с чередованием платит их на каждый вопрос и засекает; если
    # здесь их не засечь, разница режимов измерит не переключения,
    # а то, что один режим не посмотрел на свои часы.
    started_shared = time.perf_counter()
    vectors = await embed_texts(questions, client=client, settings=settings)
    llm = build_llm(settings, client)
    async with runtime.hold(llm.model_name):
        shared = time.perf_counter() - started_shared
        for question, vector in zip(questions, vectors, strict=True):
            started = time.perf_counter()
            result = await search_vectors(vector, query=question, limit=top_k, settings=settings)
            await rag_answer(question, result, llm, settings)
            run.latencies.append(time.perf_counter() - started)

    run.shared_s = shared
    run.switches = runtime.switches - before_switches
    run.loads = runtime.loads - before_loads
    logger.info(
        "Режим без чередования пройден",
        extra=safe_extra(
            questions=len(questions),
            switches=run.switches,
            shared_s=round(run.shared_s, 2),
            median_s=round(run.median, 2),
        ),
    )
    return run


async def _answer_alternating(
    questions: list[str],
    *,
    client: httpx.AsyncClient,
    settings: Settings,
    top_k: int,
) -> SwitchRun:
    """Прогон с чередованием: каждый вопрос проходит полный путь.

    Идёт через `pipeline.ask` — ту же функцию, что зовёт бот. Собрать
    здесь свою цепочку значило бы замерить не то, что работает в проде.
    """
    runtime = get_runtime(settings)
    before_switches, before_loads = runtime.switches, runtime.loads
    run = SwitchRun(mode=ALTERNATING)

    for question in questions:
        started = time.perf_counter()
        await ask(question, client=client, settings=settings, limit=top_k)
        run.latencies.append(time.perf_counter() - started)
        logger.info(
            "Вопрос с чередованием пройден",
            extra=safe_extra(
                question=question[:120],
                latency_s=round(run.latencies[-1], 2),
                switches=runtime.switches - before_switches,
            ),
        )

    run.switches = runtime.switches - before_switches
    run.loads = runtime.loads - before_loads
    return run


async def measure(
    questions: list[str],
    *,
    client: httpx.AsyncClient,
    settings: Settings | None = None,
    top_k: int | None = None,
) -> SwitchBenchmark:
    """Замерить цену переключения на одинаковых вопросах.

    Args:
        questions: вопросы. Одни и те же в обоих режимах — иначе разница
            измеряла бы разницу вопросов.
        client: HTTP-клиент к Ollama.
        settings: настройки.
        top_k: сколько продуктов отдавать модели.

    Returns:
        Замер двух режимов и разница между ними.
    """
    settings = settings or get_settings()
    top_k = top_k or settings.retrieval.top_k
    if not questions:
        raise ValueError("Замер без вопросов ничего не измеряет")

    # Прогрев: первое обращение читает веса с диска, а не из кеша страниц.
    # Результат выбрасывается — он измерил бы состояние кеша, а не железо.
    logger.info("Прогрев перед замером", extra=safe_extra(question=questions[0][:120]))
    await ask(questions[0], client=client, settings=settings, limit=top_k)

    batched = await _answer_batched(questions, client=client, settings=settings, top_k=top_k)
    alternating = await _answer_alternating(
        questions, client=client, settings=settings, top_k=top_k
    )

    benchmark = SwitchBenchmark(
        model=build_llm(settings, client).model_name,
        embedding_model=settings.ollama.embedding_model,
        batched=batched,
        alternating=alternating,
    )
    logger.info(
        "Замер переключений завершён",
        extra=safe_extra(
            questions=benchmark.questions,
            batched_median_s=round(batched.median, 2),
            alternating_median_s=round(alternating.median, 2),
            difference_s=round(benchmark.difference, 2),
            cost_per_switch_s=round(benchmark.cost_per_switch, 2),
            switches_batched=batched.switches,
            switches_alternating=alternating.switches,
        ),
    )
    return benchmark
