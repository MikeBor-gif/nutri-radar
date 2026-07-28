"""Тесты замера перед полным прогоном.

Главное здесь — экстраполяция. Она принимает решение о запуске многочасового
прогона, и ошибка в ней стоит часов GPU.
"""

from __future__ import annotations

from nutri_radar.extract.benchmark import BenchmarkResult


def _result(latencies: list[float]) -> BenchmarkResult:
    """Замер, где все вызовы удачны: `latencies` считает и неудачные тоже."""
    result = BenchmarkResult(prompt_version="v3", model_name="тест", sample_size=len(latencies))
    result.latencies = latencies
    result.input_tokens = [200] * len(latencies)
    result.output_tokens = [250] * len(latencies)
    result.sugar_forms = [1] * len(latencies)
    result.succeeded = len(latencies)
    return result


class TestЭкстраполяция:
    def test_считается_по_медиане_а_не_по_среднему(self):
        """Среднее по длинному хвосту даёт оптимистичную оценку."""
        # Медиана 10, среднее 32 — длинный хвост есть.
        hours, _ = _result([5.0, 10.0, 10.0, 10.0, 125.0]).extrapolate(3600)

        assert hours == 10.0

    def test_параллелизм_не_даёт_скидки(self):
        """Измерено: с параллелизмом 2 прогон идёт медленнее, а не вдвое быстрее."""
        result = _result([10.0] * 5)

        assert result.extrapolate(1000, 1) == result.extrapolate(1000, 4)

    def test_токены_считаются_на_весь_корпус(self):
        _, tokens = _result([10.0] * 5).extrapolate(100)

        assert tokens == 100 * (200 + 250)

    def test_пустой_замер_не_делит_на_ноль(self):
        assert _result([]).extrapolate(3000) == (0.0, 0)


class TestДоляОтказов:
    def test_невалидные_считаются_от_попыток(self):
        result = _result([10.0] * 3)
        result.invalid = 1

        assert result.invalid_share == 0.25

    def test_доля_составов_без_сахара(self):
        result = _result([10.0] * 4)
        result.sugar_forms = [0, 0, 1, 2]

        assert result.zero_sugar_share == 0.5

    def test_успехи_не_выводятся_из_числа_замеров_времени(self):
        """В `latencies` лежат и неудачные вызовы — считать их успехами нельзя."""
        result = _result([10.0] * 3)
        # Четвёртый вызов дошёл до модели, но разобран не был.
        result.latencies.append(90.0)
        result.invalid = 1

        assert result.succeeded == 3
        assert result.invalid_share == 0.25


class TestНеудачныеВызовыВходятВОценку:
    """Полный прогон оплачивает и неудачные вызовы — оценка обязана их видеть.

    Медиана к одному выбросу устойчива по построению, и это правильно. А вот
    оценка токенов и p95 обязаны его увидеть: обрезанный ответ сжигает весь
    лимит вывода, и по стоимости он на порядок дороже обычного.
    """

    def _с_обрывом(self) -> BenchmarkResult:
        result = _result([10.0] * 3)
        result.latencies.append(90.0)
        result.input_tokens.append(200)
        result.output_tokens.append(2048)
        result.invalid = 1
        return result

    def test_оценка_токенов_учитывает_сожжённый_лимит(self):
        assert self._с_обрывом().extrapolate(1000)[1] > _result([10.0] * 3).extrapolate(1000)[1]

    def test_p95_показывает_самый_долгий_вызов(self):
        assert self._с_обрывом().p95_latency == 90.0
