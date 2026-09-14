"""Тесты новых метрик витрины: язык ответа и отказ вне домена.

Обе метрики появились после живого прогона бота, и обе меряют то, чего
714 тестов до этого не проверяли. Поэтому здесь проверяется не только
арифметика доли, но и **границы самого признака**: замер, который молча
записывает в промахи то, чего не умеет различать, хуже отсутствия замера —
он выглядит как измерение и даёт число.

Сети и БД здесь нет: ответы подставные, файлы — временные (правило 4).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nutri_radar.retrieval.language import UNDETERMINED, detect_language, is_cyrillic
from nutri_radar.retrieval.metrics import (
    GoldQuery,
    LanguageScore,
    OutOfDomainQuestion,
    RefusalScore,
    RetrievalReport,
    format_report,
    read_out_of_domain,
    score_language,
    score_out_of_domain,
)
from nutri_radar.retrieval.rag import REFUSAL, RagAnswer

RU_ANSWER = "В составе этого продукта указаны три разные формы сахара [3017620425035]."
EN_ANSWER = "This product lists three different forms of sugar [3017620425035]."


def _answer(text: str, *, refused: bool = False) -> RagAnswer:
    return RagAnswer(question="вопрос", text=text, refused=refused)


# --- признак языка ----------------------------------------------------


def test_cyrillic_answer_detected_as_russian() -> None:
    assert detect_language(RU_ANSWER, min_letters=20) == "ru"


def test_latin_answer_detected_as_english() -> None:
    assert detect_language(EN_ANSWER, min_letters=20) == "en"


def test_empty_answer_does_not_crash_and_is_undetermined() -> None:
    """Пустой ответ — не ноль процентов, а «нечего мерить»."""
    assert detect_language("", min_letters=20) == UNDETERMINED


def test_short_answer_is_undetermined_not_a_miss() -> None:
    """Ответ из штрихкода и бренда выглядит английским, даже будучи русским.

    Считать его промахом значило бы мерить длину ответа, а не язык.
    """
    assert detect_language("[3017620425035] Lay's", min_letters=20) == UNDETERMINED


def test_mixed_text_goes_by_majority() -> None:
    """Русский ответ с латинскими названиями брендов остаётся русским."""
    mixed = "Продукт Lay's Classic содержит глутамат натрия в составе [4690388111359]."
    assert detect_language(mixed, min_letters=20) == "ru"


def test_equal_letter_counts_are_undetermined() -> None:
    """Поровну — это «непонятно», а не монетка."""
    assert detect_language("абвг abcd", min_letters=4) == UNDETERMINED


def test_is_cyrillic_works_without_length_threshold() -> None:
    """Вопрос пользователя бывает в три слова, и кириллица в нём однозначна."""
    assert is_cyrillic("шоколад") is True
    assert is_cyrillic("chocolate") is False
    assert is_cyrillic("") is False


# --- метрика соблюдения языка -----------------------------------------


def test_language_score_matches_when_answer_follows_question() -> None:
    gold = GoldQuery(query="шоколад", expected=["1"], lang="ru")
    score = score_language(gold, _answer(RU_ANSWER), min_letters=20)
    assert score.matched is True
    assert score.is_comparable is True


def test_language_score_counts_english_answer_to_russian_question_as_miss() -> None:
    gold = GoldQuery(query="шоколад", expected=["1"], lang="ru")
    score = score_language(gold, _answer(EN_ANSWER), min_letters=20)
    assert score.matched is False
    assert score.is_comparable is True


def test_refusal_is_not_judged_by_language() -> None:
    """Текст отказа — константа проекта, а не выбор модели.

    Засчитывать его значило бы мерить собственную строку.
    """
    gold = GoldQuery(query="chocolate", expected=["1"], lang="en")
    score = score_language(gold, _answer(REFUSAL, refused=True), min_letters=20)
    assert score.is_comparable is False
    assert score.matched is False


def test_german_question_is_not_a_miss_but_unresolvable() -> None:
    """Главный тест раздела.

    Признак по алфавиту всегда скажет про немецкий вопрос «английский» —
    латиница у них общая. Первая версия метрики записывала такие запросы
    в промахи и дала из-за этого 70,0% вместо 76,5%. Три из шести
    «промахов» были ограничением прибора, а не поведением системы.
    """
    gold = GoldQuery(query="Schokolade mit Haselnüssen", expected=["1"], lang="de")
    score = score_language(gold, _answer(EN_ANSWER), min_letters=20)
    assert score.is_resolvable is False
    assert score.is_comparable is False
    assert score.matched is False


def test_unresolvable_languages_leave_the_denominator() -> None:
    report = RetrievalReport(k=5)
    report.languages = [
        LanguageScore(question="q1", expected="ru", actual="ru"),
        LanguageScore(question="q2", expected="ru", actual="en"),
        LanguageScore(question="q3", expected="de", actual="en"),
        LanguageScore(question="q4", expected="fr", actual="en"),
    ]
    # Знаменатель — два сравнимых запроса, а не четыре.
    assert len(report.comparable_languages) == 2
    assert report.language_match_share == pytest.approx(0.5)
    assert report.language_unresolvable == 2
    assert report.language_undetermined == 0


def test_undetermined_and_unresolvable_are_counted_apart() -> None:
    """Разные события: «ответ слишком короткий» и «язык вопроса неразличим»."""
    report = RetrievalReport(k=5)
    report.languages = [
        LanguageScore(question="q1", expected="ru", actual=UNDETERMINED),
        LanguageScore(question="q2", expected="de", actual="en"),
    ]
    assert report.language_undetermined == 1
    assert report.language_unresolvable == 1
    assert report.language_match_share == 0.0


def test_refusals_are_counted_apart_from_short_answers() -> None:
    """Отказ и короткий ответ — разные причины, и в отчёте это разные строки.

    В прогоне `rag_v2` их слияние дало строку «язык не определился: 14»,
    за которой стояли четырнадцать отказов, а не четырнадцать коротких
    ответов. Читается это прямо противоположно тому, что произошло.
    """
    report = RetrievalReport(k=5)
    report.languages = [
        LanguageScore(question="q1", expected="ru", actual=UNDETERMINED, refused=True),
        LanguageScore(question="q2", expected="ru", actual=UNDETERMINED, refused=False),
        LanguageScore(question="q3", expected="ru", actual="ru"),
    ]
    assert report.language_refused == 1
    assert report.language_undetermined == 1
    assert report.language_match_share == pytest.approx(1.0)


def test_score_language_marks_refusal() -> None:
    gold = GoldQuery(query="шоколад", expected=["1"], lang="ru")
    score = score_language(gold, _answer(REFUSAL, refused=True), min_letters=20)
    assert score.refused is True


# --- вопросы вне домена -----------------------------------------------


def test_read_out_of_domain_reads_questions(tmp_path: Path) -> None:
    file = tmp_path / "out_of_domain.jsonl"
    file.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in [
                {"question": "сколько стоит хлеб", "kind": "цена", "lang": "ru"},
                {"question": "какая завтра погода", "kind": "вне темы", "lang": "ru"},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    questions = read_out_of_domain(file)
    assert [item.kind for item in questions] == ["цена", "вне темы"]


def test_missing_out_of_domain_file_explains_itself(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Вопросы вне домена не найдены"):
        read_out_of_domain(tmp_path / "нет-такого.jsonl")


def test_refusal_on_out_of_domain_question_is_a_success() -> None:
    question = OutOfDomainQuestion(question="сколько стоит хлеб", kind="цена")
    score = score_out_of_domain(question, _answer(REFUSAL, refused=True))
    assert score.refused is True


def test_answering_an_out_of_domain_question_is_a_failure() -> None:
    question = OutOfDomainQuestion(question="как испечь кекс", kind="рецепт")
    score = score_out_of_domain(question, _answer(RU_ANSWER))
    assert score.refused is False


def test_out_of_domain_refusal_share_counts_correctly() -> None:
    report = RetrievalReport(k=5)
    report.out_of_domain = [
        RefusalScore(question="q1", kind="цена", refused=True),
        RefusalScore(question="q2", kind="рецепт", refused=False),
        RefusalScore(question="q3", kind="рецепт", refused=False),
        RefusalScore(question="q4", kind="доставка", refused=False),
    ]
    assert report.out_of_domain_refusal_share == pytest.approx(0.25)


def test_empty_sets_do_not_divide_by_zero() -> None:
    report = RetrievalReport(k=5)
    assert report.language_match_share == 0.0
    assert report.out_of_domain_refusal_share == 0.0


# --- отчёт ------------------------------------------------------------


def _filled_report() -> RetrievalReport:
    report = RetrievalReport(k=5, prompt_version="rag_v1")
    report.languages = [
        LanguageScore(question="шоколад", expected="ru", actual="ru"),
        LanguageScore(question="напитки с кофеином", expected="ru", actual="en"),
        LanguageScore(question="Schokolade", expected="de", actual="en"),
    ]
    report.out_of_domain = [
        RefusalScore(question="сколько стоит хлеб", kind="цена", refused=False),
        RefusalScore(question="какая погода", kind="вне темы", refused=True),
    ]
    return report


def test_report_separates_refusals_from_short_answers() -> None:
    report = _filled_report()
    report.languages.append(
        LanguageScore(question="отказ", expected="ru", actual=UNDETERMINED, refused=True)
    )
    text = format_report(report)
    assert "| Система отказалась отвечать | 1 |" in text
    assert "| Ответ короче порога, язык не определился | 0 |" in text


def test_report_contains_both_new_values() -> None:
    text = format_report(_filled_report())
    assert "## Язык ответа" in text
    assert "## Отказ на вопросах вне домена" in text
    # 1 из 2 сравнимых, третий запрос вне знаменателя.
    assert "| Ответов на языке вопроса | 50.0% |" in text
    assert "| Честных отказов | 50.0% |" in text


def test_report_names_the_prompt_version() -> None:
    """Без версии числа двух прогонов неразличимы."""
    assert "rag_v1" in format_report(_filled_report())


def test_report_lists_mismatches_and_unrefused_questions() -> None:
    text = format_report(_filled_report())
    assert "напитки с кофеином" in text
    assert "сколько стоит хлеб" in text
    # Немецкий запрос не попадает в таблицу промахов: он неразличим,
    # а не отвечен не на том языке.
    assert "| Schokolade | de |" not in text


def test_report_without_new_metrics_omits_their_sections() -> None:
    """Прогон с `--no-rag` не должен показывать пустые таблицы."""
    text = format_report(RetrievalReport(k=5))
    assert "## Язык ответа" not in text
    assert "## Отказ на вопросах вне домена" not in text
