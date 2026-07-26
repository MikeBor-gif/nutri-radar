# Колонки Parquet-дампа Open Food Facts: что берём и почему

Источник схемы: [HF datasets-server info](https://datasets-server.huggingface.co/info?dataset=openfoodfacts%2Fproduct-database)
Снято 2026-07-26. Файл: `food.parquet`, split `food` — 4 630 773 строки, 22.4 ГБ
распакованными. Всего в схеме 100+ колонок; ниже отобрано 42.

> Имена взяты из схемы Parquet, а не из `data-fields.txt`. Последний описывает
> CSV-экспорт, и половина имён оттуда в Parquet отсутствует — подробности
> в `.ai-factory/RESEARCH.md`, раздел 2.1.

---

## Вложенные типы: три структуры, которые определяют весь код выборки

Три группы колонок — не скаляры, и это главное, что нужно понять до написания SQL.

### 1. Многоязычные тексты — `LIST<STRUCT(lang VARCHAR, text VARCHAR)>`

Такой тип у `ingredients_text`, `product_name`, `generic_name`, `packaging_text`.

```sql
-- текст состава на конкретном языке (NULL, если языка нет)
list_extract(list_filter(ingredients_text, x -> x.lang = 'ru'), 1).text

-- все языки, на которых у продукта есть непустой состав
list_transform(
  list_filter(ingredients_text, x -> x.text IS NOT NULL AND trim(x.text) <> ''),
  x -> x.lang
)
```

Из этого следует, что фильтр «непустой состав на нужных языках» — это пересечение
списков, а не `WHERE ... IS NOT NULL`:

```sql
WHERE len(list_filter(
        ingredients_text,
        x -> x.lang IN ('en','ru','de','fr','pl')
             AND x.text IS NOT NULL
             AND length(trim(x.text)) >= 10
      )) > 0
```

⚠️ Проверить на реальном файле до написания финального фильтра: содержит ли
список служебную запись `lang = 'main'`, дублирующую главный язык продукта.
Если да — исключать её, иначе языковая статистика удвоится.

### 2. Нутриенты — `LIST<STRUCT(name, value, "100g", serving, unit, prepared_*)>`

`nutriments` — **список**, а не структура с колонками. Плоских `sugars_100g`
в Parquet не существует.

```sql
-- сахар на 100 г; имя колонки "100g" начинается с цифры → всегда в кавычках
list_extract(list_filter(nutriments, x -> x.name = 'sugars'), 1)['100g'] AS sugars_100g
```

Разворачивать нужные нутриенты удобнее макросом, чтобы не плодить копипасту:

```sql
CREATE OR REPLACE MACRO nutr(nl, nm) AS
  list_extract(list_filter(nl, x -> x.name = nm), 1)['100g'];

SELECT code,
       nutr(nutriments, 'energy-kcal')   AS energy_kcal_100g,
       nutr(nutriments, 'fat')           AS fat_100g,
       nutr(nutriments, 'saturated-fat') AS saturated_fat_100g,
       nutr(nutriments, 'sugars')        AS sugars_100g,
       nutr(nutriments, 'salt')          AS salt_100g,
       nutr(nutriments, 'proteins')      AS proteins_100g,
       nutr(nutriments, 'fiber')         AS fiber_100g,
FROM read_parquet('data/food.parquet');
```

⚠️ Проверить на реальном файле: точный набор значений `name`
(`energy-kcal` против `energy`, присутствие `salt` и `sodium` одновременно,
реальное покрытие `fruits-vegetables-nuts-estimate-from-ingredients`).
Запрос для проверки:

```sql
WITH sample AS (
  FROM read_parquet('data/food.parquet') SELECT nutriments LIMIT 100000
)
SELECT u.name, count() AS n
FROM sample, UNNEST(nutriments) AS t(u)
GROUP BY ALL
ORDER BY n DESC
LIMIT 40;
```

### 3. Теги — `LIST<VARCHAR>` с языковым префиксом

Все `*_tags` — списки строк вида `en:snacks`, `en:e330`, `en:milk`. Префикс —
язык таксономии, почти всегда `en`, и он часть значения, а не мусор.

```sql
WHERE list_has_any(categories_tags, ['en:snacks', 'en:beverages', 'en:dairies'])
```

---

## Отобранные колонки

### Идентификация и служебное (9)

| Колонка | Тип | Зачем |
|---|---|---|
| `code` | VARCHAR | штрихкод, первичный ключ `products_raw` / `products` |
| `lang` | VARCHAR | главный язык продукта |
| `languages_tags` | LIST\<VARCHAR\> | все языки записи |
| `rev` | INT32 | ревизия записи, для идемпотентного upsert |
| `last_modified_t` | INT64 | UNIX-время правки; ключ для дельт |
| `created_t` | INT64 | когда продукт добавлен |
| `obsolete` | BOOL | снятые с производства — **исключаем из корпуса** |
| `completeness` | FLOAT32 | полнота записи 0..1, порог качества |
| `schema_version` | INT32 | версия схемы дампа; страховка от молчаливой смены формата |

### Тексты — ядро проекта (4)

| Колонка | Тип | Зачем |
|---|---|---|
| `ingredients_text` | LIST\<STRUCT\> | **главный вход LLM.** Без него продукт не нужен |
| `product_name` | LIST\<STRUCT\> | вход для M4 (оценка по тексту) и профиль для эмбеддингов |
| `generic_name` | LIST\<STRUCT\> | дополняет название, часто содержит тип продукта |
| `brands` | VARCHAR | бренд; для отчётов и для поиска замены в M6 |

### Классификация (5)

| Колонка | Тип | Зачем |
|---|---|---|
| `categories_tags` | LIST\<VARCHAR\> | **фильтр корпуса** и стратификация выборки |
| `categories` | VARCHAR | сырой текст категорий, для отладки фильтра |
| `food_groups_tags` | LIST\<VARCHAR\> | укрупнённые группы, устойчивее чем categories |
| `countries_tags` | LIST\<VARCHAR\> | география, коррелирует с языком состава |
| `labels_tags` | LIST\<VARCHAR\> | «без сахара», «органик» — интересно сверить с составом |

### Готовые метки для M4 — без ручной разметки (4)

| Колонка | Тип | Зачем |
|---|---|---|
| `nutriscore_grade` | VARCHAR | **целевая метка основной задачи.** Фильтровать: кроме `a`–`e` бывают `unknown`, `not-applicable` |
| `nutriscore_score` | INT32 | численный балл; регрессионный вариант задачи |
| `nova_group` | INT32 | **целевая метка второй задачи** (степень переработки), nullable |
| `nova_groups_tags` | LIST\<VARCHAR\> | та же метка тегом, для сверки |

### Числовые фичи для sanity-check M4 (3 + нутриенты)

| Колонка | Тип | Зачем |
|---|---|---|
| `nutriments` | LIST\<STRUCT\> | БЖУ/соль/сахар/энергия + `fruits-vegetables-nuts-estimate-from-ingredients` |
| `nutrition_data_per` | VARCHAR | `100g` или `serving` — без этого числа несравнимы |
| `no_nutrition_data` | BOOL | явный флаг отсутствия таблицы питательности |

Нутриенты, которые разворачиваем: `energy-kcal`, `fat`, `saturated-fat`,
`carbohydrates`, `sugars`, `fiber`, `proteins`, `salt`, `sodium`,
`fruits-vegetables-nuts-estimate-from-ingredients`.

Последний критичен: он входит в формулу Nutri-Score. Без него sanity-check
не сойдётся — см. `RESEARCH.md`, раздел 2.3.

### Baseline от собственного парсера OFF (11)

Это то, с чем наша LLM будет соревноваться. Забираем целиком — иначе не с чем
сравнивать. Обоснование в `RESEARCH.md`, раздел 4.

| Колонка | Тип | Зачем |
|---|---|---|
| `ingredients` | VARCHAR (JSON) | JSON-дерево состава, разобранного силами OFF |
| `ingredients_tags` | LIST\<VARCHAR\> | нормализованные ингредиенты — прямой аналог нашего выхода |
| `ingredients_original_tags` | LIST\<VARCHAR\> | до нормализации, полезно для словаря алиасов |
| `ingredients_n` | INT32 | сколько ингредиентов распознал OFF |
| `known_ingredients_n` | INT32 | из них нашлось в таксономии |
| `unknown_ingredients_n` | INT32 | **ключевое поле.** > 0 = парсер OFF не справился = отбор LLM-корпуса |
| `additives_n` | INT32 | baseline для нашего «числа E-добавок» |
| `additives_tags` | LIST\<VARCHAR\> | конкретные E-номера; слабая разметка для smoke-теста |
| `allergens_tags` | LIST\<VARCHAR\> | baseline для аллергенов |
| `traces_tags` | LIST\<VARCHAR\> | «может содержать следы» — отдельно от состава |
| `ingredients_analysis_tags` | LIST\<VARCHAR\> | веган / вегетарианец / пальмовое масло |

Плюс два поля по сахару и подсластителям — их стоит взять как контрольные,
хотя число *разных форм сахара* OFF не считает (это остаётся нашей фичей):

| Колонка | Тип | Зачем |
|---|---|---|
| `with_sweeteners` | INT32 | флаг наличия подсластителей по версии OFF |
| `with_non_nutritive_sweeteners` | INT32 | некалорийные подсластители |

### Приоритизация выборки (2)

| Колонка | Тип | Зачем |
|---|---|---|
| `unique_scans_n` | INT32 | сколько раз продукт сканировали — реальные, а не мусорные записи |
| `popularity_key` | INT64 | готовый ключ популярности OFF |

Полезно, чтобы LLM-корпус состоял из продуктов, которые люди действительно
покупают, а не из случайных недозаполненных записей.

### Контроль качества данных (1)

| Колонка | Тип | Зачем |
|---|---|---|
| `data_quality_errors_tags` | LIST\<VARCHAR\> | OFF сам помечает битые записи — отбрасываем на входе |

---

## Что сознательно не берём

| Группа | Колонки | Почему |
|---|---|---|
| Изображения | `images`, `max_imgid`, `last_image_t`, `photographers` | вес большой, фронтенда нет, изображения не в скоупе |
| Люди | `creator`, `editors`, `correctors_tags`, `checkers_tags`, `informers_tags`, `last_editor`, `last_modified_by`, `owner`, `owner_fields` | персональные данные вкладчиков, проекту не нужны |
| Экология | `environmental_score_*` | отдельная тема, к составу не относится, отложено |
| Упаковка | `packaging*`, `packagings`, `packagings_complete` | не про состав |
| География производства | `emb_codes*`, `cities_tags`, `manufacturing_places*`, `purchase_places_tags`, `stores*` | к задаче не относится |
| CIQUAL / Agribalyse | `ciqual_food_name_tags`, `categories_properties`, `ingredients_without_ciqual_codes*` | нужны для расчёта экологии, не для состава |
| Микронутриенты тегами | `minerals_tags`, `vitamins_tags`, `nucleotides_tags`, `unknown_nutrients_tags` | детализация без применения в текущих майлстоунах |
| Прочее служебное | `states_tags`, `misc_tags`, `entry_dates_tags`, `last_edit_dates_tags`, `data_sources_tags`, `popularity_tags`, `compared_to_category`, `link`, `quantity`, `product_quantity*`, `serving_*`, `scans_n`, `data_quality_info_tags`, `data_quality_warnings_tags`, `nutrient_levels_tags`, `ingredients_percent_analysis`, `ingredients_with_*_percent_n`, `ingredients_from_palm_oil_n`, `new_additives_n` | шум для текущих задач; при необходимости добавляются одной строкой в выборку |

Принцип: Parquet колоночный, поэтому расширить список позже стоит дешево —
перечитываем только нужные колонки. Поэтому берём необходимое, а не «всё на всякий
случай»: узкая выборка держит `products_raw` компактным, а схему — обозримой.

---

## Черновик фильтра выборки для M1

Параметры из `.ai-factory/DESCRIPTION.md`; в код попадают через
pydantic-settings, а не константами.

```sql
CREATE OR REPLACE MACRO txt(l, lg) AS
  list_extract(list_filter(l, x -> x.lang = lg), 1).text;

WITH src AS (
  FROM read_parquet('data/food.parquet')
  SELECT *
  WHERE NOT obsolete
    AND NOT coalesce(no_nutrition_data, false)      -- только для аналитического корпуса
    AND len(coalesce(data_quality_errors_tags, [])) = 0
    AND len(list_filter(
          ingredients_text,
          x -> x.lang IN ('en','ru','de','fr','pl')
               AND x.text IS NOT NULL
               AND length(trim(x.text)) >= 10
        )) > 0
    AND list_has_any(categories_tags, [
          'en:snacks', 'en:sweet-snacks', 'en:salty-snacks',
          'en:biscuits-and-cakes', 'en:chocolates', 'en:confectioneries',
          'en:beverages', 'en:sweetened-beverages',
          'en:dairies', 'en:yogurts', 'en:cheeses',
          'en:breakfasts', 'en:breakfast-cereals'
        ])
)
FROM src SELECT count() AS analytic_corpus_size;
```

Список `categories_tags` — черновой. Точные теги подбираются по факту на дампе:
таксономия OFF иерархическая, и один продукт несёт сразу несколько уровней.
Это первая задача M1 после проверки двух ⚠️-пунктов выше.
