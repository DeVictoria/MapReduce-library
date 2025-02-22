## Compgraph

Библиотека для создания и использования вычислительных графов поверх моей реализации
[MapReduce](https://ru.wikipedia.org/wiki/MapReduce) (модель распределённых вычислений от компании Google).

Новые вычислительные графы строятся в `algoritms.py`,
а новые для них операции (Map, Reduce, Join, Sort) в `operations.py`.

### Installing

```bash
# Устанавливаем библиотеку compgraph
$ pip install -e compgraph --force-reinstall
```

### Running the examples tests

В модуле examples находится скрипты для запуска четырёх вычислительных графов:

1. word_count: Требуется для каждого из слов, встречающихся в колонке text,
   посчитать количество вхождений во всю таблицу в сумме.

2. invert_index_graph: Структура данных, которая для каждого слова хранит
   список документов, в котором оно встречается, отсортированный в порядке релевантности по метрике tf-idf.

3. pmi_graph: Для каждого документа посчитать топ-10 слов, наиболее характерных для него

4. test_yandex_maps: Средняя скорость движения по городу от часа и дня недели

### Running the custom tests

Для тестов графа, создайте свой тест в файле `test_graph.py`,
а для тестов операций - `test_operations.py`.

Запустить тесты можно с помощью команды:

```bash
$ pytest compgraph
```
