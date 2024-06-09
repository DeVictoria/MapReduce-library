import heapq
import itertools
from abc import ABC, abstractmethod
import typing as tp
from collections import defaultdict

from . import Operation, TRow, TRowsGenerator, TRowsIterable


class Reducer(ABC):
    """Base class for reducers"""

    @abstractmethod
    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        """
        :param rows: table rows
        """


class Reduce(Operation):
    def __init__(self, reducer: Reducer, keys: tp.Sequence[str] = tuple()) -> None:
        self._reducer = reducer
        self._keys = keys

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for key, group in itertools.groupby(rows, lambda x: [x[k] for k in self._keys]):
            yield from self._reducer(tuple(self._keys), group)


# Reducers

class FirstReducer(Reducer):
    """Yield only first row from passed ones"""

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        for row in rows:
            yield row
            break


class TopN(Reducer):
    """Calculate top N by value"""

    def __init__(self, column: str, n: int) -> None:
        """
        :param column: column name to get top by
        :param n: number of top values to extract
        """
        self._column_max = column
        self._n = n

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        ans: list[tp.Any] = []
        count = 0
        for row in rows:
            count += 1
            heapq.heappush(ans, (row[self._column_max], count, row))
            if len(ans) > self._n:
                heapq.heappop(ans)
        for el in ans:
            yield el[-1]


class TermFrequency(Reducer):
    """Calculate frequency of values in column"""

    def __init__(self, words_column: str, result_column: str = 'tf') -> None:
        """
        :param words_column: name for column with words
        :param result_column: name for result column
        """
        self._words_column = words_column
        self._result_column = result_column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        add_word: defaultdict[str, int] = defaultdict(int)
        count = 0
        ans = []
        for row in rows:
            count += 1
            if add_word[row[self._words_column]] == 0:
                add_word[row[self._words_column]] = 1
                new_row = dict()
                for name in group_key:
                    new_row[name] = row[name]
                new_row[self._words_column] = row[self._words_column]
                ans.append(new_row)
            else:
                add_word[row[self._words_column]] += 1
        for v in ans:
            v[self._result_column] = add_word[v[self._words_column]] / count
            yield v


class Count(Reducer):
    """
    Count records by key
    Example for group_key=('a',) and column='d'
        {'a': 1, 'b': 5, 'c': 2}
        {'a': 1, 'b': 6, 'c': 1}
        =>
        {'a': 1, 'd': 2}
    """

    def __init__(self, column: str) -> None:
        """
        :param column: name for result column
        """
        self._column = column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        count = 0
        last_row: dict[str, tp.Any] = dict()
        for row in rows:
            count += 1
            last_row = row
        ans = {self._column: count}
        for k in group_key:
            ans[k] = last_row[k]
        yield ans


class Sum(Reducer):
    """
    Sum values aggregated by key
    Example for key=('a',) and column='b'
        {'a': 1, 'b': 2, 'c': 4}
        {'a': 1, 'b': 3, 'c': 5}
        =>
        {'a': 1, 'b': 5}
    """

    def __init__(self, column: str) -> None:
        """
        :param column: name for sum column
        """
        self._column = column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        count = 0
        last_row: dict[str, tp.Any] = dict()
        for row in rows:
            count += row[self._column]
            last_row = row
        ans = {self._column: count}
        for k in group_key:
            ans[k] = last_row[k]
        yield ans


class MeanSpeed(Reducer):
    """
    finds the mean speed
    of several objects according to the parameters:
    time, distance
    """

    def __init__(self, name: str, distance: str, time: str) -> None:
        """
        :param name: name for answer column
        :param distance: name for length column
        :param time: name for time column
        """
        self._name = name
        self._distance = distance
        self._time = time

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        ans: TRow = {}
        sum_time = 0
        sum_distance = 0
        for row in rows:
            if len(ans) == 0:
                for key in group_key:
                    ans[key] = row[key]
            sum_distance += row[self._distance]
            sum_time += row[self._time]
        ans[self._name] = sum_distance / sum_time
        yield ans
