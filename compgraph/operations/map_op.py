import math
import string
import re
import typing as tp
from abc import ABC, abstractmethod
import calendar

from . import Operation, TRow, TRowsGenerator, TRowsIterable, parse_datetime


class Mapper(ABC):
    """Base class for mappers"""

    @abstractmethod
    def __call__(self, row: TRow) -> TRowsGenerator:
        """
        :param row: one table row
        """


class Map(Operation):
    def __init__(self, mapper: Mapper) -> None:
        self._mapper = mapper

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for i in rows:
            yield from self._mapper(i)


# Mappers

class DummyMapper(Mapper):
    """Yield exactly the row passed"""

    def __call__(self, row: TRow) -> TRowsGenerator:
        yield row


class FilterPunctuation(Mapper):
    """Left only non-punctuation symbols"""

    def __init__(self, column: str):
        """
        :param column: name of column to process
        """
        self._column = column

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self._column] = row[self._column].translate(str.maketrans('', '', string.punctuation))
        yield row


class LowerCase(Mapper):
    """Replace column value with value in lower case"""

    def __init__(self, column: str):
        """
        :param column: name of column to process
        """
        self._column = column

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self._column] = row[self._column].lower()
        yield row


class Split(Mapper):
    """Split row on multiple rows by separator"""

    def __init__(self, column: str, separator: str | None = None) -> None:
        """
        :param column: name of column to split
        :param separator: string to separate by
        """
        self._column = column
        self._separator = separator

    def __call__(self, row: TRow) -> TRowsGenerator:
        pattern = r'[^\s]+' if self._separator is None else fr'[^{self._separator}]*'
        check = True
        for i in re.finditer(pattern, row[self._column]):
            if i[0] != '':
                check = False
                ans = row.copy()
                ans[self._column] = i[0]
                yield ans
        if check:
            ans = row.copy()
            ans[self._column] = ''
            yield ans


class Product(Mapper):
    """Calculates product of multiple columns"""

    def __init__(self, columns: tp.Sequence[str], result_column: str = 'product') -> None:
        """
        :param columns: column names to product
        :param result_column: column name to save product in
        """
        self._columns = columns
        self._result_column = result_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        count = 1
        for word in self._columns:
            count *= row[word]
        row[self._result_column] = count
        yield row


class Idf(Mapper):
    """Calculates Idf columns"""

    def __init__(self, columns: tp.Sequence[str], result_column: str = 'idf') -> None:
        """
        :param columns: column names to product
        :param result_column: column name to save product in
        """
        self._columns = columns
        self._result_column = result_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self._result_column] = math.log(row[self._columns[0]] / row[self._columns[1]])
        yield row


class Filter(Mapper):
    """Remove records that don't satisfy some condition"""

    def __init__(self, condition: tp.Callable[[TRow], bool]) -> None:
        """
        :param condition: if condition is not true - remove record
        """
        self._condition = condition

    def __call__(self, row: TRow) -> TRowsGenerator:
        if self._condition(row):
            yield row


class Project(Mapper):
    """Leave only mentioned columns"""

    def __init__(self, columns: tp.Sequence[str]) -> None:
        """
        :param columns: names of columns
        """
        self._columns = columns

    def __call__(self, row: TRow) -> TRowsGenerator:
        ans = {}
        for key in self._columns:
            ans[key] = row[key]
        yield ans


class Haversine(Mapper):
    """haversine formula"""

    EARTH_RADIUS_KM = 6373

    def __init__(self, name: str, start: str, end: str) -> None:
        """
        :param name: name of result column
        :param start: names of start
        :param end: names of end
        """
        self._name = name
        self._start = start
        self._end = end

    def __call__(self, row: TRow) -> TRowsGenerator:
        lat1 = row[self._start][1] / 180 * math.pi
        lon1 = row[self._start][0] / 180 * math.pi
        lat2 = row[self._end][1] / 180 * math.pi
        lon2 = row[self._end][0] / 180 * math.pi

        leng = math.acos(math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * math.cos(
            lon1 - lon2)) * Haversine.EARTH_RADIUS_KM
        row[self._name] = leng
        yield row


class ParseTime(Mapper):
    """parse_time to weekday and hour """

    def __init__(self, time: str, weekday: str, hour: str) -> None:
        """
        :param time: name of column to parse
        :param weekday: names of weekday res
        :param hour: names of hour res
        """
        self._time = time
        self._weekday = weekday
        self._hour = hour

    def __call__(self, row: TRow) -> TRowsGenerator:
        date = parse_datetime(row[self._time])
        row[self._weekday] = list(calendar.day_abbr)[date.weekday()]
        row[self._hour] = date.hour
        yield row


class TimeDiff(Mapper):
    """get difference between 2 date """

    def __init__(self, name: str, first_time: str, second_time: str) -> None:
        """
        :param name: name of column to parse
        :param first_time: name of first time
        :param second_time: name of second time
        """
        self._name = name
        self._first_time = first_time
        self._second_time = second_time

    def __call__(self, row: TRow) -> TRowsGenerator:
        date1 = parse_datetime(row[self._first_time])
        date2 = parse_datetime(row[self._second_time])
        row[self._name] = abs((date2 - date1).total_seconds()) / 3600
        yield row
