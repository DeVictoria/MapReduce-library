import collections.abc
import itertools
from abc import ABC, abstractmethod
import typing as tp

from . import Operation, TRowsGenerator, TRowsIterable, TRow


class Joiner(ABC):
    """Base class for joiners"""

    def __init__(self, suffix_a: str = '_1', suffix_b: str = '_2') -> None:
        self._a_suffix = suffix_a
        self._b_suffix = suffix_b
        self._duplicates: set[str] = set()

    def _do_join(self, keys: tp.Sequence[str], unpacked_a: list[TRow], rows_b: TRowsIterable) -> (
            tp.Generator)[TRow, None, bool]:
        is_empty = True
        for row1 in rows_b:
            is_empty = False
            ans = row1.copy()
            for k in row1:
                if k in self._duplicates:
                    ans[k + self._b_suffix] = ans[k]
            for row2 in unpacked_a:
                for key in row2:
                    if (key not in keys) and (key in ans):
                        self._duplicates.add(key)
                    if (key not in keys) and (key in self._duplicates):
                        ans[key + self._b_suffix] = ans[key]
                        del ans[key]
                        ans[key + self._a_suffix] = row2[key]
                    elif key not in keys:
                        ans[key] = row2[key]
                yield ans
                ans = row1.copy()
        return is_empty

    def _get_duplicates(self, rows: TRowsIterable, suffix: str) -> TRowsGenerator:
        for row in rows:
            ans = row.copy()
            for k in row:
                if k in self._duplicates:
                    ans[k + suffix] = ans[k]
                    del ans[k]
            yield ans

    @abstractmethod
    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        """
        :param keys: join keys
        :param rows_a: left table rows
        :param rows_b: right table rows
        """


class Join(Operation):
    class WrongJoinArgument(Exception):
        pass

    class NotSortedRows(Exception):
        pass

    class _End:
        """the helper class for determining the end of the table"""

    @staticmethod
    def check_sort(funk):  # type: ignore[no-untyped-def]
        def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
            k_prev = None
            reverse = None
            for k, g in funk(*args, **kwargs):
                if k is Join._End:
                    yield k, g
                    break
                if k_prev is None:
                    pass
                elif reverse is None:
                    if k > k_prev:
                        reverse = False
                    elif k < k_prev:
                        reverse = True
                else:
                    if (not reverse) and (k < k_prev):
                        raise Join.NotSortedRows('except sorted rows but take not sorted')
                    elif reverse and (k > k_prev):
                        raise Join.NotSortedRows('except sorted rows but take not sorted')
                k_prev = k
                yield k, g

        return wrapper

    def __init__(self, joiner: Joiner, keys: tp.Sequence[str] = tuple()):
        self._keys = keys
        self._joiner = joiner

    @staticmethod
    @check_sort
    def grouper(rows: TRowsIterable, keys):  # type: ignore[no-untyped-def]
        yield from itertools.groupby(rows, lambda x: [x[k] for k in keys])
        yield Join._End, Join._End

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        if len(args) == 0:
            raise Join.WrongJoinArgument("except 2 arguments: rows1, rows2, but take only 1")
        if not isinstance(args[0], collections.abc.Iterable):
            raise Join.WrongJoinArgument('except iterable rows2, but take not iterable')
        first = self.grouper(rows, self._keys)
        second = self.grouper(args[0], self._keys)
        view: tuple[int, int, TRowsIterable] | None = None
        k1, k2 = None, None
        g1: TRowsIterable = []
        g2: TRowsIterable = []
        while True:
            if view is None:
                if k1 is not Join._End:
                    k1, g1 = first.__next__()
                if k2 is not Join._End:
                    k2, g2 = second.__next__()
            else:
                if view[1] == 1:
                    k1, g1 = view[0], view[2]
                    if k2 is not Join._End:
                        k2, g2 = second.__next__()
                else:
                    k1, g1 = first.__next__()
                    if k2 is not Join._End:
                        k2, g2 = view[0], view[2]
                view = None

            if (k1 is Join._End) and (k2 is Join._End):
                break
            elif k1 is Join._End:
                yield from self._joiner(tuple(self._keys), [], g2)
            elif k2 is Join._End:
                yield from self._joiner(tuple(self._keys), g1, [])
            elif k1 == k2:
                yield from self._joiner(tuple(self._keys), g1, g2)
            elif k1 < k2:
                yield from self._joiner(tuple(self._keys), g1, [])
                view = (k2, 2, g2)
            else:
                yield from self._joiner(tuple(self._keys), [], g2)
                view = (k1, 1, g1)


# Joiners


class InnerJoiner(Joiner):
    """Join with inner strategy"""

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        unpacked_a = list(rows_a)
        if len(unpacked_a) > 0:
            yield from self._do_join(keys, unpacked_a, rows_b)


class OuterJoiner(Joiner):
    """Join with outer strategy"""

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        unpacked_a = list(rows_a)
        if len(unpacked_a) > 0:
            is_empty = yield from self._do_join(keys, unpacked_a, rows_b)
        else:
            is_empty = True
        if len(unpacked_a) == 0:
            yield from self._get_duplicates(rows_b, self._b_suffix)
        if is_empty and (len(unpacked_a) > 0):
            yield from self._get_duplicates(unpacked_a, self._a_suffix)


class LeftJoiner(Joiner):
    """Join with left strategy"""

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        unpacked_a = list(rows_a)
        if len(unpacked_a) > 0:
            is_empty = yield from self._do_join(keys, unpacked_a, rows_b)
        else:
            is_empty = True
        if is_empty and (len(unpacked_a) > 0):
            yield from self._get_duplicates(unpacked_a, self._a_suffix)


class RightJoiner(Joiner):
    """Join with right strategy"""

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        unpacked_a = list(rows_a)
        if len(unpacked_a) > 0:
            yield from self._do_join(keys, unpacked_a, rows_b)
        if len(unpacked_a) == 0:
            yield from self._get_duplicates(rows_b, self._b_suffix)
