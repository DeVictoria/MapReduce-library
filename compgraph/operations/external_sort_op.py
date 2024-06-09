import itertools
import typing as tp

from multiprocessing import Pipe, Process, connection
from operator import itemgetter

from . import Operation, TRowsGenerator, TRowsIterable


def do_sort(endpoint: connection.Connection, keys: tuple[str, ...], reverse: bool = False) -> None:
    rows = []
    while True:
        row = endpoint.recv()
        if row is None:
            break
        rows.append(row)
    rows.sort(key=itemgetter(*keys), reverse=reverse)
    for row in rows:
        endpoint.send(row)
    endpoint.send(None)


class ExternalSort(Operation):
    """
    In order to not account materialization during sorting in main process memory consumption, we delegate
    sorting to a separate process.
    This class illustrates cross-process streaming.
    """

    def __init__(self, keys: tp.Sequence[str], *, reverse: bool = False, group_keys: tp.Sequence[str] | None = None):
        self._keys = keys
        self._reverse = reverse
        self._group_keys = group_keys
        if group_keys is None:
            self._group_keys = []

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for key, group in itertools.groupby(rows, lambda x: [x[k] for k in self._group_keys]):  # type: ignore
            local_endpoint, remote_endpoint = Pipe()
            process = Process(target=do_sort, args=(remote_endpoint, self._keys, self._reverse))
            process.start()
            row_count_before = 0
            for row in group:
                local_endpoint.send(row)
                row_count_before += 1
            local_endpoint.send(None)
            row_count_after = 0
            while True:
                local_endpoint_row = local_endpoint.recv()
                if local_endpoint_row is None:
                    break
                yield local_endpoint_row
                row_count_after += 1
            assert row_count_before == row_count_after
            process.join()
