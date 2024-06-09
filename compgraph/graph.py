import copy
import json
import typing as tp

from . import operations as ops
from .operations import map_op
from .operations import reduce_op
from .operations import ExternalSort
from .operations import join_op


class Graph:
    """Computational graph implementation"""

    def __init__(self) -> None:
        self.__op: list[ops.Operation] = list()
        self._join_graphs: list[Graph] = list()

    @staticmethod
    def graph_from(name: str, from_file: bool) -> 'Graph':
        """Construct new graph which reads data from row iterator (in form of sequence of Rows
        from 'kwargs' passed to 'run' method) into graph data-flow
        Use ops.ReadIterFactory
        :param name: name of kwarg to use as data source
        :param from_file: is graph from file or from iter
        """
        if from_file:
            return Graph.graph_from_file(name, json.loads)
        else:
            return Graph.graph_from_iter(name)

    @staticmethod
    def graph_from_iter(name: str) -> 'Graph':
        """Construct new graph which reads data from row iterator (in form of sequence of Rows
        from 'kwargs' passed to 'run' method) into graph data-flow
        Use ops.ReadIterFactory
        :param name: name of kwarg to use as data source
        """
        graph = Graph()
        graph.__op.append(ops.ReadIterFactory(name))

        return graph

    @staticmethod
    def graph_from_file(filename: str, parser: tp.Callable[[str], ops.TRow]) -> 'Graph':
        """Construct new graph extended with operation for reading rows from file
        Use ops.Read
        :param filename: filename to read from
        :param parser: parser from string to Row
        """
        graph = Graph()
        graph.__op.append(ops.Read(filename, parser))

        return graph

    def map(self, mapper: map_op.Mapper) -> 'Graph':
        """Construct new graph extended with map operation with particular mapper
        :param mapper: mapper to use
        """
        new_self = copy.deepcopy(self)
        new_self.__op.append(map_op.Map(mapper))
        return new_self

    def reduce(self, reducer: reduce_op.Reducer, keys: tp.Sequence[str]) -> 'Graph':
        """Construct new graph extended with reduce operation with particular reducer
        :param reducer: reducer to use
        :param keys: keys for grouping
        """
        new_self = copy.deepcopy(self)
        new_self.__op.append(reduce_op.Reduce(reducer, keys))
        return new_self

    def sort(self, keys: tp.Sequence[str], reverse: bool = False,
             group_keys: tp.Sequence[str] | None = None) -> 'Graph':
        """Construct new graph extended with sort operation
        :param keys: sorting keys (typical is tuple of strings)
        :param reverse: reverse sorting or not
        :param group_keys: keys for grouping
        """
        new_self = copy.deepcopy(self)
        new_self.__op.append(ExternalSort(keys, reverse=reverse, group_keys=group_keys))
        return new_self

    def join(self, joiner: join_op.Joiner, join_graph: 'Graph', keys: tp.Sequence[str]) -> 'Graph':
        """Construct new graph extended with join operation with another graph
        :param joiner: join strategy to use
        :param join_graph: other graph to join with
        :param keys: keys for grouping
        """
        new_self = copy.deepcopy(self)
        new_self.__op.append(join_op.Join(joiner, keys))
        new_self._join_graphs.append(copy.deepcopy(join_graph))
        return new_self

    def run(self, **kwargs: tp.Any) -> ops.TRowsIterable:
        """Single method to start execution; data sources passed as kwargs"""
        iter_table = self.__op[0](**kwargs)
        count = 0
        for op in self.__op[1:]:
            if type(op) is join_op.Join:
                iter_table = op(iter_table, self._join_graphs[count].run(**kwargs))
                count += 1
            else:
                iter_table = op(iter_table)
        return iter_table
