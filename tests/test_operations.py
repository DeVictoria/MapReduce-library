import copy
import dataclasses
import typing as tp

import pytest
from pytest import approx

from compgraph import operations as ops


class _Key:
    def __init__(self, *args: str) -> None:
        self._items = args

    def __call__(self, d: tp.Mapping[str, tp.Any]) -> tuple[str, ...]:
        return tuple(str(d.get(key)) for key in self._items)


@dataclasses.dataclass
class MapCase:
    mapper: ops.Mapper
    data: list[ops.TRow]
    ground_truth: list[ops.TRow]
    cmp_keys: tuple[str, ...]
    mapper_item: int = 0
    mapper_ground_truth_items: tuple[int, ...] = (0,)


MAP_CASES = [
    MapCase(
        mapper=ops.Idf(['left', 'right'], 'idf'),
        data=[
            {'test_id': 1, 'left': 5, 'right': 10},
            {'test_id': 2, 'left': 60, 'right': 2},
            {'test_id': 3, 'left': 3, 'right': 15},
            {'test_id': 4, 'left': 100, 'right': 0.5},
            {'test_id': 5, 'left': 48, 'right': 15},
        ],
        ground_truth=[
            {'idf': -0.6931471805599453, 'left': 5, 'right': 10, 'test_id': 1},
            {'idf': 3.4011973816621555, 'left': 60, 'right': 2, 'test_id': 2},
            {'idf': -1.6094379124341003, 'left': 3, 'right': 15, 'test_id': 3},
            {'idf': 5.298317366548036, 'left': 100, 'right': 0.5, 'test_id': 4},
            {'idf': 1.1631508098056809, 'left': 48, 'right': 15, 'test_id': 5}],
        cmp_keys=('idf', 'left', 'right')
    ),
    MapCase(
        mapper=ops.Haversine('len', 'start', 'end'),
        data=[
            {'test_id': 1, 'start': [37.84870228730142, 55.73853974696249],
             'end': [37.8490418381989, 55.73832445777953]},
            {'test_id': 2, 'start': [37.524768467992544, 55.88785375468433],
             'end': [37.52415172755718, 55.88807155843824]},
            {'test_id': 3, 'start': [37.56963176652789, 55.846845586784184],
             'end': [37.57018438540399, 55.8469259692356]},
            {'test_id': 4, 'start': [37.41463478654623, 55.654487907886505],
             'end': [37.41442892700434, 55.654839486815035]},
            {'test_id': 5, 'start': [37.584684155881405, 55.78285809606314],
             'end': [37.58415022864938, 55.78177368734032]},

        ],
        ground_truth=[
            {'end': [37.8490418381989, 55.73832445777953], 'len': 0.03202394407224201,
             'start': [37.84870228730142, 55.73853974696249], 'test_id': 1},
            {'end': [37.52415172755718, 55.88807155843824], 'len': 0.045464188432109455,
             'start': [37.524768467992544, 55.88785375468433], 'test_id': 2},
            {'end': [37.57018438540399, 55.8469259692356], 'len': 0.035647728095922,
             'start': [37.56963176652789, 55.846845586784184], 'test_id': 3},
            {'end': [37.41442892700434, 55.654839486815035], 'len': 0.04118464617926384,
             'start': [37.41463478654623, 55.654487907886505], 'test_id': 4},
            {'end': [37.58415022864938, 55.78177368734032], 'len': 0.1251565805619792,
             'start': [37.584684155881405, 55.78285809606314], 'test_id': 5}
        ],
        cmp_keys=('len', 'start', 'end', 'test_id')
    ),
    MapCase(
        mapper=ops.ParseTime('time', 'weekday', 'hour'),
        data=[
            {'time': '20171020T112238'},
            {'time': '20171011T145553'},
            {'time': '20171020T090548'},
            {'time': '20171024T144101.879000'},
            {'time': '20171022T131828.330000'}
        ],
        ground_truth=[
            {'hour': 11, 'time': '20171020T112238', 'weekday': 'Fri'},
            {'hour': 14, 'time': '20171011T145553', 'weekday': 'Wed'},
            {'hour': 9, 'time': '20171020T090548', 'weekday': 'Fri'},
            {'hour': 14, 'time': '20171024T144101.879000', 'weekday': 'Tue'},
            {'hour': 13, 'time': '20171022T131828.330000', 'weekday': 'Sun'}],
        cmp_keys=('time', 'weekday', 'hour')
    ),
    MapCase(
        mapper=ops.TimeDiff('delta', 'enter_time', 'leave_time'),
        data=[
            {'leave_time': '20171024T144101', 'enter_time': '20171024T144059'},
            {'leave_time': '20171022T131828', 'enter_time': '20171022T131820'},
            {'leave_time': '20171014T134826.836000', 'enter_time': '20171014T134825.215000'},
            {'leave_time': '20171010T060609.897000', 'enter_time': '20171010T060608'},
            {'leave_time': '20171027T082600', 'enter_time': '20171027T082557.571000'}
        ],
        ground_truth=[{'delta': approx(0.000555, 0.1),
                       'enter_time': '20171024T144059',
                       'leave_time': '20171024T144101'},
                      {'delta': approx(0.000450, 0.1),
                       'enter_time': '20171014T134825.215000',
                       'leave_time': '20171014T134826.836000'},
                      {'delta': approx(0.000526, 0.1),
                       'enter_time': '20171010T060608',
                       'leave_time': '20171010T060609.897000'},
                      {'delta': approx(0.000674, 0.1),
                       'enter_time': '20171027T082557.571000',
                       'leave_time': '20171027T082600'},
                      {'delta': approx(0.00222, 0.1),
                       'enter_time': '20171022T131820',
                       'leave_time': '20171022T131828'}
                      ],
        cmp_keys=('delta', 'enter_time', 'leave_time')

    ),
]


@pytest.mark.parametrize('case', MAP_CASES)
def test_mapper(case: MapCase) -> None:
    mapper_data_row = copy.deepcopy(case.data[case.mapper_item])
    mapper_ground_truth_rows = [copy.deepcopy(case.ground_truth[i]) for i in case.mapper_ground_truth_items]

    key_func = _Key(*case.cmp_keys)

    mapper_result = case.mapper(mapper_data_row)
    assert isinstance(mapper_result, tp.Iterator)
    assert sorted(mapper_result, key=key_func) == sorted(mapper_ground_truth_rows, key=key_func)

    result = ops.Map(case.mapper)(iter(case.data))
    assert isinstance(result, tp.Iterator)
    assert sorted(result, key=key_func) == sorted(case.ground_truth, key=key_func)


def test_reducer_MeanSpeed() -> None:
    data: list[dict[str, tp.Any]] = [
        {'test_id': 1, 'len': 5, 'time': 10},
        {'test_id': 1, 'len': 60, 'time': 2},
        {'test_id': 1, 'len': 3, 'time': 15},
        {'test_id': 2, 'len': 100, 'time': 0.5},
        {'test_id': 2, 'len': 48, 'time': 15},
    ]
    ground_truth = [
        {'speed': 2.5185185185185186, 'test_id': 1},
        {'speed': 9.548387096774194, 'test_id': 2}]

    reducer_result = ops.MeanSpeed('speed', 'len', 'time')(('test_id',), iter(data[0:3]))
    assert isinstance(reducer_result, tp.Iterator)
    assert list(reducer_result) == ground_truth[0:1]

    result = ops.Reduce(ops.MeanSpeed('speed', 'len', 'time'), ('test_id',))(iter(data))
    assert isinstance(result, tp.Iterator)
    assert list(result) == ground_truth


def test_group_sort() -> None:
    data = [
        {'id': 1, 'count': 1, 'text': 'hm'},
        {'id': 1, 'count': 1, 'text': 'i'},
        {'id': 1, 'count': 1, 'text': 'it'},
        {'id': 1, 'count': 1, 'text': 'marrio'},
        {'id': 1, 'count': 3, 'text': 'me'},
        {'id': 2, 'count': 1, 'text': 'who'},
        {'id': 2, 'count': 2, 'text': 'am'},
        {'id': 2, 'count': 3, 'text': 'hi'},
        {'id': 2, 'count': 3, 'text': 'is'}]
    ground_truth = [
        {'id': 1, 'count': 3, 'text': 'me'},
        {'id': 1, 'count': 1, 'text': 'marrio'},
        {'id': 1, 'count': 1, 'text': 'it'},
        {'id': 1, 'count': 1, 'text': 'i'},
        {'id': 1, 'count': 1, 'text': 'hm'},
        {'id': 2, 'count': 3, 'text': 'is'},
        {'id': 2, 'count': 3, 'text': 'hi'},
        {'id': 2, 'count': 2, 'text': 'am'},
        {'id': 2, 'count': 1, 'text': 'who'}
    ]

    result = ops.ExternalSort(['count', 'text'], reverse=True, group_keys=('id',))(iter(data))
    assert isinstance(result, tp.Iterator)
    assert list(result) == ground_truth


def test_sort() -> None:
    data = [
        {'count': 1, 'text': 'hm'},
        {'count': 1, 'text': 'i'},
        {'count': 1, 'text': 'it'},
        {'count': 1, 'text': 'marrio'},
        {'count': 1, 'text': 'me'},
        {'count': 1, 'text': 'who'},
        {'count': 2, 'text': 'am'},
        {'count': 3, 'text': 'hi'},
        {'count': 3, 'text': 'is'}]
    ground_truth = [
        {'count': 3, 'text': 'is'},
        {'count': 3, 'text': 'hi'},
        {'count': 2, 'text': 'am'},
        {'count': 1, 'text': 'who'},
        {'count': 1, 'text': 'me'},
        {'count': 1, 'text': 'marrio'},
        {'count': 1, 'text': 'it'},
        {'count': 1, 'text': 'i'},
        {'count': 1, 'text': 'hm'},

    ]

    result = ops.ExternalSort(['count', 'text'], reverse=True)(iter(data))
    assert isinstance(result, tp.Iterator)
    assert list(result) == ground_truth


def test_join() -> None:
    data_left = [
        {'player': 1, 'duplicate': 'b'},
        {'player': 2, 'duplicate': 'c'}
    ]
    data_right = [
        {'player': 0, 'duplicate': 1},
        {'player': 1, 'duplicate': 2},
    ]
    ground_truth = [
        {'player': 0, 'duplicate': 1},
        {'player': 1, 'duplicate_1': 'b', 'duplicate_2': 2},
        {'player': 2, 'duplicate_1': 'c'}
    ]

    result = ops.Join(ops.OuterJoiner(suffix_a='_1', suffix_b='_2'), ['player'])(iter(data_left), iter(data_right))
    assert isinstance(result, tp.Iterator)
    assert list(result) == ground_truth
