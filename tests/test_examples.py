import json
import os

from click.testing import CliRunner

from examples import run_word_count, run_pmi_graph, run_inverted_index_graph, run_test_yandex_maps

Runner = CliRunner()


def test_run_word_count() -> None:
    docs = [
        {'doc_id': 1, 'text': 'hi hi hi, am is...'},
        {'doc_id': 2, 'text': 'HM, WHO iS i AM'},
        {'doc_id': 3, 'text': 'it Is me, MARRIO!'}
    ]

    expected = [
        {'count': 1, 'text': 'hm'},
        {'count': 1, 'text': 'i'},
        {'count': 1, 'text': 'it'},
        {'count': 1, 'text': 'marrio'},
        {'count': 1, 'text': 'me'},
        {'count': 1, 'text': 'who'},
        {'count': 2, 'text': 'am'},
        {'count': 3, 'text': 'hi'},
        {'count': 3, 'text': 'is'}
    ]

    input_filepath = 'input_f'
    with open(input_filepath, 'w') as f:
        for i in docs:
            json.dump(i, f)
            f.write('\n')

    output_filepath = 'output_f'
    with open(output_filepath, 'w') as f:
        pass

    Runner.invoke(run_word_count.main, input_filepath + ' ' + output_filepath)

    with open(output_filepath) as f:
        ans_list = json.load(f)

    assert ans_list == expected


def test_run_inverted_index_graph() -> None:
    docs = [
        {'doc_id': 1, 'text': 'hello, with world'},
        {'doc_id': 2, 'text': 'little'},
        {'doc_id': 3, 'text': 'little little little'},
        {'doc_id': 4, 'text': 'with& hello little world'},
        {'doc_id': 5, 'text': 'HELLO with! WORLD...'},
        {'doc_id': 6, 'text': 'world? world... world!!! Years!!'},
        {'doc_id': 5, 'text': 'Five Years Later...'},
        {'doc_id': 6, 'text': 'world? world..Years Later'}
    ]

    expected = [
        {'doc_id': 1, 'text': 'hello', 'tf_idf': 0.32694308433724206},
        {'doc_id': 1, 'text': 'with', 'tf_idf': 0.32694308433724206},
        {'doc_id': 1, 'text': 'world', 'tf_idf': 0.23104906018664842},
        {'doc_id': 2, 'text': 'little', 'tf_idf': 0.9808292530117262},
        {'doc_id': 3, 'text': 'little', 'tf_idf': 0.9808292530117262},
        {'doc_id': 4, 'text': 'hello', 'tf_idf': 0.24520731325293155},
        {'doc_id': 4, 'text': 'little', 'tf_idf': 0.24520731325293155},
        {'doc_id': 4, 'text': 'with', 'tf_idf': 0.24520731325293155},
        {'doc_id': 4, 'text': 'world', 'tf_idf': 0.17328679513998632},
        {'doc_id': 5, 'text': 'five', 'tf_idf': 0.3465735902799726},
        {'doc_id': 5, 'text': 'hello', 'tf_idf': 0.16347154216862103},
        {'doc_id': 5, 'text': 'later', 'tf_idf': 0.23104906018664842},
        {'doc_id': 5, 'text': 'with', 'tf_idf': 0.16347154216862103},
        {'doc_id': 5, 'text': 'years', 'tf_idf': 0.23104906018664842},
        {'doc_id': 6, 'text': 'later', 'tf_idf': 0.19804205158855578},
        {'doc_id': 6, 'text': 'world', 'tf_idf': 0.39608410317711157},
        {'doc_id': 6, 'text': 'worldyears', 'tf_idf': 0.29706307738283366},
        {'doc_id': 6, 'text': 'years', 'tf_idf': 0.19804205158855578}
    ]

    input_filepath = 'input_f'
    with open(input_filepath, 'w') as f:
        for i in docs:
            json.dump(i, f)
            f.write('\n')

    output_filepath = 'output_f'
    with open(output_filepath, 'w') as f:
        pass

    Runner.invoke(run_inverted_index_graph.main, input_filepath + ' ' + output_filepath)

    with open(output_filepath) as f:
        ans_list = json.load(f)

    os.remove(input_filepath)
    os.remove(output_filepath)

    assert ans_list == expected


def test_run_pmi_graph() -> None:
    docs = [
        {'doc_id': 1, 'text': 'hello, with world'},
        {'doc_id': 2, 'text': 'little'},
        {'doc_id': 3, 'text': 'little little little'},
        {'doc_id': 4, 'text': 'with& hello little world'},
        {'doc_id': 5, 'text': 'HELLO with! WORLD...'},
        {'doc_id': 6, 'text': 'world? world... world!!! Years!!'},
        {'doc_id': 7, 'text': 'Later Five Years Later...'},
        {'doc_id': 8, 'text': 'world?Years world..Years Later'}
    ]

    expected = [
        {'doc_id': 3, 'text': 'little', 'pmi': 1.2039728043259361},
        {'doc_id': 6, 'text': 'world', 'pmi': 1.2039728043259361},
        {'doc_id': 7, 'text': 'later', 'pmi': 1.6094379124341003},
        {'doc_id': 8, 'text': 'worldyears', 'pmi': 1.6094379124341003}
    ]

    input_filepath = 'input_f'
    with open(input_filepath, 'w') as f:
        for i in docs:
            json.dump(i, f)
            f.write('\n')

    output_filepath = 'output_f'
    with open(output_filepath, 'w') as f:
        pass

    Runner.invoke(run_pmi_graph.main, input_filepath + ' ' + output_filepath)

    with open(output_filepath) as f:
        ans_list = json.load(f)

    os.remove(input_filepath)
    os.remove(output_filepath)

    assert ans_list == expected


def test_run_yandex_maps_graph() -> None:
    docs1 = [
        {'leave_time': '20171022T131828.330000', 'enter_time': '20171022T131820.842000',
         'edge_id': 8414926848168493057},
        {'leave_time': '20171011T145553.040000', 'enter_time': '20171011T145551.957000',
         'edge_id': 8414926848168493057},
        {'leave_time': '20171020T090548.939000', 'enter_time': '20171020T090547.463000',
         'edge_id': 1293255682152955894},
        {'leave_time': '20171024T144101.879000', 'enter_time': '20171024T144059.102000',
         'edge_id': 1293255682152955894},
        {'leave_time': '20171022T131828.330000', 'enter_time': '20171022T131820.842000',
         'edge_id': 5342768494149337085},
        {'leave_time': '20171014T134826.836000', 'enter_time': '20171014T134825.215000',
         'edge_id': 5342768494149337085},
        {'leave_time': '20171024T144101.879000', 'enter_time': '20171024T144059.102000',
         'edge_id': 5342768494149337085},
        {'leave_time': '20171027T082600.201000', 'enter_time': '20171027T082557.571000', 'edge_id': 7639557040160407543}
    ]

    docs2 = [
        {'start': [37.84870228730142, 55.73853974696249], 'end': [37.8490418381989, 55.73832445777953],
         'edge_id': 8414926848168493057},
        {'start': [37.524768467992544, 55.88785375468433], 'end': [37.52415172755718, 55.88807155843824],
         'edge_id': 5342768494149337085},
        {'start': [37.56963176652789, 55.846845586784184], 'end': [37.57018438540399, 55.8469259692356],
         'edge_id': 5123042926973124604},
        {'start': [37.41463478654623, 55.654487907886505], 'end': [37.41442892700434, 55.654839486815035],
         'edge_id': 5726148664276615162},
        {'start': [37.584684155881405, 55.78285809606314], 'end': [37.58415022864938, 55.78177368734032],
         'edge_id': 451916977441439743},
        {'start': [37.736429711803794, 55.62696328852326], 'end': [37.736344216391444, 55.626937723718584],
         'edge_id': 7639557040160407543},
        {'start': [37.83196756616235, 55.76662947423756], 'end': [37.83191015012562, 55.766647034324706],
         'edge_id': 1293255682152955894}
    ]

    input_filepath1 = 'input1_f'
    with open(input_filepath1, 'w') as f:
        for i in docs1:
            json.dump(i, f)
            f.write('\n')

    input_filepath2 = 'input2_f'
    with open(input_filepath2, 'w') as f:
        for i in docs2:
            json.dump(i, f)
            f.write('\n')

    output_filepath = 'output_f'
    with open(output_filepath, 'w') as f:
        pass

    expected = [
        {'weekday': 'Fri', 'hour': 8, 'speed': 8.316328881523264},
        {'weekday': 'Fri', 'hour': 9, 'speed': 9.973211735667919},
        {'weekday': 'Sat', 'hour': 13, 'speed': 100.96920318050218},
        {'weekday': 'Sun', 'hour': 13, 'speed': 18.626954928930637},
        {'weekday': 'Tue', 'hour': 14, 'speed': 32.11947044966508},
        {'weekday': 'Wed', 'hour': 14, 'speed': 106.45078361964103}
    ]

    Runner.invoke(run_test_yandex_maps.main, input_filepath1 + ' ' + input_filepath2 + ' ' + output_filepath)

    with open(output_filepath) as f:
        ans_list = json.load(f)

    os.remove(input_filepath1)
    os.remove(input_filepath2)
    os.remove(output_filepath)

    assert ans_list == expected
