from compgraph import algorithms as alg
from pytest import approx


def test_sequent_calls1() -> None:
    rows = [
        {'doc_id': 1, 'text': 'hello, little world'},
        {'doc_id': 2, 'text': 'little'},
        {'doc_id': 3, 'text': 'little little little'},
        {'doc_id': 4, 'text': 'little? hello little world'},
        {'doc_id': 5, 'text': 'HELLO HELLO! WORLD...'},
        {'doc_id': 6, 'text': 'world? world... world!!! WORLD!!! HELLO!!!'}
    ]

    expected = [
        {'doc_id': 1, 'text': 'hello', 'tf_idf': approx(0.1351, 0.001)},
        {'doc_id': 1, 'text': 'world', 'tf_idf': approx(0.1351, 0.001)},

        {'doc_id': 2, 'text': 'little', 'tf_idf': approx(0.4054, 0.001)},

        {'doc_id': 3, 'text': 'little', 'tf_idf': approx(0.4054, 0.001)},

        {'doc_id': 4, 'text': 'hello', 'tf_idf': approx(0.1013, 0.001)},
        {'doc_id': 4, 'text': 'little', 'tf_idf': approx(0.2027, 0.001)},

        {'doc_id': 5, 'text': 'hello', 'tf_idf': approx(0.2703, 0.001)},
        {'doc_id': 5, 'text': 'world', 'tf_idf': approx(0.1351, 0.001)},

        {'doc_id': 6, 'text': 'world', 'tf_idf': approx(0.3243, 0.001)}
    ]

    graph = alg.inverted_index_graph('texts')

    result1 = graph.run(texts=lambda: iter(rows))
    result2 = graph.run(texts=lambda: iter(rows))

    assert list(result1) == expected
    assert list(result2) == expected


def test_sequent_calls2() -> None:
    rows1 = [
        {'doc_id': 1, 'text': 'hello, with world'},
        {'doc_id': 2, 'text': 'little'},
        {'doc_id': 3, 'text': 'little little little'},
        {'doc_id': 4, 'text': 'with& hello little world'},
        {'doc_id': 5, 'text': 'HELLO with! WORLD...'},
        {'doc_id': 6, 'text': 'world? world... world!!! Years!!'},
        {'doc_id': 7, 'text': 'Later Five Years Later...'},
        {'doc_id': 8, 'text': 'world?Years world..Years Later'}
    ]

    expected1 = [
        {'doc_id': 3, 'text': 'little', 'pmi': 1.2039728043259361},
        {'doc_id': 6, 'text': 'world', 'pmi': 1.2039728043259361},
        {'doc_id': 7, 'text': 'later', 'pmi': 1.6094379124341003},
        {'doc_id': 8, 'text': 'worldyears', 'pmi': 1.6094379124341003}
    ]

    rows2 = [
        {'doc_id': 1, 'text': 'hello, little world'},
        {'doc_id': 2, 'text': 'little'},
        {'doc_id': 3, 'text': 'little little little'},
        {'doc_id': 4, 'text': 'little? hello little world'},
        {'doc_id': 5, 'text': 'HELLO HELLO! WORLD...'},
        {'doc_id': 6, 'text': 'world? world... world!!! WORLD!!! HELLO!!! HELLO!!!!!!!'}
    ]

    expected2 = [
        {'doc_id': 3, 'text': 'little', 'pmi': approx(0.9555, 0.001)},
        {'doc_id': 4, 'text': 'little', 'pmi': approx(0.9555, 0.001)},
        {'doc_id': 5, 'text': 'hello', 'pmi': approx(1.1786, 0.001)},
        {'doc_id': 6, 'text': 'world', 'pmi': approx(0.7731, 0.001)},
        {'doc_id': 6, 'text': 'hello', 'pmi': approx(0.0800, 0.001)},
    ]

    graph = alg.pmi_graph('texts')

    result1 = graph.run(texts=lambda: iter(rows1))
    assert list(result1) == expected1

    result2 = graph.run(texts=lambda: iter(rows2))
    assert list(result2) == expected2
