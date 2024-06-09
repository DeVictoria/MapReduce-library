from . import Graph
from . import operations as ops


def word_count_graph(input_stream_name: str, text_column: str = 'text', count_column: str = 'count',
                     from_file: bool = False) -> Graph:
    """Constructs graph which counts words in text_column of all rows passed"""
    return (Graph.graph_from(input_stream_name, from_file)
            .map(ops.FilterPunctuation(text_column))
            .map(ops.LowerCase(text_column))
            .map(ops.Split(text_column))
            .sort([text_column])
            .reduce(ops.Count(count_column), [text_column])
            .sort([count_column, text_column]))


def inverted_index_graph(input_stream_name: str, doc_column: str = 'doc_id', text_column: str = 'text',
                         result_column: str = 'tf_idf', from_file: bool = False) -> Graph:
    """Constructs graph which calculates td-idf for every word/document pair"""

    split_word = (Graph.graph_from(input_stream_name, from_file)
                  .map(ops.FilterPunctuation(text_column))
                  .map(ops.LowerCase(text_column))
                  .map(ops.Split(text_column)))

    count_rows_column: str = 'count_rows'
    count_docs = (Graph.graph_from(input_stream_name, from_file)
                  .reduce(ops.Count(count_rows_column), []))

    count_rows_with_text_column: str = 'count_rows_with_text'
    count_idf = (split_word
                 .sort([doc_column, text_column])
                 .reduce(ops.FirstReducer(), [doc_column, text_column])
                 .sort([text_column])
                 .reduce(ops.Count(count_rows_with_text_column), [text_column])
                 .join(ops.InnerJoiner(), count_docs, [])
                 .map(ops.Idf([count_rows_column, count_rows_with_text_column])))

    tf_column: str = 'tf'
    idf_column: str = 'idf'
    count_tf = (split_word
                .sort([doc_column])
                .reduce(ops.TermFrequency(text_column), [doc_column])
                .sort([text_column])
                .join(ops.InnerJoiner(), count_idf, [text_column])
                .map(ops.Product([tf_column, idf_column], result_column))
                .map(ops.Project([doc_column, text_column, result_column]))
                .sort([text_column, result_column], reverse=True)
                .reduce(ops.TopN(result_column, 3), [text_column])
                .sort([doc_column, text_column]))

    return count_tf


def pmi_graph(input_stream_name: str, doc_column: str = 'doc_id', text_column: str = 'text',
              result_column: str = 'pmi', from_file: bool = False) -> Graph:
    """Constructs graph which gives for every document the top 10 words ranked by pointwise mutual information"""
    split_word = (Graph.graph_from(input_stream_name, from_file)
                  .map(ops.FilterPunctuation(text_column))
                  .map(ops.LowerCase(text_column))
                  .map(ops.Split(text_column))
                  .sort([doc_column, text_column]))

    count_w_column: str = 'count_w'
    filter_words = (split_word
                    .map(ops.Filter(lambda row: len(row[text_column]) > 4))
                    .sort([doc_column, text_column])
                    .reduce(ops.Count(count_w_column), [doc_column, text_column])
                    .map(ops.Filter(lambda row: row[count_w_column] >= 2))
                    .map(ops.Project([doc_column, text_column])))

    correct_words = (split_word
                     .join(ops.RightJoiner(), filter_words, [doc_column, text_column]))

    all_f_column: str = 'all_F'
    count_in_table = (correct_words
                      .sort([text_column])
                      .reduce(ops.TermFrequency(text_column, all_f_column), [])
                      .map(ops.Project([text_column, all_f_column])))

    doc_f_column: str = 'doc_F'
    count_in_doc = (correct_words
                    .sort([doc_column, text_column])
                    .reduce(ops.TermFrequency(text_column, doc_f_column), [doc_column])
                    .sort([text_column]))

    pmi = (count_in_doc
           .join(ops.InnerJoiner(), count_in_table, [text_column])
           .map(ops.Idf([doc_f_column, all_f_column], result_column))
           .sort([doc_column, result_column], reverse=True)
           .reduce(ops.TopN(result_column, 3), [doc_column])
           .sort([doc_column])
           .map(ops.Project([doc_column, text_column, result_column]))
           .sort([result_column], reverse=True, group_keys=[doc_column]))

    return pmi


def yandex_maps_graph(input_stream_name_time: str, input_stream_name_length: str,
                      enter_time_column: str = 'enter_time', leave_time_column: str = 'leave_time',
                      edge_id_column: str = 'edge_id', start_coord_column: str = 'start', end_coord_column: str = 'end',
                      weekday_result_column: str = 'weekday', hour_result_column: str = 'hour',
                      speed_result_column: str = 'speed', from_file: bool = False) -> Graph:
    """Constructs graph which measures average speed in km/h depending on the weekday and hour"""
    len_column: str = 'len'
    length = (Graph.graph_from(input_stream_name_length, from_file)
              .map(ops.Haversine(len_column, start_coord_column, end_coord_column))
              .map(ops.Project([edge_id_column, len_column]))
              .sort([edge_id_column]))

    time_diff_column: str = 'time_diff'
    parse_time = (Graph.graph_from(input_stream_name_time, from_file)
                  .map(ops.ParseTime(enter_time_column, weekday_result_column, hour_result_column))
                  .map(ops.TimeDiff(time_diff_column, enter_time_column, leave_time_column))
                  .map(ops.Project([edge_id_column, weekday_result_column, hour_result_column, time_diff_column]))
                  .sort([edge_id_column]))

    res = (length
           .join(ops.RightJoiner(), parse_time, [edge_id_column])
           .sort([weekday_result_column, hour_result_column])
           .map(ops.Project([weekday_result_column, hour_result_column, len_column, time_diff_column]))
           .reduce(ops.MeanSpeed(speed_result_column, len_column, time_diff_column),
                   [weekday_result_column, hour_result_column]))

    return res
