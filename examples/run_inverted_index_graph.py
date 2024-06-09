import json

import click

from compgraph.algorithms import inverted_index_graph


@click.command()
@click.argument('input_filepath')
@click.argument('output_filepath')
@click.argument('doc_column', default='doc_id')
@click.argument('text_column', default='text')
@click.argument('result_column', default='tf_idf')
def main(input_filepath: str, output_filepath: str, doc_column: str, text_column: str, result_column: str) -> None:
    graph = inverted_index_graph(input_stream_name=input_filepath, doc_column=doc_column, text_column=text_column,
                                 result_column=result_column, from_file=True)

    result = graph.run()
    with open(output_filepath, 'w') as out:
        json.dump(list(result), out)


if __name__ == '__main__':
    main()
