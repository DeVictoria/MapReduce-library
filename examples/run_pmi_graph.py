import json

import click

from compgraph.algorithms import pmi_graph


@click.command()
@click.argument('input_filepath')
@click.argument('output_filepath')
@click.argument('doc_column', default='doc_id')
@click.argument('count_column', default='text')
@click.argument('result_column', default='pmi')
def main(input_filepath: str, output_filepath: str, doc_column: str, count_column: str, result_column: str) -> None:
    graph = pmi_graph(input_stream_name=input_filepath, doc_column=doc_column, text_column=count_column,
                      result_column=result_column, from_file=True)

    result = graph.run()
    with open(output_filepath, 'w') as out:
        json.dump(list(result), out)


if __name__ == '__main__':
    main()
