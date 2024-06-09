import json

import click

from compgraph.algorithms import word_count_graph


@click.command()
@click.argument('input_filepath')
@click.argument('output_filepath')
@click.argument('text_column', default='text')
@click.argument('count_column', default='count')
def main(input_filepath: str, output_filepath: str, text_column: str, count_column: str) -> None:
    graph = word_count_graph(input_stream_name=input_filepath, text_column=text_column, count_column=count_column,
                             from_file=True)

    result = graph.run()
    with open(output_filepath, 'w') as out:
        json.dump(list(result), out)


if __name__ == '__main__':
    main()
