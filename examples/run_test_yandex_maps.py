import json

import click

from compgraph.algorithms import yandex_maps_graph


@click.command()
@click.argument('input_filepath1')
@click.argument('input_filepath2')
@click.argument('output_filepath')
@click.argument('enter_time_column', default='enter_time')
@click.argument('leave_time_column', default='leave_time')
@click.argument('edge_id_column', default='edge_id')
@click.argument('start_coord_column', default='start')
@click.argument('end_coord_column', default='end')
@click.argument('weekday_result_column', default='weekday')
@click.argument('hour_result_column', default='hour')
@click.argument('speed_result_column', default='speed')
def main(input_filepath1: str, input_filepath2: str, output_filepath: str, enter_time_column: str,
         leave_time_column: str, edge_id_column: str, start_coord_column: str, end_coord_column: str,
         weekday_result_column: str, hour_result_column: str, speed_result_column: str) -> None:
    graph = yandex_maps_graph(input_filepath1, input_filepath2,
                              enter_time_column=enter_time_column, leave_time_column=leave_time_column,
                              edge_id_column=edge_id_column, start_coord_column=start_coord_column,
                              end_coord_column=end_coord_column,
                              weekday_result_column=weekday_result_column, hour_result_column=hour_result_column,
                              speed_result_column=speed_result_column, from_file=True)

    result = graph.run()
    with open(output_filepath, 'w') as out:
        json.dump(list(result), out)


if __name__ == '__main__':
    main()
