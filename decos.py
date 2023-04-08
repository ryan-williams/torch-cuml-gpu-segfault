import click

quiet_opt = click.option('-q', '--quiet', is_flag=True, help='Suppress subprocess output, only print success/failure info for each iteration')
random_seed_opt = click.option('-r', '--random-seed', default=123, type=int, help='Random seed (set before `cuml.NearestNeighbors` calculation)')
shape_opt = click.option('-s', '--shape', default='10x2', help='Shape for random array passed to `cuml.NearestNeighbors`')
