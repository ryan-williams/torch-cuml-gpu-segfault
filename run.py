#!/usr/bin/env python
import shlex
from subprocess import check_call
import faulthandler

import click

faulthandler.enable()


@click.command('run.py', help='Repeatedly run a docker image, exiting on the first error', no_args_is_help=True)
@click.option('-n', '--num-repetitions', 'n', default=30)
@click.argument('img')
def main(n, img):
    cmd = [ "docker", "run", "-it", "--rm", img ]
    for i in range(n):
        print(f"Iteration {i}/{n}")
        print(f"Running: {shlex.join(cmd)}")
        try:
            check_call(cmd)
        finally:
            print(f'Done: {i}/{n}')


if __name__ == '__main__':
    main()
