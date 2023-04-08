#!/usr/bin/env python
import shlex
from functools import partial
from subprocess import check_call, DEVNULL

import click

from neighbors import repeat, run, quiet_opt, random_seed_opt, shape_opt


@click.command('run.py', help='Repeatedly run `entrypoint.sh`, either in Docker on or the host')
@click.option('-d', '--docker-img', help="Run this docker image (assumed to have been built from this repo, with ENTRYPOINT `entrypoint.sh`")
@click.option('-i', '--repeat-in-process', is_flag=True, help='Repeat `cuml.NearestNeighbors` within one process (as opposed to each one in a separate Python subprocess invocation)')
@click.option('-n', '--num-repetitions', 'n', default=30, help='Repeat `cuml.NearestNeighbors` this many times')
@click.option('-q', '--quiet', is_flag=True, help='Suppress subprocess output, only print success/failure info for each iteration')
@quiet_opt
@random_seed_opt
@shape_opt
@click.option('-x', '--exit-early', is_flag=True, help='Exit on first failure')
def main(docker_img, repeat_in_process, n, quiet, random_seed, shape, exit_early):
    log = (lambda msg: None) if quiet else print

    fail_msg = 'segfault in Docker' if docker_img else 'segfault'
    if repeat_in_process and not docker_img:
        repeat(
            partial(run.callback, quiet=quiet, random_seed=random_seed, shape=shape),
            n=n,
            log=log,
            fail_msg=fail_msg,
            exit_early=exit_early
        )
    else:
        run_args = [
            *(['-q'] if quiet else []),
            '-r', f'{random_seed}',
            '-s', shape,
        ]
        if docker_img:
            cmd = [ "docker", "run", "-it", "--rm", "--runtime", "nvidia", ]
            if repeat_in_process:
                cmd += [
                    '--entrypoint', "./run.py",
                    docker_img,
                    '-i',
                    '-n', n,
                    *run_args,
                    *(['-x'] if exit_early else []),
                ]
            else:
                cmd += [ docker_img ]
        else:
            cmd = ["./neighbors.py", *run_args]

        call_kwargs = dict(stdout=DEVNULL, stderr=DEVNULL) if quiet else dict()

        def fn():
            log(f"Running: {shlex.join(cmd)}")
            check_call(cmd, **call_kwargs)

        repeat(fn, n=n, log=log, fail_msg=fail_msg, exit_early=exit_early)


if __name__ == '__main__':
    main()
