#!/usr/bin/env python
import shlex
import sys
from functools import partial
from subprocess import check_call, DEVNULL, CalledProcessError

import click


def repeat(fn, n, log=print, fail_msg='segfault', exit_early=False):
    fmt = f"%0{len(str(n))}d"
    successes, failures = 0, 0
    for i in range(n):
        ii = fmt % (i + 1)
        log(f"Iteration {ii}/{n}")
        try:
            fn()
            print(f"✅ Success (iteration {ii}/{n})")
            successes += 1
        except CalledProcessError as e:
            if e.returncode in [-11, 139]:  # segfault
                print(f"❌ Failure (iteration {ii}/{n}): exit code {e.returncode} ({fail_msg})")
                log(f"{e}")
                if exit_early:
                    raise
                failures += 1
            else:
                raise RuntimeError(f"Unexpected returncode {e.returncode}")

    if failures:
        pct_str = "%.1f" % (failures / n * 100)
        print(f"❌ {failures}/{n} runs failed ({pct_str}%)")
        sys.exit(1)
    else:
        print(f"✅ all {successes} runs succeeded")
        sys.exit(0)


@click.command('run.py', help='Repeatedly run `neighbors:run`, either in Docker on or the host')
@click.option('-d', '--docker-img', help="Run `neighbors.py` inside the provided docker image (assumed to have been built from this repo)")
@click.option('-i', '--repeat-in-process', is_flag=True, help='Repeat `cuml.NearestNeighbors` within one process (as opposed to each one in a separate Python subprocess invocation)')
@click.option('-n', '--num-repetitions', 'n', default=30, help='Repeat `cuml.NearestNeighbors` this many times')
@click.option('-q', '--quiet', is_flag=True, help='Suppress subprocess output, only print success/failure info for each iteration')
@click.option('-q', '--quiet', is_flag=True, help='Suppress subprocess output, only print success/failure info for each iteration')
@click.option('-r', '--random-seed', default=123, type=int, help='Random seed (set before `cuml.NearestNeighbors` calculation)')
@click.option('-s', '--shape', default='10x2', help='Shape for random array passed to `cuml.NearestNeighbors`')
@click.option('-T', '--no-import-torch', is_flag=True, help='Skip the unused `import torch` (segfault no longer occurs)')
@click.option('-x', '--exit-early', is_flag=True, help='Exit on first failure')
def main(docker_img, repeat_in_process, n, quiet, random_seed, shape, no_import_torch, exit_early):
    log = (lambda msg: None) if quiet else print

    fail_msg = 'segfault in Docker' if docker_img else 'segfault'
    repeat_kwargs = dict(
        n=n,
        log=log,
        fail_msg=fail_msg,
        exit_early=exit_early
    )
    if repeat_in_process and not docker_img:
        from neighbors import run
        repeat(
            partial(run.callback, quiet=quiet, random_seed=random_seed, shape=shape, no_import_torch=no_import_torch),
            **repeat_kwargs
        )
    else:
        run_args = [
            *(['-q'] if quiet else []),
            '-r', f'{random_seed}',
            '-s', shape,
            *(['-T'] if no_import_torch else []),
        ]
        if docker_img:
            cmd = [ "docker", "run", "-it", "--rm", "--runtime", "nvidia", ]
            if repeat_in_process:
                cmd += [
                    '--entrypoint', "./run.py",
                    docker_img,
                    '-i',
                    '-n', f'{n}',
                    *run_args,
                    *(['-x'] if exit_early else []),
                ]
            else:
                cmd += [ docker_img, *run_args ]
        else:
            cmd = ["./neighbors.py", *run_args]

        call_kwargs = dict(stdout=DEVNULL, stderr=DEVNULL) if quiet else dict()

        def fn():
            log(f"Running: {shlex.join(cmd)}")
            check_call(cmd, **call_kwargs)

        if docker_img and repeat_in_process:
            fn()
        else:
            repeat(fn, **repeat_kwargs)


if __name__ == '__main__':
    main()
