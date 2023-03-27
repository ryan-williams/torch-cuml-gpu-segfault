#!/usr/bin/env python
import shlex
import sys
from subprocess import check_call, CalledProcessError
import faulthandler

import click

faulthandler.enable()


@click.command('run.py', help='Repeatedly run a docker image, exiting on the first error', no_args_is_help=True)
@click.option('-n', '--num-repetitions', 'n', default=30)
@click.option('-x', '--exit-early', is_flag=True)
@click.argument('img')
def main(n, exit_early, img):
    cmd = [ "docker", "run", "-it", "--rm", img ]
    successes, failures = 0, 0
    for i in range(n):
        ii = "%02d" % (i + 1)
        print(f"Iteration {ii}/{n}")
        try:
            print(f"Running: {shlex.join(cmd)}")
            check_call(cmd)
            print(f"✅ Success (iteration {ii}/{n})")
            successes += 1
        except CalledProcessError as e:
            if e.returncode == 139:  # Docker segfault
                print(f"❌ Failure (iteration {ii}/{n}); exit code {e.returncode} (segfault in Docker)")
                print(f"{e}")
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
        print(f"✅ runs all succeeded")
        sys.exit(0)


if __name__ == '__main__':
    main()
