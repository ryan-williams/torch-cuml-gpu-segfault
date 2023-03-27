#!/usr/bin/env python
import shlex
import sys
from subprocess import check_call, CalledProcessError
import faulthandler

import click

faulthandler.enable()


@click.command('run.py', help='Repeatedly run `entrypoint.sh`, either in Docker on or the host')
@click.option('-d', '--docker-img', help="Run this docker image (assumed to have been built from this repo, with ENTRYPOINT `entrypoint.sh`")
@click.option('-n', '--num-repetitions', 'n', default=30)
@click.option('-x', '--exit-early', is_flag=True)
def main(docker_img, n, exit_early):
    cmd = [ "docker", "run", "-it", "--rm", docker_img ] if docker_img else [ "./entrypoint.sh" ]
    fmt = f"%0{len(str(n))}d"
    successes, failures = 0, 0
    for i in range(n):
        ii = fmt % (i + 1)
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
