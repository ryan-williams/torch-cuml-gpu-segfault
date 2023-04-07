#!/usr/bin/env bash
# Invoke pipeline.py as directly as possible, via a `python` invocation that is normally nested a couple subprocesses
# deep in a Metaflow workflow run

python \
    -Xfaulthandler \
    pipeline.py \
    --quiet \
    --metadata local \
    --environment local \
    --datastore local \
    --event-logger nullSidecarLogger \
    --monitor nullSidecarMonitor \
    --datastore-root $PWD/.metaflow-example \
    step \
    start \
    --run-id example_run_id \
    --task-id 1 \
    --input-paths example_run_id/_parameters/0 \
    --retry-count 0 \
    --max-user-code-retries 0 \
    --namespace user:user
