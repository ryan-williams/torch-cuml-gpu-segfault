#!/usr/bin/env bash

python \
    -Xfaulthandler \
    pipeline.py \
    --quiet \
    --metadata local \
    --environment local \
    --datastore local \
    --event-logger nullSidecarLogger \
    --monitor nullSidecarMonitor \
    --datastore-root $PWD/.metaflow \
    step \
    start \
    --run-id example_run_id \
    --task-id 1 \
    --input-paths example_run_id/_parameters/0 \
    --retry-count 0 \
    --max-user-code-retries 0 \
    --namespace user:user
