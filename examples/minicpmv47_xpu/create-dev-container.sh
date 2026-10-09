#!/usr/bin/env bash
set -euo pipefail

prep_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
container_name=codex-minicpm47-v031-dev-20261008
image_id=$(cat "$prep_root/evidence/dev-image-id.txt")

if docker container inspect "$container_name" >/dev/null 2>&1; then
    printf 'Container already exists: %s\n' "$container_name" >&2
    exit 1
fi

# Device access is intentionally absent for this preparation phase.
docker run -d \
    --name "$container_name" \
    --label codex.task=minicpm-v47-main-prep-20261008 \
    --label codex.phase=cpu-preparation \
    --mount "type=bind,source=$prep_root,target=/workspace/preparation,readonly" \
    --workdir /workspace/preparation \
    --entrypoint /bin/bash \
    "$image_id" -lc 'exec sleep infinity'
