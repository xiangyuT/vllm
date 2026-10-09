#!/usr/bin/env bash
set -euo pipefail

prep_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$prep_root"
if test -s evidence/dev-image-id.txt; then
    image_id=$(cat evidence/dev-image-id.txt)
    docker image inspect "$image_id" --format '{{.Id}}'
    exit 0
fi

build_log=$(mktemp "$prep_root/evidence/dev-image-build.XXXXXX.log")
printf 'Build log: %s\n' "$build_log"
docker build --progress=plain \
    --build-arg HTTP_PROXY --build-arg HTTPS_PROXY --build-arg NO_PROXY \
    --build-arg http_proxy --build-arg https_proxy --build-arg no_proxy \
    -f Dockerfile.dev --iidfile evidence/dev-image-id.txt . > "$build_log" 2>&1
cat evidence/dev-image-id.txt
