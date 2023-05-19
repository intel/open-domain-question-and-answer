#!/bin/bash
set -e

# You can remove build-arg http_proxy and https_proxy if your network doesn't need it

DOCKER_BUILDKIT=0 docker build \
    -f ../../docker/Dockerfile ../../ \
    -t intel/ai-workflows:odqa-haystack-api \
    --network=host \
    --build-arg http_proxy=${http_proxy} \
    --build-arg https_proxy=${https_proxy} \
    --build-arg no_proxy=${no_proxy}
