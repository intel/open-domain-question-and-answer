#!/bin/bash
set -e

# You can remove build-arg http_proxy and https_proxy if your network doesn't need it
# no_proxy="localhost,127.0.0.0/1"
# proxy_server="" # your http proxy server
proxy_server=""

DOCKER_BUILDKIT=0 docker build \
    -f ../../docker/Dockerfile ../../ \
    -t intel/ai-workflows:odqa-haystack-api \
    --network=host \
    --build-arg http_proxy=${proxy_server} \
    --build-arg https_proxy=${proxy_server} \
    --build-arg no_proxy=${no_proxy}
