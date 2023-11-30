#!/bin/bash

export GITHUB_WORKSPACE=`pwd`
export WORKSPACE=/workspace

docker run -it \
-v ${GITHUB_WORKSPACE}:${WORKSPACE} \
amd/acdcbuild:1.1 \
bash
