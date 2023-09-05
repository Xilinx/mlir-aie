#!/usr/bin/env bash
set -xe

wget -q https://boostorg.jfrog.io/artifactory/main/release/1.72.0/source/boost_1_72_0.zip
unzip boost_1_72_0.zip
cd boost_1_72_0
./bootstrap.sh
./b2 install --with=all