#!/usr/bin/env bash
set -xe

unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     machine=linux;;
    Darwin*)    machine=macos;;
    CYGWIN*)    machine=windows;;
    MINGW*)     machine=windows;;
    MSYS_NT*)   machine=windows;;
    *)          machine="UNKNOWN:${unameOut}"
esac
echo "${machine}"

wget -q https://boostorg.jfrog.io/artifactory/main/release/1.72.0/source/boost_1_72_0.zip
unzip -q boost_1_72_0.zip
cd boost_1_72_0

if [ "$machine" == "linux" ]; then
  CXX=/usr/lib64/ccache/g++ ./bootstrap.sh --with-libraries=graph
else
  ./bootstrap.sh --with-libraries=graph
fi

./b2 install --with-graph