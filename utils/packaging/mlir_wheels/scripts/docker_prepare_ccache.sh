#!/usr/bin/env bash
set -xe

# manylinux prep
if [[ -f "/etc/centos-release" ]]; then
  yum install -y epel-release
  # sometimes the epel server is down. retry 5 times
  for i in $(seq 1 5); do
    yum install -y zlib-devel curl-devel expat-devel libpng-devel ccache wget ninja-build git && s=0 && break || s=$? && sleep 15
  done

  [ $s -eq 0 ] || exit $s

  if [[ -d "/usr/lib64/ccache" ]]; then
    [ ! -f "/usr/lib64/ccache/c++" ] && ln -s /usr/bin/ccache /usr/lib64/ccache/c++
    [ ! -f "/usr/lib64/ccache/cc" ] && ln -s /usr/bin/ccache /usr/lib64/ccache/cc
    [ ! -f "/usr/lib64/ccache/gcc" ] && ln -s /usr/bin/ccache /usr/lib64/ccache/gcc
    [ ! -f "/usr/lib64/ccache/g++" ] && ln -s /usr/bin/ccache /usr/lib64/ccache/g++
    export PATH="/usr/lib64/ccache:$PATH"
  elif [[ -d "/usr/lib/ccache" ]]; then
    [ ! -f "/usr/lib/ccache/c++" ] && ln -s /usr/bin/ccache /usr/lib/ccache/c++
    [ ! -f "/usr/lib/ccache/cc" ] && ln -s /usr/bin/ccache /usr/lib/ccache/cc
    [ ! -f "/usr/lib/ccache/gcc" ] && ln -s /usr/bin/ccache /usr/lib/ccache/gcc
    [ ! -f "/usr/lib/ccache/g++" ] && ln -s /usr/bin/ccache /usr/lib/ccache/g++
    export PATH="/usr/lib/ccache:$PATH"
  fi

elif [[ -f "/etc/alpine-release" ]]; then
  # musllinux prep
  # ccache already present
  apk add curl-dev expat-dev zlib-dev libpng-dev ccache
  export PATH="/usr/lib/ccache/bin:$PATH"
fi

# hack until https://github.com/pypa/cibuildwheel/issues/1030 is fixed
# Place ccache folder in /outputs
HOST_CCACHE_DIR="/host${HOST_CCACHE_DIR:-/home/runner/work/mlir-wheels/mlir-wheels/.ccache}"
if [ -d "$HOST_CCACHE_DIR" ]; then
  ls -l "$HOST_CCACHE_DIR"
  mkdir -p /output
  mv $HOST_CCACHE_DIR /output/.ccache
  ls -la /output/.ccache
fi

ccache -o cache_dir="/output/.ccache"
ccache -M 5 G # set cache size to 5 G
# Show ccache stats
echo "Cache stats:"
ccache --show-stats
