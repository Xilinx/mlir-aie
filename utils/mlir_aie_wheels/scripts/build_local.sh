#!/usr/bin/env bash
set -xe
HERE=$(dirname "$(realpath "$0")")

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

# rsync -avpP --exclude .git --exclude cmake-build-debug --exclude cmake-build-release ../../llvm/* llvm-project/

export APPLY_PATCHES=true

if [ "$machine" == "linux" ]; then
  export MATRIX_OS=ubuntu-20.04
  export CIBW_ARCHS=x86_64
  export CIBW_BUILD=cp311-manylinux_x86_64
  export ARCH=x86_64
  export PARALLEL_LEVEL=15
elif [ "$machine" == "macos" ]; then
  export MATRIX_OS=macos-11
  export CIBW_ARCHS=arm64
  export CIBW_BUILD=cp311-macosx_arm64
  export ARCH=arm64
  export PARALLEL_LEVEL=32
else
  export MATRIX_OS=windows-2019
  export CIBW_ARCHS=AMD64
  export CIBW_BUILD=cp311-win_amd64
  export ARCH=AMD64
fi

ccache --show-stats
ccache --print-stats
ccache --show-config

export HOST_CCACHE_DIR="$(ccache --get-config cache_dir)"
cibuildwheel "$HERE"/.. --platform "$machine"

rename 's/cp311-cp311/py3-none/' "$HERE/../wheelhouse/"mlir*whl

if [ -d "$HERE/../wheelhouse/.ccache" ]; then
  cp -R "$HERE/../wheelhouse/.ccache/"* "$HOST_CCACHE_DIR/"
fi

cp -R "$HERE/../requirements.txt" "$HERE/../python_bindings"
cp -R "$HERE/../requirements_extras.txt" "$HERE/../python_bindings"
cp -R "$HERE/../scripts" "$HERE/../python_bindings"
cp -R "$HERE/../wheelhouse/"mlir_aie*.whl "$HERE/../python_bindings"

pushd "$HERE/../python_bindings"
# escape to prevent 'Filename not matched' when both the py3-none whl and the cp311 wheel
unzip -o -q mlir_aie\*.whl
rm -rf mlir_aie*.whl

cibuildwheel --platform "$machine" --output-dir ../wheelhouse