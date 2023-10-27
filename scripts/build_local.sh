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

if [ ! -d mlir-aie ]; then
  git clone --recursive https://github.com/Xilinx/mlir-aie.git
fi

export MLIR_WHEEL_VERSION="17.0.0.2023102703+35ca6498"

if [ "$machine" == "linux" ]; then
  export CIBW_ARCHS=${CIBW_ARCHS:-x86_64}
  export PARALLEL_LEVEL=15
  export MATRIX_OS=ubuntu-20.04
elif [ "$machine" == "macos" ]; then
  export CIBW_ARCHS=${CIBW_ARCHS:-arm64}
  export MATRIX_OS=macos-11
  export PARALLEL_LEVEL=32
else
  export MATRIX_OS=windows-2019
  export CIBW_ARCHS=${CIBW_ARCHS:-AMD64}
fi

ccache --show-stats
ccache --print-stats
ccache --show-config

export HOST_CCACHE_DIR="$(ccache --get-config cache_dir)"
#export CIBW_CONTAINER_ENGINE="docker; create_args : --network=\"host\""
export PIP_FIND_LINKS="https://github.com/makslevental/mlir-wheels/releases/expanded_assets/latest https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels"

if [ x"$CIBW_ARCHS" == x"aarch64" ] || [ x"$CIBW_ARCHS" == x"arm64" ]; then
  pip install mlir-native-tools==$MLIR_WHEEL_VERSION -f -U
  export PIP_NO_BUILD_ISOLATION="false"
  pip install -r $HERE/../requirements.txt
  $HERE/../scripts/pip_install_mlir.sh

  CMAKE_GENERATOR=Ninja \
  pip wheel $HERE/.. -v -w $HERE/../wheelhouse
else
  cibuildwheel "$HERE"/.. --platform "$machine"
fi

rename 's/cp310-cp310/py3-none/' $HERE/../wheelhouse/mlir_aie-*whl
rename 's/cp311-cp311/py3-none/' $HERE/../wheelhouse/mlir_aie-*whl

cp -a $HERE/../scripts $HERE/../python_bindings/
cp -a $HERE/../requirements.txt $HERE/../python_bindings/

if [ x"$CIBW_ARCHS" == x"aarch64" ] || [ x"$CIBW_ARCHS" == x"arm64" ]; then
  pip install mlir-aie -f $HERE/../wheelhouse
  pip wheel "$HERE/../python_bindings" -v -w $HERE/../wheelhouse
else
  cibuildwheel "$HERE/../python_bindings" --platform linux --output-dir $HERE/../wheelhouse
fi
