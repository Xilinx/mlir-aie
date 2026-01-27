#!/usr/bin/env bash
set -xe

rm -rf mlir || true

# Find clone-llvm.sh to get the correct MLIR version
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
if [ -n "$MLIR_AIE_SOURCE_DIR" ]; then
    CLONE_LLVM="$MLIR_AIE_SOURCE_DIR/utils/clone-llvm.sh"
else
    # Assume we are in utils/mlir_aie_wheels/scripts
    # Check if clone-llvm.sh was copied to utils/mlir_aie_wheels (parent of scripts)
    if [ -f "$SCRIPT_DIR/../clone-llvm.sh" ]; then
        CLONE_LLVM="$SCRIPT_DIR/../clone-llvm.sh"
    else
        CLONE_LLVM="$SCRIPT_DIR/../../clone-llvm.sh"
    fi
fi

if [ ! -f "$CLONE_LLVM" ]; then
    echo "Error: clone-llvm.sh not found at $CLONE_LLVM"
    exit 1
fi

VERSION=$($CLONE_LLVM --get-wheel-version)
echo "Using MLIR version: $VERSION"

pip install mlir-native-tools==$VERSION --force -U

if [ x"$ENABLE_RTTI" == x"OFF" ]; then
  NO_RTTI="-no-rtti"
fi

if [ x"$CIBW_ARCHS" == x"arm64" ] || [ x"$CIBW_ARCHS" == x"aarch64" ]; then
  if [ x"$MATRIX_OS" == x"macos-12" ] && [ x"$CIBW_ARCHS" == x"arm64" ]; then
    PLAT=macosx_12_0_arm64
  elif [ x"$MATRIX_OS" == x"macos-14" ] && [ x"$CIBW_ARCHS" == x"arm64" ]; then
    PLAT=macosx_12_0_arm64
  elif [ x"$MATRIX_OS" == x"ubuntu-20.04" ] && [ x"$CIBW_ARCHS" == x"aarch64" ]; then
    PLAT=linux_aarch64
  fi
  pip -q download mlir$NO_RTTI==$VERSION --platform $PLAT --only-binary=:all:
else
  pip -q download mlir$NO_RTTI==$VERSION
fi

# overwrite files WITHOUT prompting
unzip -o -q mlir*whl

echo $PWD
ls -l mlir*
