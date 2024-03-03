#!/usr/bin/env bash
set -xe

rm -rf mlir || true

pip install mlir-native-tools --force -U

if [ x"$ENABLE_RTTI" == x"OFF" ]; then
  NO_RTTI="-no-rtti"
fi

if [ x"$CIBW_ARCHS" == x"arm64" ] || [ x"$CIBW_ARCHS" == x"aarch64" ]; then
  if [ x"$MATRIX_OS" == x"macos-11" ] && [ x"$CIBW_ARCHS" == x"arm64" ]; then
    PLAT=macosx_11_0_arm64
  elif [ x"$MATRIX_OS" == x"macos-14" ] && [ x"$CIBW_ARCHS" == x"arm64" ]; then
    PLAT=macosx_11_0_arm64
  elif [ x"$MATRIX_OS" == x"ubuntu-20.04" ] && [ x"$CIBW_ARCHS" == x"aarch64" ]; then
    PLAT=linux_aarch64
  fi
  pip -q download mlir$NO_RTTI --platform $PLAT --only-binary=:all:
else
  pip -q download mlir$NO_RTTI
fi

# overwrite files WITHOUT prompting
unzip -o -q mlir*whl

echo $PWD
ls -l mlir*