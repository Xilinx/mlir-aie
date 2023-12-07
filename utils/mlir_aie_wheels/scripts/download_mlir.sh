#!/usr/bin/env bash
set -xe

rm -rf mlir || true
rm -rf mlir-18* || true

SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

pip install mlir-native-tools --force -U

if [ x"$CIBW_ARCHS" == x"arm64" ] || [ x"$CIBW_ARCHS" == x"aarch64" ]; then
  if [ x"$MATRIX_OS" == x"macos-11" ] && [ x"$CIBW_ARCHS" == x"arm64" ]; then
    PLAT=macosx_11_0_arm64
  elif [ x"$MATRIX_OS" == x"ubuntu-20.04" ] && [ x"$CIBW_ARCHS" == x"aarch64" ]; then
    PLAT=linux_aarch64
  fi
  pip -q download mlir --platform $PLAT --only-binary=:all:
else
  pip -q download mlir
fi

unzip -q mlir-*whl
