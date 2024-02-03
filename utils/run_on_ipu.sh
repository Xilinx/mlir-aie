#!/bin/bash

# (c) Copyright 2023 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FIRMWARE_DIR=/lib/firmware/amdipu/1502
XRT_DIR=/opt/xilinx/xrt
source $XRT_DIR/setup.sh

rm_xclbin() {
  XCLBIN_FN=$1
  if [ -f "$XCLBIN_FN" ] && [ x"${XCLBIN_FN##*.}" == x"xclbin" ]; then
    UUID=$($XRT_DIR/bin/xclbinutil --info -i $XCLBIN_FN | grep 'UUID (xclbin)' | awk '{print $3}')
    rm -rf "$(readlink -f $FIRMWARE_DIR/$UUID.xclbin)"
    unlink $FIRMWARE_DIR/$UUID.xclbin
  fi

  # -xtype l tests for links that are broken (it is the opposite of -type)
  find $FIRMWARE_DIR -xtype l -delete;
}

if [ x"$1" == x"--clean-xclbin" ]; then
  rm_xclbin $2
  exit 0
fi

XCLBIN_FN=""
for f in "$@"; do
  if [ -f "$f" ]; then
    filename=$(basename "$f")
    extension="${f##*.}"
    if [ x"$extension" = x"xclbin" ]; then
      XCLBIN_FN="$filename"
      $XRT_DIR/amdxdna/setup_xclbin_firmware.sh -dev Phoenix -xclbin $XCLBIN_FN
    fi
  fi
done

"$@"
err=$?

if [ x"$XCLBIN_FN" != x"" ]; then
  rm_xclbin $XCLBIN_FN
fi

exit $err
