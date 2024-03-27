#!/bin/bash

# (c) Copyright 2023 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Execute a command while pre-staging for AIE driver some .xclbin firmware and
# clean up the firmware stage after execution.
# This script assumes that the path of one xclbin
# appears in the argument list

# Old AIE driver
# Where the device driver finds the firmware to be loaded.
FIRMWARE_DIR=/lib/firmware/amdipu/1502
# The AIE device model
NPU_DEVICE=Phoenix
if [[ -d /lib/firmware/amdnpu/1502_00 ]]; then
  # New AIE driver naming and firmware location starting with Linux 6.8
  FIRMWARE_DIR=/lib/firmware/amdnpu/1502_00
  NPU_DEVICE=NPU1
fi

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
# Analyze all the command arguments to find some .xclbin
for f in "$@"; do
  if [ -f "$f" ]; then
    filename=$(basename "$f")
    extension="${f##*.}"
    if [ x"$extension" = x"xclbin" ]; then
      XCLBIN_FN="$filename"
      # Stage the .xclbin to be loaded by the command execution later
      $XRT_DIR/amdxdna/setup_xclbin_firmware.sh -dev $NPU_DEVICE -xclbin $XCLBIN_FN
      break
    fi
  fi
done

# Execute the commands and its arguments
"$@"
err=$?

# TODO: use setup_xclbin_firmware.sh -clean instead
if [ x"$XCLBIN_FN" != x"" ]; then
  rm_xclbin $XCLBIN_FN
fi

exit $err
