#!/usr/bin/env bash
set -eux

NUMBER=$(lspci -D | grep "\[AMD\] AMD IPU Device" | cut -d ' ' -f1)

if [ x"$NUMBER" != x"" ]; then
  sudo modprobe -r amdxdna
  sudo modprobe drm_shmem_helper
  sudo modprobe amdxdna
else
  echo "couldn't find npu"
fi

