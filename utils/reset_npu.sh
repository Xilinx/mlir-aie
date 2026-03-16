#!/usr/bin/env bash
set -eux

# Match both older NPUs ("AMD NPU Device") and newer Strix NPUs ("Neural Processing Unit")
NUMBER=$(lspci -D | grep -E "\[AMD\] (AMD NPU Device|.*Neural Processing Unit)" | cut -d ' ' -f1)

if [ x"$NUMBER" != x"" ]; then
  sudo modprobe -r amdxdna
  sudo modprobe drm_shmem_helper
  sudo modprobe amdxdna
else
  echo "couldn't find npu"
fi
