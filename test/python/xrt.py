# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: XILINX_XRT=%XRT_DIR flock /tmp/ipu.lock /opt/xilinx/xrt/amdaie/setup_xclbin_firmware.sh -dev Phoenix -xclbin %S/final.xclbin
# RUN: %PYTHON %s | FileCheck %s
# REQUIRES: xrt_python_bindings
# REQUIRES: ryzen_ai

from pathlib import Path
import numpy as np

from aie.xrt import XCLBin

xclbin = XCLBin(str(Path(__file__).parent.absolute() / "final.xclbin"), "MLIR_AIE")
ipu_insts = [
    0x00000011,
    0x01000405,
    0x01000100,
    0x0B590100,
    0x000055FF,
    0x00000001,
    0x00000010,
    0x314E5A5F,
    0x635F5F31,
    0x676E696C,
    0x39354E5F,
    0x6E693131,
    0x5F727473,
    0x64726F77,
    0x00004573,
    0x07BD9630,
    0x000055FF,
    0x06000100,
    0x00000000,
    0x00000040,
    0x00000000,
    0x00000000,
    0x00000000,
    0x80000000,
    0x00000000,
    0x00000000,
    0x02000000,
    0x02000000,
    0x0001D214,
    0x00000000,
    0x06000121,
    0x00000000,
    0x00000040,
    0x00000000,
    0x00000000,
    0x00000000,
    0x80000000,
    0x00000000,
    0x00000000,
    0x02000000,
    0x02000000,
    0x0001D204,
    0x80000001,
    0x03000000,
    0x00010100,
]

xclbin.load_ipu_instructions(ipu_insts)
inps, outps = xclbin.mmap_buffers([(64,), (64,)], [(64,)], np.int32)

wrap_A = np.asarray(inps[0])
wrap_C = np.asarray(outps[0])

A = np.random.randint(0, 10, 64, dtype=np.int32)
C = np.zeros(64, dtype=np.int32)

np.copyto(wrap_A, A, casting="no")
np.copyto(wrap_C, C, casting="no")

xclbin.sync_buffers_to_device()
xclbin.run()
xclbin.wait()
xclbin.sync_buffers_from_device()

# CHECK: True
print(np.allclose(A + 1, wrap_C))
