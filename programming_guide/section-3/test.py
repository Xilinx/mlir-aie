# test.py -*- Python -*-
#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
import pyxrt as xrt
import time

# options
trace_size = 0
opts_xclbin = 'build/final.xclbin'
opts_kernel = 'MLIR_AIE'
do_verify = True
n_iterations = 1
n_warmup_iterations = 0

# ------------------------------------------------------
# Configure this to match your design's buffer size
# ------------------------------------------------------
INOUT0_VOLUME = 64 # Input only, 64x uint32_t in this example
INOUT1_VOLUME = 64 # Not used in this example
INOUT2_VOLUME = 64 # Output only, 64x uint32_t in this example

INOUT0_DATATYPE = np.uint32
INOUT1_DATATYPE = np.uint32
INOUT2_DATATYPE = np.uint32

INOUT0_SIZE = INOUT0_VOLUME * INOUT0_DATATYPE().itemsize
INOUT1_SIZE = INOUT1_VOLUME * INOUT1_DATATYPE().itemsize
INOUT2_SIZE = INOUT2_VOLUME * INOUT2_DATATYPE().itemsize
OUT_SIZE = INOUT2_SIZE + trace_size

# Load instruction sequence
with open('build/insts.txt', 'r') as f:
    instr_text = f.read().split('\n')
    instr_text = [l for l in instr_text if l != '']
    instr_v = np.array([int(i,16) for i in instr_text], dtype=np.uint32)

# ------------------------------------------------------
# Get device, load the xclbin & kernel and register them
# ------------------------------------------------------

# Get a device handle
device = xrt.device(0)

# Load the xclbin
xclbin = xrt.xclbin(opts_xclbin)

# Load the kernel
kernels = xclbin.get_kernels()
try:
    xkernel = [k for k in kernels if opts_kernel in k.get_name()][0]
except:
    print(f"Kernel '{opts_kernel}' not found in '{opts_xclbin}'")
    exit(-1)

# Register xclbin
device.register_xclbin(xclbin)

# Get a hardware context
context = xrt.hw_context(device, xclbin.get_uuid())

# get a kernel handle
kernel = xrt.kernel(context, xkernel.get_name())

# ------------------------------------------------------
# Initialize input/ output buffer sizes and sync them
# ------------------------------------------------------
bo_instr = xrt.bo(device, len(instr_v)*4, xrt.bo.cacheable, kernel.group_id(0))
bo_inout0 = xrt.bo(device, INOUT0_SIZE, xrt.bo.host_only, kernel.group_id(2))
bo_inout1 = xrt.bo(device, INOUT1_SIZE, xrt.bo.host_only, kernel.group_id(3))
bo_inout2 = xrt.bo(device, INOUT2_SIZE, xrt.bo.host_only, kernel.group_id(4))

# Initialize instruction buffer
bo_instr.write(instr_v, 0)

# Initialize data buffers
inout0 = np.arange(1, INOUT0_VOLUME+1, dtype=INOUT0_DATATYPE)
#inout1 = np.zeros(INOUT1_VOLUME, dtype=INOUT1_DATATYPE)
inout2 = np.zeros(INOUT2_VOLUME, dtype=INOUT2_DATATYPE)
bo_inout0.write(inout0, 0)
#bo_inout1.write(inout1, 0)
bo_inout2.write(inout2, 0)

bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
bo_inout0.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
#bo_inout1.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
bo_inout2.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

num_iter = n_iterations + n_warmup_iterations
npu_time_total = 0
npu_time_min = 9999999
npu_time_max = 0
errors = 0

for i in range(num_iter):
    print ("Running Kernel.")
    start = time.time_ns()
    h = kernel(bo_instr, len(instr_v), bo_inout0, bo_inout1, bo_inout2)
    h.wait()
    stop = time.time_ns()
    bo_inout2.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

    # if i < n_warmup_iterations:
    #     continue

    output_buffer = bo_inout2.read(OUT_SIZE, 0).view(INOUT2_DATATYPE)
    if do_verify:
        print ("Verifying results ...")
        ref = np.arange(2, INOUT0_VOLUME+2, dtype=INOUT0_DATATYPE)
        e = np.equal(output_buffer, ref)
        errors = errors + np.size(e) - np.count_nonzero(e)

    if trace_size > 0:
        print("Do something with trace!")

    npu_time = stop - start
    npu_time_total = npu_time_total + npu_time
    npu_time_min = min(npu_time_min, npu_time)
    npu_time_max = max(npu_time_max, npu_time)

print ("\nAvg NPU time: {}us.".format(int((npu_time_total / n_iterations) / 1000)))
print ("\nMin NPU time: {}us.".format(int((npu_time_min / n_iterations) / 1000)))
print ("\nMax NPU time: {}us.".format(int((npu_time_max / n_iterations) / 1000)))

if not errors:
    print("\nPASS!\n")
    exit(0)
else:
    print("\nError count: ", errors)
    print("\nFailed.\n")
    exit(-1)
