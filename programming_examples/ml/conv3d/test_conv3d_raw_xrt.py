#!/usr/bin/env python3
# Test conv3d using raw PyXRT API (not iron.tensor)

import pyxrt
import numpy as np

depth, height, width = 8, 8, 8
ci, co = 8, 8

plane_size = height * width * ci
tensor_in_size = depth * plane_size
tensor_out_size = depth * height * width * co
weights_size = ci * co  # 1x1 conv

# Create test data
input_data = np.arange(tensor_in_size, dtype=np.uint8)
weights_data = np.ones(weights_size, dtype=np.int8)
output_data = np.zeros(tensor_out_size, dtype=np.uint8)

# Load device and kernel
device = pyxrt.device(0)
xclbin = pyxrt.xclbin("/scratch/jmelber/mlir-aie/programming_examples/ml/conv3d/build/final_simple_npu2.xclbin")
device.register_xclbin(xclbin)
kernels = xclbin.get_kernels()
xkernel = [k for k in kernels if "MLIR_AIE" in k.get_name()][0]
kernel = pyxrt.kernel(device, xclbin.get_uuid(), xkernel.get_name(), pyxrt.kernel.exclusive)

# Create buffer objects
bo_input = pyxrt.bo(device, tensor_in_size, pyxrt.bo.normal, kernel.group_id(0))
bo_output = pyxrt.bo(device, tensor_out_size, pyxrt.bo.normal, kernel.group_id(1))

# Write input
bo_input.write(input_data, 0)
bo_input.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

print(f"Running conv3d_simple with raw PyXRT (8x8x8, {tensor_in_size} bytes)...")

# Run kernel
run = kernel(bo_input, bo_output)
run.wait()
state = run.state()

print(f"Kernel state: {state}")

if state == pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
    print(f"✅ SUCCESS! Kernel completed!")

    # Read output
    bo_output.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    output = np.frombuffer(bo_output.map(), dtype=np.uint8, count=tensor_out_size)

    print(f"Output range: {output.min()} to {output.max()}")
    print(f"First 16 output: {output[:16]}")
    print(f"First 16 input:  {input_data[:16]}")
else:
    print(f"❌ FAILED with state: {state}")
    exit(1)
