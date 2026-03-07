#!/usr/bin/env python3
#
# Analyze the quantization scale relationship
#

import numpy as np

# Parameters from test.py
conv_scale = 7.6294e-06
int8_scale = 0.0078

# NPU scale from kernel and conv3d.py
npu_scale_bits = 10

print("="*80)
print("QUANTIZATION SCALE ANALYSIS")
print("="*80)

print("\n1. PyTorch Quantization Flow:")
print(f"   Input:  uint8 activations, int8 weights")
print(f"   Conv:   int32 = sum(uint8 * int8)")
print(f"   Step 1: fp32 = int32 * conv_scale = int32 * {conv_scale}")
print(f"   Step 2: uint8 = clamp(round(fp32 / int8_scale), 0, 255)")
print(f"   Step 3: output = uint8 * int8_scale")
print(f"\n   Net effect: output = clamp(round(int32 * conv_scale / int8_scale), 0, 255) * int8_scale")
print(f"               output = clamp(round(int32 * {conv_scale / int8_scale:.6e}), 0, 255) * {int8_scale}")

print("\n2. NPU Kernel Quantization Flow:")
print(f"   Input:  uint8 activations, int8 weights")
print(f"   Conv:   int32 = sum(uint8 * int8)")
print(f"   Shift:  int32_shifted = (int32 + {2**(npu_scale_bits-1)}) >> {npu_scale_bits}")
print(f"           (right shift by {npu_scale_bits} = divide by {2**npu_scale_bits}, with rounding)")
print(f"   Clamp:  uint8 = clamp(int32_shifted, 0, 255)")
print(f"   Output: fp32 = uint8 * int8_scale")

print("\n3. Scale Relationship:")
print(f"   PyTorch multiplier: {conv_scale / int8_scale:.10e}")
print(f"   NPU multiplier:     {1 / (2**npu_scale_bits):.10e}")
print(f"   Ratio:              {(conv_scale / int8_scale) / (1 / (2**npu_scale_bits)):.6f}")

mismatch = (conv_scale / int8_scale) / (1 / (2**npu_scale_bits))
if abs(mismatch - 1.0) > 0.01:
    print(f"\n   WARNING: {mismatch:.2f}x mismatch in scale factors!")
else:
    print(f"\n   Scales match within 1%")

print("\n4. Expected Quantization Difference:")
print(f"   Due to rounding in NPU kernel:")
print(f"   - NPU adds {2**(npu_scale_bits-1)} before right shift (rounding)")
print(f"   - PyTorch uses round() function")
print(f"   Maximum difference per output element: ~{int8_scale:.6f} (1 LSB)")

print("\n5. Test for Systematic Bias:")
print(f"\n   Let's check if rounding direction differs:")
print(f"   NPU rounding: (x + 512) >> 10")
print(f"   PyTorch rounding: round(x / 1024)")
print(f"\n   Test cases:")
for test_val in [0, 512, 1023, 1024, 1535, 1536, 2048]:
    npu_result = (test_val + 512) >> 10
    pytorch_result = int(round(test_val / 1024))
    diff = npu_result - pytorch_result
    print(f"     x={test_val:5d}: NPU={npu_result:3d}, PyTorch={pytorch_result:3d}, diff={diff:2d}")

print("\n6. Hypothesis about the 3.5x error:")
print(f"   Tolerance:     2 * int8_scale = {2 * int8_scale:.6f}")
print(f"   Actual error:  ~{0.0546:.6f}")
print(f"   Ratio:         {0.0546 / (2 * int8_scale):.2f}x")
print(f"\n   Possible causes:")
print(f"   a) Weight reordering error (accessing wrong weights)")
print(f"   b) Input data reordering error (accessing wrong inputs)")
print(f"   c) Border handling mismatch (replicate padding vs kernel logic)")
print(f"   d) Depth accumulation issue (kernel_depth parameter)")
print(f"   e) Scale parameter not being applied correctly")

print("\n7. Check Weight Indexing:")
print(f"   Weight layout: {{O/8}}{{I/8}}KDHW{{I8}}{{O8}}")
print(f"   Formula: (kd*KH*KW*64) + (kh*KW*64) + (kw*64) + (ic*KD*KH*KW*64) + (ic8*8) + (oc_ofst*(IC/8)*KD*KH*KW*64) + oc8")
print(f"   For KD=3, KH=3, KW=3, this gives strides:")
print(f"     kd:     {3*3*64} = {3*3*64}")
print(f"     kh:     {3*64} = {3*64}")
print(f"     kw:     {64}")
print(f"     ic:     {3*3*3*64} = {3*3*3*64}")
print(f"     oc_ofst: (IC/8)*{3*3*3*64}")

print("\n8. PyTorch Weight Layout:")
print(f"   PyTorch: [co, ci, kd=1, kh, kw]")
print(f"   Reordering fills all kd positions (0,1,2) with same weight from PyTorch kd=0")
print(f"   But kernel uses kernel_depth=1, so only kd=1 is actually used (middle plane)")
print(f"\n   CHECK: Are we indexing the correct kd position in the weights?")

print("\n9. Recommended Debug Steps:")
print(f"   1. Compare first few raw int32 accumulation values")
print(f"   2. Check if error pattern is spatial (certain positions worse)")
print(f"   3. Verify weight indexing matches reordering")
print(f"   4. Test with simple known inputs (e.g., all 1s)")
print(f"   5. Check if error scales with number of accumulations")

print("\n" + "="*80)
