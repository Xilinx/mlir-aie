#!/usr/bin/env python3
#
# Analyze the actual test data to find quantization mismatch
#

import torch
import torch.nn as nn
import numpy as np

torch.use_deterministic_algorithms(True)
torch.manual_seed(0)

# Test parameters (from test.py)
ci = 8
co = 8
depth = 8
height = 8
width = 8

# Quantization scales
conv_scale = 7.6294e-06
int8_scale = 0.0078
min_val = 0
max_val = 255

print("="*80)
print("ANALYZING ACTUAL TEST DATA")
print("="*80)

# Generate same random data as test.py
int_inp = torch.randint(1, 20, (1, ci, depth, height, width)).type(torch.FloatTensor)
int_weight = torch.randint(-50, 50, (co, ci, 1, 3, 3)).type(torch.FloatTensor)

print(f"\nInput statistics:")
print(f"  Min: {int_inp.min():.1f}, Max: {int_inp.max():.1f}")
print(f"  Mean: {int_inp.mean():.2f}, Std: {int_inp.std():.2f}")

print(f"\nWeight statistics:")
print(f"  Min: {int_weight.min():.1f}, Max: {int_weight.max():.1f}")
print(f"  Mean: {int_weight.mean():.2f}, Std: {int_weight.std():.2f}")

# PyTorch model
class Conv3dModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(ci, co, kernel_size=(1,3,3), padding=0, bias=False)

    def forward(self, x):
        out_int = self.conv(x)
        out_quant = out_int * conv_scale
        out_float = int8_scale * torch.clamp(
            torch.round(out_quant / int8_scale), min_val, max_val
        )
        return out_float, out_int, out_quant

model = Conv3dModel()
model.eval()
model.conv.weight.data.copy_(int_weight)

# Apply replication padding
int_inp_padded = torch.nn.functional.pad(int_inp, (1, 1, 1, 1, 0, 0), mode='replicate')
golden_output, raw_conv_out, quant_intermediate = model(int_inp_padded)

print(f"\nRaw convolution output statistics:")
print(f"  Min: {raw_conv_out.min():.1f}, Max: {raw_conv_out.max():.1f}")
print(f"  Mean: {raw_conv_out.mean():.2f}, Std: {raw_conv_out.std():.2f}")

print(f"\nQuantized output (uint8 range, before scaling back):")
quant_uint8 = torch.clamp(torch.round(quant_intermediate / int8_scale), min_val, max_val)
print(f"  Min: {quant_uint8.min():.1f}, Max: {quant_uint8.max():.1f}")
print(f"  Mean: {quant_uint8.mean():.2f}, Std: {quant_uint8.std():.2f}")

print(f"\nFinal output (after scaling):")
print(f"  Min: {golden_output.min():.6f}, Max: {golden_output.max():.6f}")
print(f"  Mean: {golden_output.mean():.6f}, Std: {golden_output.std():.6f}")

# Now let's manually compute what NPU should produce
print("\n" + "="*80)
print("MANUAL NPU COMPUTATION FOR FIRST OUTPUT")
print("="*80)

# Get first output channel, first depth, center position (4,4) to avoid borders
c_idx, d_idx, h_idx, w_idx = 0, 0, 4, 4

print(f"\nComputing output for position [c={c_idx}, d={d_idx}, h={h_idx}, w={w_idx}]")
print("This is a center position, so full 3x3 kernel applies")

# Manual computation
int32_sum = 0
for ic in range(ci):
    for kh in range(3):
        for kw in range(3):
            # Input index (with padding)
            inp_h = h_idx - 1 + kh + 1  # +1 for padding offset
            inp_w = w_idx - 1 + kw + 1  # +1 for padding offset
            inp_val = int_inp[0, ic, d_idx, inp_h, inp_w].item()
            wt_val = int_weight[c_idx, ic, 0, kh, kw].item()
            int32_sum += inp_val * wt_val
            if ic == 0 and kh < 2 and kw < 2:  # Print first few
                print(f"  inp[{ic},{d_idx},{inp_h},{inp_w}]={inp_val:.0f} * wt[{c_idx},{ic},0,{kh},{kw}]={wt_val:.0f} = {inp_val*wt_val:.0f}")

print(f"\nTotal int32 accumulation: {int32_sum:.0f}")

# NPU quantization
int32_sum = int(int32_sum)
npu_shifted = (int32_sum + 512) >> 10
npu_clamped = max(0, min(255, npu_shifted))
npu_output = npu_clamped * int8_scale

print(f"After right shift: ({int32_sum:.0f} + 512) >> 10 = {npu_shifted}")
print(f"After clamp to [0,255]: {npu_clamped}")
print(f"After scaling by {int8_scale}: {npu_output:.6f}")

# PyTorch value for same position
pytorch_val = golden_output[0, c_idx, d_idx, h_idx, w_idx].item()
pytorch_raw = raw_conv_out[0, c_idx, d_idx, h_idx, w_idx].item()

print(f"\nPyTorch for same position:")
print(f"  Raw conv output: {pytorch_raw:.1f}")
print(f"  After quantization: {pytorch_val:.6f}")

print(f"\nDifference: {abs(npu_output - pytorch_val):.6f}")
print(f"Tolerance (2x int8_scale): {2*int8_scale:.6f}")

if abs(npu_output - pytorch_val) < 2*int8_scale:
    print("PASS: Center pixel matches within tolerance")
else:
    print("FAIL: Center pixel does NOT match")
    print(f"  Error is {abs(npu_output - pytorch_val) / (2*int8_scale):.2f}x tolerance")

# Check if raw convolution matches
if abs(int32_sum - pytorch_raw) < 0.1:
    print("\nRaw convolution matches PyTorch - problem is in quantization")
else:
    print(f"\nRaw convolution MISMATCH: NPU={int32_sum:.1f}, PyTorch={pytorch_raw:.1f}")
    print("Problem is in the convolution itself, not quantization")

print("\n" + "="*80)
print("HYPOTHESIS TESTING")
print("="*80)

# Test if the scale relationship is causing systematic bias
print("\nScale relationship:")
print(f"  conv_scale / int8_scale = {conv_scale / int8_scale:.10e}")
print(f"  1 / 1024 = {1/1024:.10e}")
print(f"  Ratio = {(conv_scale / int8_scale) / (1/1024):.6f}")

# The ratio is ~1.0016, meaning NPU scale is slightly too large
# This would cause NPU outputs to be ~0.16% smaller than PyTorch

# Check maximum possible accumulation value
max_sum = ci * 3 * 3 * 20 * 50  # max_inp * max_wt * num_muls
min_sum = ci * 3 * 3 * 1 * (-50)  # min_inp * min_wt * num_muls
print(f"\nAccumulation range:")
print(f"  Maximum possible: {max_sum} (right shift: {max_sum >> 10})")
print(f"  Minimum possible: {min_sum} (right shift: {(min_sum + 512) >> 10} after clamp)")

print("\n" + "="*80)
