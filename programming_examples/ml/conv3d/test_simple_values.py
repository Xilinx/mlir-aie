#!/usr/bin/env python3
#
# Test Conv3D with simple known values to debug quantization
#

import torch
import torch.nn as nn
import numpy as np

torch.use_deterministic_algorithms(True)
torch.manual_seed(0)

# Test parameters
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
print("SIMPLE TEST CASE: Verify Quantization with Known Values")
print("="*80)

# Test 1: All ones (simplest case)
print("\nTest 1: All-ones input and weights")
print("-" * 80)

int_inp = torch.ones(1, ci, depth, height, width)
int_weight = torch.ones(co, ci, 1, 3, 3)

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

print(f"Expected raw convolution output (center pixel, no borders):")
print(f"  Sum = {ci} channels * 3*3 kernel = {ci * 9}")
print(f"  (since all inputs and weights are 1)")

print(f"\nActual PyTorch output (first element):")
print(f"  Raw conv:         {raw_conv_out.flatten()[0]:.1f}")
print(f"  After conv_scale: {quant_intermediate.flatten()[0]:.6e}")
print(f"  After quant:      {golden_output.flatten()[0]:.6f}")

print(f"\nExpected NPU computation:")
print(f"  int32 sum = {ci * 9}")
print(f"  Shifted = ({ci * 9} + 512) >> 10 = {(ci * 9 + 512) >> 10}")
print(f"  Output = {(ci * 9 + 512) >> 10} * {int8_scale} = {((ci * 9 + 512) >> 10) * int8_scale:.6f}")

diff = golden_output.flatten()[0].item() - ((ci * 9 + 512) >> 10) * int8_scale
print(f"\nExpected difference: {diff:.6f}")
if abs(diff) < 2 * int8_scale:
    print("  PASS: Within tolerance")
else:
    print(f"  FAIL: Exceeds tolerance of {2*int8_scale:.6f}")

# Test 2: Check border vs center pixels
print("\n\nTest 2: Border vs Center Pixel Comparison")
print("-" * 80)

output_np = golden_output.squeeze().detach().cpu().numpy()
print(f"Output shape: {output_np.shape}  (channels, depth, height, width)")

# Check a few positions
print(f"\nChannel 0, Depth 0 values:")
print(f"  Top-left corner (0,0):     {output_np[0, 0, 0, 0]:.6f}")
print(f"  Top-right corner (0,{width-1}):    {output_np[0, 0, 0, width-1]:.6f}")
print(f"  Bottom-left corner ({height-1},0): {output_np[0, 0, height-1, 0]:.6f}")
print(f"  Center ({height//2},{width//2}):        {output_np[0, 0, height//2, width//2]:.6f}")

# Check if corners have different values (they should due to border replication)
unique_values = np.unique(output_np[0, 0])
print(f"\nNumber of unique values in first output plane: {len(unique_values)}")
print(f"Unique values: {unique_values}")

if len(unique_values) == 1:
    print("  WARNING: All values are identical - border handling may not be working")
else:
    print("  OK: Different values detected (border handling is active)")

# Test 3: Manual calculation for corner
print("\n\nTest 3: Manual Calculation for Top-Left Corner (0,0)")
print("-" * 80)

print("Top-left corner has reduced receptive field due to borders:")
print("  - kw loop starts at kw=1 (skip kw=0 due to left border)")
print("  - kh loop: kh=0 uses top border (replicated), kh=1,2 use actual rows")
print("")

# For all-ones input with replicate padding, corner pixel sees:
# kw=1,2 (not kw=0), kh=0,1,2, all input channels
# That's 2 * 3 * ci = 2 * 3 * 8 = 48 multiply-adds
corner_sum = 2 * 3 * ci
print(f"Expected corner accumulation: {corner_sum}")
print(f"After right shift: ({corner_sum} + 512) >> 10 = {(corner_sum + 512) >> 10}")
print(f"After scaling: {((corner_sum + 512) >> 10) * int8_scale:.6f}")

# Test 4: Weight reordering verification
print("\n\nTest 4: Weight Reordering Check")
print("-" * 80)

wts_orig = int_weight.data.numpy().astype(np.int8)  # [co, ci, 1, 3, 3]
co8, ci8 = co // 8, ci // 8
wts_reordered = np.zeros((co8, ci8, 3, 3, 3, 8, 8), dtype=np.int8)

for oc8 in range(co8):
    for ic8 in range(ci8):
        for kd in range(3):
            for kh in range(3):
                for kw in range(3):
                    for i in range(8):
                        for o in range(8):
                            wts_reordered[oc8, ic8, kd, kh, kw, i, o] = wts_orig[
                                oc8 * 8 + o, ic8 * 8 + i, 0, kh, kw
                            ]

# Check if weight at position (oc=0, ic=0, kd=1, kh=1, kw=1, ic8=0, oc8=0) is 1
# This is the center of the 3D kernel
center_weight = wts_reordered[0, 0, 1, 1, 1, 0, 0]
print(f"Center kernel weight [oc8=0, ic8=0, kd=1, kh=1, kw=1, ic8_inner=0, oc8_inner=0]: {center_weight}")

# Compute the index this weight would have in flattened array
wts_flat = wts_reordered.flatten()
# Index calculation based on kernel formula:
# (kd*KH*KW*64) + (kh*KW*64) + (kw*64) + (ic*KD*KH*KW*64) + (ic8*8) + (oc_ofst*IC/8*KD*KH*KW*64) + oc8
oc8_val, ic8_val, kd, kh, kw, ic8_inner, oc8_inner = 0, 0, 1, 1, 1, 0, 0
wts_indx = (kd * 3 * 3 * 64) + (kh * 3 * 64) + (kw * 64) + \
           (ic8_val * 3 * 3 * 3 * 64) + (ic8_inner * 8) + \
           (oc8_val * ci8 * 3 * 3 * 3 * 64) + oc8_inner

print(f"Computed index: {wts_indx}")
print(f"Weight at computed index: {wts_flat[wts_indx]}")

if wts_flat[wts_indx] == center_weight:
    print("  PASS: Weight indexing formula is correct")
else:
    print("  FAIL: Weight indexing formula is INCORRECT!")
    print(f"  Expected {center_weight}, got {wts_flat[wts_indx]}")

print("\n" + "="*80)
