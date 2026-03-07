#!/usr/bin/env python3
#
# Debug the padding and reordering issue
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

# Generate same random data
int_inp = torch.randint(1, 20, (1, ci, depth, height, width)).type(torch.FloatTensor)
int_weight = torch.randint(-50, 50, (co, ci, 1, 3, 3)).type(torch.FloatTensor)

print("="*80)
print("DEBUGGING PADDING AND INPUT INDEXING")
print("="*80)

print(f"\nOriginal input shape: {int_inp.shape}")
print("Format: [batch, channels, depth, height, width]")

# Apply replication padding
int_inp_padded = torch.nn.functional.pad(int_inp, (1, 1, 1, 1, 0, 0), mode='replicate')
print(f"\nPadded input shape: {int_inp_padded.shape}")
print("Padding: (left=1, right=1, top=1, bottom=1, front=0, back=0)")

print(f"\nOriginal input [0,0,0,:,:]:")
print(int_inp[0,0,0,:,:])

print(f"\nPadded input [0,0,0,:,:]:")
print(int_inp_padded[0,0,0,:,:])

# Now let's compute what PyTorch actually computes
class Conv3dModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(ci, co, kernel_size=(1,3,3), padding=0, bias=False)

model = Conv3dModel()
model.eval()
model.conv.weight.data.copy_(int_weight)

out = model.conv(int_inp_padded)
print(f"\nConv output shape: {out.shape}")
print("Format: [batch, out_channels, depth, height, width]")

# The output height/width should be same as input (8x8) because we padded
print(f"Expected output height/width: {height} x {width}")
print(f"Actual output height/width: {out.shape[3]} x {out.shape[4]}")

# Now let's manually compute one output position
print("\n" + "="*80)
print("MANUAL COMPUTATION FOR POSITION [0, 0, 0, 4, 4]")
print("="*80)

# Output at [batch=0, out_channel=0, depth=0, h=4, w=4]
manual_sum = 0.0
print("\nConvolution window:")
for ic in range(ci):
    for kh in range(3):
        for kw in range(3):
            # For output position (h=4, w=4), the kernel window covers:
            # Input positions: (h+kh-1, w+kw-1) for kh,kw in [0,1,2]
            # Since padding adds 1 pixel on each side (top, bottom, left, right),
            # the padded input indices are just h+kh, w+kw
            h_padded = 4 + kh
            w_padded = 4 + kw

            inp_val = int_inp_padded[0, ic, 0, h_padded, w_padded].item()
            wt_val = int_weight[0, ic, 0, kh, kw].item()
            manual_sum += inp_val * wt_val

            if ic == 0:  # Print for first channel
                print(f"  kh={kh}, kw={kw}: inp_padded[0,{ic},0,{h_padded},{w_padded}]={inp_val:.0f} * wt[0,{ic},0,{kh},{kw}]={wt_val:.0f} = {inp_val*wt_val:.0f}")

print(f"\nManual sum: {manual_sum:.1f}")
print(f"PyTorch output: {out[0, 0, 0, 4, 4].item():.1f}")
print(f"Match: {abs(manual_sum - out[0, 0, 0, 4, 4].item()) < 0.1}")

# Now check what the NPU kernel sees
print("\n" + "="*80)
print("NPU KERNEL INPUT INDEXING")
print("="*80)

print("\nNPU kernel receives input WITHOUT external padding.")
print("It handles borders internally using replicate logic.")
print("")
print("For NPU at output position (h=4, w=4):")
print("  This is a center position (1 <= h < height-1, 1 <= w < width-1)")
print("  Kernel accesses input at (y-1+kh, x-1+kw) for kh,kw in [0,1,2]")
print("  That's positions: (3,3), (3,4), (3,5), (4,3), ..., (5,5)")
print("")

# What does the NPU see?
print("NPU input (no padding, original):")
for kh in range(3):
    for kw in range(3):
        y_npu = 4 - 1 + kh  # Output y=4, kernel offset kh
        x_npu = 4 - 1 + kw  # Output x=4, kernel offset kw
        val = int_inp[0, 0, 0, y_npu, x_npu].item()
        print(f"  kh={kh}, kw={kw}: inp[0,0,0,{y_npu},{x_npu}] = {val:.0f}")

print("\nPyTorch input (with padding):")
for kh in range(3):
    for kw in range(3):
        h_padded = 4 + kh
        w_padded = 4 + kw
        val = int_inp_padded[0, 0, 0, h_padded, w_padded].item()
        print(f"  kh={kh}, kw={kw}: inp_padded[0,0,0,{h_padded},{w_padded}] = {val:.0f}")

# Check if they match
print("\nDo they match?")
match = True
for kh in range(3):
    for kw in range(3):
        y_npu = 4 - 1 + kh
        x_npu = 4 - 1 + kw
        h_padded = 4 + kh
        w_padded = 4 + kw
        npu_val = int_inp[0, 0, 0, y_npu, x_npu].item()
        pt_val = int_inp_padded[0, 0, 0, h_padded, w_padded].item()
        if npu_val != pt_val:
            print(f"  MISMATCH at kh={kh}, kw={kw}: NPU={npu_val:.0f}, PyTorch={pt_val:.0f}")
            match = False

if match:
    print("  YES - inputs match for center position")

print("\n" + "="*80)
