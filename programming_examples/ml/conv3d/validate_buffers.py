#!/usr/bin/env python3
"""
Standalone validation script for multi-core buffer allocation logic.
This can be run without NPU hardware to verify the buffer sizes are correct.
"""

import numpy as np

def validate_multicore_buffers(depth, height, width, ci, co, n_cores):
    """Validate buffer allocation for multi-core conv3d"""

    print(f"\n{'='*60}")
    print(f"Validating {n_cores}-core configuration")
    print(f"{'='*60}")
    print(f"Volume: {depth}x{height}x{width}")
    print(f"Channels: {ci} -> {co}")

    # Calculate dimensions
    ci8 = ci // 8
    co8 = co // 8
    oc_per_core = co // n_cores
    oc8_per_core = oc_per_core // 8

    assert co % (n_cores * 8) == 0, f"co={co} must be divisible by {n_cores * 8}"

    # Input shape and size
    shape_in_act = (depth, ci8, height, 8, width)
    input_size = np.prod(shape_in_act)

    # Weight shapes
    wts_shape_full = (co8, ci8, 3, 3, 3, 8, 8)
    wts_shape_per_core = (oc8_per_core, ci8, 3, 3, 3, 8, 8)
    wts_size_full = np.prod(wts_shape_full)
    wts_size_per_core = np.prod(wts_shape_per_core)

    # Output shapes
    shape_out_full = (depth, co8, height, 8, width)
    shape_out_per_core = (depth, oc8_per_core, height, 8, width)
    out_size_full = np.prod(shape_out_full)
    out_size_per_core = np.prod(shape_out_full) // n_cores

    print(f"\nBuffer Sizes:")
    print(f"  Input per core:      {input_size:8d} elements (duplicated)")
    print(f"  Weights per core:    {wts_size_per_core:8d} elements")
    print(f"  Output per core:     {out_size_per_core:8d} elements")
    print(f"  Total output:        {out_size_full:8d} elements")

    # Verify weight split
    print(f"\nWeight Split Verification:")
    assert wts_size_per_core * n_cores == wts_size_full, \
        f"Weight split error: {wts_size_per_core} * {n_cores} != {wts_size_full}"
    print(f"  ✓ {wts_size_per_core} * {n_cores} = {wts_size_full}")

    # Verify output split
    print(f"\nOutput Split Verification:")
    assert out_size_per_core * n_cores == out_size_full, \
        f"Output split error: {out_size_per_core} * {n_cores} != {out_size_full}"
    print(f"  ✓ {out_size_per_core} * {n_cores} = {out_size_full}")

    # Simulate weight slicing
    print(f"\nWeight Slicing Test:")
    wts_dummy = np.arange(wts_size_full, dtype=np.int8).reshape(wts_shape_full)
    for c in range(n_cores):
        wts_start = c * oc8_per_core
        wts_end = (c + 1) * oc8_per_core
        wts_core = wts_dummy[wts_start:wts_end]
        assert wts_core.shape == wts_shape_per_core, \
            f"Core {c}: expected shape {wts_shape_per_core}, got {wts_core.shape}"
        print(f"  Core {c}: wts[{wts_start}:{wts_end}] -> {wts_core.shape}, {wts_core.size} elements ✓")

    # Simulate output concatenation
    print(f"\nOutput Concatenation Test:")
    out_tensors = []
    for c in range(n_cores):
        out_dummy = np.zeros(shape_out_per_core, dtype=np.uint8)
        out_tensors.append(out_dummy)

    concatenated = np.concatenate(out_tensors, axis=1)
    assert concatenated.shape == shape_out_full, \
        f"Concatenation error: expected {shape_out_full}, got {concatenated.shape}"
    print(f"  Concatenated shape: {concatenated.shape}")
    print(f"  Expected shape:     {shape_out_full}")
    print(f"  ✓ Match!")

    flattened = concatenated.flatten()
    assert flattened.size == out_size_full, \
        f"Flatten error: expected {out_size_full}, got {flattened.size}"
    print(f"  Flattened size: {flattened.size} ✓")

    print(f"\n{'='*60}")
    print(f"✓ All validations passed for {n_cores}-core configuration!")
    print(f"{'='*60}\n")
    return True

if __name__ == "__main__":
    # Test single-core configuration
    print("\n" + "="*60)
    print("SINGLE-CORE VALIDATION (8 output channels)")
    print("="*60)
    validate_multicore_buffers(
        depth=8, height=8, width=8,
        ci=8, co=8, n_cores=1
    )

    # Test multi-core configuration
    print("\n" + "="*60)
    print("MULTI-CORE VALIDATION (32 output channels, 4 cores)")
    print("="*60)
    validate_multicore_buffers(
        depth=8, height=8, width=8,
        ci=8, co=32, n_cores=4
    )

    print("\n✓✓✓ ALL TESTS PASSED ✓✓✓\n")
