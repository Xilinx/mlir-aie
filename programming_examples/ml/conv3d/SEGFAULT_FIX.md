# Segfault Fix for Multi-Core Conv3D (32 Output Channels)

## Root Cause Analysis

The test.py segfault was caused by **four critical buffer allocation and indexing bugs** in the multi-core execution path (lines 147-221).

## Bugs Identified and Fixed

### Bug #1: Incorrect Weight Reshaping (Line 170-173)
**Problem:**
```python
wts_core = wts.reshape(co8, ci8, 3, 3, 3, 8, 8)[wts_start:wts_end].flatten()
```
The `wts` array was **already shaped** as `(co8, ci8, 3, 3, 3, 8, 8)` from line 131. Calling `.reshape()` again with the same shape is redundant, but more critically, this caused confusion about the array structure.

**Fix:**
```python
# wts is already shaped as (co8, ci8, 3, 3, 3, 8, 8)
wts_core = wts[wts_start:wts_end].flatten()
```
Directly slice the first dimension (co8) to extract weights for each core.

**Impact:** This was causing incorrect weight extraction, potentially leading to out-of-bounds memory access.

---

### Bug #2: Incorrect Buffer Size Calculation (Line 176-178)
**Problem:**
```python
out_size_per_core = (np.prod(shape_out) // n_cores) * dtype_out.itemsize
buffers.append(iron.zeros(out_size_per_core, dtype=dtype_out))
```
`iron.zeros()` expects **number of elements**, not **number of bytes**. Multiplying by `dtype_out.itemsize` (which is 1 for uint8) created buffers that were technically the same size by accident, but the code was semantically wrong. For other data types (e.g., int32), this would allocate 4x too much memory.

**Fix:**
```python
# Output buffers per core (in elements, not bytes)
out_size_per_core = np.prod(shape_out) // n_cores
buffers.append(iron.zeros(out_size_per_core, dtype=dtype_out))
```

**Impact:** Incorrect buffer allocation that would fail with non-byte data types.

---

### Bug #3: Incorrect Output Concatenation (Line 217-221)
**Problem:**
```python
# Multi-core: concatenate outputs from all cores
out_tensors = []
for c in range(n_cores):
    out_idx = n_cores * 2 + c
    out_t = buffers[out_idx]
    if not isinstance(out_t, np.ndarray):
        out_t = out_t.numpy()
    out_tensors.append(out_t)
# Concatenate along channel dimension
data_buffer = np.concatenate(out_tensors, axis=0) * int8_scale
```
This attempted to concatenate **flat 1D buffers** along axis=0, which simply chains them sequentially. However, the outputs need to be:
1. Reshaped to their proper 5D layout: `(depth, co8_per_core, height, 8, width)`
2. Concatenated along the **channel dimension** (axis=1, the co8 dimension)
3. Then flattened back to 1D for the subsequent reordering step

**Fix:**
```python
# Multi-core: concatenate outputs from all cores
# Each core produces shape (depth, co8_per_core, height, 8, width)
oc8_per_core = (co // n_cores) // 8
out_shape_per_core = (depth, oc8_per_core, height, 8, width)
out_tensors = []
for c in range(n_cores):
    out_idx = n_cores * 2 + c  # After inputs and weights
    out_t = buffers[out_idx]
    if not isinstance(out_t, np.ndarray):
        out_t = out_t.numpy()
    # Reshape to proper layout
    out_t_reshaped = out_t.reshape(out_shape_per_core)
    out_tensors.append(out_t_reshaped)
# Concatenate along channel dimension (axis=1, the co8 dimension)
data_buffer = np.concatenate(out_tensors, axis=1).flatten() * int8_scale
```

**Impact:** This was likely causing the segfault because the improperly concatenated buffer would have the wrong shape for the subsequent reordering step (line 245), leading to out-of-bounds access.

---

### Bug #4: Trace Configuration Variable Reference (Line 189-190)
**Problem:**
```python
if enable_trace:
    trace_config = TraceConfig(
        ...
        last_tensor_shape=out.shape,
        last_tensor_dtype=out.dtype,
    )
```
In multi-core mode, the variable `out` was never defined. Only the single-core path creates an `out` variable.

**Fix:**
```python
if enable_trace:
    last_tensor = buffers[-1]
    trace_config = TraceConfig(
        ...
        last_tensor_shape=last_tensor.shape,
        last_tensor_dtype=last_tensor.dtype,
    )
```

**Impact:** Would crash immediately if tracing was enabled in multi-core mode.

---

## Buffer Layout Summary

### Single-Core Buffer Layout:
```
buffers = [
    in1,      # Full input: (depth, ci8, height, 8, width) flattened
    in2,      # Full weights: (co8, ci8, 3, 3, 3, 8, 8) flattened
    out       # Full output: (depth, co8, height, 8, width) flattened
]
```

### Multi-Core Buffer Layout (4 cores):
```
buffers = [
    # Inputs (duplicated for each core):
    in_core0,  # Full input duplicated
    in_core1,  # Full input duplicated
    in_core2,  # Full input duplicated
    in_core3,  # Full input duplicated

    # Weights (split across cores):
    wts_core0,  # Weights for channels 0-7:   (co8/4, ci8, 3, 3, 3, 8, 8) flattened
    wts_core1,  # Weights for channels 8-15:  (co8/4, ci8, 3, 3, 3, 8, 8) flattened
    wts_core2,  # Weights for channels 16-23: (co8/4, ci8, 3, 3, 3, 8, 8) flattened
    wts_core3,  # Weights for channels 24-31: (co8/4, ci8, 3, 3, 3, 8, 8) flattened

    # Outputs (split across cores):
    out_core0,  # Output channels 0-7:   (depth, co8/4, height, 8, width) flattened
    out_core1,  # Output channels 8-15:  (depth, co8/4, height, 8, width) flattened
    out_core2,  # Output channels 16-23: (depth, co8/4, height, 8, width) flattened
    out_core3,  # Output channels 24-31: (depth, co8/4, height, 8, width) flattened
]
```

Buffer indexing in multi-core:
- Inputs: `buffers[0:n_cores]`
- Weights: `buffers[n_cores:n_cores*2]`
- Outputs: `buffers[n_cores*2:n_cores*3]`

## Testing Verification

After these fixes, the test should:
1. Correctly allocate all buffers without overflow
2. Split weights properly across cores
3. Concatenate outputs correctly along the channel dimension
4. Not segfault during execution or output processing

## Files Modified
- `/scratch/jmelber/mlir-aie/programming_examples/ml/conv3d/test.py` (lines 157, 170-179, 184-227)
