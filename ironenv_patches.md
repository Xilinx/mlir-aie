# MLIR-AIE Python Extras Compatibility Patches

This document describes the patches applied to fix version incompatibilities in the mlir-aie wheel version `0.0.1.2026030604` (released 2026-03-06).

## Root Cause

The `aie/extras/` package (upstream MLIR Python bindings) expects functions and methods that don't exist in the mlir-aie wheel's dialect implementations. This causes import errors when trying to use `aie.iron` or any IRON API.

## Error Messages Before Patching

```python
ImportError: cannot import name '_is_integer_like_type' from 'aie.dialects.arith'
ImportError: cannot import name '_is_complex_type' from 'aie.dialects.linalg.opdsl.lang.emitter'
AttributeError: type object 'aie._mlir_libs._mlir.ir.ShapedType' has no attribute 'isinstance'
```

## Patches Applied

All patches are in `/scratch/jmelber/mlir-aie/ironenv/lib/python3.12/site-packages/`

---

### Patch 1: Add `_is_integer_like_type` to arith.py

**File:** `mlir_aie/python/aie/dialects/arith.py`

**Location:** End of file (after line 90, after the `constant` function)

**Added:**
```python
# Compatibility function for extras package
def _is_integer_like_type(type):
    """Check if a type is integer-like (IntegerType or IndexType)."""
    return isinstance(type, (IntegerType, IndexType))
```

**Reason:** The `aie/extras/dialects/arith.py` (line 24) tries to import this function but it doesn't exist in the mlir-aie version.

---

### Patch 2: Add type checking functions to linalg emitter

**File:** `mlir_aie/python/aie/dialects/linalg/opdsl/lang/emitter.py`

**Location:** End of file (after the `_is_bool_type` function, around line 300+)

**Added:**
```python
# Compatibility functions for extras package
def _is_integer_like_type(t: Type) -> bool:
    """Check if a type is integer-like (IntegerType or IndexType)."""
    return isinstance(t, (IntegerType, IndexType))


def _is_index_type(t: Type) -> bool:
    """Check if a type is IndexType."""
    return isinstance(t, IndexType)


def _is_floating_point_type(t: Type) -> bool:
    """Check if a type is FloatType."""
    return isinstance(t, FloatType)


def _is_complex_type(t: Type) -> bool:
    """Check if a type is ComplexType."""
    return isinstance(t, ComplexType)
```

**Reason:** The `aie/extras/dialects/arith.py` (lines 25-28) tries to import these functions:
```python
from ...dialects.linalg.opdsl.lang.emitter import (
    _is_complex_type,
    _is_floating_point_type,
    _is_index_type,
)
```

---

### Patch 3: Fix `ShapedType.isinstance` usage

**File:** `aie/extras/dialects/arith.py`

**Location:** Lines 104-113 (in the `constant` function)

**Original code:**
```python
    if _is_floating_point_type(type) and not isinstance(value, np.ndarray):
        value = float(value)

    if ShapedType.isinstance(type) and isinstance(value, (int, float, bool)):
        ranked_tensor_type = ShapedType(type)
        value = np.full(
            ranked_tensor_type.shape,
            value,
            dtype=mlir_type_to_np_dtype(ranked_tensor_type.element_type),
        )
```

**Patched code:**
```python
    if _is_floating_point_type(type) and not isinstance(value, np.ndarray):
        value = float(value)

    # Compatibility fix: Use try/except instead of ShapedType.isinstance
    try:
        ranked_tensor_type = ShapedType(type)
        is_shaped = True
    except:
        is_shaped = False

    if is_shaped and isinstance(value, (int, float, bool)):
        ranked_tensor_type = ShapedType(type)
        value = np.full(
            ranked_tensor_type.shape,
            value,
            dtype=mlir_type_to_np_dtype(ranked_tensor_type.element_type),
        )
```

**Reason:** The static method `ShapedType.isinstance()` doesn't exist in the current MLIR Python bindings. The workaround uses a try/except block to check if a type can be converted to `ShapedType`.

---

## Verification

After applying these patches, the following imports work successfully:

```bash
$ /scratch/jmelber/mlir-aie/ironenv/bin/python3 -c "from aie.iron import Worker; print('SUCCESS')"
Failed to import PyXRT: No module named 'pyxrt', proceeding without runtime libraries.
SUCCESS
```

The PyXRT warning is expected and doesn't affect MLIR generation.

## Testing

Successfully generated MLIR for conv3d example:
```bash
$ cd programming_examples/ml/conv3d
$ /scratch/jmelber/mlir-aie/ironenv/bin/python3 conv3d.py npu 8 8 8 8 8 0 > build/aie2.mlir
```

Result: 82-line MLIR file generated with no errors.

---

## Recommendation

These patches should be reported to the mlir-aie maintainers as they indicate a version incompatibility in the wheel release process. The proper fix would be to:

1. Update the mlir-aie dialect implementations to include the missing functions
2. Or update the extras package to not rely on these functions
3. Or pin the MLIR Python bindings to a compatible version

## Files Modified

1. `/scratch/jmelber/mlir-aie/ironenv/lib/python3.12/site-packages/mlir_aie/python/aie/dialects/arith.py`
2. `/scratch/jmelber/mlir-aie/ironenv/lib/python3.12/site-packages/mlir_aie/python/aie/dialects/linalg/opdsl/lang/emitter.py`
3. `/scratch/jmelber/mlir-aie/ironenv/lib/python3.12/site-packages/aie/extras/dialects/arith.py`

## Date Applied

2026-03-06

## Wheel Version

- `mlir-aie`: 0.0.1.2026030604+ce37267
- `llvm-aie`: 20.0.0.2026030601+0778a83d
