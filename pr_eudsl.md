# fix: replace deprecated np.bool with np.bool_

`np.bool` was removed in NumPy 1.24. Replace the dict key in
`_np_dtype_to_mlir_type_ctor` with `np.bool_`, which works across all
supported NumPy versions (1.19.5+).

**File:** `projects/eudsl-python-extras/mlir/extras/util.py:135`
