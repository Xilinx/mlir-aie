# Add One - C++ Driver Test

This test demonstrates using the **C++ aiecc driver** instead of the Python aiecc.py.

## Test Overview

This test performs a simple operation: adding 41 to each element of an input array using AIE cores.

## Key Differences from Python Driver Tests

The main difference is in `run.lit`:

**Python driver (old):**
```
RUN: %python aiecc.py --no-aiesim --aie-generate-xclbin --aie-generate-npu-insts ...
```

**C++ driver (new):**
```
RUN: aiecc --aie-generate-xclbin --aie-generate-npu-insts ...
```

## Benefits of C++ Driver

1. **Performance**: Native C++ implementation with direct MLIR API access
2. **Integration**: Better integration with LLVM/MLIR C++ infrastructure
3. **Maintenance**: Type-safe C++ code with compile-time checks
4. **Consistency**: Same command-line interface as Python version

## Test Structure

- `aie.mlir`: AIE design with objectFifos and runtime sequence
- `run.lit`: LIT test configuration using C++ aiecc
- `test.cpp`: Host test program (same as Python driver version)

## Running

This test is executed as part of the mlir-aie test suite:

```bash
cd build
ninja check-aie
```

Or run specifically:

```bash
lit test/npu-xrt/add_one_cpp_driver
```
