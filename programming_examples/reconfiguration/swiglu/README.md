# SwiGLU Reconfiguration Variants

This directory contains three implementations of the SwiGLU operator, each demonstrating different NPU execution patterns:

1. **separate_xclbins** - Each kernel compiled to a separate xclbin, executed sequentially
2. **runlist** - All kernels combined into a single xclbin, executed via runlist
3. **fused_transactions** - All kernels compiled into a single ELF file with fused transactions

## Directory Structure

```
swiglu/
├── separate_xclbins/
│   ├── Makefile
│   └── test.cpp
├── runlist/
│   ├── Makefile
│   └── test.cpp
└── fused_transactions/
    ├── Makefile
    ├── test.cpp
    └── combine_mlir.py
```

## Building

Each variant can be built independently:

```bash
cd separate_xclbins && make
cd runlist && make
cd fused_transactions && make
```

## Running Individual Tests

```bash
cd separate_xclbins && make test
cd runlist && make test
cd fused_transactions && make test
```

## Running Benchmarks

From the parent `reconfiguration` directory:

```bash
./run_benchmarks.sh
```

This will:
- Run all three variants
- Collect timing data (100 iterations each)
- Save results to `benchmark_results.csv`

## Visualizing Results

```bash
python3 visualize_results.py
```

This will:
- Read `benchmark_results.csv`
- Generate a bar chart (`benchmark_results.png`)
- Print performance statistics and comparisons to stdout

## Configuration

Key parameters can be adjusted at the top of each `Makefile` and `test.cpp`:

### Makefile
- `KERNEL_NAMES` - Names of the kernels
- Tool paths (PEANO_CLANG, AIECC, etc.)

### test.cpp
- `EMBEDDING_DIM` (default: 2048)
- `HIDDEN_DIM` (default: 8192)
- `NUM_ITERATIONS` (default: 100)

## SwiGLU Operation Sequence

The SwiGLU operator executes the following sequence:

1. **swiglu_gemv_1**: Matrix-vector multiply (weights_1 × input → left)
2. **swiglu_gemv_1**: Matrix-vector multiply (weights_2 × input → right)
3. **swiglu_silu**: SiLU activation (left → left_swished)
4. **swiglu_eltwise_mul**: Element-wise multiply (left_swished × right → intermediate)
5. **swiglu_gemv_2**: Matrix-vector multiply (weights_3 × intermediate → output)

## Clean

```bash
cd separate_xclbins && make clean
cd runlist && make clean
cd fused_transactions && make clean
```
