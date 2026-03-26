# Dynamic Single-Core Matrix Multiplication

Single-core bf16 GEMM with **runtime-configurable matrix dimensions**. One compiled XCLBIN supports any M/K/N that are multiples of 32 — matrix sizes are determined at runtime, not compile time.

## Quick Start

```bash
# Build everything (XCLBIN + C++ TXN code + test executable)
make build/dynsize/final_dynamic.xclbin M=128 K=128 N=128 devicename=npu2
make dynamic_generated.exe M=128 K=128 N=128 devicename=npu2

# Run multiple sizes from the same XCLBIN
./dynamic_generated.exe -x build/dynsize/final_dynamic.xclbin -k MLIR_AIE -M 32 -K 32 -N 32 -v 1
./dynamic_generated.exe -x build/dynsize/final_dynamic.xclbin -k MLIR_AIE -M 64 -K 64 -N 64 -v 1
./dynamic_generated.exe -x build/dynsize/final_dynamic.xclbin -k MLIR_AIE -M 128 -K 128 -N 128 -v 1

# Or run all at once
make run_dynamic_generated M=128 K=128 N=128 devicename=npu2
```

## Performance

All sizes from a single XCLBIN on NPU Strix Halo (NPU2):

| Size | TXN Instructions | Time | GFLOPS | Status |
|------|-----------------|------|--------|--------|
| 32x32x32 | 225 words | 1275 us | 0.05 | PASS |
| 64x32x64 | 357 words | 1096 us | 0.24 | PASS |
| 64x64x64 | 357 words | 866 us | 0.61 | PASS |
| 96x96x96 | 566 words | 760 us | 2.33 | PASS |
| 128x64x128 | 698 words | 1103 us | 1.90 | PASS |
| 128x128x128 | 698 words | 1575 us | 2.66 | PASS |

## How It Works

The design has a fixed **tile size** (32x32x32 bf16->f32) but **variable problem size** (M, K, N). The core runs an infinite loop that reads iteration counts from RTP (Runtime Tunable Parameters), and the host generates DMA instruction sequences at runtime for each problem size.

### Architecture

```
                    ┌──────────────────────────┐
                    │   single_core_dynamic.py  │
                    │      --dynamic-txn        │
                    └────────────┬─────────────┘
                                 │
                       aie_gemm_dynamic.mlir
                    (one MLIR, three regions)
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                   │
         aie.core            aie.device          aie.runtime_sequence
     (compute kernel)     (hw configuration)    (DMA orchestration)
     scf.while loops      objectFIFOs, locks    SSA M/K/N params
     RTP-driven bounds    buffer placement      scf.for/scf.if loops
              │                  │                   │
              ▼                  ▼                   ▼
    ┌─────────────────┐ ┌──────────────────┐ ┌─────────────────────┐
    │ Peano clang++   │ │ CDO/PDI/xclbin   │ │ EmitC → C++ codegen │
    │ → core_0_2.elf  │ │ → final.xclbin   │ │ → generated_txn.h   │
    └─────────────────┘ └──────────────────┘ └─────────────────────┘
              │                  │                   │
              └────────┬────────┘                   │
                       ▼                            ▼
              final_dynamic.xclbin          dynamic_generated.exe
              (load once)                   (calls generate_txn_sequence
                                             with runtime M, K, N)
```

### Compilation Flow

A single `aiecc` invocation produces both the XCLBIN and the C++ TXN code:

```bash
aiecc --aie-generate-xclbin --xclbin-name=final_dynamic.xclbin \
      --aie-generate-txn-cpp --txn-cpp-name=generated_gemm_txn.h \
      aie_gemm_dynamic.mlir
```

This works because `aie.runtime_sequence` has the `IsolatedFromAbove` trait, which:
- Prevents the SCF-to-CF pass from entering the runtime sequence (preserving SCF ops for C++ codegen)
- Prevents constant hoisting across the isolation boundary
- Allows both XCLBIN and TXN generation from the same MLIR with identical buffer addresses

#### XCLBIN Path (hardware configuration)

```
MLIR → objectFifo lowering → buffer allocation → routing
     → SCF→CF (core body only) → AIECoreToStandard → LLVM IR
     → Peano opt/llc → ELF → CDO → PDI → xclbin
```

The XCLBIN contains the core ELF, tile configuration, DMA descriptors, locks, and routing. It does NOT contain the runtime DMA instruction sequence.

#### TXN C++ Path (runtime instruction generation)

```
MLIR (clone) → NPU lowering pipeline
  aiex.npu.dma_memcpy_nd (with SSA sizes/strides)
    → arith ops computing BD words (d0_size * elemWidth / addrGran, ...)
    → npu.write32 for each BD word
    → npu.address_patch for buffer addresses
    → npu.maskwrite32 for S2MM tokens
    → npu.sync for completion
  → ConvertAIEXToEmitCPass
    → emitc.call_opaque("txn_append_write32", ...)
    → emitc.for / emitc.if (from SCF)
    → emitc.variable + emitc.assign (for iter_args/results)
  → translateToCpp
    → generated_gemm_txn.h
```

The generated C++ function `generate_txn_sequence(M, K, N)` returns a `std::vector<uint32_t>` of NPU instruction words.

### Runtime Flow

```
1. Load XCLBIN (configures AIE array, loads core ELF)
2. Allocate host buffers A[M×K bf16], B[K×N bf16], C[M×N f32]
3. Call generate_txn_sequence(M, K, N)
   → Writes RTP values (K/32, M/32 * N/32)
   → Programs shim DMA BDs for A, B, C with dynamic strides/offsets
   → Issues queue pushes with completion tokens
   → Returns instruction word vector
4. Submit instructions to NPU via XRT
5. NPU executes: DMA streams tiles to core, core computes, results DMA back
6. Read result buffer
```

## Design Variants

| File | Description |
|------|-------------|
| `single_core_dynamic.py` | Low-level dialect with `--dynamic-txn` flag |
| `single_core_dynamic_placed.py` | Placed API variant |
| `single_core_dynamic_iron.py` | IRON high-level API variant |
| `dynamic_gemm_txn.h` | Hand-written C++ TXN encoder (reference) |

## Key Design Decisions

**Fixed tile size (32x32x32)**: The AIE core microkernel (`mm.cc`) operates on fixed 32x32 tiles. The dynamic part is how many tiles are processed and the DMA pattern for streaming them.

**RTP for loop bounds**: The core reads K_div_k and total_tiles from an RTP buffer via `memref.load`. The host writes these before issuing DMAs. The DMA start acts as an implicit ordering barrier.

**Pingpong DMA pattern**: Two sets of BDs (even/odd) overlap compute and data movement. Each half-block processes up to 2 tile rows. The `scf.for` in the runtime_sequence iterates over tile-row blocks, with `scf.if` guards for boundary conditions.

**IsolatedFromAbove**: The `aie.runtime_sequence` is marked `IsolatedFromAbove` so the module-level SCF-to-CF pass (needed for core compilation) doesn't enter it. The EmitC path needs the SCF ops preserved to generate C++ for/if statements.

## Constraints

- M, K, N must be multiples of 32 (the tile size)
- bf16 input, f32 output only (kernel constraint)
- Single core (tile 0,2) — no multi-core parallelism
- NPU2 (Strix Halo) target only

## Files

```
single_core_dynamic/
├── single_core_dynamic.py      # Python design (static + --dynamic-txn)
├── single_core_dynamic_placed.py
├── single_core_dynamic_iron.py
├── test_dynamic.cpp            # Test harness (hand-written & auto-generated paths)
├── dynamic_gemm_txn.h          # Hand-written C++ TXN encoder
├── Makefile
├── tests/
│   ├── run_strix_makefile.lit
│   ├── run_strix_makefile_placed.lit
│   ├── run_strix_makefile_iron.lit
│   └── run_strix_makefile_generated.lit
└── build/dynsize/              # Generated artifacts
    ├── aie_gemm_dynamic.mlir   # Dynamic MLIR (with SSA M/K/N)
    ├── final_dynamic.xclbin    # Hardware configuration
    └── generated_gemm_txn.h    # Auto-generated C++ TXN function
```
