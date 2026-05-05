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

This works because `aiecc` explicitly marks `aie.runtime_sequence` legal for
the module-level SCF-to-CF conversion, which:
- Preserves SCF ops in the runtime sequence for C++ codegen
- Still lowers `aie.core` bodies for normal code generation
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

### Detailed TXN Walkthrough

The dynamic TXN path does not generate a new GEMM kernel. It generates the
instruction stream that tells the already-compiled kernel how many 32x32x32
tiles to process and how to move those tiles through shim DMA.

#### 1. Fixed compute kernel

The AIE core kernel in `aie_kernels/aie2p/mm.cc` is fixed at a
**32x32x32 bf16->f32** tile shape. It uses `aie::mmul` and accumulates one
output tile across multiple K-slices.

The dynamic part is therefore not the math itself. The dynamic part is:

- how many output tiles exist: `(M / 32) * (N / 32)`
- how many K tiles each output tile accumulates: `K / 32`
- which regions of the host buffers A, B, and C each DMA descriptor covers

#### 2. Core-side runtime control

Inside `single_core_dynamic.py`, the core reads two RTP values at runtime:

- `rtp[0] = K_div_k`
- `rtp[1] = total_tiles`

Those values control the nested loops on the core:

- outer loop over output tiles
- inner loop over K accumulation steps

So one fixed ELF can execute many GEMM sizes as long as M, K, and N remain
multiples of 32.

#### 3. Runtime sequence as a parameterized TXN program

With `--dynamic-txn`, the runtime sequence itself takes `M`, `K`, and `N` as
SSA values:

```mlir
aie.runtime_sequence(A, B, C, M, K, N)
```

The runtime sequence computes:

- `M_div_m = M / 32`
- `K_div_k = K / 32`
- `N_div_n = N / 32`
- `tiles = M_div_m * N_div_n`

It then writes RTP values and emits DMA orchestration. The important thing is
that the orchestration is still written in high-level MLIR using `arith`, `scf`,
and `aiex.npu.dma_memcpy_nd`.

#### 4. What NPU lowering turns that into

Each `aiex.npu.dma_memcpy_nd` expands into explicit descriptor programming:

- BD register writes (`npu.write32`)
- host address patching (`npu.address_patch`)
- queue setup and completion-token configuration (`npu.maskwrite32`)
- completion waits (`npu.sync`)

This is why the emitted C++ is verbose: by the time EmitC runs, the compiler is
no longer expressing “copy a multidimensional slice”; it is expressing exact NPU
transaction words.

#### 5. What the generated C++ function does

The emitted function has the shape:

```cpp
std::vector<uint32_t> generate_txn_sequence(int32_t M, int32_t K, int32_t N)
```

Conceptually it does:

1. Compute `M/32`, `K/32`, `N/32`, and `tiles`.
2. Emit two RTP writes.
3. Loop over tile-row blocks.
4. For each ping-pong half-block:
   - compute DMA descriptor fields,
   - append BD programming writes,
   - append address patches,
   - append queue pushes,
   - append syncs as needed.
5. Prepend the TXN header and return the word vector.

The generated code is mechanically lowered from MLIR, so it contains many local
temporaries (`v17`, `v18`, …). These are just SSA values printed as C++ locals.
#### 6. Ping-pong scheduling

The dynamic sequence processes output rows in blocks:

- `rows_per_block = 4`
- each ping-pong half handles up to 2 tile rows
- two BD banks are alternated so one half-block can execute while the next is
  being prepared

The schedule looks like:

- program output C DMA
- program A and B input DMAs for one or two tile rows
- optionally wait for the previous batch before BD reuse
- push DMA queues
- continue to the next half-block

This is what lets one core stream larger GEMMs while reusing a fixed microkernel.

#### 7. Why this feature matters

Without runtime-parameterized TXN generation, you typically need:

- one XCLBIN
- one fixed instruction stream
- one fixed problem size

With this feature, you get:

- one XCLBIN
- one fixed core ELF
- one generated host function that builds a fresh instruction stream for each
  runtime `M`, `K`, `N`

In practice, that means the expensive parts stay fixed:

- placement
- routing
- core code generation
- XCLBIN creation

while the cheap part changes at runtime:

- the transaction stream that sets RTPs and programs DMA descriptors

That is the core value of the feature: **compile once, vary GEMM shape at
runtime by regenerating only the TXN program**.

## Design Variants

| File | Description |
|------|-------------|
| `single_core_dynamic.py` | Low-level dialect with `--dynamic-txn` flag |
| `single_core_dynamic_placed.py` | Placed API variant |
| `single_core_dynamic_iron.py` | IRON high-level API variant |

## Key Design Decisions

**Fixed tile size (32x32x32)**: The AIE core microkernel (`mm.cc`) operates on fixed 32x32 tiles. The dynamic part is how many tiles are processed and the DMA pattern for streaming them.

**RTP for loop bounds**: The core reads K_div_k and total_tiles from an RTP buffer via `memref.load`. The host writes these before issuing DMAs. The DMA start acts as an implicit ordering barrier.

**Pingpong DMA pattern**: Two sets of BDs (even/odd) overlap compute and data movement. Each half-block processes up to 2 tile rows. The `scf.for` in the runtime_sequence iterates over tile-row blocks, with `scf.if` guards for boundary conditions.

**SCF preservation**: `aiecc` explicitly keeps `aie.runtime_sequence` legal
during the module-level SCF-to-CF conversion, so SCF stays available for the
EmitC path while `aie.core` regions continue through the normal lowering flow.

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
├── test_dynamic.cpp            # Test harness (auto-generated TXN path)
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
