# Dynamic Single-Core Matrix Multiplication

Single-core GEMM with **runtime-configurable matrix dimensions**, written in the
high-level IRON API (`@iron.jit`). One compiled XCLBIN supports any `(M, K, N)`
that are multiples of the tile size — the problem size is supplied to the runtime
sequence as SSA values, not baked in at compile time.

This is a minimal delta over [`../single_core/single_core.py`](../single_core/single_core.py).
The two differences are:

1. The runtime sequence takes `M, K, N` as scalar arguments and derives its loop
   trip counts and DMA geometry from them using `range_` / `if_` (which emit
   `scf.for` / `scf.if`) and arithmetic on the SSA values.
2. The core reads its loop trip counts from an RTP `Buffer` that the host
   populates at the start of the sequence, so one fixed core ELF runs any size.

## Quick Start

```bash
# Build the XCLBIN + generated C++ TXN header + test executable, then run a
# sweep of sizes from the single XCLBIN.
make run_dynamic M=128 K=128 N=128 devicename=npu2
```

This builds `build/dynsize/final_dynamic.xclbin` and `generated_gemm_txn.h` in
one `aiecc` invocation, compiles `test_dynamic.cpp` against the generated TXN
header, and runs 32³ / 64³ / 96³ / 128³ from the same XCLBIN.

## How It Works

The design has a fixed **tile size** (e.g. 32x32x32) but **variable problem
size** (M, K, N). The core loops over output tiles, reading the iteration counts
from RTP (Runtime Tunable Parameters); the host generates the DMA instruction
stream at runtime for each problem size via the generated
`generate_txn_sequence(M, K, N)` C++ function.

### Architecture

```
                    ┌──────────────────────────────┐
                    │  single_core_dynamic_txn.py   │
                    │  (.as_mlir on the @iron.jit   │
                    │   single_core_dynamic design) │
                    └────────────┬─────────────────┘
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
multiples of the tile size.

#### 3. Runtime sequence as a parameterized TXN program

The runtime sequence itself takes `M`, `K`, and `N` as SSA values:

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

## Key Design Decisions

**RTP for loop bounds**: The core reads `K_div_k` and `tiles` from an RTP `Buffer`
(`use_write_rtp=True`) via `memref.load`. The host writes these at the start of the
runtime sequence; the DMA start acts as an implicit ordering barrier.

**Pingpong DMA pattern**: Two `TaskGroup`s overlap compute and data movement —
the previous group is finished (`prev.resolve()`) only after the next group's
transfers are issued. The `range_` in the runtime sequence iterates over tile-row
blocks, with `if_` guards for boundary conditions.

**SCF preservation**: `aiecc` explicitly keeps `aie.runtime_sequence` legal
during the module-level SCF-to-CF conversion, so SCF stays available for the
EmitC / TXN-C++ path while `aie.core` regions continue through the normal
lowering flow.

## Constraints

- M, K, N must be multiples of the tile size (`m`, `k`, `n`)
- input/output dtype must both be integral or both float; output ≥ input width
- Single core — no multi-core parallelism
- The dynamic (SSA-size) sequence is lowered through the TXN-C++ flow, not a
  static `insts.bin`

## Files

```
single_core_dynamic/
├── single_core_dynamic.py       # @iron.jit design (SSA M/K/N runtime sequence)
├── single_core_dynamic_txn.py   # Driver: emits MLIR / compiles XCLBIN + TXN header
├── test_dynamic.cpp             # Host test harness (auto-generated TXN path)
├── Makefile
├── run_makefile_dynamic.lit
├── run_strix_makefile_dynamic.lit
└── build/dynsize/               # Generated artifacts
    ├── aie_gemm_dynamic.mlir    # Dynamic MLIR (with SSA M/K/N)
    ├── final_dynamic.xclbin     # Hardware configuration
    └── generated_gemm_txn.h     # Auto-generated C++ TXN function
```
