# Prototype 7: MemTile Pool Allocator

Demonstrates a **pool allocation pattern** in MemTile where a single
`aie.buffer()` covers all sub-regions, and `dma_bd(pool, offset=N, len=M)`
selects which portion each buffer descriptor accesses.

## Key Difference from Prototype 6

| | Prototype 6 | Prototype 7 |
|---|---|---|
| **MemTile buffers** | 4 separate `aie.buffer()` | 1 pool `aie.buffer()` |
| **Sub-region selection** | Each BD references its own buffer | `dma_bd(pool, offset=N)` |
| **MLIR objects** | 4 buffers + 8 locks | 1 buffer + 8 locks |

This proves that one contiguous memory allocation can serve multiple logical
buffers via offset-based addressing in the BD configuration — the foundation
for a runtime pool allocator.

## Pool Layout

```
pool buffer (2048 bytes = memref<2048xui8>)
┌─────────────┬─────────────┬──────────────┬──────────────┬─────────────┐
│ data_a      │ data_b      │ result_a     │ result_b     │ free        │
│ [0:256]     │ [256:512]   │ [512:768]    │ [768:1024]   │ [1024:2048] │
│ → Core A in │ → Core B in │ ← Core A out │ ← Core B out │             │
└─────────────┴─────────────┴──────────────┴──────────────┴─────────────┘
```

## Architecture

```
DDR ──[Shim MM2S:0]──► MemTile S2MM:0 ──► pool[0:256], pool[256:512]
                                                │
        MemTile MM2S:0 ◄── pool[0:256] ─────────┘  (to Core A)
        MemTile MM2S:2 ◄── pool[256:512] ───────┘  (to Core B)
             │                   │
         Core(0,2)           Core(0,3)
         passthrough         passthrough
             │                   │
MemTile S2MM:2 ◄── pool[512:768]  MemTile S2MM:4 ◄── pool[768:1024]
                                        │
        MemTile MM2S:4 ◄────────────────┘  (BD chain: [512:768] then [768:1024])
             │
DDR ◄──[Shim S2MM:0]
```

## How It Works

1. **Compile time**: One pool buffer declared in MemTile. Flows pre-wired.
   Core DMAs are static (loop forever).
2. **Runtime**: `dma_configure_task()` programs MemTile BDs with different
   offsets into the same pool buffer. Locks synchronize phases.
3. **Data flow**: DDR → pool input regions → cores → pool result regions → DDR.

All MemTile DMA programming happens at runtime via `dma_configure_task()` —
no `@memtile_dma` decorator is used.

## Generated MLIR

The key MLIR pattern — all BDs reference the same `%pool` buffer with different
offsets:

```mlir
%pool = aie.buffer(%mem_tile_0_1) {sym_name = "pool"} : memref<2048xui8>

// Phase 1: DDR → pool (BD chain, two offsets)
aie.dma_bd(%pool : memref<2048xui8>, 0, 256)      // pool[0:256]
aie.dma_bd(%pool : memref<2048xui8>, 256, 256)    // pool[256:512]

// Phase 2: pool → cores (different offsets, different channels)
aie.dma_bd(%pool : memref<2048xui8>, 0, 256)      // MM2S:0 → Core A
aie.dma_bd(%pool : memref<2048xui8>, 256, 256)    // MM2S:2 → Core B

// Phase 3: cores → pool results
aie.dma_bd(%pool : memref<2048xui8>, 512, 256)    // S2MM:2 ← Core A
aie.dma_bd(%pool : memref<2048xui8>, 768, 256)    // S2MM:4 ← Core B

// Phase 4: pool → DDR (BD chain)
aie.dma_bd(%pool : memref<2048xui8>, 512, 256)    // result_a
aie.dma_bd(%pool : memref<2048xui8>, 768, 256)    // result_b
```

## Building and Running

```bash
source ../env.sh
NPU2=1 make all
NPU2=1 make run_py
```

## Limitations

- Offsets in `dma_bd()` are compile-time constants (not SSA values). Each
  distinct offset requires a separate BD in the runtime sequence.
- For truly dynamic offsets, use `npu_writebd()` or the
  `dynamic-runtime-sequences` branch which adds `npu_write32_dynamic`.
- Only even MemTile DMA channels (0, 2, 4) are used due to a compiler BD ID
  allocation bug on odd channels.

## Files

| File | Description |
|------|-------------|
| `aie2.py` | MLIR generation — pool buffer + offset-based BDs |
| `test.py` | NPU hardware test — passthrough verification |
| `Makefile` | Build recipe (kernel compile, MLIR gen, xclbin) |
