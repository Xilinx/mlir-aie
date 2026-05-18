# dma_compression

A silicon-level probe for AIE-ML compute-tile DMA `Enable_Compression`,
written as an IRON design. Single-tile passthrough (shim → compute(0,2) →
shim) on Phoenix npu1, with per-BD and per-channel compression registers
flipped from the host runtime sequence via `aiex.npu.npu_maskwrite32`,
surfaced through IRON's `Runtime.inline_ops` escape hatch.

What this probe establishes about the AIE-ML compute-tile compression hardware:

- Enabling compression on a direction requires **two** register writes — the
  per-BD `Enable_Compression` bit (`MEMORY_MODULE_DMA_BD{n}_1` bit 31) AND
  the per-channel `(De)compression_Enable` bit
  (`MEMORY_MODULE_DMA_{S2MM,MM2S}_0_CTRL` bit 4). The driver's
  `XAie_DmaEnableCompression()` / `XAie_DmaChannelEnCompression()` pair
  (`xaie_dma.c:522,1747`) makes this explicit; mlir-aie's existing passes
  plumb neither for the compute-tile path.
- With compression enabled on either direction, the consumer shim BD length
  must match the compressed byte count, or the DMA stalls.

## Files

- `dma_compression.py` — IRON design (~160 lines). One `build_design()` style
  function that takes a config name + input/output tensors and returns an
  MLIR module. Uses `ObjectFifo.forward(tile=compute_tile)` for the DMA-link
  passthrough and `Runtime.inline_ops` for register writes.
- `kernel.cc` — peano core-side kernel used by the `core_*` configs.
  Calls `aiev2intrin.h`'s `write_tm` to write the compression registers from
  inside the compute tile's core, with `__builtin_aiev2_sched_barrier()`
  between each `st.tm` to avoid the VLIW-bundle hazard documented on
  [issue #2346](https://github.com/Xilinx/mlir-aie/issues/2346).
- `test_dma_compression.py` — Python driver. JIT-compiles + dispatches each
  config via `iron.jit`, validates the output buffer against a per-config
  expected matches/mismatches/untouched breakdown.
- `run_jit.lit` — lit driver; runs `test_dma_compression.py` on npu1.

## Configs

All configs use arange input and a deterministic per-config expected output.
All complete in well under one second.

| Config           | MM2S compress | S2MM decompress | Configured from | Expected (matches / mismatches / untouched) |
|------------------|:-------------:|:---------------:|-----------------|---------------------------------------------|
| `base`           | —             | —               | —               | 4096 / 0 / 0                                |
| `cmp_only`       | ✓             | —               | host runtime    | 1024 / 1920 / 1152                          |
| `dcmp_only`      | —             | ✓               | host runtime    | 1024 / 3072 / 0                             |
| `both`           | ✓             | ✓               | host runtime    | 1043 / 1901 / 1152                          |
| `core_cmp_only`  | ✓             | —               | **core (peano `write_tm`)** | 1024 / 1920 / 1152              |
| `core_dcmp_only` | —             | ✓               | **core (peano `write_tm`)** | 1024 / 3072 / 0                 |
| `core_both`      | ✓             | ✓               | **core (peano `write_tm`)** | 1043 / 1901 / 1152              |
| `memtile_base`       | —         | —               | host runtime    | 4096 / 0 / 0                                |
| `memtile_cmp_only`   | ✓         | —               | host runtime    | 1024 / 1920 / 1152                          |
| `memtile_dcmp_only`  | —         | ✓               | host runtime    | 1024 / 3072 / 0                             |
| `memtile_both`       | ✓         | ✓               | host runtime    | 1043 / 1901 / 1152                          |
| `lossless_roundtrip` | CT MM2S   | memtile S2MM    | host runtime    | **4096 / 0 / 0** (true lossless roundtrip)  |
| `multi_base`               | —     | —               | —               | 4096 / 0 / 0                                |
| `multi_cmp_only`           | CT MM2S | —             | host runtime    | 1024 / 1920 / 1152 (asymmetric, low-level dialect) |
| `multi_lossless_roundtrip` | CT MM2S | CT S2MM       | host runtime    | **4096 / 0 / 0** (CT(0,2)→CT(0,3) roundtrip)|

Shim BD lengths: 4096 in, 4096 out for `base`. Configs with MM2S compression
use a 2944-int output TAP; configs with S2MM decompression use a 2944-int
input TAP. 2944 is the empirical compressed byte count for arange 0..N-1
on Phoenix (~1.39× ratio). The asymmetric configs use this as a
`TensorAccessPattern` on the shim side so the BD lengths match what the
compressor actually emits — otherwise the consumer DMA stalls.

**Host-side configs** (`base`/`cmp_only`/`dcmp_only`/`both` and the
`memtile_*` family) flip the compression registers from the runtime
sequence via `aiex.npu.npu_maskwrite32`. **Core-side configs**
(`core_*`) flip the same registers from inside the compute tile's core
via peano's `write_tm` intrinsic (the peano-side companion to PR
[#2348](https://github.com/Xilinx/mlir-aie/pull/2348)'s chess-only
`test/npu-xrt/tile_mapped_read/`). The `memtile_*` configs route through
memtile(0,1) instead of compute(0,2); the memtile has its own register
layout (BD compression bit lives in BD?_4 instead of BD?_1) but presents
identical observable behaviour. Both alternate-path families produce
bit-identical output to the baseline compute-tile host-side counterparts.

### Single-dispatch lossless roundtrip

`lossless_roundtrip` chains two tiles in one dispatch:
```
shim --[raw]--> CT(0,2) --[compressed]--> memtile(0,1) --[raw]--> shim
```
CT has a Worker that copies bytes from its input fifo to its output fifo
(passthrough); the CT MM2S DMA compresses on the way out to memtile.
Memtile's S2MM decompresses on receive, and memtile's link forwards
the recovered raw bytes to shim. The compressed bytes only exist on the
wire between CT and memtile; both shim ends see the original 4096 raw
int32s. matches=4096 on arange input proves the compression is
losslessly invertible on non-trivial data — not just the degenerate
all-zero case.

(IRON constraint hit during development: a single ObjectFifo can't be
both the destination of one `ObjectFifoLinkOp` and the source of
another. So CT can't simply forward to memtile; instead CT runs a
trivial Worker so its outgoing fifo `b_ct_to_mt` is produced by the
worker rather than by a link.)

`multi_lossless_roundtrip` does the same shape on a CT-to-CT link (no
memtile). `multi_cmp_only` is the asymmetric-compression-engagement
proof: it compresses on CT(0,2) MM2S but does NOT decompress on
CT(0,3) S2MM, so the inter-tile wire carries fewer bytes than the
consumer would normally expect. To avoid the consumer DMA stalling,
the CT(0,3) BDs are hand-sized to `RATIOED_PER_LINE = 736` ints/BD.
IRON's `forward()` / object-fifo lowering doesn't expose per-side BD
sizing (the linked fifo's `line_ty` sizes both ends), so this config
is built from low-level `aie.mem` / `aie.dma_start` / `aie.flow` ops
in `_build_multi_cmp_only()` instead of the IRON `ObjectFifo` API. The
asymmetric output pattern (1024 / 1920 / 1152, matching single-tile
`cmp_only`) proves DMA compression actually engages on the inter-CT
link.

### Scheduler barrier hazard (#2346)

`kernel.cc` includes an `enable_mm2s_compression_no_barrier` function (not
wired into any runtime config) that omits `__builtin_aiev2_sched_barrier()`
between consecutive `write_tm` calls. The peano backend then packs the
`st.tm` ops into a single VLIW bundle — the hazard
[@joeldushouyu documented](https://github.com/Xilinx/mlir-aie/issues/2346#issuecomment-2922081498)
on the issue thread.

Disassemble the kernel with `llvm-objdump -d` to see the difference:

```
# WITH barriers (enable_mm2s_compression) — each st.tm in its own bundle:
00000000 <enable_mm2s_compression>:
       0: ... movxm r0, #-0x80000000
      12: st.tm r0, [p0];  movxm p1, ...
      26: st.tm r0, [p1]
      2c: st.tm r1, [p2]
      32: ret lr

# WITHOUT barriers (enable_mm2s_compression_no_barrier) — bundle-packed:
00000000 <enable_mm2s_compression_no_barrier>:
       0: ... movxm r0, #-0x80000000
      14: st.tm r0, [p0];  movxm p1, ...
      1e: mova r1, #0x10;  st.tm r0, [p1];  movxm p0, ...   <-- 3 ops in one VLIW bundle
      2a: st.tm r1, [p0]
```

The bundled `st.tm r0, [p1]` at offset 0x1e issues in parallel with `mova`
and `movxm`, and the next `st.tm` follows immediately at 0x2a — close
enough that the second store may issue before the first reaches the
processor bus, stalling the kernel.

To regenerate the asm:

```bash
clang++ -O2 -std=c++20 --target=aie2-none-unknown-elf -DNDEBUG \
    -c kernel.cc -o kernel.o
llvm-objdump -d --disassemble-symbols=enable_mm2s_compression kernel.o
llvm-objdump -d --disassemble-symbols=enable_mm2s_compression_no_barrier kernel.o
```

The `core_*` configs use the with-barriers functions only; the no-barrier
function is kept in `kernel.cc` for objdump reproducibility but is not
exposed via an IRON `ExternalFunction`.

### Two-dispatch roundtrip

In addition to the single-dispatch `lossless_roundtrip` config, the test
driver also runs a `roundtrip` test that chains the existing single-tile
configs across two iron.jit dispatches:

1. `cmp_only` takes raw arange input → output buffer holds 2944 ints
   worth of compressed bytes (rest is sentinel)
2. Those compressed bytes are fed back as input to `dcmp_only`
3. Asserts the recovered 4096 ints equal the original arange

This is the "host orchestrates compression and decompression separately"
path; `lossless_roundtrip` is the "hardware does the roundtrip in one
dispatch with the compressed stream living on the wire" path. Both end
up with matches=4096, demonstrating two different ways to show that AIE-ML
compression is invertible on arbitrary input.

The reason single-tile `both` (decompress→compress) can't be lossless on
arbitrary input: on a given tile, S2MM only has `Decompression_Enable`
and MM2S only has `Compression_Enable`. You can do
decompress-then-compress on one tile (what `both` does — feeds raw to
the decompressor, gets garbage, then compresses the garbage) but
compress-then-decompress requires two tiles, hence the
`lossless_roundtrip` topology.

**Why these aren't all `matches=4096`:** with arange input, the compressor
on the output side emits ~2944 bytes; only the first BD's worth of
elements (~1024) passes through close to identity (a state-machine warm-up
artifact) before subsequent BDs return compressed bytes that don't
re-interpret as the original int32s. The compressor is lossless, but the
output stream is not the input stream — that's the whole point of
compression. To see lossless round-trip on this hardware, you need
all-zero input (a degenerate case where compress + decompress = identity
because the byte counts trivially match); the prior C++ probe used that.

## How to run

### Prereqs

- Phoenix npu1 device, `amdxdna` driver loaded, `/dev/accel/accel0` present
- XRT at `/opt/xilinx/xrt`
- mlir-aie built per `docs/Building.md`, with an activated `ironenv`

### Run all four configs

```bash
source ironenv/bin/activate
source /opt/xilinx/xrt/setup.sh
source utils/env_setup.sh install /path/to/llvm-aie
cd programming_examples/basic/dma_compression
python3 test_dma_compression.py
```

Total wall time on Phoenix is well under a second; each config dispatches
in ~150 ms.

### Run a single config

```bash
python3 test_dma_compression.py both -v
```

### Inspect the lowered MLIR

```bash
python3 -c "
import numpy as np
from aie.iron.device import NPU1Col1
from aie.extras.context import mlir_mod_ctx
from dma_compression import dma_compression
a = np.arange(4096, dtype=np.int32); c = np.zeros(4096, dtype=np.int32)
with mlir_mod_ctx() as ctx:
    print(dma_compression(a, c, config='both', dev=NPU1Col1()))
"
```

## Register reference

| Register                                  | Tile-relative addr | Bit | Purpose                              |
|-------------------------------------------|--------------------|:---:|--------------------------------------|
| `MEMORY_MODULE_DMA_BD{n}_1`               | 0x1D004 + n·0x20   | 31  | BD-level `Enable_Compression`        |
| `MEMORY_MODULE_DMA_S2MM_0_CTRL`           | 0x1DE00            | 4   | Channel-level `Decompression_Enable` |
| `MEMORY_MODULE_DMA_MM2S_0_CTRL`           | 0x1DE10            | 4   | Channel-level `Compression_Enable`   |

Source: `third_party/aie-rt/driver/src/global/xaiemlgbl_params.h` and
`xaiemlgbl_reginit.c`.

After IRON's `ObjectFifo.forward()` lowering, BDs land at:

| BD slot | Channel               |
|---------|-----------------------|
| 0, 1    | S2MM ch 0 (in from shim) |
| 2, 3    | MM2S ch 0 (out to shim)  |
