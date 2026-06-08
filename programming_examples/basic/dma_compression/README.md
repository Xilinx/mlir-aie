# dma_compression

A silicon-level probe for compute-tile and memtile DMA
`Enable_Compression` on AIE-ML (Phoenix npu1) and AIE2P (Strix / Strix
Halo / Krackan npu2), written as an IRON design. Single-tile passthrough
(shim → compute(0,2) → shim) and two-tile chains, with per-BD and
per-channel compression registers flipped from the host runtime sequence
via `aiex.npu.npu_maskwrite32` (surfaced through IRON's
`Runtime.inline_ops` escape hatch) or from inside the core via peano's
`write_tm` intrinsic.

What this probe establishes about the AIE compute-tile compression hardware:

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

- `dma_compression.py` — IRON design. `dma_compression(in_tensor, out_tensor,
  config=...)` returns an MLIR module per config. Most configs use
  `ObjectFifo.forward(tile=...)` + `Runtime.inline_ops`; `multi_cmp_only`
  drops to low-level `aie.mem` / `aie.dma_start` for per-side BD sizing,
  and `regdump` uses an output-only Worker that calls the kernel.
- `kernel.cc` — peano core-side kernel used by the `core_*` and `regdump`
  configs. Arch-guarded include selects `aie2pintrin.h` on AIE2P or
  `aiev2intrin.h` on AIE-ML; both expose `write_tm` and `read_tm`. A
  `SCHED_BARRIER()` macro between back-to-back `st.tm` / `lda.tm`
  instructions avoids the VLIW-bundle hazard documented on
  [issue #2346](https://github.com/Xilinx/mlir-aie/issues/2346).
- `test_dma_compression.py` — Python driver. Auto-detects arch from XRT,
  JIT-compiles + dispatches each config via `iron.jit`, validates the
  output buffer against per-config matches/mismatches/untouched buckets
  AND byte-exact sha256 against per-arch goldens for the
  `*cmp_only` / `*dcmp_only` / `*both` payloads.
- `run_jit.lit` — lit driver; runs `test_dma_compression.py` on npu1 and npu2.

## Configs

All configs use arange (or compressed-arange for `*both`, see the same-tile
section below) input and a deterministic per-config expected output. All
complete in well under one second (data dispatches) or ~1.5 s
(`core_*` configs, dominated by the per-config peano kernel build).

| Config           | MM2S compress | S2MM decompress | Configured from | Expected (matches / mismatches / untouched) |
|------------------|:-------------:|:---------------:|-----------------|---------------------------------------------|
| `base`           | —             | —               | —               | 4096 / 0 / 0                                |
| `cmp_only`       | ✓             | —               | host runtime    | 1024 / 1920 / 1152                          |
| `dcmp_only`      | —             | ✓               | host runtime    | 1024 / 3072 / 0                             |
| `both`           | ✓             | ✓               | host runtime    | 2944 / 0 / 1152 (lossless same-tile roundtrip on compressed-arange) |
| `core_cmp_only`  | ✓             | —               | **core (peano `write_tm`)** | 1024 / 1920 / 1152              |
| `core_dcmp_only` | —             | ✓               | **core (peano `write_tm`)** | 1024 / 3072 / 0                 |
| `core_both`      | ✓             | ✓               | **core (peano `write_tm`)** | 2944 / 0 / 1152              |
| `memtile_base`       | —         | —               | host runtime    | 4096 / 0 / 0                                |
| `memtile_cmp_only`   | ✓         | —               | host runtime    | 1024 / 1920 / 1152                          |
| `memtile_dcmp_only`  | —         | ✓               | host runtime    | 1024 / 3072 / 0                             |
| `memtile_both`       | ✓         | ✓               | host runtime    | 2944 / 0 / 1152                          |
| `lossless_roundtrip` | CT MM2S   | memtile S2MM    | host runtime    | **4096 / 0 / 0** (true lossless roundtrip)  |
| `multi_base`               | —     | —               | —               | 4096 / 0 / 0                                |
| `multi_cmp_only`           | CT MM2S | —             | host runtime    | 1024 / 1920 / 1152 (asymmetric, low-level dialect) |
| `multi_lossless_roundtrip` | CT MM2S | CT S2MM       | host runtime    | **4096 / 0 / 0** (CT(0,2)→CT(0,3) roundtrip)|
| `regdump`        | —             | —               | core `write_tm`+`read_tm` | 4× post-write reads of BD0/1/2/3_1 == 0x80000000 |

Shim BD lengths: 4096 in, 4096 out for `base`. Configs with MM2S compression
use a 2944-int output TAP; configs with S2MM decompression use a 2944-int
input TAP. 2944 is the empirical compressed byte count for arange 0..N-1
(~1.39× ratio, identical on AIE-ML and AIE2P). The asymmetric configs use
this as a `TensorAccessPattern` on the shim side so the BD lengths match
what the compressor actually emits — otherwise the consumer DMA stalls.

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
wired into any runtime config) that omits the `SCHED_BARRIER()` between
consecutive `write_tm` calls. The peano backend then packs the `st.tm` ops
into a single VLIW bundle — the hazard
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

To regenerate the asm (substitute `aie2p` for `aie2` when targeting AIE2P):

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
up with matches=4096, demonstrating two different ways to show that AIE
compression is invertible on arbitrary input (verified on both AIE-ML
and AIE2P).

Single-tile `both` (decompress→compress) is a true lossless roundtrip
when fed real compressed bytes: the test driver pre-runs `cmp_only` once
to capture the compressed-arange payload, then feeds those bytes (zero-
padded to N) as the input tensor for the `both` configs. S2MM
decompresses real compressed data to arange in the CT buffer, MM2S
re-compresses to the same compressed bytes, and out byte-equals input
on the first `RATIOED_N` int32s. This is the same lossless invariant the
cross-tile `lossless_roundtrip` proves, but on a single tile.

(For `*cmp_only`/`*dcmp_only` the output is intentionally NOT
`matches=4096`: with arange input, the compressor emits ~2944 bytes;
only the first BD (~1024 elements) passes through close to identity (a
state-machine warm-up artifact) before subsequent BDs return compressed
bytes that don't re-interpret as the original int32s. The compressor is
lossless — the `roundtrip` and `*both` configs prove that — but the
output stream isn't the input stream, which is the whole point of
compression.)

## How to run

### Prereqs

- Phoenix npu1 OR Strix / Strix Halo / Krackan npu2 device,
  `amdxdna` driver loaded, `/dev/accel/accel0` present
- XRT at `/opt/xilinx/xrt`
- mlir-aie built per `docs/Building.md`, with an activated `ironenv`

### Run all configs

```bash
source ironenv/bin/activate
source /opt/xilinx/xrt/setup.sh
source utils/env_setup.sh install /path/to/llvm-aie
cd programming_examples/basic/dma_compression
python3 test_dma_compression.py
```

Each compute-tile dispatch is sub-200 ms; the `core_*` configs take
~1.5 s each for the per-config peano kernel build. The full 16-config
sweep + roundtrip completes in well under 15 seconds on either arch.

### Run a single config

```bash
python3 test_dma_compression.py both -v
```

### Inspect the lowered MLIR

```bash
python3 -c "
from aie.iron.device import NPU1Col1
from aie.utils.hostruntime import set_current_device
from dma_compression import dma_compression
set_current_device(NPU1Col1())
print(dma_compression.as_mlir(config='both'))
"
```

`dma_compression` is a ``@iron.jit``-decorated ``CallableDesign``;
its ``.as_mlir(**compile_kwargs)`` runs the generator and returns the
serialised MLIR without any aiecc work.  See
[`compilation_stages.md`](../../../programming_guide/compilation_stages.md)
§Inspecting an intermediate stage for the other introspection knobs.

## Strix npu2 (AIE2P) support

All 16 configs run on Strix / Strix Halo / Krackan. The DMA register
addresses and field semantics are identical between AIE-ML and AIE2P, so
the host plumbing (register addresses, BD/CTRL bit assignments, BD-vs-
channel pairing) ports as-is; `kernel.cc` arch-guards its intrinsic
include so the `core_*` configs compile on both. The compressed
*payload bytes* differ between arches even though the overall ratio is
identical — see `### sha256 goldens` below.

### Same-tile `*both` configs

`both`, `memtile_both`, and `core_both` do a single-dispatch lossless
roundtrip on real compressed bytes: the driver pre-runs `cmp_only` once
to capture the compressed-arange payload, then feeds those bytes (zero-
padded to N) as input. S2MM decompresses to arange in the CT buffer,
MM2S re-compresses, and `out[:RATIOED_N]` byte-equals the input.

(Earlier revisions fed raw arange directly into the decompressor; AIE2P
raises a `Decompression_underflow` event and stalls the DMA, while
AIE-ML completes with deterministic garbage.)

### `regdump`

`regdump` is a core-side `write_tm` + `read_tm` self-test. The kernel
writes `COMPRESS_BIT` to each `BD?_1` and reads it back; the driver
asserts each post-write read equals `0x80000000`. This directly verifies
the mechanism the `core_*` configs depend on.

Host-side `npu_maskwrite32` to BD/CTRL registers is verified
operationally by the `cmp_only` / `dcmp_only` / `*both` data assertions
(not by `regdump`-style direct readback — empirically the NPU
controller's BD/CTRL writes only commit on `writebd` queueing).

### sha256 goldens

`GOLDEN_SHA_BY_ARCH` in `test_dma_compression.py` is empirically derived
and keyed by `npu_str` (`npu1` / `npu2`). The overall compression ratio
is identical on AIE-ML and AIE2P (1.391× for arange), but the per-chunk
encoding differs (header semantics, payload byte ordering, and the
within-chunk int rotation), so the byte-exact payloads diverge.
Per-arch shas guard against silent regressions on either side.

## Register reference

| Register                                  | Tile-relative addr | Bit | Purpose                              |
|-------------------------------------------|--------------------|:---:|--------------------------------------|
| `MEMORY_MODULE_DMA_BD{n}_1`               | 0x1D004 + n·0x20   | 31  | BD-level `Enable_Compression`        |
| `MEMORY_MODULE_DMA_S2MM_0_CTRL`           | 0x1DE00            | 4   | Channel-level `Decompression_Enable` |
| `MEMORY_MODULE_DMA_MM2S_0_CTRL`           | 0x1DE10            | 4   | Channel-level `Compression_Enable`   |

Source: `third_party/aie-rt/driver/src/global/xaiemlgbl_params.h` and
`xaiemlgbl_reginit.c` (AIE2P uses the same addresses and bit positions —
see `xaie2pgbl_params.h`).

After IRON's `ObjectFifo.forward()` lowering, BDs land at:

| BD slot | Channel               |
|---------|-----------------------|
| 0, 1    | S2MM ch 0 (in from shim) |
| 2, 3    | MM2S ch 0 (out to shim)  |
