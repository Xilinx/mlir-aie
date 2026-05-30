<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//-->

# <ins>Dispatch-Overhead Bisector</ins>

A diagnostic example for measuring the per-launch dispatch floor on
AIE2P silicon.

This example exposes a TRIVIAL kernel (`passthrough.cc` — a pure
64-byte-vectorised memcpy with no arithmetic) wrapped in a parameterised
IRON topology. By running the kernel under different `(n_chunks,
dense_bytes)` values and timing each launch end-to-end, a driver script
can attribute the wall-time per launch to the four diagnostic suspects
that show up on AIE2P:

| Suspect | What it covers |
|---|---|
| (a) | Per-launch `xrt::run.wait()` return-path overhead |
| (b) | Per-launch instruction-stream upload cost |
| (c) | Per-chunk shim-DMA setup overhead × N_CHUNKS |
| (d) | AIE2P firmware dispatcher per-launch handshake |

## Topology

```
shim(0,0) --(of_in)--> Tile (0,2) passThroughLine --(of_out)--> shim(0,0)
```

Single compute tile, single FIFO each direction, no memtiles, no
cross-column routing. The simplest possible IRON topology that still
emits a real shim → compute → shim DMA chain per launch — which is
the only way to attribute per-launch wall to the dispatch layer rather
than kernel compute.

## Bisection methodology

Two ORTHOGONAL knobs the host driver sweeps:

- **`n_chunks`** at fixed `dense_bytes`: each extra chunk emits one
  extra `aiex.dma_configure_task_for` op pair (one shim BD in + one
  shim BD out) in the runtime sequence. The slope
  `d(wall) / d(n_chunks)` quantifies suspect **(c)** per-chunk
  shim-DMA setup *plus* the per-extra-BD payload DMA cost.
- **`dense_bytes`** at fixed `n_chunks`: payload bytes per BD scales
  while shim BD count stays at 2 (one in, one out) and the instruction
  stream stays a fixed size. The slope `d(wall) / d(dense_bytes)` is
  the per-byte payload-DMA throughput; subtracting it from the
  `n_chunks` slope leaves the per-chunk shim-DMA setup cost in
  isolation.

A linear regression on each knob gives `slope` + `intercept`. At the
reference shape (n=1, b=4096), the per-launch wall decomposes as:

```
ref_wall_us  =  payload_dma_throughput_us
             +  suspect_c_per_chunk_shim_dma_us
             +  fixed_per_launch_floor_us       (suspects a + b + d combined)
```

The two knobs separate `payload_dma_throughput` and
`suspect_c_per_chunk_shim_dma`; the residual `fixed_per_launch_floor`
is the (a) + (b) + (d) combination, which can be further bisected by
varying the instruction-stream byte count via additional topology
shapes if needed.

## Build

```sh
source /opt/xilinx/xrt/setup.sh
export PEANO_INSTALL_DIR=<path/to/llvm-aie/install>

# Single-variant build (n=1, b=4096):
make NPU2=1

# Or build all bisection variants in one shot:
make NPU2=1 all-variants
```

Each variant lives under `build/n<N>_b<B>/{aie.mlir,final.xclbin,insts.bin}`.

## Run

```sh
./dispatch_overhead_bisector \
    -x build/n1_b4096/final.xclbin \
    -i build/n1_b4096/insts.bin \
    --dense-bytes 4096 \
    --n-chunks 1 \
    --iters 100 --warmup 5
```

The runner prints a per-variant summary as `KEY=VALUE` lines (for
machine parsing) plus the per-iteration distribution:

```
dispatch_iters=100
dispatch_warmup=5
dispatch_dense_bytes=4096
dispatch_n_chunks=1
dispatch_instr_bytes=<instruction-stream byte count>
dispatch_total_bytes=4096
dispatch_wall_us_total=<sum>
dispatch_wall_us_avg=<sum/N>
dispatch_wall_us_min=<min>
dispatch_wall_us_max=<max>
dispatch_wall_us_p50=<p50>
dispatch_wall_us_p90=<p90>
PASS!
```

`--csv-out path/to/file.csv` additionally writes one row per iteration
for offline analysis.

## Sweeping all variants

Once `make all-variants` has produced every `build/n<N>_b<B>/` tree,
loop the host runner over each variant from a shell script of your
choosing:

```sh
for V in build/n*_b*/; do
    n=$(basename "$V" | sed 's/^n\([0-9]*\)_b.*/\1/')
    b=$(basename "$V" | sed 's/^n[0-9]*_b\([0-9]*\)/\1/')
    ./dispatch_overhead_bisector \
        -x "$V/final.xclbin" -i "$V/insts.bin" \
        --dense-bytes "$b" --n-chunks "$n" \
        --iters 100 --warmup 5 \
        --csv-out "results/${n}_${b}.csv"
done
```

The driver script can then regress `dispatch_wall_us_p50` against
`(n_chunks, dense_bytes)` to attribute per-launch wall as described
above.

## Notes on the AIE2P shim BD pool

The lowering pass complains about unassigned BD chain IDs once the
per-channel shim BD count exceeds the auto-chain threshold;
empirically this is hit around `n_chunks=16` (32 total shim BDs).
The `n_chunks` knob is therefore typically capped at 4 for a useful
bisection sweep.

Below `dense_bytes ≈ 256`, AIE2P silicon has been observed to flush
small-payload BDs through a slower firmware path on some firmware
revisions; the regression should either restrict the `dense_bytes`
sweep to ≥ 512 or report the small-payload variants as a separate
"sub-cliff" bucket.
