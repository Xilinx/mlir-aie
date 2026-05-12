# compression_silicon_check

A silicon-level probe for AIE-ML compute-tile DMA `Enable_Compression`. Single-tile
passthrough (shim → compute(0,2) → shim) on Phoenix npu1, with optional per-BD
and per-channel compression registers flipped via `aiex.npu.maskwrite32` from
the runtime sequence. No mlir-aie C++ patches required — uses off-the-shelf
ops to poke the registers directly from the host instruction stream.

Built to answer one question for [PR #3039](https://github.com/Xilinx/mlir-aie/pull/3039)'s
SparseFifo plumbing: **does the AIE-ML compression hardware actually do anything
when the BD-level `Enable_Compression` bit is set?** Spoiler: yes, but only when
a *channel-level* enable bit is also set on the same direction's CTRL register —
which the PR does not plumb.

This is the only DMA-compression test in mlir-aie today; intended as a regression
guard if/when the C++ extraction lands.

## Configurations

There are six configurations across two groups.

### Symmetric / unsized (`run_sym`)

| Config       | Producer MM2S BD bit | Consumer S2MM BD bit | MM2S CTRL bit | S2MM CTRL bit | Shim BD lengths |
|--------------|:--------------------:|:--------------------:|:-------------:|:-------------:|:---------------:|
| `base`       | —                    | —                    | —             | —             | in=4096 out=4096 |
| `cmp_only`   | ✓                    | —                    | ✓             | —             | in=4096 out=4096 |
| `dcmp_only`  | —                    | ✓                    | —             | ✓             | in=4096 out=4096 |
| `both`       | ✓                    | ✓                    | ✓             | ✓             | in=4096 out=4096 |

`cmp_only` and `dcmp_only` here use *unsized* shim BDs (in_n = out_n = N).
This intentionally creates a length mismatch — the compressor emits fewer
bytes than the consumer S2MM expects, the BD never completes, `amdxdna` times
out at 10 s and retries ~3 times. Wall time ≈ 30 s. The slowness is **purely
driver timeout, not compression latency**.

### Ratioed asymmetric (`run_ratioed`)

| Config              | Compression bits set | Shim BD lengths    |
|---------------------|----------------------|--------------------|
| `cmp_only_ratioed`  | producer side only   | in=4096 out=2944   |
| `dcmp_only_ratioed` | consumer side only   | in=2944 out=4096   |

Sized so the shim BD lengths match the empirical compressed byte count for
arange input (~1.39× ratio for ramp 0..N-1). Demonstrates that with correct
length math, asymmetric configs complete in 1 ms — confirming compression
hardware itself adds no latency.

## Expected results (Phoenix npu1)

Run on a machine with `cat /sys/module/amdxdna/parameters/timeout_in_sec` = 10
or higher. Lower values will skew the wall times for the unsized asymmetric
configs.

```
--- config: base ---
[base] running, N=4096... done in 0ms
[base] result: matches=4096 mismatches=0 untouched(sentinel)=0 (TDR=no)

--- config: cmp_only ---                                # SLOW: BD-length mismatch
[cmp_only] running, N=4096... done in 27000ms
[cmp_only] result: matches=1024 mismatches=1920 untouched(sentinel)=1152

--- config: dcmp_only ---                               # SLOW: BD-length mismatch
[dcmp_only] running, N=4096... done in 27000ms
[dcmp_only] result: matches=1024 mismatches=3072 untouched(sentinel)=0

--- config: both ---                                    # FAST: symmetric round-trip
[both] running, N=4096... done in 1ms
[both] result: matches=4096 mismatches=0 untouched(sentinel)=0

--- config: cmp_only_ratioed ---                        # FAST: ratio-sized
[cmp_only_ratioed] running, N=4096... done in 1ms
[cmp_only_ratioed] result: matches=1024 mismatches=1920 untouched(sentinel)=1152

--- config: dcmp_only_ratioed ---                       # FAST: ratio-sized
[dcmp_only_ratioed] running, N=4096... done in 1ms
[dcmp_only_ratioed] result: matches=1024 mismatches=3072 untouched(sentinel)=0
```

`base` and `both` are the only two configurations whose output bit-equals input.
`both` being fast and lossless is the proof that compress + decompress compose
into an identity in flight on real silicon. `cmp_only_ratioed` and
`dcmp_only_ratioed` being fast (vs the unsized variants' 27 s) is the proof
that the apparent slowness is BD-length mismatch, not compression latency.

The first 1024 elements always match the input ramp across every config that
engages (de)compression — first-BD-after-channel-config pass-through, looks
like a state-machine warm-up or a write-ordering race between the host's
`maskwrite32` and the tile starting its DMA. Reproducible artifact, doesn't
change the conclusions.

## How to replicate

### Prereqs

- Phoenix npu1 (or compatible AIE-ML device) accessible via `/dev/accel/accel0`
- `amdxdna` driver loaded
- `mlir-aie` and `llvm-aie` (Peano) installed
- XRT installed at `/opt/xilinx/xrt` (or wherever `setup.sh` lives)

### Environment

```bash
source /opt/xilinx/xrt/setup.sh
source /path/to/mlir-aie/utils/env_setup.sh /path/to/mlir-aie/install
# If you use a virtualenv (e.g., the project's ironenv):
source /path/to/mlir-aie/ironenv/bin/activate
```

### Build + run everything

```bash
cd programming_examples/basic/compression_silicon_check
make test_all
```

This builds the host executable, the six xclbins, and runs all six configs in
sequence. Total time on Phoenix is ~1 minute (most of which is the two unsized
asymmetric configs' driver-timeout wait).

### Run a single config

```bash
make CONFIG=both run         # build + run one config
make run_sym                  # build + run the four symmetric configs
make run_ratioed              # build + run the two ratioed asymmetric configs
make clean
```

### Inspect the lowered MLIR

```bash
python3 compression_check.py both 4096 4096 \
  | aie-opt --aie-objectFifo-stateful-transform --aie-assign-bd-ids \
  | grep -E 'maskwrite32|dma_bd|next_bd'
```

### Register details

| Register                                  | Tile-relative addr | Bit | Purpose                       |
|-------------------------------------------|--------------------|-----|-------------------------------|
| `MEMORY_MODULE_DMA_BD{n}_1`               | 0x1D004 + n·0x20   | 31  | BD-level Enable_Compression   |
| `MEMORY_MODULE_DMA_S2MM_0_CTRL`           | 0x1DE00            | 4   | Channel-level Decompression_Enable |
| `MEMORY_MODULE_DMA_MM2S_0_CTRL`           | 0x1DE10            | 4   | Channel-level Compression_Enable   |

Source: `third_party/aie-rt/driver/src/global/xaiemlgbl_params.h` and
`xaiemlgbl_reginit.c`. The driver's `XAie_DmaEnableCompression()` /
`XAie_DmaChannelEnCompression()` API pair (`xaie_dma.c:522,1747`) makes the
two-register requirement explicit; mlir-aie's existing passes today plumb
neither for the compute-tile path.
