# yolo26n-opt — fresh benchmark record (2026-06-19)

Authoritative performance numbers from a clean, from-scratch rebuild of the
PR #3142 branch. Supersedes the incremental, mid-sprint figures scattered
through the commit log and README (those were captured at different points
against a `install/` that no longer matched the branch source).

## Provenance

| Item | Value |
|---|---|
| Date (UTC) | 2026-06-19 |
| Branch | `yolo26n-opt` (PR #3142 head), local `yolo26n-opt-bench` |
| Commit | `b7e499a51` (Merge branch 'main' into yolo26n-opt) |
| mlir-aie build | from source via `utils/build-mlir-aie-from-wheels.sh` |
| MLIR base wheel | `23.0.0.2026051220+ea2f5081` (pinned by `utils/clone-llvm.sh`) |
| Peano (llvm-aie) | `21.0.0.2026061901+a76244b4` (nightly) |
| nanobind | `2.12.0` (from hash-pinned `python/requirements_dev.lock`) |
| Python deps | `requirements.txt` + `requirements_dev.lock` (CI-matched) + `requirements_ml.txt` |
| NPU | AMD Strix (`[0000:65:00.1] NPU Strix`), firmware 255.0.5.35 |
| XRT | 2.20.0 |
| Power mode | **Turbo** (`sudo xrt-smi configure --pmode turbo`) — confirmed on at start and end |
| Chain default | `M8_TILES=4`; m9 standalone uses `M9_STAGE=10` |
| Timing | `time_chain.py` / `time_block.py`, `--n-warmup 3 --n-iters 20`, per-sample median |

### Build-environment note (root-cause of a false start)

The pre-existing `install/` + `ironenv/` were inconsistent with this branch:
the installed MLIR wheel was `21.0.0` (branch pins `23.0.0`), and a loose
`requirements_dev.txt` install pulled **nanobind 2.13.0** while the bindings
must be built against **2.12.0** (what CI pins in `requirements_dev.lock`).
The 2.13↔2.12 ABI skew made `aie.extras` `ScalarValue` incompatible with the
freshly-built `_mlir` bindings (`TypeError: incompatible function arguments
… ScalarValue vs Value`). Fix: install `requirements_dev.lock` with
`--require-hashes` (matching CI), relink the bindings, rebuild. After that the
full chain is bit-exact vs ORT.

## Correctness gate

Full chain m0..m10 at N=15: **bit-exact vs ONNXRuntime**
(`mismatches=0/30, max|diff|=0`, all 15 samples). Timing below was only taken
after this passed.

## Chain — full prefix sweep (per-sample median, turbo)

`CHAIN_BLOCKS="m0,…,mN" CHAIN_N_SAMPLES=N make time_chain`. fps = 1000 / per-sample ms.

| Chain | N=1 ms | N=1 fps | N=4 ms | N=4 fps | N=15 ms | N=15 fps |
|---|---:|---:|---:|---:|---:|---:|
| m0        | 2.89 | 346.14 | 2.80 | 356.85 | 2.78 | 360.17 |
| m0..m1    | 2.86 | 350.07 | 2.79 | 358.02 | 2.78 | 359.69 |
| m0..m2    | 2.98 | 335.91 | 2.81 | 356.05 | 2.78 | 359.21 |
| m0..m3    | 3.00 | 333.22 | 2.83 | 353.20 | 2.78 | 359.14 |
| m0..m4    | 3.07 | 325.55 | 2.85 | 351.08 | 2.79 | 358.11 |
| m0..m5    | 3.11 | 321.89 | 2.87 | 348.66 | 2.79 | 357.84 |
| m0..m6    | 3.26 | 306.42 | 2.90 | 344.92 | 2.80 | 356.77 |
| m0..m7    | 3.48 | 287.31 | 2.96 | 338.04 | 2.82 | 354.15 |
| m0..m8    | 3.91 | 255.83 | 3.05 | 327.48 | 2.85 | 351.02 |
| m0..m9    | 6.21 | 160.95 | 3.80 | 263.48 | 3.19 | 313.29 |
| **m0..m10 (FULL)** | **6.70** | **149.15** | **3.96** | **252.74** | **3.28** | **304.90** |

Full chain m0..m10: **304.90 fps** at N=15 (3.28 ms/sample), **149.15 fps**
single-dispatch (N=1). m9 (attention) is the dominant per-stage cost — the
N=1 jump from m0..m8 (3.91 ms) to m0..m9 (6.21 ms) is the attention block;
batching (N≥4) hides most of it.

## Per-block standalone (median of n=20, turbo)

`make BLOCK=mN time`. m8 = `M8_TILES=4`, m9 = `M9_STAGE=10`.

| Block | Topology | Median (ms) | fps |
|---|---|---:|---:|
| m0  | conv2dk3 stem (3x3 s2)         | 2.89 | 345.91 |
| m1  | conv_stride2                  | 2.52 | 396.45 |
| m2  | c3k2_small                    | 2.38 | 420.89 |
| m3  | conv_stride2 (chunked)        | 1.63 | 612.40 |
| m4  | c3k2_small                    | 1.74 | 574.51 |
| m5  | conv_stride2 (chunked)        | 1.82 | 550.83 |
| m6  | c3k2_heavy                    | 0.98 | 1024.24 |
| m7  | conv_stride2 (chunked)        | 1.62 | 616.03 |
| m8  | c3k2_heavy (4-tile megakernel)| 2.12 | 471.06 |
| m9  | attention (qkv/attn/ffn)      | 2.72 | 367.01 |
| m10 | head conv2dk1 + pool + gemm   | 1.76 | 567.17 |

## vs README / PR-description claims

The README's chain table (e.g. m0..m10 N=15 = 62.32 fps; m0 N=15 = 90.81 fps)
and the commit-log progression (200–221 fps range) are **stale and now
superseded**. This fresh full-rebuild measures the full chain at
**304.90 fps (N=15)** — materially higher than every previously-recorded
figure. Likely contributors: the matched MLIR-base + Peano + nanobind stack
built coherently from source, and a single consistent measurement pass rather
than deltas stitched across sprints.

## Reproduce

```bash
source /opt/xilinx/xrt/setup.sh
source ironenv/bin/activate
source utils/env_setup.sh install
export AIETOOLS_ROOT=/proj/xbuilds/2024.2_INT_daily_latest/installs/lin64/Vitis/2024.2
export PATH=$PATH:${AIETOOLS_ROOT}/bin
cd programming_examples/ml/yolo26n
make data
CHAIN_N_SAMPLES=15 make run_chain          # bit-exact gate
CHAIN_BLOCKS="m0,…,m10" CHAIN_N_SAMPLES=15 make time_chain   # chain timing
make BLOCK=mN time                          # per-block timing
```
