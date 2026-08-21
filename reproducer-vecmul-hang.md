---
type: note
date: 2026-07-14
tags: [mlir-aie, bug, npu, aie2p, strix, reproducer]
---
# Reproducer — vectorized `vector_scalar_mul` hangs the NPU (ERT timeout)

Self-contained reproducer for the stock mlir-aie `vector_scalar_mul` example hanging
on Strix/`aie2p` hardware. Concrete paths are for our shared lab environment
(`/proj/xbuilds` Vitis snapshots, XRT under `/opt/xilinx`); substitute your own where
marked `<...>` (your mlir-aie checkout and Python venv).

## TL;DR
The **stock** `programming_examples/basic/vector_scalar_mul` example **hangs the NPU**
(`ERT_CMD_STATE_TIMEOUT`) at its **default config** on a healthy Strix/`aie2p` device.
It is:
- **compiler-invariant** — hangs with peano *and* with two distinct `chesscc` builds
  (`#241219` Dec 2024 and `#250729` Jul 2025), across Vitis 2025.1 / 2025.2;
- **firmware/driver/kernel-invariant** — reproduced on two independent Strix/`aie2p`
  environments spanning an NPU-firmware, kernel, and amdxdna-driver delta;
- **backend- and dtype-independent** (int16 and int32);
- **deterministic** — every attempt hangs.

The single differentiator is the **vectorized `scale` kernel**: the same design built
with the **scalar** `scale` kernel, and an IRON `transform` kernel, both **run fine**
on the same device in the same session. A trivial `out = in · scalar` failing at stock
config is a real regression.

## Target stack
- **Hardware:** AMD Ryzen AI ("Strix") NPU — `npu2` / `aie2p`, 8 columns.
- **OS:** Ubuntu 24.04.1, Linux 6.14.x.
- **Runtime:** XRT 2.20.0 + amdxdna driver 2.20.0.
- **Compiler / tooling:** `mlir_aie` 1.3.4 wheel + peano `llvm-aie 21.0.0.2026062301+cb664e8c`
  (the default backend), or chess from a Vitis 2025.1 / 2025.2 snapshot on `/proj/xbuilds`
  carrying the `aie2p` target (exact `settings64.sh` paths in the step-3 chess block).
- **Sources:** the in-tree `programming_examples/basic/vector_scalar_mul` +
  `aie_kernels/aie2/scale.cc`, unmodified.

## Reproduced across two independent environments
The hang is not tied to one machine's firmware/driver state — it reproduced on two
independent Strix/`aie2p` setups that differ in NPU firmware, Linux kernel, and driver
build:

| host | NPU firmware | Linux kernel | amdxdna driver | XRT | result |
|---|---|---|---|---|---|
| `xcoradaie213` | `255.0.2.7` | `6.14.0-061400-generic` | `2.20.0_20251110` | `2.20.0` | **HANG** (deterministic, 4/4) |
| `xcoradaie211` | `255.0.5.35` | `6.14.0-28-generic` | `2.20.0_20251008` | `2.20.0` | **HANG** (deterministic, 2/2) |

Held identical across both: chip (Strix/`aie2p`/`npu2`), Ubuntu 24.04.1, `mlir_aie 1.3.4`,
peano `21.0.0.2026062301+cb664e8c`, and the stock example + `scale.cc`. Note
`xcoradaie211`'s driver build is *older* (Oct 08) than `xcoradaie213`'s (Nov 10) while its
firmware is *newer* — independent machine states, not a single linear update.

## Prerequisites
- The **target stack** above, with **XRT installed and `xrt-smi` on your PATH**
  (`source /opt/xilinx/xrt/setup.sh` if it isn't), and a **Python venv containing the
  `mlir_aie` 1.3.4 wheel** (peano is pulled in with it). If you don't already have that
  venv, build it per the mlir-aie install / quickstart docs first.

## 1. Set up the shell (do this first — later steps assume it)
```bash
cd <mlir-aie checkout>                           # stock/unmodified example + kernels
source <venv>/bin/activate                       # Python venv with the mlir_aie 1.3.4 wheel
export MLIR_AIE_INSTALL_DIR="$(pip show mlir_aie | awk '/^Location/{print $2}')/mlir_aie"
export PATH="$MLIR_AIE_INSTALL_DIR/bin:$PATH"
```

## 2. Capture your environment (please report it with your result)
Firmware / driver / kernel / toolchain all matter for this bug — read them **live on
your box** (with the venv from step 1 active) and include the output with whatever
result you get, so every run is self-describing:
```bash
hostname; uname -r; . /etc/os-release && echo "$PRETTY_NAME"
xrt-smi examine | grep -E 'BDF|NPU|Processor|Model|Version|Branch|Hash|amdxdna|Firmware'
python3 -c "import importlib.metadata as m; print('mlir_aie', m.version('mlir_aie')); print('llvm-aie', m.version('llvm-aie'))"
git rev-parse --short HEAD
git status --short programming_examples/basic/vector_scalar_mul aie_kernels   # confirm example + kernel are unmodified
```
Note the NPU **BDF** from the `Device(s) Present` table (e.g. `0000:65:00.1`) — you pass
it to `xrt-smi validate` in step 3.

## 3. Build and run (default is peano — no Vitis needed)
```bash
cd programming_examples/basic/vector_scalar_mul
make clean && make all                           # int16, 8192 B, peano, VECTORIZED scale kernel

# Clean protocol: prove the device is healthy immediately before the run.
xrt-smi validate -d <BDF> --run latency          # -> [PASSED]
python3 test.py --xclbin build/final_8192.xclbin --instr build/insts_8192.bin \
        --kernel MLIR_AIE --in1-size 8192 --in2-size 4 --out-size 8192
# EXPECTED (bug):
#   HostRuntimeError: Kernel returned ert_cmd_state.ERT_CMD_STATE_TIMEOUT
#   (raised from DefaultNPURuntime.run -> .../xrtruntime/hostruntime.py:284)
# WANTED (if fixed): PASS!
```
The failure is inside `run()` (the hardware executing the xclbin); `load()` is fine.

**Optional — chess backend** (to confirm compiler-invariance). Source a Vitis snapshot
from `/proj/xbuilds` that ships the `aie2p` chess target, then rebuild with `CHESS=true`.
The two genuinely distinct `chesscc` builds tested (mind the differing install layout):
```bash
# chesscc #241219 (built 20-Dec-2024) — Vitis 2025.2_INT:
source /proj/xbuilds/2025.2_INT_qualified_latest/installs/lin64/Vitis/HEAD/settings64.sh
# --- or chesscc #250729 (built 30-Jul-2025) — Vitis 2025.2_REL (note: .../HEAD/Vitis layout):
source /proj/xbuilds/2025.2_REL_qualified_latest/installs/lin64/HEAD/Vitis/settings64.sh

export PATH="$XILINX_VITIS/bin:$XILINX_VITIS/aietools/bin:$PATH"
xchesscc +v                                      # confirm which chesscc build you got
make clean && make all CHESS=true                # rebuild with chess
xrt-smi validate -d <BDF> --run latency          # -> [PASSED]
python3 test.py --xclbin build/final_8192.xclbin --instr build/insts_8192.bin \
        --kernel MLIR_AIE --in1-size 8192 --in2-size 4 --out-size 8192   # also hangs
```
(`2025.1_INT_daily_latest` currently resolves to the same `chesscc #241219`, so it is not
an independent compiler point right now — the axis that matters is the `chesscc` build.)

## Scope matrix (each row: build → `xrt-smi validate` [PASSED] → run)
| example / config | compute kernel | compiler | dtype | result |
|---|---|---|---|---|
| vector_scalar_mul (default) | `scale` **vectorized** | peano `21.0.0.2026062301` | int16 | **HANG** |
| vector_scalar_mul `-bw 32` | `scale` **vectorized** | peano | int32 | **HANG** |
| vector_scalar_mul `CHESS=true` | `scale` **vectorized** | chess `#241219` (Dec 2024; Vitis 2025.2_INT, also 2025.1_INT) | int16 | **HANG** |
| vector_scalar_mul `CHESS=true` | `scale` **vectorized** | chess `#250729` (Jul 2025; Vitis 2025.2_REL) | int16 | **HANG** |
| event_trace | `scale` **scalar** | peano / chess | int32 | PASS |
| vector_scalar_add | IRON `transform` (x+1) | peano | int32 | PASS |
| `xrt-smi validate` (latency/throughput) | builtin | — | — | PASS |

The invariance axis is the **`chesscc` build, not the Vitis label** — `2025.1_INT_daily_latest`
and `2025.2_INT_qualified_latest` currently ship the *same* `chesscc #241219`, so the two
rows above are the two genuinely distinct chess compilers tested (`#241219`, `#250729`),
plus peano. Confirm yours with `xchesscc +v`.

## What's already ruled out
- **Device / inherited state** — `xrt-smi validate` PASSES immediately before and after
  every hang; two other designs run fine in the same session; the queue-wedge from a
  timeout is fully cleared by the next `validate` (device undamaged).
- **Buffer size** — fails at the stock 8192 B default.
- **Compiler / toolchain version** — peano + two distinct `chesscc` builds (`#241219`
  Dec 2024, `#250729` Jul 2025) all hang. Note the **default build is peano**, bundled in
  the `mlir_aie` 1.3.4 wheel and **Vitis-independent** — no Vitis choice changes it; and
  switching Vitis version does not help.
- **dtype** — int16 and int32 both hang.
- **Firmware / driver / kernel** — reproduced across the two-host delta above.
- **Topology / liveness** — `vector_scalar_mul` (hangs) and `event_trace` (passes) lower
  to the **same design**: objectfifos `in`/`out` `1024×`(depth 2) + `factor` `1×`(depth 2);
  core = persistent outer `scf.for`, inner `scf.for 0..4` → acquire in+out →
  `func.call <scale>` → release; runtime DMA 4096 in / 4096 out. Static liveness is clean
  and the output element count matches the DMA (no transfer-count mismatch).

## The one real difference — the compute kernel path (both from `aie_kernels/aie2/scale.cc`)
Selected by the design's `vectorized=True|False` flag (`kernels.scale(...)`), which links
a different `extern "C"` entry from the **same** source file:
- **Hangs** — `vector_scalar_mul_vector` → `scale_vectorized<int16_t>`: `vec_factor=32`,
  `aie::load_v<32>` → `aie::mul(A0, factor)` (`acc32`) → `store_v(cout.to_vector<int16_t>(0))`,
  finite `for i < N/32`. (The int32 `-bw 32` path is the `scale_vectorized<int32_t>`
  specialization — `vec_factor=16` + `acc64` — and also hangs.)
- **Works** — `vector_scalar_mul_scalar` → `scale_scalar<T>`: plain `for i < N: c[i] = factor*a[i]`.

The finite compute loop should not infinite-loop, so the working hypothesis is that the
**vectorized kernel executes on the core and stalls it** (HW context times out) —
consistent with the failure being invariant to the whole compiler axis *and* to
firmware/driver, while the scalar path is fine.

## What did NOT localize it
- **aiesim (SystemC ISS)** — the sim package builds and the AIE2 ISS starts, but NPU
  objectfifo / runtime-sequence designs aren't wired for the aiesim testbench: the
  generated `ps.so` has `ps_main` **undefined**, and the design maps to an AIE-ML sim
  model rather than `aie2p`. An `aie2p` ISS device exists
  (`aietools/data/aie2p/devices/aie2p_8x4_device.json`) but needs regen-for-npu2, a
  `genwrapper_for_ps.cpp` link fix, and a stimulus/done testbench.

## Suggested next steps for debugging
1. **Confirm known-good combos** — is the vectorized `scale` kernel expected to work on
   `aie2p`/`npu2` at these firmware levels with `mlir_aie 1.3.4` + peano
   `21.0.0.2026062301`? Any (firmware, wheel, peano) combo where it passes?
2. **On-core post-mortem** — read the core's PC / event / exception state after the
   timeout via any amdxdna/XRT debug interface, to distinguish an infinite-loop / fault
   inside the kernel from a DMA/lock stall.
3. **Kernel-source bisect** — in `aie_kernels/aie2/scale.cc`, reduce `scale_vectorized`
   toward `scale_scalar` (e.g. drop the `aie::mul`/`acc32`, or the `store_v(to_vector<>)`
   cast, or shrink `vec_factor`) to find the exact op/pattern that triggers the stall.
4. **`aie2p` aiesim** — finish the sim build (regen for npu2, link `ps_main`, add a
   stimulus/done testbench) so the kernel can run in simulation, separating a
   codegen/kernel-logic fault from a HW/firmware one.

## Operational notes
- A timed-out run wedges the NPU command queue until the next `xrt-smi validate` (or a
  reset). Re-validate before each attempt.
- `xrt-smi reset --force` on a shared box may be access-gated — prefer `validate` to
  clear the queue between runs.
- **Only count a run whose pre-run `validate` PASSED.** If the pre-run `validate` itself
  fails — e.g. `DRM_IOCTL_AMDXDNA_GET_INFO IOCTL failed (err=-12): Cannot allocate
  memory` — the device is in a bad/transient state, not exhibiting this bug. Run
  `xrt-smi validate` again until it PASSES, then retry; an `err=-12`/ENOMEM from
  `test.py` is that device-state issue, *not* the `ERT_CMD_STATE_TIMEOUT` this report is
  about. (Seen once during a chess sweep; a second `validate` cleared it and the gated
  retry then reproduced the timeout.)
