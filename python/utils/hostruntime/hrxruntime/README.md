<!---//===- README.md --------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Running IRON (Python) & C++ tests on the **HRX** runtime

This package (`aie.utils.hostruntime.hrxruntime`) is the **HRX** host-runtime
backend for IRON. It dispatches IRON designs and the C++ testbench on the
AMD **XDNA2 NPU** through **`libhrx`** (the IREE-based runtime with an `amdxdna`
HAL), consuming the `aiecc` artifacts (`final.xclbin` + `insts.bin`).

This README is a **self-contained runbook**: follow it top-to-bottom to build and
run both the IRON (Python) end-to-end tests and the C++ on-hardware tests with
HRX. No source edits to example/test files are needed — HRX is selected by an
environment/make variable.

> TL;DR
> - **IRON / Python:** `IRON_RUNTIME=hrx python <example>.py …`
> - **C++:** `RUNTIME=hrx make run`
> - HRX is **auto-detected** (no env vars required if it's in a standard/sibling
>   location). `libhrx` builds the amdxdna XADX executable **internally** via
>   `hrx_amdxdna_executable_create` — there is no separate helper to build.
> - `aiebu-asm` / XRT are **not** required at runtime (libhrx derives the patch
>   table straight from the `insts.bin` transaction).
> - For **XRT-free xclbin packaging** at build time, build the bundled
>   `hrx-xclbinutil` (`-DAIE_BUILD_HRXXCLBINUTIL=ON`, §3b).

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| **XDNA2 NPU** (Strix / `npu4` / aie2p) | `/dev/accel/accel0` present; `amdxdna` driver loaded. |
| **HRX built** (`libhrx.so`) | See §2. This is the runtime we dispatch through; it exports `hrx_amdxdna_executable_create`, which builds the amdxdna XADX package internally. |
| **MLIR-AIE installed from *this* branch** | So `import aie` works, `aiecc` can build examples, and the wheel includes this `hrxruntime` package + the `IRON_RUNTIME` switch. See §3. |
| **A C++ toolchain + CMake ≥ 3.30** | Only for the C++ tests. (`pip install "cmake>=3.30"` into the venv if the system cmake is older.) |
| **Peano / llvm-aie** | Needed by `aiecc` to build the example `.xclbin`/`insts.bin` (same as the XRT flow). |
| **`xclbinutil`** (xclbin packaging) | `aiecc` packages the `.xclbin` with the tool named by **`AIE_XCLBINUTIL`** / **`--xclbinutil-path`** (explicit, deterministic — see §3b), falling back to a `PATH` lookup only if neither is set. Provided by XRT, **or** build the bundled XRT-free `hrx-xclbinutil` (§3b) — recommended on an XRT-free box. |
| XRT userspace (pyxrt) | **Optional.** Only needed for the XRT baseline. HRX itself does not need it. |

The NPU + driver + Peano requirements are exactly the same as the normal (XRT)
IRON flow — if XRT examples build/run on this box, HRX has what it needs too,
plus a built `libhrx`. With the bundled `hrx-xclbinutil` (§3b) instead of XRT's,
the whole HRX flow — build *and* run — needs **no XRT install at all**.

---

## 2. Provision HRX (`libhrx.so`)

Run the fetch helper from the repo root:

```bash
utils/fetch-hrx-release.sh
```

It downloads, checksum-verifies, and extracts the pinned HRX release
(coordinates in `utils/hrx-release.env`) and prints the path to an
`env.sh`. The release ships a relocatable install prefix (`include/hrx` +
`lib/libhrx.so` + `lib/cmake/hrx`); the helper synthesizes an `env.sh` that
exports `HRX_DIR` / `LD_LIBRARY_PATH` / `CMAKE_PREFIX_PATH`, which §4's
auto-detection consumes:

```bash
source "$(utils/fetch-hrx-release.sh)"
# or, in one step:  eval "$(utils/fetch-hrx-release.sh --print-env)"
```

This is exactly how the pure-HRX CI job provisions `libhrx`. Once sourced, skip
to §3 — you're done with HRX provisioning.

---

## 3. Install MLIR-AIE (from this branch)

Install the `mlir_aie` + `llvm_aie` wheels per the project's normal instructions,
**built from this branch** so the wheel ships this `hrxruntime` package and the
`IRON_RUNTIME` selection logic in `aie/utils/__init__.py`.

Verify the imported `aie.utils` is the one from this branch (it must expose the
HRX hooks):

```bash
python3 - <<'PY'
import aie.utils as u, inspect
print("aie.utils from:", inspect.getfile(u))
print("has hrx hooks  :", hasattr(u, "has_hrx"))   # must be True
PY
```

> **Two-trees caveat.** If you instead use a *prebuilt* wheel and edit this source
> tree directly, the edits won't take effect until copied into the imported
> package dir (the path printed above). The robust fix is to **(re)build/install
> the wheel from this branch**. If you must hot-patch, copy:
> `python/utils/__init__.py` → `<aie>/utils/__init__.py` and
> `python/utils/hostruntime/hrxruntime/*` → `<aie>/utils/hostruntime/hrxruntime/`,
> then `rm -rf <aie>/utils/hostruntime/hrxruntime/__pycache__`.

### 3b. Build xclbins without XRT — the bundled `hrx-xclbinutil`

`aiecc` packages the final `.xclbin` by shelling out to a bare `xclbinutil`.
Upstream that tool ships with XRT (and pulls in Boost + a full XRT install). This
branch can instead build a self-contained, **XRT-free / Boost-free** `xclbinutil`
from the `third_party/hrx-xclbinutil` submodule and install it as `xclbinutil`
next to `aiecc` in the wheel's `bin/`, so packaging needs no system XRT.

Enable it when building MLIR-AIE from source:

```bash
git submodule update --init third_party/hrx-xclbinutil
cmake … -DAIE_BUILD_HRXXCLBINUTIL=ON     # add to your normal mlir-aie configure
```

- It is a **trimmed copy of XRT's `xclbinutil`** — only the section handlers an
  NPU xclbin needs (MEM_TOPOLOGY, AIE_PARTITION, EMBEDDED_METADATA, IP_LAYOUT,
  CONNECTIVITY, GROUP_TOPOLOGY, GROUP_CONNECTIVITY) — with Boost replaced by a
  small `std`-backed shim. It builds with just a C++17 compiler + CMake ≥ 3.20.
- **Selecting which `xclbinutil` `aiecc` uses (the primary contract).** Point
  `aiecc` at a specific binary explicitly — this is deterministic and does not
  depend on `PATH` ordering:
  - **`AIE_XCLBINUTIL=<path>`** (environment variable), **or**
  - **`aiecc --xclbinutil-path <path>`** (command-line flag).

  If set, `aiecc` uses exactly that binary and **fails loudly if it is missing**,
  so a `RUNTIME=hrx` build never silently falls back to XRT's tool. This is what
  `programming_examples/makefile-common` does: on `RUNTIME=hrx` it sets
  `AIE_XCLBINUTIL` to the `xclbinutil` installed next to `aiecc` (the bundled
  `hrx-xclbinutil`), resolved as `aiecc`'s sibling.
- **Fallback (only when neither of the above is set):** `aiecc` locates the tool
  via `findAieTool("xclbinutil")`, which searches `PATH` first, then `aiecc`'s
  own directory. Relying on this is discouraged for HRX because a system XRT
  `xclbinutil` earlier on `PATH` wins; prefer `AIE_XCLBINUTIL` /
  `--xclbinutil-path` to make the choice explicit.
- Combined with a built `libhrx` (§2), this makes the entire HRX flow — build
  *and* run — need **no XRT at all** (libhrx derives the patch table itself; see §7).
- **Regression check:** `test/aiecc/hrx_xclbin_sections.mlir` packages a real
  `.xclbin` with the bundled tool and asserts (`xclbinutil --info`) that it
  carries `MEM_TOPOLOGY`, `AIE_PARTITION`, and `EMBEDDED_METADATA`. It is gated
  on the `hrxxclbinutil` lit feature, so it runs only when the tool was built
  (`-DAIE_BUILD_HRXXCLBINUTIL=ON`) and `peano` is available — an in-tree,
  hardware-free xclbin section check.

> **Standalone build** (outside the mlir-aie build — e.g. to shadow XRT's tool on
> `PATH` without rebuilding the wheel):
> ```bash
> cmake -G Ninja -B bld -S third_party/hrx-xclbinutil -DCMAKE_BUILD_TYPE=Release
> cmake --build bld --target hrx-xclbinutil          # -> bld/tools/hrx-xclbinutil
> ln -sf "$PWD/bld/tools/hrx-xclbinutil" <dir-first-on-PATH>/xclbinutil
> ```

---

## 4. Environment (optional — only if auto-detection fails)

Auto-detection (`FindHRX.cmake` for C++, `hrxruntime/discovery.py` for Python)
probes standard locations and a sibling `../hrx-system/build/hrx-install` prefix,
and prefers the shipped `hrx` CMake package (`find_package(hrx CONFIG)` →
`hrx::hrx`). If your HRX lives elsewhere, export these **hints** (highest
priority):

```bash
export HRX_DIR=<hrx-install>                  # install prefix w/ include/hrx + lib/libhrx.so
export LIBHRX_DIR=$HRX_DIR/lib                # dir containing libhrx.so
export LD_LIBRARY_PATH=$LIBHRX_DIR:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=$HRX_DIR:$CMAKE_PREFIX_PATH   # so find_package(hrx) resolves (C++)
```

**Verify HRX is discoverable.** Primary check (robust — prints `True`/`False`
even when HRX is missing; this only checks that `libhrx.so` can be *located* on
disk — **no `dlopen`, no device init, no kernel run**):

```bash
python3 -c "import aie.utils as u; print('has_hrx:', u.has_hrx)"   # -> True
```

If `has_hrx` is `True`, you're ready. To see the resolved paths (only meaningful
once HRX is found):

```bash
python3 -c "from aie.utils.hostruntime.hrxruntime import discovery as d; \
print('libhrx :', d.find_libhrx()); print('root   :', d.find_hrx_dir())"
```

No XADX helper is needed: `libhrx` builds the amdxdna XADX package internally via
`hrx_amdxdna_executable_create`, so linking `libhrx.so` (C++) or dlopen-ing it
(Python) is all the host stack requires.

---

## 5. Run IRON (Python) end-to-end tests with HRX

Select the backend with **`IRON_RUNTIME=hrx`**. Everything else is the normal
IRON flow. Two common shapes:

### 5a. `test.py`-driven example

```bash
cd programming_examples/basic/vector_scalar_mul
make                                  # builds build/final_8192.xclbin + build/insts_8192.bin (aiecc)
IRON_RUNTIME=hrx python3 test.py \
  --xclbin build/final_8192.xclbin --instr build/insts_8192.bin \
  --kernel MLIR_AIE --in1-size 8192 --in2-size 4 --out-size 8192
# expect: PASS!
```

### 5b. `@iron.jit` self-running example

```bash
cd programming_examples/basic/vector_vector_add
IRON_RUNTIME=hrx python3 vector_vector_add.py     # expect: PASS!
```

### Backend selection semantics (`IRON_RUNTIME`)

- `hrx`  — force HRX; clear error if `libhrx` can't be found. **HRX is opt-in:
  this is the only value that selects it.**
- `xrt`  — force XRT (default upstream behavior).
- `auto` *(default when unset)* — XRT if available, else CPU. **`auto` never
  selects HRX** — the product contract is "XRT is the default, HRX is opt-in", so
  an XRT-less host degrades to CPU rather than silently switching to HRX.

Quick smoke test (no hardware dispatch — just import + selection):

```bash
IRON_RUNTIME=hrx python3 -c "import aie.utils as u; print(u.DEFAULT_TENSOR_CLASS.__name__)"
# -> HRXTensor
```

Designs known to pass on HRX: `vector_scalar_mul`, `vector_vector_add`,
`vector_scalar_add`, `vector_reduce_add`, `passthrough_dmas`,
`vector_reduce_max`.

### 5c. Multi-dispatch chains / runlists (`run_chain`)

`HRXHostRuntime.run_chain([(handle, args), ...])` records several
dispatches into one HRX command buffer — in order, with an execution + memory
barrier between them — and submits the whole batch with a single
`synchronize`. The amdxdna HAL lowers the multi-dispatch command buffer into one
`ERT_CMD_CHAIN`. Because of the barrier, a later run observes an earlier run's
device writes, so producer→consumer chains work (one run's output buffer is the
next run's input). Entries may share one `handle` (re-dispatch the same kernel)
or use different handles (a true multi-kernel pipeline).

A self-building backend test mirroring `test_runlist.cpp` (`run0: out0 = in+1`,
`run1: out1 = out0+1`, plus a deeper N-link chain) lives at
`test/python/npu-hrx/test_chain_hrx.py`. It builds its own design via
`@iron.jit`, so it needs no pre-built artifacts:

```bash
IRON_RUNTIME=hrx python3 -m pytest test/python/npu-hrx/test_chain_hrx.py
# expect: passed
```

---

## 6. Run C++ on-hardware tests with HRX

Select the backend with **`RUNTIME=hrx`** on the example's `make`. This builds the
C++ host exe against `libhrx` (no XRT SDK headers needed) and runs it on the NPU.

```bash
cd programming_examples/basic/vector_reduce_max/single_core_designs
make all                       # aiecc: build/final.xclbin + build/insts.bin
RUNTIME=hrx make run           # build C++ host vs HRX + run on the NPU
# expect: ... PASS!
```

What `RUNTIME=hrx` does under the hood (no per-example edits):
- `makefile-common` adds `-DUSE_HRX=ON` to every `build_host_exe` cmake call.
- `programming_examples/common.cmake` runs `find_package(HRX)` (which prefers the
  shipped `hrx` CMake package), links `libhrx.so`, and defines
  `TEST_UTILS_USE_HRX` so the shared `xrt_test_wrapper.h` pulls in
  `hrx_test_wrapper.h`. A dummy `xrt_coreutil` INTERFACE target neutralizes the
  examples' `-lxrt_coreutil`. No XADX helper is compiled — `hrx_test_wrapper.h`
  calls `hrx_amdxdna_executable_create` directly.

> If the system `cmake` is < 3.30, install a newer one (`pip install "cmake>=3.30"`)
> and ensure it's first on `PATH` before `make`.

`make run` builds the host exe into `_build/`, copies it to `./<targetname>.exe`,
and runs it. You can re-run it directly (ensure `libhrx.so` is on the loader
path, e.g. via `LD_LIBRARY_PATH=$HRX_DIR/lib`):

```bash
./vector_reduce_max.exe -x build/final.xclbin -i build/insts.bin -k MLIR_AIE
```

### 6b. C++ multi-dispatch chains / runlists

`hrx_test::dispatch_chain({{&lk, {&a, &b}}, ...})` (in `hrx_test_wrapper.h`)
records several kernel dispatches into one HRX command
buffer — in order, with an execution + memory barrier between them — then submits
the batch with a single `synchronize` (the amdxdna HAL lowers it into one
`ERT_CMD_CHAIN`). A later run sees an earlier run's device writes, so
producer→consumer chains work; entries may share one `LoadedKernel` or use
different ones (a multi-kernel pipeline).

A chained testbench mirroring `test_runlist.cpp` (`run0: out0 = in+1`,
`run1: out1 = out0+1`, where `run1`'s input is `run0`'s output) lives in
`vector_scalar_add`. Its make target self-selects the HRX backend
(`-DUSE_HRX=ON`), so you don't even need `RUNTIME=hrx`:

```bash
cd programming_examples/basic/vector_scalar_add
make all                       # build/final.xclbin + build/insts.bin
make run_runlist_hrx           # build the HRX chain testbench + run on the NPU
# expect: ... PASS!
```

> The HRX runlist testbench reads `insts.bin` directly and hands the raw TXN to
> `hrx_amdxdna_executable_create` (libhrx derives the patch table — no aiebu/ELF
> needed), unlike the XRT `run_runlist` target which consumes `insts.elf`.

---

## 7. How patch tables work (and why aiebu isn't required)

HRX's `amdxdna` COMMAND_CHAIN path host-patches each I/O buffer's device address
into the control code using a **patch table** of `(offset, arg_idx, addend)`
triples. As of the `amdxdna-hal-native-rel` API, **libhrx derives this patch
table itself** inside `hrx_amdxdna_executable_create`, by scanning the XAie
transaction's `BLOCKWRITE`/`DDR_PATCH` ops — the exact same derivation the host
stack used to do. The host therefore just passes the raw `insts.bin` transaction
and the xclbin; no host-side patch-table extraction or XADX serialization
happens anymore.

Without the patch table the NPU writes to address 0 → **all-zero output**, so if
you ever see all-zero output confirm the transaction bytes reached libhrx intact.

- The host does no patch-table extraction of its own. For a raw `insts.bin` it
  passes the transaction straight through; for an ELF input
  (`aiecc --aie-generate-elf`) it extracts `.ctrltext` (the TXN verbatim) via
  the small `control_code_from_elf` helper (defined in Python `_bindings.py` and
  re-exported from the package `__init__.py`; `hrx_test_wrapper.h` on the C++
  side) and hands that to libhrx as the transaction.

---

## 8. Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `IRON_RUNTIME=hrx … ImportError: libhrx.so could not be located` | HRX not found. Build it (§2), place it as a sibling `../hrx`, or set `HRX_DIR`/`LIBHRX_DIR` (§4). Verify with the §4 probe. |
| C++ configure: `USE_HRX=ON but the HRX runtime was not found` | Same as above (CMake side). Set `HRX_DIR`/`CMAKE_PREFIX_PATH` or co-locate `../hrx-system/build/hrx-install`. |
| C++ link/load: `undefined symbol: hrx_amdxdna_executable_create` | Your `libhrx.so` predates the `amdxdna-hal-native-rel` API. Refresh HRX from the pinned release (§2) and re-check with `nm -D`. |
| Output is **all zeros** but no error | The transaction didn't reach libhrx intact (or the xclbin/insts pair is mismatched). Confirm you're on this branch and passing the raw `insts.bin`. |
| `has_hrx`/import works but run hangs or `hrx_stream_synchronize … INTERNAL` | The dispatch may not have completed. Recover the device with a driver reload (`sudo rmmod amdxdna && sudo modprobe amdxdna`). |
| `import aie.utils` lacks `has_hrx` | You're importing an upstream wheel, not this branch (§3 two-trees caveat). |
| C++ build: `CMake 3.30 or higher is required` | Old system cmake; install `cmake>=3.30` and put it first on `PATH`. |
| aiecc: `xclbinutil not found, skipping xclbin generation` (no `.xclbin` produced) | No `xclbinutil` selected. Point `aiecc` at one explicitly with `AIE_XCLBINUTIL=<path>` or `--xclbinutil-path <path>` (§3b); install XRT; or build the bundled one (`-DAIE_BUILD_HRXXCLBINUTIL=ON`, §3b). |
| aiecc: fails because `AIE_XCLBINUTIL` / `--xclbinutil-path` points at a missing file | Intentional loud failure so an HRX build never silently uses XRT's tool. Fix the path, or unset it to fall back to a `PATH` lookup. |

---

## 9. Files in this package

| File | Role |
|---|---|
| `__init__.py` | Package entry point; re-exports `HRXContext`, `HRXError`, and `control_code_from_elf`. Import is side-effect-free (no `dlopen`). |
| `_bindings.py` | The C ABI layer: enum/flag constants, `ctypes` struct mirrors, lazy `libhrx` `dlopen`, the bound `hrx_*` entry points, `HRXError`, and `control_code_from_elf` for the ELF `.ctrltext` input path. |
| `context.py` | `HRXContext` — the process-wide device/stream/buffer/exe/dispatch singleton, with `create_executable` calling `hrx_amdxdna_executable_create`. |
| `hostruntime.py` | `HRXHostRuntime` / `CachedHRXRuntime` (the IRON `HostRuntime` implementation). |
| `tensor.py` | `HRXTensor` (persistent host-mapped device buffer, zero-copy numpy view). |
| `discovery.py` | Path-only HRX discovery (no dlopen): `find_libhrx`/`find_hrx_dir`/`hrx_available`. |

Related (outside this package):
- `cmake/modules/FindHRX.cmake` — HRX auto-detection for CMake (prefers the shipped `hrx` package → `hrx::hrx`).
- `programming_examples/common.cmake` — `USE_HRX` wiring (links `libhrx`; no helper build).
- `programming_examples/makefile-common` — the `RUNTIME=xrt|hrx` switch.
- `runtime_lib/test_lib/hrx_test_wrapper.h` — C++ HRX backend.
- `tools/hrx-xclbinutil/CMakeLists.txt` + `third_party/hrx-xclbinutil` — the bundled
  XRT-free `xclbinutil` for xclbin packaging (`-DAIE_BUILD_HRXXCLBINUTIL=ON`, §3b).
- `test/aiecc/hrx_xclbin_sections.mlir` — lit section-check test for the bundled
  `xclbinutil` (gated on the `hrxxclbinutil` feature added in `test/lit.cfg.py`).
