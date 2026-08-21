<!-- Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# hrx-xclbinutil

An in-tree component of mlir-aie (not a standalone repository): a self-contained,
**Boost-free**, **NPU-minimal** build of the XRT `xclbinutil` packaging tool. It
wraps a PDI (produced by bootgen) into the final `.xclbin` container for AIE/NPU
designs — with **no XRT install and no Boost**.

Normally it is built as part of mlir-aie by configuring with
`-DAIE_BUILD_HRXXCLBINUTIL=ON`, which installs it as `xclbinutil` next to `aiecc`
(see `tools/hrx-xclbinutil/CMakeLists.txt`). The standalone build below is for
developing/testing this tool in isolation.

## Why is this in-tree?

This tool was previously carried as a git submodule pointing at
[`jtuyls/hrx-xclbinutil`](https://github.com/jtuyls/hrx-xclbinutil), last pinned
at commit `3940dd23bda7a941df9f2d10015fd66c9410b7af`. mlir-aie is its only
consumer, so a separate one-user repository added friction (an extra clone /
`git submodule update --init`, a second place to land changes, and pinning
churn) without any benefit. The sources were folded in here as regular tracked
files so that:

- the build and the on-device HRX tests work from a plain mlir-aie checkout,
  with no submodule init step;
- changes to the tool are reviewed and versioned alongside the code that uses
  it, in a single commit/PR; and
- CI can build and test it directly (see
  `.github/workflows/buildAndTestHrxXclbinutil.yml`).

The `vendor/` subtree is upstream XRT source (kept under its original
Apache-2.0 / MIT licenses via `REUSE.toml`); everything else here is
first-party. If upstream XRT changes are ever needed, re-trim from the pinned
XRT revision noted below rather than reintroducing a submodule.

## Layout

```
vendor/                 trimmed XRT source (tracked), pinned to nod-ai/XRT @ a9fdf618
  xclbinutil/             30 .cxx — only the section handlers an NPU xclbin needs
                          (+ aie-pdi-transform/ = transformcdo). FPGA/flash/BMC/
                          SmartNIC/MCS sections removed.
  include/xclbin.h        axlf container format
  version.h.in            version header template
util/                   the hrx:: util module (replaces Boost)
  hrx_util.h              types + templates: hrx::format, hrx::property_tree::ptree,
                          hrx::program_options, hrx::algorithm, hrx::optional,
                          hrx::uuids, hrx::lexical_cast, ... (+ XML writer)
  hrx_util.cpp            non-template JSON read/write engine
CMakeLists.txt          self-contained build
```

The vendored source `#include "hrx_util.h"` and call `hrx::...` directly — no Boost
masquerade. Two off-path functions (`exec`, `search_path`) were reimplemented
without `boost::process`/`asio`.

## Build

Prerequisites (system): CMake ≥3.20, Ninja, and a C++17 compiler (clang or gcc) —
that's it. On Debian/Ubuntu: `sudo apt-get install cmake ninja-build clang`.
(No `uuid-dev`: the vendored `xclbin.h` self-defines `uuid_t` since only the type
is used — see the note in `vendor/include/xclbin.h`.)

```bash
cmake -G Ninja -B bld -S . -DCMAKE_BUILD_TYPE=Release
cmake --build bld --target hrx-xclbinutil
# -> bld/tools/hrx-xclbinutil   (reports "XRT Build Version: 2.18.0")
ctest --test-dir bld --output-on-failure   # functional smoke test
```

## Use

Expose it as `xclbinutil` on `PATH` (the name the AIE toolchain looks for):

```bash
mkdir -p bin && ln -sf "$PWD/bld/tools/hrx-xclbinutil" bin/xclbinutil
export PATH="$PWD/bin:$PATH"
```

## Validation

All 7 functional xclbin sections (MEM_TOPOLOGY, AIE_PARTITION, EMBEDDED_METADATA,
IP_LAYOUT, CONNECTIVITY, GROUP_TOPOLOGY, GROUP_CONNECTIVITY) are **byte-identical**
to a real-Boost build of the same tool. The only diffs are non-functional: the
axlf build timestamp (wall-clock) and a cosmetic version-metadata string (Boost's
`format("%d") % uint8` prints raw bytes; the hrx shim prints `"2.18.0"`).
