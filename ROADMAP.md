<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->
<!-- Editors: cite at most one durable anchor per item (a tracking discussion,
     an open umbrella issue, or a label) — never an individual implementation
     PR or a closed "fixed #N" issue, both of which go stale as soon as they
     merge/close. -->
# Roadmap

This roadmap outlines where the project is headed. It reflects current
priorities, not firm commitments — plans may change as the project evolves.

_Last updated: 2026-09-17_

## Now
Things actively being worked on.

- [ ] Dynamic runtime sequences — compile an `aie.runtime_sequence` once and run it at many problem sizes. The compiler foundation and eager IRON `Runtime` callback API are merged; follow-on work targets runtime-scalar JIT dispatch and standalone sequence artifacts, with overlay/sequence caching still under discussion ([#3222](https://github.com/Xilinx/mlir-aie/discussions/3222))
- [ ] Mature native Windows support — complete host-C++ example migration to CMake/CTest, verify actual device execution in CI, and broaden coverage toward Linux parity while preserving supported Make-based entry points ([#3459](https://github.com/Xilinx/mlir-aie/issues/3459))
- [ ] Unify DMA validation and legalization across static and dynamic paths — catch or legalize unsupported layouts, transfer granularity, and address-width limits rather than silently truncating them; improve task-queue safety to prevent dropped transfers ([`dma-programming`](https://github.com/Xilinx/mlir-aie/labels/dma-programming))
- [ ] Pursue a 10/10 developer/maintainer experience — CI improvements, linting, typing, validation, code coverage, reduction of duplicated/stale code, and efforts to document the roadmap, tools, and expectations for contributors
- [ ] Mature modular ObjectFIFO lowering — build on the merged pass decomposition to clarify repetition and depth semantics, separate allocation concerns, and add Python bindings and examples for the mid-level representation ([#3620](https://github.com/Xilinx/mlir-aie/issues/3620))
- [ ] Improve robustness of existing features, including JIT caching and tile-memory diagnostics — refine stack analysis and preserve explicit bank-placement requirements for kernel statics and lookup tables

## Next
Planned for the near future.

- [ ] Consolidate and validate reusable AIE2/AIE2P kernels — build on the merged IRON kernel migration; expand portable transformer/conv coverage and establish numerical contracts, reference testing, benchmarks, and consistent rounding conventions ([#3412](https://github.com/Xilinx/mlir-aie/discussions/3412))
- [ ] Improve agentic workflows for IRON programming — mature the initial skill chain for porting models to AIE/NPU with an LLM coding agent (baseline → dataflow → kernel optimization → validation); still experimental and untested beyond single-dispatch designs ([#3426](https://github.com/Xilinx/mlir-aie/issues/3426))
- [ ] Explore repo boundaries and ecosystem health from a cross-repo, contract-first perspective ([#3390](https://github.com/Xilinx/mlir-aie/discussions/3390))

## Later
Ideas we want to pursue eventually. Not yet scheduled.

- [ ] Express access patterns in the compiler — move DMA tiling from Python `taplib` into MLIR (affine maps + structured control flow) so one program lowers to both static and dynamic data movement ([#3239](https://github.com/Xilinx/mlir-aie/discussions/3239))
- [ ] Tracing mode-1 (EVENT_PC) support — a community-contributed decoder for mode-1 trace streams has been validated against hardware capture; wiring it into the existing trace pipeline's conventions is still being discussed ([#3365](https://github.com/Xilinx/mlir-aie/discussions/3365))
- [ ] Further reduce the project's dependency on Chess/Vitis — build on ongoing Peano migrations, distinguish remaining simulator, compiler, and placement gaps from intentionally Chess-specific coverage, and migrate tests where possible ([#3479](https://github.com/Xilinx/mlir-aie/issues/3479))
- [ ] Support for VEK385

## How to contribute

Want to help or suggest something?

- New to the project? Check items labeled [`good first issue`](https://github.com/Xilinx/mlir-aie/labels/good%20first%20issue).
- Already familiar with the codebase? Look for [`help wanted`](https://github.com/Xilinx/mlir-aie/labels/help%20wanted) or [`needs more 👀 (eyes)`](https://github.com/Xilinx/mlir-aie/labels/needs%20more%20%F0%9F%91%80%20(eyes)).
- Propose new ideas by [opening an issue](https://github.com/Xilinx/mlir-aie/issues/new).
- Questions and discussion go in [Discussions](https://github.com/Xilinx/mlir-aie/discussions).

Items marked below reflect their status:

- **Now** — actively being worked on
- **Next** — planned, help welcome
- **Later** — aspirational, open to input
