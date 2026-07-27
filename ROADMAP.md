<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->
<!-- Editors: cite at most one durable anchor per item (a tracking discussion,
     an open umbrella issue, or a label) — never an individual implementation
     PR or a closed "fixed #N" issue, both of which go stale as soon as they
     merge/close. -->
# Roadmap

This roadmap outlines where the project is headed. It reflects current
priorities, not firm commitments — plans may change as the project evolves.

_Last updated: 2026-07-27_

## Now
Things actively being worked on.

- [ ] Dynamic runtime sequences — compile an `aie.runtime_sequence` once and run it at many problem sizes. The core milestone (standalone TXN encoding, AIEX→EmitC codegen, SSA-operand control ops, dynamic BD allocation, multi-column GEMM, dynamic trace) is merged; what remains is reworking the IRON `Runtime` Python API into an eager callback body so the control flow this milestone introduces can be expressed without hacks, plus a handful of optional EmitC coverage-gap ops ([#3222](https://github.com/Xilinx/mlir-aie/discussions/3222))
- [ ] Mature native Windows support — broaden example and CI coverage, and close feature-parity gaps with Linux (builds on the [native Windows guide](docs/buildHostWinNative.md))
- [ ] Unify DMA buffer-descriptor validation and legalization — close gaps where hardware-illegal access patterns are silently truncated or rejected outright instead of caught or legalized, tracked under the [`dma-programming`](https://github.com/Xilinx/mlir-aie/labels/dma-programming) label; broader layout-transformation checks remain open in [#2566](https://github.com/Xilinx/mlir-aie/issues/2566)

## Next
Planned for the near future.

- [ ] Make ObjectFIFO lowering fully dynamic — replace the pass's custom static unrolling/analysis with always-dynamic bookkeeping plus a general-purpose unrolling pass that folds constants back where possible, so arbitrary control flow around object FIFOs is supported instead of rejected ([#3298](https://github.com/Xilinx/mlir-aie/pull/3298))
- [ ] Consolidate reusable aie2p kernels — upstream generic transformer/conv kernels (transpose, depthwise conv, higher-precision activations, fused matmul epilogues) authored for an external inference engine into `aie_kernels/aie2p/` ([#3412](https://github.com/Xilinx/mlir-aie/discussions/3412))
- [ ] Improve agentic workflows for IRON programming — mature the initial skill chain for porting models to AIE/NPU with an LLM coding agent (baseline → dataflow → kernel optimization → validation); still experimental and untested beyond single-dispatch designs ([#3426](https://github.com/Xilinx/mlir-aie/issues/3426))

## Later
Ideas we want to pursue eventually. Not yet scheduled.

- [ ] Express access patterns in the compiler — move DMA tiling from Python `taplib` into MLIR (affine maps + structured control flow) so one program lowers to both static and dynamic data movement ([#3239](https://github.com/Xilinx/mlir-aie/discussions/3239))
- [ ] Tracing mode-1 (EVENT_PC) support — a community-contributed decoder for mode-1 trace streams has been validated against hardware capture; wiring it into the existing trace pipeline's conventions is still being discussed ([#3365](https://github.com/Xilinx/mlir-aie/discussions/3365))

## How to contribute

Want to help or suggest something?

- New to the project? Check items labeled [`good first issue`](https://github.com/Xilinx/mlir-aie/labels/good%20first%20issue).
- Already familiar with the codebase? Look for [`help wanted`](https://github.com/Xilinx/mlir-aie/labels/help%20wanted) or [`needs more 👀 (eyes)`](https://github.com/Xilinx/mlir-aie/labels/needs%20more%20%F0%9F%91%80%20(eyes)).
- Propose new ideas by [opening an issue](https://github.com/Xilinx/mlir-aie/issues/new).
- Questions and discussion go in [Discussions](https://github.com/Xilinx/mlir-aie/discussions).

Items marked below reflect their status:

- **Now** — committed and in progress
- **Next** — planned, help welcome
- **Later** — aspirational, open to input
