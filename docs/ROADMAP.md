<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->
# Roadmap

This roadmap outlines where the project is headed. It reflects current
priorities, not firm commitments — plans may change as the project evolves.

_Last updated: 2026-07-14_

## Now
Things actively being worked on.

- [ ] Dynamic runtime sequences — compile an `aie.runtime_sequence` once and run it at many problem sizes, via standalone TXN encoding and EmitC C++ generation ([#3222](https://github.com/Xilinx/mlir-aie/discussions/3222))

## Next
Planned for the near future.

- [ ] Mature native Windows support — broaden example and CI coverage, and close feature-parity gaps with Linux (builds on the [native Windows guide](buildHostWinNative.md))

## Later
Ideas we want to pursue eventually. Not yet scheduled.

- [ ] Express access patterns in the compiler — move DMA tiling from Python `taplib` into MLIR (affine maps + structured control flow) so one program lowers to both static and dynamic data movement ([#3239](https://github.com/Xilinx/mlir-aie/discussions/3239))

## How to contribute

Want to help or suggest something?

- Check items labeled [`good first issue`](https://github.com/Xilinx/mlir-aie/labels/good%20first%20issue).
- Propose new ideas by [opening an issue](https://github.com/Xilinx/mlir-aie/issues/new).
- Questions and discussion go in [Discussions](https://github.com/Xilinx/mlir-aie/discussions).

Items marked below reflect their status:

- **Now** — committed and in progress
- **Next** — planned, help welcome
- **Later** — aspirational, open to input
