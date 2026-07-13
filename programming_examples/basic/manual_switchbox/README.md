<!---//===- README.md --------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# <ins>Manual Switchbox Routing</ins>

This reference design pins **every** stream-switch hop of a
shim → compute → shim passthrough by hand, instead of letting
`--aie-create-pathfinder-flows` route it from `ObjectFifo` / `Flow`
endpoints. It targets a Ryzen™ AI NPU (npu2, single column).

Almost every design should route with `ObjectFifo` or `Flow` — automatic
routing is correct and far less error-prone. This example exists to show
the escape hatch for the rare case where an exact, router-independent
path is required (reproducing a specific hardware configuration, or
steering around a resource the router would otherwise take).

## No new IRON API

Manual routing needs no dedicated primitive. The raw `aie.switchbox`,
`aie.connect`, and `aie.shim_mux` dialect ops are already available, and
IRON already resolves any object implementing the `Resolvable` protocol
(`tiles()` + `resolve()`) at device scope when it is handed to a `Worker`
via `fn_args`. The [design](./manual_switchbox.py) defines two tiny
user-side classes — `ManualSwitchbox` and `ManualShimMux` — that emit the
switchbox configuration in their `resolve()`; the compute core's function
simply ignores them.

## Data path

```
DDR --shim DMA--> shim_mux --> shim SB --> memtile SB --> compute SB
    --> compute DMA (S2MM) --> [core copies buffer] --> compute DMA (MM2S)
    --> compute SB --> memtile SB --> shim SB --> shim_mux --> shim DMA --> DDR
```

Every switchbox / shim_mux connection in the source is exactly what the
pathfinder would have emitted for the equivalent two `aie.flow`s — this
design just writes them out.

## Two rules for manual routing

1. **Ports must match what the router would pick.** A hand-written
   connection only carries data if its ports and DMA channels line up
   with the rest of the path. The reliable way to get them right is to
   build the equivalent `Flow` / `ObjectFifo` design and dump the
   generated connections:

   ```
   aie-opt --aie-place-tiles --aie-objectFifo-stateful-transform \
           --aie-create-pathfinder-flows <design>.mlir
   ```

   then reproduce those exact ports.

2. **Manual and automatic routing don't share a hop.** A pinned
   `connect` that competes with a `flow` for the same ports fails
   routing ("Unable to find a legal routing"). Pin on a disjoint segment
   (the pathfinder augments the rest of that tile's switchbox around it),
   or pin the whole path and use no `flow`, as this example does.

See [section 2g of the programming guide](../../../programming_guide/section-2/section-2g/)
for the conceptual background.

## Usage

```bash
make -f Makefile run devicename=npu2
```

The design's `run_and_verify` path compiles, runs on the NPU, and checks
the output against a numpy reference in a single call.
