<!---//===- README.md --------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# <ins>Route Vias</ins>

An `aie.flow` can name the exact stream-switch ports that its route must use at
selected tiles. These vias provide reproducible routes, steer critical flows
through chosen resources, preserve routes across compiler changes, and make
routes easier to inspect and compare. A via is a hard constraint. The router
can fail when the pinned ports cannot form a legal route.

The [design](./route_vias.py) passes one `int32` vector from a shim to a compute
tile and back. The input flow has no vias, so the router chooses its complete
route. The return flow pins its physical path:

```mlir
aie.flow(%core, DMA : 0, %shim, DMA : 0)
  via (%core : DMA : 0 -> South : 0,
     %mem : North : 0 -> South : 0,
     %shim : North : 0 -> DMA : 0)
```

The return flow therefore records the exact route. A via list can also contain
only selected hops. The router assigns each gap between those constraints.

## Generate and route the design

Generate the logical design without running it:

```bash
python3 route_vias.py --dev npu2 --emit-mlir > route_vias.mlir
```

Place the tiles, split each pinned hop into routable flow segments, and route
all segments:

```bash
aie-opt route_vias.mlir \
  --aie-place-tiles \
  --aie-split-flow-vias \
  --aie-create-pathfinder-flows \
  -o routed.mlir
```

`routed.mlir` contains the pinned return path. The router assigns the
switchbox ports for the regular input flow.

Run the pass-through on an NPU:

```bash
python3 route_vias.py --dev npu2
```

## Lift and replay a routed design

`--aie-find-flows=emit-vias=true` replaces routed switchbox configuration with
logical flows that record every switchbox hop as a via:

```bash
aie-opt routed.mlir --aie-find-flows=emit-vias=true -o lifted.mlir
```

The lifted file is suitable for inspection, comparison, and storage as a
known route. Split and route it to reproduce the switchbox configuration:

```bash
aie-opt lifted.mlir \
  --aie-split-flow-vias \
  --aie-create-pathfinder-flows \
  -o replayed.mlir
```

Delete selected entries from a lifted flow's `via` list to retain only the
constraints that matter. The router assigns each resulting gap.