<!---//===- README.md --------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Route Vias

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

Use `aiecc` to place and route the design, and retain the physical MLIR:

```bash
aiecc -j2 \
  --get=input_physical.mlir \
  --output-dir=build \
  route_vias.mlir
```

`build/input_physical.mlir` contains the pinned return path and the route that
the router assigned to the regular input flow.

## Lift, edit, and rebuild the route

Recover the routed switchbox configuration as logical flows. Each recovered
flow records its switchbox hops as vias:

```bash
aie-opt build/input_physical.mlir \
  --aie-find-flows=emit-vias=true \
  -o lifted.mlir
```

Edit `lifted.mlir` to remove the constraints that the router may reassign. For
example, remove the compute-tile hop from the return flow:

```mlir
aie.flow(%core, DMA : 0, %shim, DMA : 0)
  via (%mem : North : 0 -> South : 0,
       %shim : North : 0 -> South : 2)
```

Save the result as `edited.mlir`, then compile it into runnable artifacts:

```bash
aiecc -j2 \
  --get-xclbin \
  --get-npu-insts \
  --output-dir=edited-build \
  edited.mlir
```

This command produces `edited-build/aie.xclbin` and
`edited-build/insts_main_sequence.bin`. During compilation, the router assigns
the gap created by the deleted via and retains the remaining constraints.

Run the rerouted design on an NPU2 and verify that its output matches its
input:

```bash
python3 route_vias.py --dev npu2 \
  --run-xclbin edited-build/aie.xclbin \
  --run-insts edited-build/insts_main_sequence.bin
```

These options load the supplied artifacts directly. They do not compile the
`route_vias` function again.
