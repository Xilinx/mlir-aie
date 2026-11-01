// RUN: aie-opt --pass-pipeline='builtin.module(aie.device(aie-place-tiles,aie-objectFifo-stateful-transform,aie-create-pathfinder-flows,aie-assign-lock-ids,aie-assign-bd-ids,aie-assign-buffer-addresses))' %S/Inputs/light.mlir -o %t.image.mlir
// RUN: aie-opt --pass-pipeline='builtin.module(aie.device(aie-place-tiles,aie-objectFifo-stateful-transform,aie-create-pathfinder-flows,aie-assign-lock-ids,aie-assign-bd-ids,aie-assign-buffer-addresses))' %S/Inputs/heavy.mlir -o %t.stream.mlir
// RUN: not aie-opt --aie-check-against-image=image=%t.image.mlir %t.stream.mlir -o /dev/null 2>&1 | FileCheck %s
// RUN: aie-opt --aie-check-against-image=image=%t.image.mlir %t.image.mlir -o /dev/null

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Inputs/light.mlir and Inputs/heavy.mlir share a static design, but heavy's
// runtime tasks hold more BDs than one shim has, so the placer puts in_b's
// shim end on another shim. An instruction stream built from heavy for the
// image built from light would program a shim channel the image never set
// up; the check names it and says to pin it. The image checked against
// itself passes.

// CHECK:      error: this instruction stream would program resources its image ({{.*}}.image.mlir) does not have; {{[0-9]+}} differ:
// CHECK-NEXT:   in_b's shim end: image (0, 0) MM2S 1, this stream (1, 0) MM2S 0; pin it to keep it stable across shapes
// CHECK:        shim mux (1, 0): image connects none, this stream DMA:0->North:3
// CHECK:        the routes follow the placement and channels above
// CHECK-NOT:  in_a's shim end
