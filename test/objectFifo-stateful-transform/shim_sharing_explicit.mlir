// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: sed 's/NPUDEVICE/npu2_1col/g' %S/../npu-xrt/objectfifo_pack_shim_sharing/aie.mlir > %t.mlir
// RUN: aie-opt --aie-objectfifo-split %t.mlir | FileCheck %s --check-prefix=SPLIT
// RUN: aie-opt --aie-objectFifo-stateful-transform %t.mlir | FileCheck %s
// RUN: sed 's/ {aie.shareable_with = \[@out_[ab]\]}//' %t.mlir | aie-opt --aie-objectFifo-stateful-transform | FileCheck %s
// RUN: sed 's/ {aie.shareable_with = \[@out_b\]}//' %t.mlir | aie-opt --aie-objectFifo-stateful-transform | FileCheck %s
// RUN: sed 's/ {aie.shareable_with = \[@out_[ab]\]}//' %t.mlir | not aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" 2>&1 | FileCheck %s --check-prefix=DENIED
// RUN: sed 's/\[@out_b\]/[]/' %t.mlir | not aie-opt --aie-objectFifo-stateful-transform 2>&1 | FileCheck %s --check-prefix=DENIED

// The input-gated hardware fixture permits S2MM sharing in both directions.
// Without the permissions, the gates aie-objectfifo-verify proves let the same
// ends share; without the proofs too, or with `[]`, they do not.
// SPLIT: aie.route_endpoint @out_a__cons__dma
// SPLIT-SAME: aie.shareable_with = [@out_b__cons__dma]
// SPLIT: aie.route_endpoint @out_b__cons__dma
// SPLIT-SAME: aie.shareable_with = [@out_a__cons__dma]
// CHECK-DAG: aie.shim_dma_allocation @out_a__shim_alloc(%{{.*}}, S2MM, [[CH:[0-9]]])
// CHECK-DAG: aie.shim_dma_allocation @out_b__shim_alloc(%{{.*}}, S2MM, [[CH]])
// DENIED: error: 'aie.tile' op number of input DMA channel exceeded!
