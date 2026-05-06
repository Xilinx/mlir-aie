//===- cpp_expand_load_pdis.mlir -------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Test full ELF generation with --expand-load-pdis.
// This verifies that the single-clone approach correctly:
//   1. Creates empty device PDIs for device reset
//   2. Assigns PDI IDs in module iteration order (empties first)
//   3. Generates CDO/PDI for ALL devices (empties + originals)
//   4. Generates NPU instructions from the same expanded module
//   5. Produces a valid full ELF with all PDIs
// This matches the old Python FlowRunner behavior.

// REQUIRES: peano

// RUN: aiecc --no-xchesscc --no-xbridge --generate-full-elf --expand-load-pdis --verbose %s 2>&1 | FileCheck %s

// Verify Phase A: core compilation for original devices only
// CHECK: Processing device: main
// CHECK: No cores to compile in device main
// CHECK: Processing device: add_one
// CHECK: Compiling 1 core

// Verify Phase B: single expanded clone with empty devices
// CHECK: Expanded module PDI ID mapping:
// CHECK-DAG: @empty_{{[01]}} -> PDI ID {{[12]}}
// CHECK-DAG: @main -> PDI ID
// CHECK-DAG: @add_one -> PDI ID
// CHECK: Assigned PDI id=

// Verify artifacts generated for empty devices
// CHECK: Generating artifacts for device: empty_{{[01]}} (from expanded module)
// CHECK: Generating CDO artifacts for device: empty_

// Verify artifacts generated for original devices from expanded module
// CHECK: Generating artifacts for device: main (from expanded module)
// CHECK: Generating artifacts for device: add_one (from expanded module)

// Verify full ELF generation with all devices
// CHECK: Generating full ELF with {{[3-5]}} device
// CHECK: Generated config.json
// CHECK: Generated full ELF
// CHECK: Compilation completed successfully

module {

    aie.device(npu2) @main {
        aie.runtime_sequence @sequence(%arg : memref<16xi32>) {
            aiex.configure @add_one {
                aiex.run @add_one_seq (%arg) : (memref<16xi32>)
            }
        }
    }

    aie.device(npu2) @add_one {
        %t00 = aie.tile(0, 0)
        %t02 = aie.tile(0, 2)

        aie.objectfifo @of_in (%t00, {%t02}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
        aie.objectfifo @of_out(%t02, {%t00}, 1 : i32) : !aie.objectfifo<memref<16xi32>>

        aie.core(%t02) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c16 = arith.constant 16 : index
            %c1_i32 = arith.constant 1 : i32
            %c_intmax = arith.constant 0xFFFFFE : index

            scf.for %niter = %c0 to %c_intmax step %c1 {
                %subview_in  = aie.objectfifo.acquire @of_in (Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
                %subview_out = aie.objectfifo.acquire @of_out(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
                %elem_in  = aie.objectfifo.subview.access %subview_in [0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                %elem_out = aie.objectfifo.subview.access %subview_out[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                scf.for %i = %c0 to %c16 step %c1 {
                    %0 = memref.load %elem_in[%i] : memref<16xi32>
                    %1 = arith.addi %0, %c1_i32 : i32
                    memref.store %1, %elem_out[%i] : memref<16xi32>
                }
                aie.objectfifo.release @of_in (Consume, 1)
                aie.objectfifo.release @of_out(Produce, 1)
            }
            aie.end
        }

        aie.runtime_sequence @add_one_seq(%a : memref<16xi32>) {
            %t_in = aiex.dma_configure_task_for @of_in {
                aie.dma_bd(%a : memref<16xi32>, 0, 16)
                aie.end
            }
            %t_out = aiex.dma_configure_task_for @of_out {
                aie.dma_bd(%a : memref<16xi32>, 0, 16)
                aie.end
            } {issue_token = true}
            aiex.dma_start_task(%t_in)
            aiex.dma_start_task(%t_out)
            aiex.dma_await_task(%t_out)
        }
    }
}
