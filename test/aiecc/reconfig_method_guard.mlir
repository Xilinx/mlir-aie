//===- reconfig_method_guard.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The --reconfig-method selector validates its arguments up front, before any
// flow runs:
//   * an unrecognized method value is rejected;
//   * the fold requires --get-full-elf;
//   * write32 (out-of-band direct-write delivery) is mutually exclusive with
//     --load-pdi-to-ctrl-pkt (in-band control-packet delivery) -- contradictory
//     transports for the same reconfiguration.

// RUN: not aiecc --get-full-elf --reconfig-method=bogus %s 2>&1 | FileCheck %s --check-prefix=BADMETHOD
// BADMETHOD: --reconfig-method must be loadpdi|write32|ctrlpkt

// RUN: not aiecc --reconfig-method=ctrlpkt %s 2>&1 | FileCheck %s --check-prefix=NOFULLELF
// NOFULLELF: --reconfig-method requires --get-full-elf

// RUN: not aiecc --get-full-elf --reconfig-method=write32 --load-pdi-to-ctrl-pkt %s 2>&1 | FileCheck %s

// CHECK: --reconfig-method=write32 is mutually exclusive with {{.*}}--load-pdi-to-ctrl-pkt

// A two-config idiomatic module (host device @main configuring config device
// @add_one), reused from test/aiecc/cpp_expand_load_pdis.mlir, so the guard
// sees a real design and fires before any lowering.
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
                %elem_in = aie.objectfifo.acquire @of_in (Consume, 1) : memref<16xi32>
                %elem_out = aie.objectfifo.acquire @of_out(Produce, 1) : memref<16xi32>
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
                aie.dma_bd(%a : memref<16xi32> offset = 0 len = 16)
                aie.end
            }
            %t_out = aiex.dma_configure_task_for @of_out {
                aie.dma_bd(%a : memref<16xi32> offset = 0 len = 16)
                aie.end
            } {issue_token = true}
            aiex.dma_start_task(%t_in)
            aiex.dma_start_task(%t_out)
            aiex.dma_await_task(%t_out)
        }
    }
}
