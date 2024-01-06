# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

# RUN: HOST_RUNTIME_LIB_DIR=%host_runtime_lib% WORKDIR=%T XRT_DIR=%XRT_DIR %PYTHON %s | FileCheck %s
# REQUIRES: xrt_python_bindings
# REQUIRES: ryzen_ai

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from aie.extras.dialects.ext import memref, arith, func
from aie.extras.runtime.passes import run_pipeline
from aie.extras.util import bb

import aie.extras.types as T
from aie.compiler.aiecc.main import (
    generate_cores_list,
    emit_partition,
    emit_design_bif,
    emit_design_kernel_json,
    mem_topology,
    chesshack,
)
from aie.dialects import aie
from aie.dialects.aie import (
    AIEDevice,
    DMAChannelDir,
    LockAction,
    WireBundle,
    device,
    generate_bcf,
    generate_cdo,
    ipu_instgen,
    mem,
    memtile_dma,
    tile,
    translate_mlir_to_llvmir,
    aie_llvm_link,
)
from aie.dialects.aiex import ipu_sync, ipu_dma_memcpy_nd
from aie.dialects.scf import for_
from aie.dialects.scf import yield_
from aie.xrt import XCLBin
from util import construct_and_print_module

range_ = for_

DMA = WireBundle.DMA
S2MM = DMAChannelDir.S2MM
MM2S = DMAChannelDir.MM2S
Acquire = LockAction.Acquire
AcquireGreaterEqual = LockAction.AcquireGreaterEqual
Release = LockAction.Release


def extract_input_files(core_bcf):
    return re.findall(r"^_include _file (.*)", core_bcf, re.MULTILINE)


# CHECK-LABEL: add_one_using_dma
@construct_and_print_module
def add_one_using_dma(module):
    @device(AIEDevice.ipu)
    def ipu():
        memref.global_("objFifo_in0", T.memref(16, T.i32()), sym_visibility="public")
        memref.global_(
            "objFifo_in0_cons", T.memref(16, T.i32()), sym_visibility="public"
        )
        memref.global_("objFifo_in1", T.memref(8, T.i32()), sym_visibility="public")
        memref.global_(
            "objFifo_in1_cons", T.memref(8, T.i32()), sym_visibility="public"
        )
        memref.global_("objFifo_out0", T.memref(16, T.i32()), sym_visibility="public")
        memref.global_(
            "objFifo_out0_cons", T.memref(16, T.i32()), sym_visibility="public"
        )
        memref.global_("objFifo_out1", T.memref(8, T.i32()), sym_visibility="public")
        memref.global_(
            "objFifo_out1_cons", T.memref(8, T.i32()), sym_visibility="public"
        )

        tile_0_0 = tile(0, 0)
        tile_0_1 = tile(0, 1)
        tile_0_2 = tile(0, 2)

        objFifo_in0_cons_buff_0 = aie.buffer(
            T.memref(16, T.i32()), tile_0_1, sym_name="objFifo_in0_cons_buff_0"
        )
        objFifo_in0_cons_buff_1 = aie.buffer(
            T.memref(16, T.i32()), tile_0_1, sym_name="objFifo_in0_cons_buff_1"
        )
        objFifo_out0_buff_0 = aie.buffer(
            T.memref(16, T.i32()), tile_0_1, sym_name="objFifo_out0_buff_0"
        )
        objFifo_out0_buff_1 = aie.buffer(
            T.memref(16, T.i32()), tile_0_1, sym_name="objFifo_out0_buff_1"
        )

        objFifo_in1_cons_buff_0 = aie.buffer(
            T.memref(8, T.i32()), tile_0_2, sym_name="objFifo_in1_cons_buff_0"
        )
        objFifo_in1_cons_buff_1 = aie.buffer(
            T.memref(8, T.i32()), tile_0_2, sym_name="objFifo_in1_cons_buff_1"
        )
        objFifo_out1_buff_0 = aie.buffer(
            T.memref(8, T.i32()), tile_0_2, sym_name="objFifo_out1_buff_0"
        )
        objFifo_out1_buff_1 = aie.buffer(
            T.memref(8, T.i32()), tile_0_2, sym_name="objFifo_out1_buff_1"
        )

        objFifo_in0_prod_lock = aie.lock(
            tile_0_0, lock_id=0, init=0, sym_name="objFifo_in0_prod_lock"
        )
        objFifo_in0_cons_lock = aie.lock(
            tile_0_0, lock_id=1, init=0, sym_name="objFifo_in0_cons_lock"
        )
        objFifo_out0_cons_prod_lock = aie.lock(
            tile_0_0, lock_id=2, init=0, sym_name="objFifo_out0_cons_prod_lock"
        )
        objFifo_out0_cons_cons_lock = aie.lock(
            tile_0_0, lock_id=3, init=0, sym_name="objFifo_out0_cons_cons_lock"
        )

        objFifo_in0_cons_prod_lock = aie.lock(
            tile_0_1, lock_id=0, init=2, sym_name="objFifo_in0_cons_prod_lock"
        )
        objFifo_in0_cons_cons_lock = aie.lock(
            tile_0_1, lock_id=1, init=0, sym_name="objFifo_in0_cons_cons_lock"
        )
        objFifo_out0_prod_lock = aie.lock(
            tile_0_1, lock_id=2, init=2, sym_name="objFifo_out0_prod_lock"
        )
        objFifo_out0_cons_lock = aie.lock(
            tile_0_1, lock_id=3, init=0, sym_name="objFifo_out0_cons_lock"
        )

        objFifo_in1_cons_prod_lock = aie.lock(
            tile_0_2, lock_id=0, init=2, sym_name="objFifo_in1_cons_prod_lock"
        )
        objFifo_in1_cons_cons_lock = aie.lock(
            tile_0_2, lock_id=1, init=0, sym_name="objFifo_in1_cons_cons_lock"
        )
        objFifo_out1_prod_lock = aie.lock(
            tile_0_2, lock_id=2, init=2, sym_name="objFifo_out1_prod_lock"
        )
        objFifo_out1_cons_lock = aie.lock(
            tile_0_2, lock_id=3, init=0, sym_name="objFifo_out1_cons_lock"
        )

        aie.flow(tile_0_0, DMA, 0, tile_0_1, DMA, 0)
        aie.flow(tile_0_1, DMA, 0, tile_0_2, DMA, 0)
        aie.flow(tile_0_1, DMA, 1, tile_0_0, DMA, 0)
        aie.flow(tile_0_2, DMA, 0, tile_0_1, DMA, 1)

        @aie.core(tile_0_2)
        def core():
            c1_i32 = arith.constant(1)
            for i in range_(0, 8, 2):
                # TODO(max): fix the ordering in the asm to match the ordering in the `ins`
                aie.use_lock(objFifo_in1_cons_cons_lock, 1, AcquireGreaterEqual)
                aie.use_lock(objFifo_out1_prod_lock, 1, AcquireGreaterEqual)

                for arg1 in range_(0, 8, 1):
                    v0 = memref.load(objFifo_in1_cons_buff_0, [arg1])
                    v1 = arith.addi(v0, c1_i32)
                    memref.store(v1, objFifo_out1_buff_0, [arg1])
                    yield_([])

                aie.use_lock(objFifo_in1_cons_prod_lock, 1, Release)
                aie.use_lock(objFifo_out1_cons_lock, 1, Release)

                aie.use_lock(objFifo_in1_cons_cons_lock, 1, AcquireGreaterEqual)
                aie.use_lock(objFifo_out1_prod_lock, 1, AcquireGreaterEqual)

                for arg1 in range_(0, 8, 1):
                    v0 = memref.load(objFifo_in1_cons_buff_1, [arg1])
                    v1 = arith.addi(v0, c1_i32)
                    memref.store(v1, objFifo_out1_buff_1, [arg1])
                    yield_([])

                aie.use_lock(objFifo_in1_cons_prod_lock, 1, Release)
                aie.use_lock(objFifo_out1_cons_lock, 1, Release)

                yield_([])

        aie.shim_dma_allocation("objFifo_in0", MM2S, 0, 0)

        @func.func(emit=True)
        def bobsyouruncle(
            arg0: T.memref(64, T.i32()),
            arg1: T.memref(32, T.i32()),
            arg2: T.memref(64, T.i32()),
        ):
            ipu_dma_memcpy_nd(
                "objFifo_in0",
                0,
                arg0,
                [0, 0, 0, 0],
                [1, 1, 1, 64],
                [0, 0, 0],
            )

            ipu_dma_memcpy_nd(
                "objFifo_out0",
                1,
                arg2,
                [0, 0, 0, 0],
                [1, 1, 1, 64],
                [0, 0, 0],
            )
            ipu_sync(channel=0, column=0, column_num=1, direction=0, row=0, row_num=1)

        @memtile_dma(tile_0_1)
        def memtile_dma_0_1():
            bb1, bb3 = aie.dma_start(S2MM, 0)
            with bb(bb1):  # 2 preds: bb0, bb2
                aie.use_lock(objFifo_in0_cons_prod_lock, 1, AcquireGreaterEqual)
                aie.dma_bd(objFifo_in0_cons_buff_0, 0, 16)
                aie.use_lock(objFifo_in0_cons_cons_lock, 1, Release)
                bb2 = aie.next_bd()
            with bb(bb2):  # pred: bb1
                aie.use_lock(objFifo_in0_cons_prod_lock, 1, AcquireGreaterEqual)
                aie.dma_bd(objFifo_in0_cons_buff_1, 0, 16)
                aie.use_lock(objFifo_in0_cons_cons_lock, 1, Release)
                aie.next_bd(bb1)
            with bb(bb3):  # pred: bb0
                bb4, bb6 = aie.dma_start(MM2S, 0)
            with bb(bb4):  # 2 preds: bb3, bb5
                aie.use_lock(objFifo_in0_cons_cons_lock, 1, AcquireGreaterEqual)
                aie.dma_bd(objFifo_in0_cons_buff_0, 0, 16)
                aie.use_lock(objFifo_in0_cons_prod_lock, 1, Release)
                bb5 = aie.next_bd()
            with bb(bb5):  # pred: bb4
                aie.use_lock(objFifo_in0_cons_cons_lock, 1, AcquireGreaterEqual)
                aie.dma_bd(objFifo_in0_cons_buff_1, 0, 16)
                aie.use_lock(objFifo_in0_cons_prod_lock, 1, Release)
                aie.next_bd(bb4)
            with bb(bb6):  # pred: bb3
                bb7, bb9 = aie.dma_start(MM2S, 1)
            with bb(bb7):  # 2 preds: bb6, bb8
                aie.use_lock(objFifo_out0_cons_lock, 1, AcquireGreaterEqual)
                aie.dma_bd(objFifo_out0_buff_0, 0, 16)
                aie.use_lock(objFifo_out0_prod_lock, 1, Release)
                bb8 = aie.next_bd()
            with bb(bb8):  # pred: bb7
                aie.use_lock(objFifo_out0_cons_lock, 1, AcquireGreaterEqual)
                aie.dma_bd(objFifo_out0_buff_1, 0, 16)
                aie.use_lock(objFifo_out0_prod_lock, 1, Release)
                aie.next_bd(bb7)
            with bb(bb9):  # pred: bb6
                bb10, bb12 = aie.dma_start(S2MM, 1)
            with bb(bb10):  # 2 preds: bb9, bb11
                aie.use_lock(objFifo_out0_prod_lock, 1, AcquireGreaterEqual)
                aie.dma_bd(objFifo_out0_buff_0, 0, 16)
                aie.use_lock(objFifo_out0_cons_lock, 1, Release)
                bb11 = aie.next_bd()
            with bb(bb11):  # pred: bb10
                aie.use_lock(objFifo_out0_prod_lock, 1, AcquireGreaterEqual)
                aie.dma_bd(objFifo_out0_buff_1, 0, 16)
                aie.use_lock(objFifo_out0_cons_lock, 1, Release)
                aie.next_bd(bb10)
            with bb(bb12):  # pred: bb9
                aie.end()

        aie.shim_dma_allocation("objFifo_out0", S2MM, 0, 0)

        @mem(tile_0_2)
        def mem_0_2():
            bb1, bb3 = aie.dma_start(S2MM, 0)
            with bb(bb1):  # 2 preds: bb0, bb2
                aie.use_lock(objFifo_in1_cons_prod_lock, 1, AcquireGreaterEqual)
                aie.dma_bd(objFifo_in1_cons_buff_0, 0, 8)
                aie.use_lock(objFifo_in1_cons_cons_lock, 1, Release)
                bb2 = aie.next_bd()
            with bb(bb2):  # pred: bb1
                aie.use_lock(objFifo_in1_cons_prod_lock, 1, AcquireGreaterEqual)
                aie.dma_bd(objFifo_in1_cons_buff_1, 0, 8)
                aie.use_lock(objFifo_in1_cons_cons_lock, 1, Release)
                aie.next_bd(bb1)
            with bb(bb3):  # pred: bb0
                bb4, bb6 = aie.dma_start(MM2S, 0)
            with bb(bb4):  # 2 preds: bb3, bb5
                aie.use_lock(objFifo_out1_cons_lock, 1, AcquireGreaterEqual)
                aie.dma_bd(objFifo_out1_buff_0, 0, 8)
                aie.use_lock(objFifo_out1_prod_lock, 1, Release)
                bb5 = aie.next_bd()
            with bb(bb5):  # pred: bb4
                aie.use_lock(objFifo_out1_cons_lock, 1, AcquireGreaterEqual)
                aie.dma_bd(objFifo_out1_buff_1, 0, 8)
                aie.use_lock(objFifo_out1_prod_lock, 1, Release)
                aie.next_bd(bb4)
            with bb(bb6):  # pred: bb3
                aie.end()

    print(module)
    pass_pipeline = ",".join(
        [
            "lower-affine",
            "aie-canonicalize-device",
            "aie.device(" + "aie-assign-lock-ids",
            "aie-register-objectFifos",
            "aie-objectFifo-stateful-transform",
            "aie-lower-broadcast-packet",
            "aie-create-packet-flows",
            "aie-lower-multicast",
            "aie-assign-buffer-addresses)",
            "convert-scf-to-cf",
        ]
    )
    # CHECK: %objFifo_in0_cons_buff_0 = aie.buffer(%tile_0_1) {address = 0 : i32, sym_name = "objFifo_in0_cons_buff_0"} : memref<16xi32>
    # CHECK: %objFifo_in0_cons_buff_1 = aie.buffer(%tile_0_1) {address = 64 : i32, sym_name = "objFifo_in0_cons_buff_1"} : memref<16xi32>
    # CHECK: %objFifo_out0_buff_0 = aie.buffer(%tile_0_1) {address = 128 : i32, sym_name = "objFifo_out0_buff_0"} : memref<16xi32>
    # CHECK: %objFifo_out0_buff_1 = aie.buffer(%tile_0_1) {address = 192 : i32, sym_name = "objFifo_out0_buff_1"} : memref<16xi32>
    # CHECK: %objFifo_in1_cons_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, sym_name = "objFifo_in1_cons_buff_0"} : memref<8xi32>
    # CHECK: %objFifo_in1_cons_buff_1 = aie.buffer(%tile_0_2) {address = 1056 : i32, sym_name = "objFifo_in1_cons_buff_1"} : memref<8xi32>
    # CHECK: %objFifo_out1_buff_0 = aie.buffer(%tile_0_2) {address = 1088 : i32, sym_name = "objFifo_out1_buff_0"} : memref<8xi32>
    # CHECK: %objFifo_out1_buff_1 = aie.buffer(%tile_0_2) {address = 1120 : i32, sym_name = "objFifo_out1_buff_1"} : memref<8xi32>
    input_with_addresses = run_pipeline(module, "builtin.module(" + pass_pipeline + ")")
    print(input_with_addresses)

    generated_ipu_insts = run_pipeline(
        input_with_addresses, "builtin.module(aie.device(aie-dma-to-ipu))"
    )
    # CHECK: aiex.ipu.writebd_shimtile {bd_id = 0 : i32, buffer_length = 64 : i32, buffer_offset = 0 : i32, column = 0 : i32, column_num = 1 : i32, d0_stepsize = 0 : i32, d0_wrap = 0 : i32, d1_stepsize = 0 : i32, d1_wrap = 0 : i32, d2_stepsize = 0 : i32, ddr_id = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_stepsize = 0 : i32, iteration_wrap = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
    # CHECK: aiex.ipu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 0 : ui32}
    # CHECK: aiex.ipu.writebd_shimtile {bd_id = 1 : i32, buffer_length = 64 : i32, buffer_offset = 0 : i32, column = 0 : i32, column_num = 1 : i32, d0_stepsize = 0 : i32, d0_wrap = 0 : i32, d1_stepsize = 0 : i32, d1_wrap = 0 : i32, d2_stepsize = 0 : i32, ddr_id = 2 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_stepsize = 0 : i32, iteration_wrap = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
    # CHECK: aiex.ipu.write32 {address = 119300 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483649 : ui32}
    # CHECK: aiex.ipu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
    print(generated_ipu_insts)

    cores = generate_cores_list(str(input_with_addresses))
    print(cores)

    aie_opt_lower_to_llvm_passes = [
        "canonicalize",
        "cse",
        "convert-vector-to-llvm",
        "expand-strided-metadata",
        "lower-affine",
        "convert-math-to-llvm",
        "convert-arith-to-llvm",
        "finalize-memref-to-llvm",
        "convert-func-to-llvm{ use-bare-ptr-memref-call-conv }",
        "convert-cf-to-llvm",
        "canonicalize",
        "cse",
    ]

    pass_pipeline = ", ".join(
        [
            "aie.device(aie-localize-locks",
            "aie-normalize-address-spaces)",
            "aie-standard-lowering",
            "aiex-standard-lowering",
            *aie_opt_lower_to_llvm_passes,
        ]
    )
    input_opt_with_addresses = run_pipeline(
        input_with_addresses, "builtin.module(" + pass_pipeline + ")"
    )
    # CHECK: module attributes {llvm.target_triple = "aie2"} {
    # CHECK:   llvm.mlir.global external @objFifo_out1_buff_1() {addr_space = 0 : i32} : !llvm.array<8 x i32>
    # CHECK:   llvm.mlir.global external @objFifo_out1_buff_0() {addr_space = 0 : i32} : !llvm.array<8 x i32>
    # CHECK:   llvm.mlir.global external @objFifo_in1_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<8 x i32>
    # CHECK:   llvm.mlir.global external @objFifo_in1_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<8 x i32>
    # CHECK:   llvm.mlir.global external @objFifo_out0_buff_1() {addr_space = 0 : i32} : !llvm.array<16 x i32>
    # CHECK:   llvm.mlir.global external @objFifo_out0_buff_0() {addr_space = 0 : i32} : !llvm.array<16 x i32>
    # CHECK:   llvm.mlir.global external @objFifo_in0_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<16 x i32>
    # CHECK:   llvm.mlir.global external @objFifo_in0_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<16 x i32>
    # CHECK:   llvm.func @debug_i32(i32) attributes {sym_visibility = "private"}
    # CHECK:   llvm.func @llvm.aie2.put.ms(i32, i32) attributes {sym_visibility = "private"}
    # CHECK:   llvm.func @llvm.aie2.get.ss() -> !llvm.struct<(i32, i32)> attributes {sym_visibility = "private"}
    # CHECK:   llvm.func @llvm.aie2.mcd.write.vec(vector<16xi32>, i32) attributes {sym_visibility = "private"}
    # CHECK:   llvm.func @llvm.aie2.scd.read.vec(i32) -> vector<16xi32> attributes {sym_visibility = "private"}
    # CHECK:   llvm.func @llvm.aie2.acquire(i32, i32) attributes {sym_visibility = "private"}
    # CHECK:   llvm.func @llvm.aie2.release(i32, i32) attributes {sym_visibility = "private"}
    # CHECK:   llvm.mlir.global external @objFifo_in0() {addr_space = 0 : i32} : !llvm.array<16 x i32>
    # CHECK:   llvm.mlir.global external @objFifo_in0_cons() {addr_space = 0 : i32} : !llvm.array<16 x i32>
    # CHECK:   llvm.mlir.global external @objFifo_in1() {addr_space = 0 : i32} : !llvm.array<8 x i32>
    # CHECK:   llvm.mlir.global external @objFifo_in1_cons() {addr_space = 0 : i32} : !llvm.array<8 x i32>
    # CHECK:   llvm.mlir.global external @objFifo_out0() {addr_space = 0 : i32} : !llvm.array<16 x i32>
    # CHECK:   llvm.mlir.global external @objFifo_out0_cons() {addr_space = 0 : i32} : !llvm.array<16 x i32>
    # CHECK:   llvm.mlir.global external @objFifo_out1() {addr_space = 0 : i32} : !llvm.array<8 x i32>
    # CHECK:   llvm.mlir.global external @objFifo_out1_cons() {addr_space = 0 : i32} : !llvm.array<8 x i32>
    # CHECK:   llvm.func @bobsyouruncle(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
    # CHECK:     llvm.return
    # CHECK:   }
    # CHECK:   llvm.func @core_0_2() {
    # CHECK:     %0 = llvm.mlir.constant(31 : index) : i64
    # CHECK:     %1 = llvm.mlir.constant(2 : index) : i64
    # CHECK:     %2 = llvm.mlir.constant(8 : index) : i64
    # CHECK:     %3 = llvm.mlir.constant(51 : i32) : i32
    # CHECK:     %4 = llvm.mlir.constant(48 : i32) : i32
    # CHECK:     %5 = llvm.mlir.constant(50 : i32) : i32
    # CHECK:     %6 = llvm.mlir.constant(49 : i32) : i32
    # CHECK:     %7 = llvm.mlir.constant(1 : index) : i64
    # CHECK:     %8 = llvm.mlir.constant(-1 : i32) : i32
    # CHECK:     %9 = llvm.mlir.constant(1 : i32) : i32
    # CHECK:     %10 = llvm.mlir.constant(0 : index) : i64
    # CHECK:     llvm.br ^bb1(%10 : i64)
    # CHECK:   ^bb1(%11: i64):  // 2 preds: ^bb0, ^bb8
    # CHECK:     %12 = llvm.icmp "slt" %11, %2 : i64
    # CHECK:     llvm.cond_br %12, ^bb2, ^bb9
    # CHECK:   ^bb2:  // pred: ^bb1
    # CHECK:     llvm.call @llvm.aie2.acquire(%6, %8) : (i32, i32) -> ()
    # CHECK:     llvm.call @llvm.aie2.acquire(%5, %8) : (i32, i32) -> ()
    # CHECK:     llvm.br ^bb3(%10 : i64)
    # CHECK:   ^bb3(%13: i64):  // 2 preds: ^bb2, ^bb4
    # CHECK:     %14 = llvm.icmp "slt" %13, %2 : i64
    # CHECK:     llvm.cond_br %14, ^bb4, ^bb5
    # CHECK:   ^bb4:  // pred: ^bb3
    # CHECK:     %15 = llvm.mlir.addressof @objFifo_in1_cons_buff_0 : !llvm.ptr
    # CHECK:     %16 = llvm.getelementptr %15[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x i32>
    # CHECK:     %17 = llvm.ptrtoint %16 : !llvm.ptr to i64
    # CHECK:     %18 = llvm.and %17, %0  : i64
    # CHECK:     %19 = llvm.icmp "eq" %18, %10 : i64
    # CHECK:     "llvm.intr.assume"(%19) : (i1) -> ()
    # CHECK:     %20 = llvm.getelementptr %16[%13] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    # CHECK:     %21 = llvm.load %20 : !llvm.ptr -> i32
    # CHECK:     %22 = llvm.add %21, %9  : i32
    # CHECK:     %23 = llvm.mlir.addressof @objFifo_out1_buff_0 : !llvm.ptr
    # CHECK:     %24 = llvm.getelementptr %23[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x i32>
    # CHECK:     %25 = llvm.ptrtoint %24 : !llvm.ptr to i64
    # CHECK:     %26 = llvm.and %25, %0  : i64
    # CHECK:     %27 = llvm.icmp "eq" %26, %10 : i64
    # CHECK:     "llvm.intr.assume"(%27) : (i1) -> ()
    # CHECK:     %28 = llvm.getelementptr %24[%13] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    # CHECK:     llvm.store %22, %28 : i32, !llvm.ptr
    # CHECK:     %29 = llvm.add %13, %7  : i64
    # CHECK:     llvm.br ^bb3(%29 : i64)
    # CHECK:   ^bb5:  // pred: ^bb3
    # CHECK:     llvm.call @llvm.aie2.release(%4, %9) : (i32, i32) -> ()
    # CHECK:     llvm.call @llvm.aie2.release(%3, %9) : (i32, i32) -> ()
    # CHECK:     llvm.call @llvm.aie2.acquire(%6, %8) : (i32, i32) -> ()
    # CHECK:     llvm.call @llvm.aie2.acquire(%5, %8) : (i32, i32) -> ()
    # CHECK:     llvm.br ^bb6(%10 : i64)
    # CHECK:   ^bb6(%30: i64):  // 2 preds: ^bb5, ^bb7
    # CHECK:     %31 = llvm.icmp "slt" %30, %2 : i64
    # CHECK:     llvm.cond_br %31, ^bb7, ^bb8
    # CHECK:   ^bb7:  // pred: ^bb6
    # CHECK:     %32 = llvm.mlir.addressof @objFifo_in1_cons_buff_1 : !llvm.ptr
    # CHECK:     %33 = llvm.getelementptr %32[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x i32>
    # CHECK:     %34 = llvm.ptrtoint %33 : !llvm.ptr to i64
    # CHECK:     %35 = llvm.and %34, %0  : i64
    # CHECK:     %36 = llvm.icmp "eq" %35, %10 : i64
    # CHECK:     "llvm.intr.assume"(%36) : (i1) -> ()
    # CHECK:     %37 = llvm.getelementptr %33[%30] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    # CHECK:     %38 = llvm.load %37 : !llvm.ptr -> i32
    # CHECK:     %39 = llvm.add %38, %9  : i32
    # CHECK:     %40 = llvm.mlir.addressof @objFifo_out1_buff_1 : !llvm.ptr
    # CHECK:     %41 = llvm.getelementptr %40[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x i32>
    # CHECK:     %42 = llvm.ptrtoint %41 : !llvm.ptr to i64
    # CHECK:     %43 = llvm.and %42, %0  : i64
    # CHECK:     %44 = llvm.icmp "eq" %43, %10 : i64
    # CHECK:     "llvm.intr.assume"(%44) : (i1) -> ()
    # CHECK:     %45 = llvm.getelementptr %41[%30] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    # CHECK:     llvm.store %39, %45 : i32, !llvm.ptr
    # CHECK:     %46 = llvm.add %30, %7  : i64
    # CHECK:     llvm.br ^bb6(%46 : i64)
    # CHECK:   ^bb8:  // pred: ^bb6
    # CHECK:     llvm.call @llvm.aie2.release(%4, %9) : (i32, i32) -> ()
    # CHECK:     llvm.call @llvm.aie2.release(%3, %9) : (i32, i32) -> ()
    # CHECK:     %47 = llvm.add %11, %1  : i64
    # CHECK:     llvm.br ^bb1(%47 : i64)
    # CHECK:   ^bb9:  // pred: ^bb1
    # CHECK:     llvm.return
    # CHECK:   }
    # CHECK: }
    print(input_opt_with_addresses)

    input_ll = translate_mlir_to_llvmir(input_opt_with_addresses.operation)

    with open(Path(__file__).parent / "chess_intrinsic_wrapper.ll") as f:
        chess_intrinsic_wrapper = f.read()
        input_llchesslinked_ll = chesshack(
            aie_llvm_link([input_ll, chess_intrinsic_wrapper])
        )

    # CHECK: ; ModuleID = 'aie-llvm-link'
    # CHECK: source_filename = "aie-llvm-link"
    # CHECK: target triple = "aie2"
    #
    # CHECK: %struct.ipd.custom_type.uint2_t.uint2_t = type { i2 }
    #
    # CHECK: @objFifo_out1_buff_1 = external global [8 x i32]
    # CHECK: @objFifo_out1_buff_0 = external global [8 x i32]
    # CHECK: @objFifo_in1_cons_buff_1 = external global [8 x i32]
    # CHECK: @objFifo_in1_cons_buff_0 = external global [8 x i32]
    #
    # CHECK: define void @bobsyouruncle(ptr %0, ptr %1, ptr %2) {
    # CHECK:   ret void
    # CHECK: }
    #
    # CHECK: define void @core_0_2() {
    # CHECK:   br label %1
    #
    # CHECK: 1:                                                ; preds = %32, %0
    # CHECK:   %2 = phi i64 [ %33, %32 ], [ 0, %0 ]
    # CHECK:   %3 = icmp slt i64 %2, 8
    # CHECK:   br i1 %3, label %4, label %34
    #
    # CHECK: 4:                                                ; preds = %1
    # CHECK:   call void @llvm.aie2.acquire(i32 49, i32 -1)
    # CHECK:   call void @llvm.aie2.acquire(i32 50, i32 -1)
    # CHECK:   br label %5
    #
    # CHECK: 5:                                                ; preds = %8, %4
    # CHECK:   %6 = phi i64 [ %17, %8 ], [ 0, %4 ]
    # CHECK:   %7 = icmp slt i64 %6, 8
    # CHECK:   br i1 %7, label %8, label %18
    #
    # CHECK: 8:                                                ; preds = %5
    # CHECK:   %9 = and i64 ptrtoint (ptr @objFifo_in1_cons_buff_0 to i64), 31
    # CHECK:   %10 = icmp eq i64 %9, 0
    # CHECK:   call void @llvm.assume(i1 %10)
    # CHECK:   %11 = getelementptr i32, ptr @objFifo_in1_cons_buff_0, i64 %6
    # CHECK:   %12 = load i32, ptr %11, align 4
    # CHECK:   %13 = add i32 %12, 1
    # CHECK:   %14 = and i64 ptrtoint (ptr @objFifo_out1_buff_0 to i64), 31
    # CHECK:   %15 = icmp eq i64 %14, 0
    # CHECK:   call void @llvm.assume(i1 %15)
    # CHECK:   %16 = getelementptr i32, ptr @objFifo_out1_buff_0, i64 %6
    # CHECK:   store i32 %13, ptr %16, align 4
    # CHECK:   %17 = add i64 %6, 1
    # CHECK:   br label %5
    #
    # CHECK: 18:                                               ; preds = %5
    # CHECK:   call void @llvm.aie2.release(i32 48, i32 1)
    # CHECK:   call void @llvm.aie2.release(i32 51, i32 1)
    # CHECK:   call void @llvm.aie2.acquire(i32 49, i32 -1)
    # CHECK:   call void @llvm.aie2.acquire(i32 50, i32 -1)
    # CHECK:   br label %19
    #
    # CHECK: 19:                                               ; preds = %22, %18
    # CHECK:   %20 = phi i64 [ %31, %22 ], [ 0, %18 ]
    # CHECK:   %21 = icmp slt i64 %20, 8
    # CHECK:   br i1 %21, label %22, label %32
    #
    # CHECK: 22:                                               ; preds = %19
    # CHECK:   %23 = and i64 ptrtoint (ptr @objFifo_in1_cons_buff_1 to i64), 31
    # CHECK:   %24 = icmp eq i64 %23, 0
    # CHECK:   call void @llvm.assume(i1 %24)
    # CHECK:   %25 = getelementptr i32, ptr @objFifo_in1_cons_buff_1, i64 %20
    # CHECK:   %26 = load i32, ptr %25, align 4
    # CHECK:   %27 = add i32 %26, 1
    # CHECK:   %28 = and i64 ptrtoint (ptr @objFifo_out1_buff_1 to i64), 31
    # CHECK:   %29 = icmp eq i64 %28, 0
    # CHECK:   call void @llvm.assume(i1 %29)
    # CHECK:   %30 = getelementptr i32, ptr @objFifo_out1_buff_1, i64 %20
    # CHECK:   store i32 %27, ptr %30, align 4
    # CHECK:   %31 = add i64 %20, 1
    # CHECK:   br label %19
    #
    # CHECK: 32:                                               ; preds = %19
    # CHECK:   call void @llvm.aie2.release(i32 48, i32 1)
    # CHECK:   call void @llvm.aie2.release(i32 51, i32 1)
    # CHECK:   %33 = add i64 %2, 2
    # CHECK:   br label %1
    #
    # CHECK: 34:                                               ; preds = %1
    # CHECK:   ret void
    # CHECK: }
    #
    # CHECK: declare void @llvm.aie2.acquire(i32, i32)
    #
    # CHECK: ; Function Attrs: nocallback nofree nosync nounwind willreturn inaccessiblememonly writeonly
    # CHECK: declare void @llvm.assume(i1 noundef) #0
    #
    # CHECK: declare void @llvm.aie2.release(i32, i32)
    #
    # CHECK: ; Function Attrs: mustprogress nounwind
    # CHECK: define dso_local void @llvm___aie2___acquire(i32 noundef %0, i32 noundef %1) local_unnamed_addr addrspace(1) #1 {
    # CHECK:   tail call addrspace(1) void @llvm.chess_memory_fence()
    # CHECK:   tail call addrspace(1) void @_Z25chess_separator_schedulerv() #5
    # CHECK:   tail call x86_regcallcc addrspace(1) void @__regcall3__chessintr_void_acquire_guarded___uint___uint(i32 zeroext %0, i32 zeroext %1) #5
    # CHECK:   tail call addrspace(1) void @_Z25chess_separator_schedulerv() #5
    # CHECK:   tail call addrspace(1) void @llvm.chess_memory_fence()
    # CHECK:   ret void
    # CHECK: }
    #
    # CHECK: ; Function Attrs: mustprogress nounwind willreturn
    # CHECK: declare void @llvm.chess_memory_fence() addrspace(1) #2
    #
    # CHECK: ; Function Attrs: nounwind inaccessiblememonly
    # CHECK: declare dso_local void @_Z25chess_separator_schedulerv() local_unnamed_addr addrspace(1) #3
    #
    # CHECK: ; Function Attrs: nounwind inaccessiblememonly
    # CHECK: declare dso_local x86_regcallcc void @__regcall3__chessintr_void_acquire_guarded___uint___uint(i32 zeroext, i32 zeroext) local_unnamed_addr addrspace(1) #3
    #
    # CHECK: ; Function Attrs: mustprogress nounwind
    # CHECK: define dso_local void @llvm___aie2___release(i32 noundef %0, i32 noundef %1) local_unnamed_addr addrspace(1) #1 {
    # CHECK:   tail call addrspace(1) void @llvm.chess_memory_fence()
    # CHECK:   tail call addrspace(1) void @_Z25chess_separator_schedulerv() #5
    # CHECK:   tail call x86_regcallcc addrspace(1) void @__regcall3__chessintr_void_release_guarded___uint___sint(i32 zeroext %0, i32 signext %1) #5
    # CHECK:   tail call addrspace(1) void @_Z25chess_separator_schedulerv() #5
    # CHECK:   tail call addrspace(1) void @llvm.chess_memory_fence()
    # CHECK:   ret void
    # CHECK: }
    #
    # CHECK: ; Function Attrs: nounwind inaccessiblememonly
    # CHECK: declare dso_local x86_regcallcc void @__regcall3__chessintr_void_release_guarded___uint___sint(i32 zeroext, i32 signext) local_unnamed_addr addrspace(1) #3
    #
    # CHECK: ; Function Attrs: nounwind
    # CHECK: define dso_local void @llvm___aie___event0() local_unnamed_addr addrspace(1) #4 {
    # CHECK:   tail call x86_regcallcc addrspace(1) void @__regcall3__chessintr_void_event_uint2_t(%struct.ipd.custom_type.uint2_t.uint2_t zeroinitializer) #5
    # CHECK:   ret void
    # CHECK: }
    #
    # CHECK: ; Function Attrs: nounwind inaccessiblememonly
    # CHECK: declare dso_local x86_regcallcc void @__regcall3__chessintr_void_event_uint2_t(%struct.ipd.custom_type.uint2_t.uint2_t) local_unnamed_addr addrspace(1) #3
    #
    # CHECK: ; Function Attrs: nounwind
    # CHECK: define dso_local void @llvm___aie___event1() local_unnamed_addr addrspace(1) #4 {
    # CHECK:   tail call x86_regcallcc addrspace(1) void @__regcall3__chessintr_void_event_uint2_t(%struct.ipd.custom_type.uint2_t.uint2_t { i2 1 }) #5
    # CHECK:   ret void
    # CHECK: }
    #
    # CHECK: attributes #0 = { nocallback nofree nosync nounwind willreturn inaccessiblememonly writeonly }
    # CHECK: attributes #1 = { mustprogress nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtin-memcpy" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
    # CHECK: attributes #2 = { mustprogress nounwind willreturn }
    # CHECK: attributes #3 = { nounwind inaccessiblememonly "frame-pointer"="all" "no-builtin-memcpy" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
    # CHECK: attributes #4 = { nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtin-memcpy" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
    # CHECK: attributes #5 = { nounwind inaccessiblememonly "no-builtin-memcpy" }
    #
    # CHECK: !llvm.module.flags = !{!0, !1, !2}
    # CHECK: !llvm.linker.options = !{}
    # CHECK: !llvm.ident = !{!3}
    #
    # CHECK: !0 = !{i32 2, !"Debug Info Version", i32 3}
    # CHECK: !1 = !{i32 1, !"wchar_size", i32 4}
    # CHECK: !2 = !{i32 7, !"frame-pointer", i32 2}
    # CHECK: !3 = !{!"clang version 15.0.5 (/u/sgasip/ipd/repositories/llvm_ipd 3a25925e0239306412dac02da5e4c8c51ae722e8)"}
    print(input_llchesslinked_ll)

    pass_pipeline = ", ".join(
        [
            "aie.device(aie-create-pathfinder-flows",
            "aie-lower-broadcast-packet",
            "aie-create-packet-flows",
            "aie-lower-multicast)",
        ]
    )
    input_physical = run_pipeline(
        input_with_addresses, "builtin.module(" + pass_pipeline + ")"
    )
    # CHECK: %tile_0_0 = aie.tile(0, 0)
    # CHECK: %switchbox_0_0 = aie.switchbox(%tile_0_0) {
    # CHECK:   aie.connect<South : 3, North : 0>
    # CHECK:   aie.connect<North : 0, South : 2>
    # CHECK: }
    # CHECK: %tile_0_1 = aie.tile(0, 1)
    # CHECK: %switchbox_0_1 = aie.switchbox(%tile_0_1) {
    # CHECK:   aie.connect<South : 0, DMA : 0>
    # CHECK:   aie.connect<DMA : 0, North : 0>
    # CHECK:   aie.connect<DMA : 1, South : 0>
    # CHECK:   aie.connect<North : 0, DMA : 1>
    # CHECK: }
    # CHECK: %tile_0_2 = aie.tile(0, 2)
    # CHECK: %switchbox_0_2 = aie.switchbox(%tile_0_2) {
    # CHECK:   aie.connect<South : 0, DMA : 0>
    # CHECK:   aie.connect<DMA : 0, South : 0>
    # CHECK: }
    # CHECK: %shim_mux_0_0 = aie.shim_mux(%tile_0_0) {
    # CHECK:   aie.connect<DMA : 0, North : 3>
    # CHECK:   aie.connect<North : 2, DMA : 0>
    # CHECK: }
    # CHECK: aie.wire(%shim_mux_0_0 : North, %switchbox_0_0 : South)
    # CHECK: aie.wire(%tile_0_0 : DMA, %shim_mux_0_0 : DMA)
    # CHECK: aie.wire(%tile_0_1 : Core, %switchbox_0_1 : Core)
    # CHECK: aie.wire(%tile_0_1 : DMA, %switchbox_0_1 : DMA)
    # CHECK: aie.wire(%switchbox_0_0 : North, %switchbox_0_1 : South)
    # CHECK: aie.wire(%tile_0_2 : Core, %switchbox_0_2 : Core)
    # CHECK: aie.wire(%tile_0_2 : DMA, %switchbox_0_2 : DMA)
    # CHECK: aie.wire(%switchbox_0_1 : North, %switchbox_0_2 : South)
    print(input_physical)

    aie_control_cpp = generate_cdo(input_physical.operation)
    # CHECK: /************************** Constants/Macros *****************************/
    # CHECK: #define HW_GEN                   XAIE_DEV_GEN_AIEML
    # CHECK: #define XAIE_NUM_ROWS            6
    # CHECK: #define XAIE_NUM_COLS            5
    # CHECK: #define XAIE_BASE_ADDR           0x40000000
    # CHECK: #define XAIE_COL_SHIFT           25
    # CHECK: #define XAIE_ROW_SHIFT           20
    # CHECK: #define XAIE_SHIM_ROW            0
    # CHECK: #define XAIE_MEM_TILE_ROW_START  1
    # CHECK: #define XAIE_MEM_TILE_NUM_ROWS   1
    # CHECK: #define XAIE_AIE_TILE_ROW_START  2
    # CHECK: #define XAIE_AIE_TILE_NUM_ROWS   4
    # CHECK: #define FOR_WRITE                0
    # CHECK: #define FOR_READ                 1
    # CHECK: #define XAIE_PARTITION_BASE_ADDR 0x0
    #
    # CHECK: /***************************** Includes *********************************/
    # CHECK: //#include <fstream>
    # CHECK: extern "C"
    # CHECK: {
    # CHECK:   #include <xaiengine.h>
    # CHECK: }
    # CHECK: //#include "adf/adf_api/AIEControlConfig.h"
    #
    # CHECK: #define __mlir_aie_try(x) x
    # CHECK: static XAie_DmaDimDesc *__mlir_aie_alloc_dim_desc(size_t ndims) {
    # CHECK:   XAie_DmaDimDesc *ret = NULL;
    # CHECK:   ret = (XAie_DmaDimDesc *)calloc(sizeof(XAie_DmaDimDesc), ndims);
    # CHECK:   if(NULL == ret) {
    # CHECK:     fprintf(stderr, "Allocating DmaDimDesc failed.\n");
    # CHECK:   }
    # CHECK:   return ret;
    # CHECK: }
    # CHECK: XAie_InstDeclare(DevInst, &ConfigPtr);   // Declare global device instance
    #
    # CHECK: bool ppgraph_load_elf(const std::string& work_path, std::vector<std::string>& elfInfoPath)
    # CHECK: {
    # CHECK: std::string work_dir = (work_path.empty() ?  "Work" : work_path);
    # CHECK: {
    # CHECK: if (XAie_LoadElf(&DevInst, XAie_TileLoc(0,2), (work_dir + "/core_0_2.elf").c_str(), XAIE_ENABLE) != XAIE_OK)
    # CHECK: {
    # CHECK:     std::cerr << "ERROR: Failed to load elf for core(%d,%d)" << std::endl;
    # CHECK:     return false;
    # CHECK: }
    # CHECK: }
    # CHECK:     return true;
    # CHECK: } // ppgraph_load_elf
    #
    # CHECK: void ppgraph_core_enable()
    # CHECK: {
    # CHECK: XAie_CoreEnable(&DevInst, XAie_TileLoc(0,2));
    # CHECK:     return;
    # CHECK: } // ppgraph_core_enable
    #
    # CHECK: void enableErrorHandling()
    # CHECK: {
    # CHECK:     XAie_ErrorHandlingInit(&DevInst);
    # CHECK: } // enableErrorHandling
    #
    # CHECK: void ppgraph_init(const std::string& work_path)
    # CHECK: {
    # CHECK: XAie_CoreReset(&DevInst, XAie_TileLoc(0,2));
    # CHECK: XAie_CoreUnreset(&DevInst, XAie_TileLoc(0,2));
    # CHECK: for (int l=0; l<16; l++)
    # CHECK:   XAie_LockSetValue(&DevInst, XAie_TileLoc(0,2), XAie_LockInit(l, 0));
    # CHECK: XAie_LockSetValue(&DevInst, XAie_TileLoc(0,0), XAie_LockInit(0, 0));
    # CHECK: XAie_LockSetValue(&DevInst, XAie_TileLoc(0,0), XAie_LockInit(1, 0));
    # CHECK: XAie_LockSetValue(&DevInst, XAie_TileLoc(0,0), XAie_LockInit(2, 0));
    # CHECK: XAie_LockSetValue(&DevInst, XAie_TileLoc(0,0), XAie_LockInit(3, 0));
    # CHECK: XAie_LockSetValue(&DevInst, XAie_TileLoc(0,1), XAie_LockInit(0, 2));
    # CHECK: XAie_LockSetValue(&DevInst, XAie_TileLoc(0,1), XAie_LockInit(1, 0));
    # CHECK: XAie_LockSetValue(&DevInst, XAie_TileLoc(0,1), XAie_LockInit(2, 2));
    # CHECK: XAie_LockSetValue(&DevInst, XAie_TileLoc(0,1), XAie_LockInit(3, 0));
    # CHECK: XAie_LockSetValue(&DevInst, XAie_TileLoc(0,2), XAie_LockInit(0, 2));
    # CHECK: XAie_LockSetValue(&DevInst, XAie_TileLoc(0,2), XAie_LockInit(1, 0));
    # CHECK: XAie_LockSetValue(&DevInst, XAie_TileLoc(0,2), XAie_LockInit(2, 2));
    # CHECK: XAie_LockSetValue(&DevInst, XAie_TileLoc(0,2), XAie_LockInit(3, 0));
    # CHECK: XAie_DmaDesc dma_tile02_bd0;
    # CHECK: XAie_DmaDescInit(&DevInst, &(dma_tile02_bd0), XAie_TileLoc(0,2));
    # CHECK: XAie_DmaSetLock(&(dma_tile02_bd0), XAie_LockInit(0,-1),XAie_LockInit(1,1));
    # CHECK: XAie_DmaSetAddrLen(&(dma_tile02_bd0), /* addrA */ 0x400,  /* len */ 8 * 4);
    # CHECK: XAie_DmaSetNextBd(&(dma_tile02_bd0),  /* nextbd */ 1,  /* enableNextBd */ 1);
    # CHECK: XAie_DmaEnableBd(&(dma_tile02_bd0));
    # CHECK: XAie_DmaWriteBd(&DevInst, &(dma_tile02_bd0), XAie_TileLoc(0,2),  /* bd */ 0);
    # CHECK: XAie_DmaDesc dma_tile02_bd1;
    # CHECK: XAie_DmaDescInit(&DevInst, &(dma_tile02_bd1), XAie_TileLoc(0,2));
    # CHECK: XAie_DmaSetLock(&(dma_tile02_bd1), XAie_LockInit(0,-1),XAie_LockInit(1,1));
    # CHECK: XAie_DmaSetAddrLen(&(dma_tile02_bd1), /* addrA */ 0x420,  /* len */ 8 * 4);
    # CHECK: XAie_DmaSetNextBd(&(dma_tile02_bd1),  /* nextbd */ 0,  /* enableNextBd */ 1);
    # CHECK: XAie_DmaEnableBd(&(dma_tile02_bd1));
    # CHECK: XAie_DmaWriteBd(&DevInst, &(dma_tile02_bd1), XAie_TileLoc(0,2),  /* bd */ 1);
    # CHECK: XAie_DmaDesc dma_tile02_bd2;
    # CHECK: XAie_DmaDescInit(&DevInst, &(dma_tile02_bd2), XAie_TileLoc(0,2));
    # CHECK: XAie_DmaSetLock(&(dma_tile02_bd2), XAie_LockInit(3,-1),XAie_LockInit(2,1));
    # CHECK: XAie_DmaSetAddrLen(&(dma_tile02_bd2), /* addrA */ 0x440,  /* len */ 8 * 4);
    # CHECK: XAie_DmaSetNextBd(&(dma_tile02_bd2),  /* nextbd */ 3,  /* enableNextBd */ 1);
    # CHECK: XAie_DmaEnableBd(&(dma_tile02_bd2));
    # CHECK: XAie_DmaWriteBd(&DevInst, &(dma_tile02_bd2), XAie_TileLoc(0,2),  /* bd */ 2);
    # CHECK: XAie_DmaDesc dma_tile02_bd3;
    # CHECK: XAie_DmaDescInit(&DevInst, &(dma_tile02_bd3), XAie_TileLoc(0,2));
    # CHECK: XAie_DmaSetLock(&(dma_tile02_bd3), XAie_LockInit(3,-1),XAie_LockInit(2,1));
    # CHECK: XAie_DmaSetAddrLen(&(dma_tile02_bd3), /* addrA */ 0x460,  /* len */ 8 * 4);
    # CHECK: XAie_DmaSetNextBd(&(dma_tile02_bd3),  /* nextbd */ 2,  /* enableNextBd */ 1);
    # CHECK: XAie_DmaEnableBd(&(dma_tile02_bd3));
    # CHECK: XAie_DmaWriteBd(&DevInst, &(dma_tile02_bd3), XAie_TileLoc(0,2),  /* bd */ 3);
    # CHECK: XAie_DmaChannelPushBdToQueue(&DevInst, XAie_TileLoc(0,2), /* ChNum */0, /* dmaDir */ DMA_S2MM, /* BdNum */0);
    # CHECK: XAie_DmaChannelEnable(&DevInst, XAie_TileLoc(0,2), /* ChNum */ 0, /* dmaDir */ DMA_S2MM);
    # CHECK: XAie_DmaChannelPushBdToQueue(&DevInst, XAie_TileLoc(0,2), /* ChNum */0, /* dmaDir */ DMA_MM2S, /* BdNum */2);
    # CHECK: XAie_DmaChannelEnable(&DevInst, XAie_TileLoc(0,2), /* ChNum */ 0, /* dmaDir */ DMA_MM2S);
    # CHECK: XAie_DmaDesc dma_tile01_bd0;
    # CHECK: XAie_DmaDescInit(&DevInst, &(dma_tile01_bd0), XAie_TileLoc(0,1));
    # CHECK: XAie_DmaSetLock(&(dma_tile01_bd0), XAie_LockInit(64,-1),XAie_LockInit(65,1));
    # CHECK: XAie_DmaSetAddrLen(&(dma_tile01_bd0), /* addrA */ 0x80000,  /* len */ 16 * 4);
    # CHECK: XAie_DmaSetNextBd(&(dma_tile01_bd0),  /* nextbd */ 1,  /* enableNextBd */ 1);
    # CHECK: XAie_DmaEnableBd(&(dma_tile01_bd0));
    # CHECK: XAie_DmaWriteBd(&DevInst, &(dma_tile01_bd0), XAie_TileLoc(0,1),  /* bd */ 0);
    # CHECK: XAie_DmaDesc dma_tile01_bd1;
    # CHECK: XAie_DmaDescInit(&DevInst, &(dma_tile01_bd1), XAie_TileLoc(0,1));
    # CHECK: XAie_DmaSetLock(&(dma_tile01_bd1), XAie_LockInit(64,-1),XAie_LockInit(65,1));
    # CHECK: XAie_DmaSetAddrLen(&(dma_tile01_bd1), /* addrA */ 0x80040,  /* len */ 16 * 4);
    # CHECK: XAie_DmaSetNextBd(&(dma_tile01_bd1),  /* nextbd */ 0,  /* enableNextBd */ 1);
    # CHECK: XAie_DmaEnableBd(&(dma_tile01_bd1));
    # CHECK: XAie_DmaWriteBd(&DevInst, &(dma_tile01_bd1), XAie_TileLoc(0,1),  /* bd */ 1);
    # CHECK: XAie_DmaDesc dma_tile01_bd2;
    # CHECK: XAie_DmaDescInit(&DevInst, &(dma_tile01_bd2), XAie_TileLoc(0,1));
    # CHECK: XAie_DmaSetLock(&(dma_tile01_bd2), XAie_LockInit(65,-1),XAie_LockInit(64,1));
    # CHECK: XAie_DmaSetAddrLen(&(dma_tile01_bd2), /* addrA */ 0x80000,  /* len */ 16 * 4);
    # CHECK: XAie_DmaSetNextBd(&(dma_tile01_bd2),  /* nextbd */ 3,  /* enableNextBd */ 1);
    # CHECK: XAie_DmaEnableBd(&(dma_tile01_bd2));
    # CHECK: XAie_DmaWriteBd(&DevInst, &(dma_tile01_bd2), XAie_TileLoc(0,1),  /* bd */ 2);
    # CHECK: XAie_DmaDesc dma_tile01_bd3;
    # CHECK: XAie_DmaDescInit(&DevInst, &(dma_tile01_bd3), XAie_TileLoc(0,1));
    # CHECK: XAie_DmaSetLock(&(dma_tile01_bd3), XAie_LockInit(65,-1),XAie_LockInit(64,1));
    # CHECK: XAie_DmaSetAddrLen(&(dma_tile01_bd3), /* addrA */ 0x80040,  /* len */ 16 * 4);
    # CHECK: XAie_DmaSetNextBd(&(dma_tile01_bd3),  /* nextbd */ 2,  /* enableNextBd */ 1);
    # CHECK: XAie_DmaEnableBd(&(dma_tile01_bd3));
    # CHECK: XAie_DmaWriteBd(&DevInst, &(dma_tile01_bd3), XAie_TileLoc(0,1),  /* bd */ 3);
    # CHECK: XAie_DmaDesc dma_tile01_bd24;
    # CHECK: XAie_DmaDescInit(&DevInst, &(dma_tile01_bd24), XAie_TileLoc(0,1));
    # CHECK: XAie_DmaSetLock(&(dma_tile01_bd24), XAie_LockInit(67,-1),XAie_LockInit(66,1));
    # CHECK: XAie_DmaSetAddrLen(&(dma_tile01_bd24), /* addrA */ 0x80080,  /* len */ 16 * 4);
    # CHECK: XAie_DmaSetNextBd(&(dma_tile01_bd24),  /* nextbd */ 25,  /* enableNextBd */ 1);
    # CHECK: XAie_DmaEnableBd(&(dma_tile01_bd24));
    # CHECK: XAie_DmaWriteBd(&DevInst, &(dma_tile01_bd24), XAie_TileLoc(0,1),  /* bd */ 24);
    # CHECK: XAie_DmaDesc dma_tile01_bd25;
    # CHECK: XAie_DmaDescInit(&DevInst, &(dma_tile01_bd25), XAie_TileLoc(0,1));
    # CHECK: XAie_DmaSetLock(&(dma_tile01_bd25), XAie_LockInit(67,-1),XAie_LockInit(66,1));
    # CHECK: XAie_DmaSetAddrLen(&(dma_tile01_bd25), /* addrA */ 0x800C0,  /* len */ 16 * 4);
    # CHECK: XAie_DmaSetNextBd(&(dma_tile01_bd25),  /* nextbd */ 24,  /* enableNextBd */ 1);
    # CHECK: XAie_DmaEnableBd(&(dma_tile01_bd25));
    # CHECK: XAie_DmaWriteBd(&DevInst, &(dma_tile01_bd25), XAie_TileLoc(0,1),  /* bd */ 25);
    # CHECK: XAie_DmaDesc dma_tile01_bd26;
    # CHECK: XAie_DmaDescInit(&DevInst, &(dma_tile01_bd26), XAie_TileLoc(0,1));
    # CHECK: XAie_DmaSetLock(&(dma_tile01_bd26), XAie_LockInit(66,-1),XAie_LockInit(67,1));
    # CHECK: XAie_DmaSetAddrLen(&(dma_tile01_bd26), /* addrA */ 0x80080,  /* len */ 16 * 4);
    # CHECK: XAie_DmaSetNextBd(&(dma_tile01_bd26),  /* nextbd */ 27,  /* enableNextBd */ 1);
    # CHECK: XAie_DmaEnableBd(&(dma_tile01_bd26));
    # CHECK: XAie_DmaWriteBd(&DevInst, &(dma_tile01_bd26), XAie_TileLoc(0,1),  /* bd */ 26);
    # CHECK: XAie_DmaDesc dma_tile01_bd27;
    # CHECK: XAie_DmaDescInit(&DevInst, &(dma_tile01_bd27), XAie_TileLoc(0,1));
    # CHECK: XAie_DmaSetLock(&(dma_tile01_bd27), XAie_LockInit(66,-1),XAie_LockInit(67,1));
    # CHECK: XAie_DmaSetAddrLen(&(dma_tile01_bd27), /* addrA */ 0x800C0,  /* len */ 16 * 4);
    # CHECK: XAie_DmaSetNextBd(&(dma_tile01_bd27),  /* nextbd */ 26,  /* enableNextBd */ 1);
    # CHECK: XAie_DmaEnableBd(&(dma_tile01_bd27));
    # CHECK: XAie_DmaWriteBd(&DevInst, &(dma_tile01_bd27), XAie_TileLoc(0,1),  /* bd */ 27);
    # CHECK: XAie_DmaChannelPushBdToQueue(&DevInst, XAie_TileLoc(0,1), /* ChNum */0, /* dmaDir */ DMA_S2MM, /* BdNum */0);
    # CHECK: XAie_DmaChannelEnable(&DevInst, XAie_TileLoc(0,1), /* ChNum */ 0, /* dmaDir */ DMA_S2MM);
    # CHECK: XAie_DmaChannelPushBdToQueue(&DevInst, XAie_TileLoc(0,1), /* ChNum */0, /* dmaDir */ DMA_MM2S, /* BdNum */2);
    # CHECK: XAie_DmaChannelEnable(&DevInst, XAie_TileLoc(0,1), /* ChNum */ 0, /* dmaDir */ DMA_MM2S);
    # CHECK: XAie_DmaChannelPushBdToQueue(&DevInst, XAie_TileLoc(0,1), /* ChNum */1, /* dmaDir */ DMA_MM2S, /* BdNum */24);
    # CHECK: XAie_DmaChannelEnable(&DevInst, XAie_TileLoc(0,1), /* ChNum */ 1, /* dmaDir */ DMA_MM2S);
    # CHECK: XAie_DmaChannelPushBdToQueue(&DevInst, XAie_TileLoc(0,1), /* ChNum */1, /* dmaDir */ DMA_S2MM, /* BdNum */26);
    # CHECK: XAie_DmaChannelEnable(&DevInst, XAie_TileLoc(0,1), /* ChNum */ 1, /* dmaDir */ DMA_S2MM);
    # CHECK:   int x, y;
    # CHECK: // Core Stream Switch column 0 row 0
    # CHECK: x = 0;
    # CHECK: y = 0;
    # CHECK: XAie_StrmConnCctEnable(&DevInst, XAie_TileLoc(x,y), CTRL, 0, SOUTH, 0);
    # CHECK: {
    # CHECK:   //configure DMA_<S2MM/MM2S>_<N>_Ctrl register
    # CHECK:   XAie_DmaChannelDesc DmaChannelDescInst;
    # CHECK:   XAie_DmaChannelDescInit(&DevInst, &DmaChannelDescInst, XAie_TileLoc(x,y));
    # CHECK:   XAie_DmaChannelSetControllerId(&DmaChannelDescInst, 0);
    # CHECK:   XAie_DmaWriteChannel(&DevInst, &DmaChannelDescInst, XAie_TileLoc(x,y), 0, DMA_S2MM);
    # CHECK: }
    #
    # CHECK: {
    # CHECK:   //configure DMA_<S2MM/MM2S>_<N>_Ctrl register
    # CHECK:   XAie_DmaChannelDesc DmaChannelDescInst;
    # CHECK:   XAie_DmaChannelDescInit(&DevInst, &DmaChannelDescInst, XAie_TileLoc(x,y));
    # CHECK:   XAie_DmaChannelSetControllerId(&DmaChannelDescInst, 0);
    # CHECK:   XAie_DmaWriteChannel(&DevInst, &DmaChannelDescInst, XAie_TileLoc(x,y), 1, DMA_S2MM);
    # CHECK: }
    #
    # CHECK: XAie_AieToPlIntfEnable (&DevInst, XAie_TileLoc(x, y), 0, PLIF_WIDTH_32);
    # CHECK: XAie_StrmConnCctEnable(&DevInst, XAie_TileLoc(x,y), SOUTH, 3, NORTH, 0);
    # CHECK: XAie_StrmConnCctEnable(&DevInst, XAie_TileLoc(x,y), NORTH, 0, SOUTH, 2);
    # CHECK: // Core Stream Switch column 0 row 1
    # CHECK: x = 0;
    # CHECK: y = 1;
    # CHECK: XAie_StrmConnCctEnable(&DevInst, XAie_TileLoc(x,y), SOUTH, 0, DMA, 0);
    # CHECK: XAie_StrmConnCctEnable(&DevInst, XAie_TileLoc(x,y), DMA, 0, NORTH, 0);
    # CHECK: XAie_StrmConnCctEnable(&DevInst, XAie_TileLoc(x,y), DMA, 1, SOUTH, 0);
    # CHECK: XAie_StrmConnCctEnable(&DevInst, XAie_TileLoc(x,y), NORTH, 0, DMA, 1);
    # CHECK: // Core Stream Switch column 0 row 2
    # CHECK: x = 0;
    # CHECK: y = 2;
    # CHECK: XAie_StrmConnCctEnable(&DevInst, XAie_TileLoc(x,y), SOUTH, 0, DMA, 0);
    # CHECK: XAie_StrmConnCctEnable(&DevInst, XAie_TileLoc(x,y), DMA, 0, SOUTH, 0);
    # CHECK: // ShimMux column 0 row 0
    # CHECK: // NOTE ShimMux always connects from the south as directions are defined relative to the tile stream switch
    # CHECK: x = 0;
    # CHECK: y = 0;
    # CHECK: XAie_EnableShimDmaToAieStrmPort(&DevInst, XAie_TileLoc(x,y), 3);
    # CHECK: XAie_EnableAieToShimDmaStrmPort(&DevInst, XAie_TileLoc(x,y), 2);
    # CHECK: } // ppgraph_init
    #
    #
    #
    # CHECK:   class InitializeAIEControl
    # CHECK:   {
    # CHECK:   public:
    # CHECK:     InitializeAIEControl()
    # CHECK:     {
    # CHECK:       XAie_SetupConfig(ConfigPtr, HW_GEN, XAIE_BASE_ADDR, XAIE_COL_SHIFT,
    # CHECK:                        XAIE_ROW_SHIFT, XAIE_NUM_COLS, XAIE_NUM_ROWS,
    # CHECK:                        XAIE_SHIM_ROW, XAIE_MEM_TILE_ROW_START,
    # CHECK:                        XAIE_MEM_TILE_NUM_ROWS, XAIE_AIE_TILE_ROW_START,
    # CHECK:                        XAIE_AIE_TILE_NUM_ROWS);
    #
    # CHECK:       XAie_SetupPartitionConfig(&DevInst, XAIE_PARTITION_BASE_ADDR, 1, 1);
    #
    # CHECK:       XAie_CfgInitialize(&(DevInst), &ConfigPtr);
    #
    # CHECK: #if defined(__AIESIM__)
    # CHECK: #if defined(__CDO__)
    # CHECK:       XAie_SetIOBackend(&(DevInst), XAIE_IO_BACKEND_CDO); // Set aiengine driver library to run for CDO Mode
    # CHECK:       XAie_UpdateNpiAddr(&(DevInst), 0x0);
    # CHECK: #else
    # CHECK:       //AIE driver currently error out XAie_UpdateNpiAddr for AIESIM
    # CHECK: #endif
    # CHECK: #else
    # CHECK:       XAie_UpdateNpiAddr(&(DevInst), 0x0);
    # CHECK: #endif
    #
    # CHECK: #if defined(__AIESIM__) && !defined(__CDO__)
    # CHECK:       XAie_TurnEccOff(&DevInst);
    # CHECK: #endif
    #
    # CHECK: #if defined(__AIESIM__) && !defined(__CDO__)
    # CHECK:       extern unsigned ess_debug;
    # CHECK: #else
    # CHECK:       unsigned ess_debug = false;
    # CHECK: #endif
    #
    # CHECK: #ifdef __EXCLUDE_PL_CONTROL__
    # CHECK:       bool exclude_pl_control = true;
    # CHECK: #else
    # CHECK:       bool exclude_pl_control = false;
    # CHECK: #endif
    #
    # CHECK: #ifdef __CDO__
    # CHECK:       int trace_config_stream_option = 2;
    # CHECK: #else
    # CHECK:       int trace_config_stream_option = 0;
    # CHECK: #endif
    # CHECK:     }
    # CHECK:   } initAIEControl;
    print(aie_control_cpp)

    col, row, _ = cores[0]
    core_0_2_bcf = generate_bcf(input_with_addresses.operation, col, row)

    # CHECK: _entry_point _main_init
    # CHECK: _symbol core_0_2 _after _main_init
    # CHECK: _symbol      _main_init 0
    # CHECK: _reserved DMb      0x00000 0x40000 //Don't put data in code memory
    # CHECK: _reserved DMb 0x40000 0x10000  // No tile with memory exists to the south.
    # CHECK: _reserved DMb 0x50000 0x10000  // No tile with memory exists to the west.
    # CHECK: _reserved DMb 0x60000 0x10000  // Don't allocate variables outside of local memory.
    # CHECK: _symbol objFifo_in1_cons_buff_0 0x70400 0x20
    # CHECK: _extern objFifo_in1_cons_buff_0
    # CHECK: _reserved DMb 0x70400 0x20
    # CHECK: _symbol objFifo_in1_cons_buff_1 0x70420 0x20
    # CHECK: _extern objFifo_in1_cons_buff_1
    # CHECK: _reserved DMb 0x70420 0x20
    # CHECK: _symbol objFifo_out1_buff_0 0x70440 0x20
    # CHECK: _extern objFifo_out1_buff_0
    # CHECK: _reserved DMb 0x70440 0x20
    # CHECK: _symbol objFifo_out1_buff_1 0x70460 0x20
    # CHECK: _extern objFifo_out1_buff_1
    # CHECK: _reserved DMb 0x70460 0x20
    # CHECK: _stack    DM_stack 0x70000  0x400 //stack for core
    # CHECK: _reserved DMb 0x80000 0x80000 // And everything else the core can't see
    # CHECK: _resolve _main core_0_2
    print(core_0_2_bcf)

    # awk /_include _file/ {print($3)} /home/mlevental/dev_projects/mlir-aie/cmake-build-debug/test/ipu-xrt/add_one_using_dma/aie.mlir.prj/core_0_2.bcf
    re.findall(r"^_include _file (.*)", core_0_2_bcf, re.MULTILINE)

    WORKDIR = Path(os.getenv("WORKDIR", str(Path(".").absolute()))).absolute()
    HOST_RUNTIME_LIB_DIR = Path(os.getenv("HOST_RUNTIME_LIB_DIR")).absolute()
    AIETOOLS_DIR = Path(os.getenv("AIETOOLS")).absolute()
    # bootgen and xclbinutil
    VITIS_BIN_DIR = AIETOOLS_DIR.parent / "bin"
    RDI_DATADIR = f"{AIETOOLS_DIR}/data"
    XRT_DIR = Path(os.getenv("XRT_DIR", "/opt/xilinx/xrt")).absolute()
    XILINXD_LICENSE_FILE = Path(os.getenv("XILINXD_LICENSE_FILE")).absolute()
    ld_path = [
        os.getenv("LD_LIBRARY_PATH"),
        f"{AIETOOLS_DIR}/lib/lnx64.o",
        f"{AIETOOLS_DIR}/lnx64/tools/dot/lib",
        f"{HOST_RUNTIME_LIB_DIR}/xaiengine/cdo",
        f"{XRT_DIR}/lib",
    ]
    ld_path = ":".join(list(filter(None, ld_path)))
    path = [
        os.getenv("PATH"),
        f"{AIETOOLS_DIR}/bin/unwrapped/lnx64.o",
        f"{AIETOOLS_DIR}/tps/lnx64/target/bin/LNa64bin",
        str(VITIS_BIN_DIR),
    ]
    path = ":".join(list(filter(None, path)))
    env = {
        "LD_LIBRARY_PATH": ld_path,
        "RDI_DATADIR": RDI_DATADIR,
        "PATH": path,
        "XILINXD_LICENSE_FILE": XILINXD_LICENSE_FILE,
        "XILINX_XRT": XRT_DIR,
    }

    xchess_args = [
        f"{AIETOOLS_DIR}/bin/unwrapped/lnx64.o/xchesscc",
        "+P",
        "4",  # parallel compilation (function + file level)
        "-p",
        "me",  # parallel compilation (function level only)
        "-C",
        "Release_LLVM",  # configuration
        "-D__AIENGINE__",
        "-D__AIE_ARCH__=20",
        "-D__AIEARCH__=20",
        "-Y",
        f"clang={AIETOOLS_DIR}/tps/lnx64/target/bin/LNa64bin/chess-clang",
        "-P",
        f"{AIETOOLS_DIR}/data/aie_ml/lib",  # processor model directory
        "-d",  # disassemble output
        "-f",  # use LLVM frontend
        # "+f", only run LLVM frontend (emits IR)
        "+w",
        str(WORKDIR),
    ]

    with open(WORKDIR / "input.llchesslinked.ll", "w") as f:
        f.write(input_llchesslinked_ll)

    cmd = [
        *xchess_args,
        "-c",  # compile/assemble only, do not link
        "input.llchesslinked.ll",
        "-o",
        "input.o",
    ]
    subprocess.run(cmd, check=True, cwd=WORKDIR, env=env)

    with open(WORKDIR / "core_0_2.bcf", "w") as f:
        f.write(core_0_2_bcf)

    cmd = [
        *xchess_args,
        "input.o",
        *extract_input_files(core_0_2_bcf),
        "+l",  # linker configuration file
        "core_0_2.bcf",
        "-o",
        "core_0_2.elf",
    ]
    # print(f"LD_LIBRARY_PATH={ld_path} PATH={path} {RDI_DATADIR=} {' '.join(cmd)}")

    subprocess.run(cmd, check=True, cwd=WORKDIR, env=env)

    cmd = [
        "g++",
        "-fPIC",
        "-std=gnu++17",
        "-D__AIEARCH__=20",
        "-D__AIESIM__",
        "-D__CDO__",
        "-D__PS_INIT_AIE__",
        "-D__LOCK_FENCE_MODE__=2",
        "-Wl,--no-as-needed",
        "-lxaienginecdo",
        "-lcdo_driver",
        "-DAIE_OPTION_SCALAR_FLOAT_ON_VECTOR",
        "-DAIE2_FP32_EMULATION_ACCURACY_FAST",
        f"-I{HOST_RUNTIME_LIB_DIR}/xaiengine/cdo/include",
        f"-L{HOST_RUNTIME_LIB_DIR}/xaiengine/cdo",
        f"-I{AIETOOLS_DIR}",
        f"-L{AIETOOLS_DIR}/lib/lnx64.o",
        "-o",
        "cdo_main",
        str(Path(__file__).parent.absolute() / "cdo_main.cpp"),
    ]
    subprocess.run(cmd, check=True, cwd=WORKDIR)
    subprocess.run(
        [f"{os.getenv('WORKDIR')}/cdo_main", "-w", WORKDIR],
        check=True,
        cwd=WORKDIR,
        env=env,
    )

    with open(WORKDIR / "mem_topology.json", "w") as f:
        json.dump(mem_topology, f, indent=2)
    with open(WORKDIR / "aie_partition.json", "w") as f:
        json.dump(emit_partition(str(module)), f, indent=2)
    with open(WORKDIR / "kernels.json", "w") as f:
        json.dump(emit_design_kernel_json(), f, indent=2)
    with open(WORKDIR / "design.bif", "w") as f:
        f.write(emit_design_bif(WORKDIR))

    cmd = [
        "bootgen",
        "-arch",
        "versal",
        "-image",
        WORKDIR / "design.bif",
        "-w",  # force overwrite
        "-o",
        WORKDIR / "design.pdi",
    ]
    subprocess.run(cmd, check=True, cwd=WORKDIR, env=env)

    cmd = [
        "xclbinutil",
        "--add-replace-section",
        f"MEM_TOPOLOGY:JSON:{WORKDIR / 'mem_topology.json'}",
        "--add-kernel",
        str(WORKDIR / "kernels.json"),
        "--add-replace-section",
        f"AIE_PARTITION:JSON:{WORKDIR / 'aie_partition.json'}",
        "--force",
        "--output",
        f"{WORKDIR / 'final.xclbin'}",
    ]
    subprocess.run(cmd, check=True, cwd=WORKDIR, env=env)

    ipu_insts = ipu_instgen(generated_ipu_insts.operation)
    # CHECK: 00000011
    # CHECK: 01000405
    # CHECK: 01000100
    # CHECK: 0B590100
    # CHECK: 000055FF
    # CHECK: 00000001
    # CHECK: 00000010
    # CHECK: 314E5A5F
    # CHECK: 635F5F31
    # CHECK: 676E696C
    # CHECK: 39354E5F
    # CHECK: 6E693131
    # CHECK: 5F727473
    # CHECK: 64726F77
    # CHECK: 00004573
    # CHECK: 07BD9630
    # CHECK: 000055FF
    # CHECK: 06000100
    # CHECK: 00000000
    # CHECK: 00000040
    # CHECK: 00000000
    # CHECK: 00000000
    # CHECK: 00000000
    # CHECK: 80000000
    # CHECK: 00000000
    # CHECK: 00000000
    # CHECK: 02000000
    # CHECK: 02000000
    # CHECK: 0001D214
    # CHECK: 00000000
    # CHECK: 06000121
    # CHECK: 00000000
    # CHECK: 00000040
    # CHECK: 00000000
    # CHECK: 00000000
    # CHECK: 00000000
    # CHECK: 80000000
    # CHECK: 00000000
    # CHECK: 00000000
    # CHECK: 02000000
    # CHECK: 02000000
    # CHECK: 0001D204
    # CHECK: 80000001
    # CHECK: 03000000
    # CHECK: 00010100
    print("\n".join(ipu_insts))

    handle = subprocess.run(
        [
            "flock",
            "/tmp/ipu.lock",
            "/opt/xilinx/xrt/amdaie/setup_xclbin_firmware.sh",
            "-dev",
            "Phoenix",
            "-xclbin",
            f"{WORKDIR / 'final.xclbin'}",
        ],
        capture_output=True,
        cwd=WORKDIR,
        env=env,
    )
    stderr = handle.stderr.decode("utf-8").strip()
    if len(stderr):
        print(f"{stderr=}", file=sys.stderr)
        assert False

    xclbin = XCLBin(f"{WORKDIR / 'final.xclbin'}", "MLIR_AIE")
    ipu_insts = [int(inst, 16) for inst in ipu_insts]
    xclbin.load_ipu_instructions(ipu_insts)
    inps, outps = xclbin.mmap_buffers([(64,), (64,)], [(64,)], np.int32)

    wrap_A = np.asarray(inps[0])
    wrap_C = np.asarray(outps[0])

    A = np.random.randint(0, 10, 64, dtype=np.int32)
    C = np.zeros(64, dtype=np.int32)

    np.copyto(wrap_A, A, casting="no")
    np.copyto(wrap_C, C, casting="no")

    xclbin.sync_buffers_to_device()
    xclbin.run()
    xclbin.wait()
    xclbin.sync_buffers_from_device()

    # CHECK: True
    print(np.allclose(A + 1, wrap_C))
