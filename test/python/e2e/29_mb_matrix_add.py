# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

# RUN: HAS_RYZEN_AI="%run_on_ipu" HOST_RUNTIME_LIB_DIR=%host_runtime_lib% WORKDIR=%T XRT_DIR=%XRT_DIR %PYTHON %s | FileCheck %s
# REQUIRES: xrt_python_bindings

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from aie.extras.dialects.ext import memref, arith, func, linalg
from aie.extras.runtime.passes import run_pipeline
from aie.extras.util import bb, find_ops

import aie.extras.types as T
from aie.compiler.aiecc.main import (
    generate_cores_list,
    emit_partition,
    emit_design_bif,
    emit_design_kernel_json,
    mem_topology,
    chesshack,
)
from aie.dialects import aie, builtin, pdl
from aie.dialects.aie import (
    AIEDevice,
    DMAChannelDir,
    LockAction,
    WireBundle,
    device,
    generate_bcf,
    generate_cdo,
    ipu_instgen,
    lock,
    mem,
    memtile_dma,
    tile,
    translate_mlir_to_llvmir,
    aie_llvm_link,
    BDDimLayout,
)
from aie.dialects.aie import translate_aie_vec_to_cpp
from aie.dialects.aiex import ipu_sync, ipu_dma_memcpy_nd_
from aie.dialects.scf import for_
from aie.dialects.scf import yield_
from aie.dialects.transform import (
    get_parent_op,
    apply_registered_pass,
    any_op_t,
)
from aie.dialects.transform.extras import named_sequence
from aie.dialects.transform.loop import loop_unroll
from aie.dialects.transform.structured import structured_match
from aie.ir import Context, Location, StringAttr, UnitAttr
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


lower_to_cf_with_addresses_and_routed = [
    "lower-affine",
    "aie-canonicalize-device",
    "aie.device(" + "aie-assign-lock-ids",
    "aie-create-pathfinder-flows",
    "aie-lower-broadcast-packet",
    "aie-create-packet-flows",
    "aie-lower-multicast",
    "aie-assign-buffer-addresses)",
    "convert-scf-to-cf",
]

lower_to_llvm = [
    "aie.device(aie-localize-locks",
    "aie-normalize-address-spaces)",
    "aie-standard-lowering",
    "aiex-standard-lowering",
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

WORKDIR = Path(os.getenv("WORKDIR", str(Path(".").absolute()))).absolute()
HOST_RUNTIME_LIB_DIR = Path(os.getenv("HOST_RUNTIME_LIB_DIR")).absolute()
AIETOOLS_DIR = Path(os.getenv("AIETOOLS")).absolute()
# bootgen and xclbinutil
VITIS_BIN_DIR = AIETOOLS_DIR.parent / "bin"
RDI_DATADIR = f"{AIETOOLS_DIR}/data"
XRT_DIR = Path(os.getenv("XRT_DIR", "/opt/xilinx/xrt")).absolute()
XILINXD_LICENSE_FILE = Path(os.getenv("XILINXD_LICENSE_FILE")).absolute()
LD_PATH = [
    os.getenv("LD_LIBRARY_PATH"),
    f"{AIETOOLS_DIR}/lib/lnx64.o",
    f"{AIETOOLS_DIR}/lnx64/tools/dot/lib",
    f"{HOST_RUNTIME_LIB_DIR}/xaiengine/cdo",
    f"{XRT_DIR}/lib",
]
LD_PATH = ":".join(list(filter(None, LD_PATH)))
PATH = [
    os.getenv("PATH"),
    f"{AIETOOLS_DIR}/bin/unwrapped/lnx64.o",
    f"{AIETOOLS_DIR}/tps/lnx64/target/bin/LNa64bin",
    str(VITIS_BIN_DIR),
]
PATH = ":".join(list(filter(None, PATH)))
ENV = {
    "LD_LIBRARY_PATH": LD_PATH,
    "RDI_DATADIR": RDI_DATADIR,
    "PATH": PATH,
    "XILINXD_LICENSE_FILE": XILINXD_LICENSE_FILE,
    "XILINX_XRT": XRT_DIR,
}

XCHESS_ARGS = [
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
    "+s",  # print commands being executed
    # "+f", only run LLVM frontend (emits IR)
    "+w",
    str(WORKDIR),
]


def aie_buffer(tile, type):
    return aie.buffer(type, tile)


def aie_lock(tile, lock_id, init, sym_name):
    return aie.lock(tile, lock_id=lock_id, init=init, sym_name=sym_name)


def aie_use_lock(lock, action, value):
    return aie.use_lock(lock, value, action)


def memref_global(sym_visibility, sym_name, type):
    return memref.global_(sym_name, type, sym_visibility=sym_visibility)


IMAGE_WIDTH = 128
IMAGE_HEIGHT = 16
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT

TILE_WIDTH = 16
TILE_HEIGHT = 8
TILE_SIZE = TILE_WIDTH * TILE_HEIGHT

NUM_3D = IMAGE_WIDTH / TILE_WIDTH
NUM_4D = IMAGE_HEIGHT / TILE_HEIGHT

M = IMAGE_HEIGHT
K = IMAGE_WIDTH
N = 128
m = 64
k = 32
n = 64
r = 4
s = 4
t = 4
word_size_in = 2
word_size_out = 2

A_sz_in_i32s = M * K * word_size_in // 4
B_sz_in_i32s = K * N * word_size_in // 4
C_sz_in_bytes = M * N * word_size_out
C_sz_in_i32s = C_sz_in_bytes // 4

M_div_m = M // m
K_div_k = K // k
N_div_n = N // n
slices = M_div_m * N_div_n

# Matrix A: MxK, submatrices a: mxk
k_in_i32s = k * word_size_in // 4
K_in_i32s = K * word_size_in // 4

# Matrix B: KxN, submatrices b: kxn
n_in_i32s = n * word_size_in // 4
N_in_i32s = N * word_size_in // 4
k_x_N_in_i32s = k * N * word_size_in // 4

# Output Matrix C: MxN
n_in_i32s_out = n * word_size_out // 4
N_in_i32s_out = N * word_size_out // 4
m_x_N_in_i32s_out = m * N * word_size_out // 4


# CHECK-LABEL: mb_matrix_add
@lambda x: print(x.__name__) or x()
def mb_matrix_add():
    with Context(), Location.unknown():
        i16 = T.i16()

        @builtin.module
        def module():
            @builtin.module(sym_name="mod_aie")
            def mod_aie():
                @device(AIEDevice.ipu)
                def ipu():
                    tile_0_0 = aie.tile(0, 0)
                    tile_0_2 = aie.tile(0, 2)

                    # forward
                    aie.flow(tile_0_0, DMA, 0, tile_0_2, DMA, 0)
                    aie.flow(tile_0_0, DMA, 1, tile_0_2, DMA, 1)
                    # backward
                    aie.flow(tile_0_2, DMA, 0, tile_0_0, DMA, 0)
                    aie.flow(tile_0_2, DMA, 1, tile_0_0, DMA, 1)

                    aie.shim_dma_allocation("inA", MM2S, 0, 0)
                    aie.shim_dma_allocation("inB", MM2S, 1, 0)
                    aie.shim_dma_allocation("outC", S2MM, 0, 0)

                    memref_global("public", "inA", T.memref(64, 32, i16))
                    memref_global("public", "inB", T.memref(32, 64, i16))
                    memref_global("public", "outC", T.memref(64, 64, i16))

                    @func.func(emit=True)
                    def bobsyouruncle(
                        arg0: T.memref(8192, T.i32()),
                        arg1: T.memref(8192, T.i32()),
                        arg2: T.memref(8192, T.i32()),
                    ):
                        ipu_dma_memcpy_nd_(
                            0,
                            0,
                            arg0,
                            offsets=[0, 0, 0, 0],
                            wraps=[2, 4, 64, 16],
                            strides=[0, 16, 64],
                            id=1,
                            metadata="inA",
                        )
                        ipu_dma_memcpy_nd_(
                            0,
                            0,
                            arg0,
                            offsets=[0, 0, 4096, 4096],
                            wraps=[2, 4, 64, 16],
                            strides=[0, 16, 64],
                            id=3,
                            metadata="inA",
                        )
                        ipu_dma_memcpy_nd_(
                            0,
                            0,
                            arg1,
                            offsets=[0, 0, 0, 0],
                            wraps=[2, 4, 32, 32],
                            strides=[32, 2048, 64],
                            id=2,
                            metadata="inB",
                        )
                        ipu_dma_memcpy_nd_(
                            0,
                            0,
                            arg1,
                            offsets=[0, 0, 0, 0],
                            wraps=[2, 4, 32, 32],
                            strides=[32, 2048, 64],
                            id=4,
                            metadata="inB",
                        )
                        ipu_dma_memcpy_nd_(
                            0,
                            0,
                            arg2,
                            offsets=[0, 0, 0, 0],
                            wraps=[2, 2, 64, 32],
                            strides=[4096, 32, 64],
                            id=0,
                            metadata="outC",
                        )
                        ipu_sync(
                            channel=0,
                            column=0,
                            column_num=1,
                            direction=0,
                            row=0,
                            row_num=1,
                        )

                    ping_a = aie_buffer(tile_0_2, T.memref(128, T.i32()))
                    pong_a = aie_buffer(tile_0_2, T.memref(128, T.i32()))
                    ping_b = aie_buffer(tile_0_2, T.memref(128, T.i32()))
                    pong_b = aie_buffer(tile_0_2, T.memref(128, T.i32()))
                    ping_c = aie_buffer(tile_0_2, T.memref(128, T.i32()))
                    pong_c = aie_buffer(tile_0_2, T.memref(128, T.i32()))

                    lock_0_2_0 = aie.lock(tile_0_2, lock_id=0)
                    lock_0_2_1 = aie.lock(tile_0_2, lock_id=1)
                    lock_0_2_2 = aie.lock(tile_0_2, lock_id=2)
                    lock_0_2_3 = aie.lock(tile_0_2, lock_id=3)
                    lock_0_2_4 = aie.lock(tile_0_2, lock_id=4)
                    lock_0_2_5 = aie.lock(tile_0_2, lock_id=5)

                    @aie.mem(tile_0_2)
                    def m02():
                        bd0, src1 = aie.dma_start(S2MM, 0)
                        with src1:
                            bd4, dma0 = aie.dma_start(S2MM, 1)
                        with dma0:
                            bd2, end = aie.dma_start(MM2S, 0)

                        # ping/pong a
                        with bd0:
                            aie_use_lock(lock_0_2_0, Acquire, 0)
                            aie.dma_bd(ping_a, 0, 128)
                            aie_use_lock(lock_0_2_0, Release, 1)
                            bd1 = aie.next_bd()
                        with bd1:
                            aie_use_lock(lock_0_2_1, Acquire, 0)
                            aie.dma_bd(pong_a, 0, 128)
                            aie_use_lock(lock_0_2_1, Release, 1)
                            aie.next_bd(bd0)

                        # ping/pong b
                        with bd4:
                            aie_use_lock(lock_0_2_4, Acquire, 0)
                            aie.dma_bd(ping_b, 0, 128)
                            aie_use_lock(lock_0_2_4, Release, 1)
                            bd5 = aie.next_bd()
                        with bd5:
                            aie_use_lock(lock_0_2_5, Acquire, 0)
                            aie.dma_bd(pong_b, 0, 128)
                            aie_use_lock(lock_0_2_5, Release, 1)
                            aie.next_bd(bd4)

                        # ping/pong c
                        with bd2:
                            aie_use_lock(lock_0_2_2, Acquire, 1)
                            aie.dma_bd(ping_c, 0, 128)
                            aie_use_lock(lock_0_2_2, Release, 0)
                            bd3 = aie.next_bd()
                        with bd3:
                            aie_use_lock(lock_0_2_3, Acquire, 1)
                            aie.dma_bd(pong_c, 0, 128)
                            aie_use_lock(lock_0_2_3, Release, 0)
                            aie.next_bd(bd2)
                        with end:
                            aie.end()

                    @aie.core(tile_0_2)
                    def core():
                        for i in range_(0, 16):
                            aie_use_lock(lock_0_2_0, Acquire, 1)
                            aie_use_lock(lock_0_2_4, Acquire, 1)
                            aie_use_lock(lock_0_2_2, Acquire, 0)
                            for arg3 in range_(0, 128):
                                v0 = memref.load(ping_a, [arg3])
                                v1 = memref.load(ping_b, [arg3])
                                v2 = arith.addi(v0, v1)
                                memref.store(v2, ping_c, [arg3])
                                yield_([])
                            aie_use_lock(lock_0_2_0, Release, 0)
                            aie_use_lock(lock_0_2_4, Release, 0)
                            aie_use_lock(lock_0_2_2, Release, 1)

                            aie_use_lock(lock_0_2_1, Acquire, 1)
                            aie_use_lock(lock_0_2_5, Acquire, 1)
                            aie_use_lock(lock_0_2_3, Acquire, 0)
                            for arg4 in range_(0, 128):
                                v3 = memref.load(pong_a, [arg4])
                                v4 = memref.load(pong_b, [arg4])
                                v5 = arith.addi(v3, v4)
                                memref.store(v5, pong_c, [arg4])
                                yield_([])
                            aie_use_lock(lock_0_2_1, Release, 0)
                            aie_use_lock(lock_0_2_5, Release, 0)
                            aie_use_lock(lock_0_2_3, Release, 1)
                            yield_([])

                # CHECK: module {
                # CHECK:   aie.device(ipu) {
                # CHECK:     %tile_0_0 = aie.tile(0, 0)
                # CHECK:     %tile_0_2 = aie.tile(0, 2)
                # CHECK:     aie.flow(%tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
                # CHECK:     aie.flow(%tile_0_0, DMA : 1, %tile_0_2, DMA : 1)
                # CHECK:     aie.flow(%tile_0_2, DMA : 0, %tile_0_0, DMA : 0)
                # CHECK:     aie.flow(%tile_0_2, DMA : 1, %tile_0_0, DMA : 1)
                # CHECK:     %buffer_0_2 = aie.buffer(%tile_0_2) : memref<128xi32>
                # CHECK:     %buffer_0_2_0 = aie.buffer(%tile_0_2) : memref<128xi32>
                # CHECK:     %buffer_0_2_1 = aie.buffer(%tile_0_2) : memref<128xi32>
                # CHECK:     %buffer_0_2_2 = aie.buffer(%tile_0_2) : memref<128xi32>
                # CHECK:     %buffer_0_2_3 = aie.buffer(%tile_0_2) : memref<128xi32>
                # CHECK:     %buffer_0_2_4 = aie.buffer(%tile_0_2) : memref<128xi32>
                # CHECK:     %lock_0_2 = aie.lock(%tile_0_2, 0)
                # CHECK:     %lock_0_2_5 = aie.lock(%tile_0_2, 1)
                # CHECK:     %lock_0_2_6 = aie.lock(%tile_0_2, 2)
                # CHECK:     %lock_0_2_7 = aie.lock(%tile_0_2, 3)
                # CHECK:     %lock_0_2_8 = aie.lock(%tile_0_2, 4)
                # CHECK:     %lock_0_2_9 = aie.lock(%tile_0_2, 5)
                # CHECK:     %mem_0_2 = aie.mem(%tile_0_2) {
                # CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb3, ^bb1)
                # CHECK:     ^bb1:  // pred: ^bb0
                # CHECK:       %1 = aie.dma_start(S2MM, 1, ^bb5, ^bb2)
                # CHECK:     ^bb2:  // pred: ^bb1
                # CHECK:       %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
                # CHECK:     ^bb3:  // 2 preds: ^bb0, ^bb4
                # CHECK:       aie.use_lock(%lock_0_2, Acquire, 0)
                # CHECK:       aie.dma_bd(%buffer_0_2 : memref<128xi32>, 0, 128)
                # CHECK:       aie.use_lock(%lock_0_2, Release, 1)
                # CHECK:       aie.next_bd ^bb4
                # CHECK:     ^bb4:  // pred: ^bb3
                # CHECK:       aie.use_lock(%lock_0_2_5, Acquire, 0)
                # CHECK:       aie.dma_bd(%buffer_0_2_0 : memref<128xi32>, 0, 128)
                # CHECK:       aie.use_lock(%lock_0_2_5, Release, 1)
                # CHECK:       aie.next_bd ^bb3
                # CHECK:     ^bb5:  // 2 preds: ^bb1, ^bb6
                # CHECK:       aie.use_lock(%lock_0_2_8, Acquire, 0)
                # CHECK:       aie.dma_bd(%buffer_0_2_1 : memref<128xi32>, 0, 128)
                # CHECK:       aie.use_lock(%lock_0_2_8, Release, 1)
                # CHECK:       aie.next_bd ^bb6
                # CHECK:     ^bb6:  // pred: ^bb5
                # CHECK:       aie.use_lock(%lock_0_2_9, Acquire, 0)
                # CHECK:       aie.dma_bd(%buffer_0_2_2 : memref<128xi32>, 0, 128)
                # CHECK:       aie.use_lock(%lock_0_2_9, Release, 1)
                # CHECK:       aie.next_bd ^bb5
                # CHECK:     ^bb7:  // 2 preds: ^bb2, ^bb8
                # CHECK:       aie.use_lock(%lock_0_2_6, Acquire, 1)
                # CHECK:       aie.dma_bd(%buffer_0_2_3 : memref<128xi32>, 0, 128)
                # CHECK:       aie.use_lock(%lock_0_2_6, Release, 0)
                # CHECK:       aie.next_bd ^bb8
                # CHECK:     ^bb8:  // pred: ^bb7
                # CHECK:       aie.use_lock(%lock_0_2_7, Acquire, 1)
                # CHECK:       aie.dma_bd(%buffer_0_2_4 : memref<128xi32>, 0, 128)
                # CHECK:       aie.use_lock(%lock_0_2_7, Release, 0)
                # CHECK:       aie.next_bd ^bb7
                # CHECK:     ^bb9:  // pred: ^bb2
                # CHECK:       aie.end
                # CHECK:     }
                # CHECK:     %core_0_2 = aie.core(%tile_0_2) {
                # CHECK:       %c0 = arith.constant 0 : index
                # CHECK:       %c16 = arith.constant 16 : index
                # CHECK:       %c1 = arith.constant 1 : index
                # CHECK:       scf.for %arg0 = %c0 to %c16 step %c1 {
                # CHECK:         aie.use_lock(%lock_0_2, Acquire, 1)
                # CHECK:         aie.use_lock(%lock_0_2_8, Acquire, 1)
                # CHECK:         aie.use_lock(%lock_0_2_6, Acquire, 0)
                # CHECK:         %c0_10 = arith.constant 0 : index
                # CHECK:         %c128 = arith.constant 128 : index
                # CHECK:         %c1_11 = arith.constant 1 : index
                # CHECK:         scf.for %arg1 = %c0_10 to %c128 step %c1_11 {
                # CHECK:           %0 = memref.load %buffer_0_2[%arg1] : memref<128xi32>
                # CHECK:           %1 = memref.load %buffer_0_2_1[%arg1] : memref<128xi32>
                # CHECK:           %2 = arith.addi %0, %1 : i32
                # CHECK:           memref.store %2, %buffer_0_2_3[%arg1] : memref<128xi32>
                # CHECK:         }
                # CHECK:         aie.use_lock(%lock_0_2, Release, 0)
                # CHECK:         aie.use_lock(%lock_0_2_8, Release, 0)
                # CHECK:         aie.use_lock(%lock_0_2_6, Release, 1)
                # CHECK:         aie.use_lock(%lock_0_2_5, Acquire, 1)
                # CHECK:         aie.use_lock(%lock_0_2_9, Acquire, 1)
                # CHECK:         aie.use_lock(%lock_0_2_7, Acquire, 0)
                # CHECK:         %c0_12 = arith.constant 0 : index
                # CHECK:         %c128_13 = arith.constant 128 : index
                # CHECK:         %c1_14 = arith.constant 1 : index
                # CHECK:         scf.for %arg1 = %c0_12 to %c128_13 step %c1_14 {
                # CHECK:           %0 = memref.load %buffer_0_2_0[%arg1] : memref<128xi32>
                # CHECK:           %1 = memref.load %buffer_0_2_2[%arg1] : memref<128xi32>
                # CHECK:           %2 = arith.addi %0, %1 : i32
                # CHECK:           memref.store %2, %buffer_0_2_4[%arg1] : memref<128xi32>
                # CHECK:         }
                # CHECK:         aie.use_lock(%lock_0_2_5, Release, 0)
                # CHECK:         aie.use_lock(%lock_0_2_9, Release, 0)
                # CHECK:         aie.use_lock(%lock_0_2_7, Release, 1)
                # CHECK:       }
                # CHECK:       aie.end
                # CHECK:     }

            # CHECK:   }
            # CHECK: }

    print(module)
