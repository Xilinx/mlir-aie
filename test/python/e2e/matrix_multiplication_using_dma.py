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
from functools import reduce
from pathlib import Path
from pprint import pprint

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
    generate_cdo_direct,
    lock,
    mem,
    memtile_dma,
    tile,
    translate_mlir_to_llvmir,
    aie_llvm_link,
    BDDimLayout,
)
from aie.dialects.aie import translate_aie_vec_to_cpp
from aie.dialects.aiex import ipu_sync, ipu_dma_memcpy_nd
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


def memref_global(sym_visibility, sym_name, type):
    return memref.global_(sym_name, type, sym_visibility=sym_visibility)


def aie_buffer(tile, sym_name, type):
    return aie.buffer(type, tile, sym_name=sym_name)


def aie_lock(tile, lock_id, init, sym_name):
    return aie.lock(tile, lock_id=lock_id, init=init, sym_name=sym_name)


def aie_use_lock(lock, action, value):
    return aie.use_lock(lock, value, action)


scale = 8

M = 128 // scale
K = 128 // scale
N = 128 // scale
m = 64 // scale
k = 32 // scale
n = 64 // scale
r = 4
s = 4
t = 4

i32_size_in_bytes = 4
# i16 = 2 bytes
word_size = 2  # bytes


def four_by_four_tiling(*matrix_dims):
    return [
        [(matrix_dims[1] // 4), (matrix_dims[0] // word_size) * 4],
        [(matrix_dims[0] // word_size // 4), 4],
        [4, (matrix_dims[1] // word_size)],
        [4, 1],
    ]


tiling = four_by_four_tiling(M, K)
print("sizes=", [a for a, b in tiling])
print("strides=", [b for a, b in tiling])


@func.func(emit=False, sym_visibility="private")
def matmul_i16_i16(
    A: f"T.memref({m}, {k}, T.i16())",
    B: f"T.memref({k}, {n}, T.i16())",
    C: f"T.memref({m}, {n}, T.i16())",
):
    linalg.matmul(A, B, C)


A_sz_in_i32s = M * K * word_size // i32_size_in_bytes
B_sz_in_i32s = K * N * word_size // i32_size_in_bytes
C_sz_in_bytes = M * N * word_size
C_sz_in_i32s = C_sz_in_bytes // i32_size_in_bytes

M_div_m = M // m
K_div_k = K // k
N_div_n = N // n
slices = M_div_m * N_div_n

# Matrix A: MxK, submatrices a: mxk
k_in_i32s = k * word_size // i32_size_in_bytes
K_in_i32s = K * word_size // i32_size_in_bytes

# Matrix B: KxN, submatrices b: kxn
n_in_i32s = n * word_size // i32_size_in_bytes
N_in_i32s = N * word_size // i32_size_in_bytes
k_x_N_in_i32s = k * N * word_size // i32_size_in_bytes

# Output Matrix C: MxN
n_in_i32s_out = n * word_size // i32_size_in_bytes
N_in_i32s_out = N * word_size // i32_size_in_bytes
m_x_N_in_i32s_out = m * N * word_size // i32_size_in_bytes


def num_els(shape):
    return reduce(lambda acc, v: acc * v, shape, 1)


# CHECK-LABEL: matrix_multiplication_using_dma
@lambda x: print(x.__name__) or x()
def matrix_multiplication_using_dma():
    with Context(), Location.unknown():
        i16 = T.i16()

        @builtin.module
        def mod():
            @builtin.module(sym_name="mod_aie")
            def mod_aie():
                @device(AIEDevice.ipu)
                def ipu():
                    matmul_i16_i16.emit(decl=True)

                    tile_0_0 = tile(0, 0)
                    tile_0_1 = tile(0, 1)
                    tile_0_2 = tile(0, 2)

                    # forward
                    aie.flow(tile_0_0, DMA, 0, tile_0_1, DMA, 0)
                    aie.flow(tile_0_0, DMA, 1, tile_0_1, DMA, 1)

                    aie.flow(tile_0_1, DMA, 0, tile_0_2, DMA, 0)
                    aie.flow(tile_0_1, DMA, 1, tile_0_2, DMA, 1)

                    # backward
                    aie.flow(tile_0_2, DMA, 0, tile_0_1, DMA, 2)
                    aie.flow(tile_0_1, DMA, 2, tile_0_0, DMA, 0)

                    aie.shim_dma_allocation("inA", MM2S, 0, 0)
                    aie.shim_dma_allocation("inB", MM2S, 1, 0)
                    aie.shim_dma_allocation("outC", S2MM, 0, 0)

                    memref_global("public", "inA", T.memref(m, k, i16))
                    memref_global("public", "inB", T.memref(k, n, i16))
                    memref_global("public", "outC", T.memref(m, n, i16))

                    @func.func(emit=True)
                    def bobsyouruncle(
                        A: T.memref(A_sz_in_i32s, T.i32()),
                        B: T.memref(B_sz_in_i32s, T.i32()),
                        C: T.memref(C_sz_in_i32s, T.i32()),
                    ):
                        global M, N, K
                        # only do 5 tile rows at a time before synchronizing, so we can reuse BDs
                        rows_per_block = 5
                        for tile_row_block in range(
                            (M_div_m + rows_per_block - 1) // rows_per_block
                        ):
                            C_row_offset_in_i32s = (
                                tile_row_block
                                * rows_per_block
                                * m
                                * N
                                * word_size
                                // i32_size_in_bytes
                            )
                            num_tile_rows = min(
                                rows_per_block,
                                M_div_m - tile_row_block * rows_per_block,
                            )
                            ipu_dma_memcpy_nd(
                                metadata="outC",
                                bd_id=0,
                                mem=C,
                                offsets=[0, 0, 0, C_row_offset_in_i32s],
                                sizes=[4, 2, 4, 4],
                                strides=[32, 4, 8],
                            )
                            for tile_row in range(num_tile_rows):
                                A_row_offset_in_i32s = (
                                    ((tile_row_block * rows_per_block) + tile_row)
                                    * m
                                    * K
                                    * word_size
                                    // i32_size_in_bytes
                                )
                                ipu_dma_memcpy_nd(
                                    metadata="inA",
                                    bd_id=2 * tile_row + 1,
                                    mem=A,
                                    offsets=[0, 0, 0, A_row_offset_in_i32s],
                                    sizes=[4, 2, 4, 4],
                                    strides=[32, 4, 8],
                                )
                                ipu_dma_memcpy_nd(
                                    metadata="inB",
                                    bd_id=2 * tile_row + 2,
                                    mem=B,
                                    sizes=[4, 2, 4, 4],
                                    strides=[32, 4, 8],
                                )

                            ipu_sync(column=0, row=0, direction=0, channel=0)

                    inA_cons_buff_0 = aie_buffer(
                        tile_0_1, "inA_cons_buff_0", T.memref(m, k, i16)
                    )
                    inA_cons_buff_1 = aie_buffer(
                        tile_0_1, "inA_cons_buff_1", T.memref(m, k, i16)
                    )
                    inB_cons_buff_0 = aie_buffer(
                        tile_0_1, "inB_cons_buff_0", T.memref(k, n, i16)
                    )
                    inB_cons_buff_1 = aie_buffer(
                        tile_0_1, "inB_cons_buff_1", T.memref(k, n, i16)
                    )
                    memC_cons_buff_0 = aie_buffer(
                        tile_0_1, "memC_cons_buff_0", T.memref(m, n, i16)
                    )
                    memC_cons_buff_1 = aie_buffer(
                        tile_0_1, "memC_cons_buff_1", T.memref(m, n, i16)
                    )

                    inA_cons_cons_lock = aie_lock(
                        tile_0_1, 1, init=0, sym_name="inA_cons_cons_lock"
                    )
                    inA_cons_prod_lock = aie_lock(
                        tile_0_1, 0, init=2, sym_name="inA_cons_prod_lock"
                    )
                    inB_cons_cons_lock = aie_lock(
                        tile_0_1, 3, init=0, sym_name="inB_cons_cons_lock"
                    )
                    inB_cons_prod_lock = aie_lock(
                        tile_0_1, 2, init=2, sym_name="inB_cons_prod_lock"
                    )
                    memC_cons_cons_lock = aie_lock(
                        tile_0_1, 5, init=0, sym_name="memC_cons_cons_lock"
                    )
                    memC_cons_prod_lock = aie_lock(
                        tile_0_1, 4, init=2, sym_name="memC_cons_prod_lock"
                    )

                    @memtile_dma(tile_0_1)
                    def memtile_dma_0_1():
                        global r, s, t
                        bb1, bb3 = aie.dma_start(S2MM, 0)
                        with bb(bb1):  # 2 preds: ^bb0, ^bb2
                            aie_use_lock(inA_cons_prod_lock, AcquireGreaterEqual, 1)
                            aie.dma_bd(inA_cons_buff_0, 0, inA_cons_buff_0.n_elements)
                            aie_use_lock(inA_cons_cons_lock, Release, 1)
                            bb2 = aie.next_bd()
                        with bb(bb2):  # pred: ^bb1
                            aie_use_lock(inA_cons_prod_lock, AcquireGreaterEqual, 1)
                            aie.dma_bd(inA_cons_buff_1, 0, inA_cons_buff_1.n_elements)
                            aie_use_lock(inA_cons_cons_lock, Release, 1)
                            aie.next_bd(bb1)

                        with bb(bb3):  # pred: ^bb0
                            bb4, bb6 = aie.dma_start(MM2S, 0)
                        with bb(bb4):  # 2 preds: ^bb3, ^bb5
                            aie_use_lock(inA_cons_cons_lock, AcquireGreaterEqual, 1)
                            aie.dma_bd(
                                inA_cons_buff_0,
                                0,
                                inA_cons_buff_0.n_elements,
                                dimensions=[
                                    BDDimLayout(
                                        size=m // r,
                                        stride=(r * k * word_size // i32_size_in_bytes),
                                    ),
                                    BDDimLayout(
                                        size=k // s,
                                        stride=s * word_size // i32_size_in_bytes,
                                    ),
                                    BDDimLayout(
                                        size=r,
                                        stride=k * word_size // i32_size_in_bytes,
                                    ),
                                    BDDimLayout(
                                        size=s * word_size // i32_size_in_bytes,
                                        stride=1,
                                    ),
                                ],
                            )
                            aie_use_lock(inA_cons_prod_lock, Release, 1)
                            bb5 = aie.next_bd()
                        with bb(bb5):  # pred: ^bb4
                            aie_use_lock(inA_cons_cons_lock, AcquireGreaterEqual, 1)
                            aie.dma_bd(
                                inA_cons_buff_1,
                                0,
                                inA_cons_buff_1.n_elements,
                                dimensions=[
                                    BDDimLayout(
                                        size=m // r,
                                        stride=(r * k * word_size // i32_size_in_bytes),
                                    ),
                                    BDDimLayout(
                                        size=k // s,
                                        stride=s * word_size // i32_size_in_bytes,
                                    ),
                                    BDDimLayout(
                                        size=r,
                                        stride=k * word_size // i32_size_in_bytes,
                                    ),
                                    BDDimLayout(
                                        size=s * word_size // i32_size_in_bytes,
                                        stride=1,
                                    ),
                                ],
                            )
                            aie_use_lock(inA_cons_prod_lock, Release, 1)
                            aie.next_bd(bb4)

                        with bb(bb6):  # pred: ^bb3
                            bb7, bb9 = aie.dma_start(S2MM, 1)
                        with bb(bb7):  # 2 preds: ^bb6, ^bb8
                            aie_use_lock(inB_cons_prod_lock, AcquireGreaterEqual, 1)
                            aie.dma_bd(inB_cons_buff_0, 0, inB_cons_buff_0.n_elements)
                            aie_use_lock(inB_cons_cons_lock, Release, 1)
                            bb8 = aie.next_bd()
                        with bb(bb8):  # pred: ^bb7
                            aie_use_lock(inB_cons_prod_lock, AcquireGreaterEqual, 1)
                            aie.dma_bd(inB_cons_buff_1, 0, inB_cons_buff_1.n_elements)
                            aie_use_lock(inB_cons_cons_lock, Release, 1)
                            aie.next_bd(bb7)

                        with bb(bb9):  # pred: ^bb6
                            bb10, bb12 = aie.dma_start(MM2S, 1)
                        with bb(bb10):  # 2 preds: ^bb9, ^bb11
                            aie_use_lock(inB_cons_cons_lock, AcquireGreaterEqual, 1)
                            aie.dma_bd(
                                inB_cons_buff_0,
                                0,
                                inB_cons_buff_0.n_elements,
                                dimensions=[
                                    BDDimLayout(
                                        size=k // s,
                                        stride=(s * n * word_size // i32_size_in_bytes),
                                    ),
                                    BDDimLayout(
                                        size=n // t,
                                        stride=t * word_size // i32_size_in_bytes,
                                    ),
                                    BDDimLayout(
                                        size=s,
                                        stride=n * word_size // i32_size_in_bytes,
                                    ),
                                    BDDimLayout(
                                        size=t * word_size // i32_size_in_bytes,
                                        stride=1,
                                    ),
                                ],
                            )
                            aie_use_lock(inB_cons_prod_lock, Release, 1)
                            bb11 = aie.next_bd()
                        with bb(bb11):  # pred: ^bb10
                            aie_use_lock(inB_cons_cons_lock, AcquireGreaterEqual, 1)
                            aie.dma_bd(
                                inB_cons_buff_1,
                                0,
                                inB_cons_buff_1.n_elements,
                                dimensions=[
                                    BDDimLayout(
                                        size=k // s,
                                        stride=(s * n * word_size // i32_size_in_bytes),
                                    ),
                                    BDDimLayout(
                                        size=n // t,
                                        stride=t * word_size // i32_size_in_bytes,
                                    ),
                                    BDDimLayout(
                                        size=s,
                                        stride=n * word_size // i32_size_in_bytes,
                                    ),
                                    BDDimLayout(
                                        size=t * word_size // i32_size_in_bytes,
                                        stride=1,
                                    ),
                                ],
                            )
                            aie_use_lock(inB_cons_prod_lock, Release, 1)
                            aie.next_bd(bb10)

                        with bb(bb12):  # pred: ^bb9
                            bb13, bb15 = aie.dma_start(S2MM, 2)
                        with bb(bb13):  # 2 preds: ^bb12, ^bb14
                            aie_use_lock(memC_cons_prod_lock, AcquireGreaterEqual, 1)
                            aie.dma_bd(memC_cons_buff_0, 0, memC_cons_buff_0.n_elements)
                            aie_use_lock(memC_cons_cons_lock, Release, 1)
                            bb14 = aie.next_bd()
                        with bb(bb14):  # pred: ^bb13
                            aie_use_lock(memC_cons_prod_lock, AcquireGreaterEqual, 1)
                            aie.dma_bd(memC_cons_buff_1, 0, memC_cons_buff_1.n_elements)
                            aie_use_lock(memC_cons_cons_lock, Release, 1)
                            aie.next_bd(bb13)

                        with bb(bb15):  # pred: ^bb12
                            bb16, bb18 = aie.dma_start(MM2S, 2)
                        with bb(bb16):  # 2 preds: ^bb15, ^bb17
                            aie_use_lock(memC_cons_cons_lock, AcquireGreaterEqual, 1)
                            aie.dma_bd(
                                memC_cons_buff_0,
                                0,
                                memC_cons_buff_0.n_elements,
                                dimensions=[
                                    BDDimLayout(
                                        size=m // r,
                                        stride=(r * n * word_size // i32_size_in_bytes),
                                    ),
                                    BDDimLayout(
                                        size=r,
                                        stride=t * word_size // i32_size_in_bytes,
                                    ),
                                    BDDimLayout(
                                        size=n // t,
                                        stride=(r * t * word_size // i32_size_in_bytes),
                                    ),
                                    BDDimLayout(
                                        size=t * word_size // i32_size_in_bytes,
                                        stride=1,
                                    ),
                                ],
                            )
                            aie_use_lock(memC_cons_prod_lock, Release, 1)
                            bb17 = aie.next_bd()
                        with bb(bb17):  # pred: ^bb16
                            aie_use_lock(memC_cons_cons_lock, AcquireGreaterEqual, 1)
                            aie.dma_bd(
                                memC_cons_buff_1,
                                0,
                                memC_cons_buff_1.n_elements,
                                dimensions=[
                                    BDDimLayout(
                                        size=m // r,
                                        stride=(r * n * word_size // i32_size_in_bytes),
                                    ),
                                    BDDimLayout(
                                        size=r,
                                        stride=t * word_size // i32_size_in_bytes,
                                    ),
                                    BDDimLayout(
                                        size=n // t,
                                        stride=r * t * word_size // i32_size_in_bytes,
                                    ),
                                    BDDimLayout(
                                        size=t * word_size // i32_size_in_bytes,
                                        stride=1,
                                    ),
                                ],
                            )
                            aie_use_lock(memC_cons_prod_lock, Release, 1)
                            aie.next_bd(bb16)

                        with bb(bb18):  # pred: ^bb15
                            aie.end()

                    memA_cons_buff_0 = aie_buffer(
                        tile_0_2, "memA_cons_buff_0", T.memref(m, k, i16)
                    )
                    memA_cons_buff_1 = aie_buffer(
                        tile_0_2, "memA_cons_buff_1", T.memref(m, k, i16)
                    )
                    memB_cons_buff_0 = aie_buffer(
                        tile_0_2, "memB_cons_buff_0", T.memref(k, n, i16)
                    )
                    memB_cons_buff_1 = aie_buffer(
                        tile_0_2, "memB_cons_buff_1", T.memref(k, n, i16)
                    )
                    memC_buff_0 = aie_buffer(
                        tile_0_2, "memC_buff_0", T.memref(m, n, i16)
                    )
                    memC_buff_1 = aie_buffer(
                        tile_0_2, "memC_buff_1", T.memref(m, n, i16)
                    )

                    memA_cons_cons_lock = aie_lock(
                        tile_0_2, 1, init=0, sym_name="memA_cons_cons_lock"
                    )
                    memA_cons_prod_lock = aie_lock(
                        tile_0_2, 0, init=2, sym_name="memA_cons_prod_lock"
                    )
                    memB_cons_cons_lock = aie_lock(
                        tile_0_2, 3, init=0, sym_name="memB_cons_cons_lock"
                    )
                    memB_cons_prod_lock = aie_lock(
                        tile_0_2, 2, init=2, sym_name="memB_cons_prod_lock"
                    )
                    memC_cons_lock = aie_lock(
                        tile_0_2, 5, init=0, sym_name="memC_cons_lock"
                    )
                    memC_prod_lock = aie_lock(
                        tile_0_2, 4, init=2, sym_name="memC_prod_lock"
                    )

                    @aie.mem(tile_0_2)
                    def mem_0_2():
                        # double buffer memA
                        bb1, bb3 = aie.dma_start(S2MM, 0)
                        with bb(bb1):  # 2 preds: ^bb0, ^bb2
                            aie_use_lock(memA_cons_prod_lock, AcquireGreaterEqual, 1)
                            aie.dma_bd(memA_cons_buff_0, 0, memA_cons_buff_0.n_elements)
                            aie_use_lock(memA_cons_cons_lock, Release, 1)
                            bb2 = aie.next_bd()
                        with bb(bb2):  # pred: ^bb1
                            aie_use_lock(memA_cons_prod_lock, AcquireGreaterEqual, 1)
                            aie.dma_bd(memA_cons_buff_1, 0, memA_cons_buff_1.n_elements)
                            aie_use_lock(memA_cons_cons_lock, Release, 1)
                            aie.next_bd(bb1)

                        # double buffer memB
                        with bb(bb3):  # pred: ^bb0
                            bb4, bb6 = aie.dma_start(S2MM, 1)
                        with bb(bb4):  # 2 preds: ^bb3, ^bb5
                            aie_use_lock(memB_cons_prod_lock, AcquireGreaterEqual, 1)
                            aie.dma_bd(memB_cons_buff_0, 0, memB_cons_buff_0.n_elements)
                            aie_use_lock(memB_cons_cons_lock, Release, 1)
                            bb5 = aie.next_bd()
                        with bb(bb5):  # pred: ^bb4
                            aie_use_lock(memB_cons_prod_lock, AcquireGreaterEqual, 1)
                            aie.dma_bd(memB_cons_buff_1, 0, memB_cons_buff_1.n_elements)
                            aie_use_lock(memB_cons_cons_lock, Release, 1)
                            aie.next_bd(bb4)

                        # double buffer memC
                        with bb(bb6):  # pred: ^bb3
                            bb7, bb9 = aie.dma_start(MM2S, 0)
                        with bb(bb7):  # 2 preds: ^bb6, ^bb8
                            aie_use_lock(memC_cons_lock, AcquireGreaterEqual, 1)
                            aie.dma_bd(memC_buff_0, 0, memC_buff_0.n_elements)
                            aie_use_lock(memC_prod_lock, Release, 1)
                            bb8 = aie.next_bd()
                        with bb(bb8):  # pred: ^bb7
                            aie_use_lock(memC_cons_lock, AcquireGreaterEqual, 1)
                            aie.dma_bd(memC_buff_1, 0, memC_buff_1.n_elements)
                            aie_use_lock(memC_prod_lock, Release, 1)
                            aie.next_bd(bb7)
                        with bb(bb9):  # pred: ^bb6
                            aie.end()

                    @aie.core(tile_0_2)
                    def core():
                        # TODO(max): use while-true?
                        for arg0 in range_(0, 2 << 31):
                            for arg1 in range_(0, slices, 2):
                                aie_use_lock(memC_prod_lock, AcquireGreaterEqual, 1)

                                for arg2 in range_(0, K_div_k, 2):
                                    aie_use_lock(
                                        memA_cons_cons_lock, AcquireGreaterEqual, 1
                                    )
                                    aie_use_lock(
                                        memB_cons_cons_lock, AcquireGreaterEqual, 1
                                    )

                                    matmul_i16_i16(
                                        memA_cons_buff_0, memB_cons_buff_0, memC_buff_0
                                    )

                                    aie_use_lock(memA_cons_prod_lock, Release, 1)
                                    aie_use_lock(memB_cons_prod_lock, Release, 1)

                                    aie_use_lock(
                                        memA_cons_cons_lock, AcquireGreaterEqual, 1
                                    )
                                    aie_use_lock(
                                        memB_cons_cons_lock, AcquireGreaterEqual, 1
                                    )

                                    matmul_i16_i16(
                                        memA_cons_buff_1, memB_cons_buff_1, memC_buff_0
                                    )

                                    aie_use_lock(memA_cons_prod_lock, Release, 1)
                                    aie_use_lock(memB_cons_prod_lock, Release, 1)

                                    yield_([])

                                aie_use_lock(memC_cons_lock, Release, 1)
                                aie_use_lock(memC_prod_lock, AcquireGreaterEqual, 1)

                                for arg2 in range_(0, K_div_k, 2):
                                    aie_use_lock(
                                        memA_cons_cons_lock, AcquireGreaterEqual, 1
                                    )
                                    aie_use_lock(
                                        memB_cons_cons_lock, AcquireGreaterEqual, 1
                                    )

                                    matmul_i16_i16(
                                        memA_cons_buff_0, memB_cons_buff_0, memC_buff_1
                                    )

                                    aie_use_lock(memA_cons_prod_lock, Release, 1)
                                    aie_use_lock(memB_cons_prod_lock, Release, 1)

                                    aie_use_lock(
                                        memA_cons_cons_lock, AcquireGreaterEqual, 1
                                    )
                                    aie_use_lock(
                                        memB_cons_cons_lock, AcquireGreaterEqual, 1
                                    )

                                    matmul_i16_i16(
                                        memA_cons_buff_1, memB_cons_buff_1, memC_buff_1
                                    )

                                    aie_use_lock(memA_cons_prod_lock, Release, 1)
                                    aie_use_lock(memB_cons_prod_lock, Release, 1)

                                    yield_([])

                                aie_use_lock(memC_cons_lock, Release, 1)

                                yield_([])
                            yield_([])

            @builtin.module(sym_name="mod_aievecc")
            def mod_aievecc():
                @builtin.module(
                    attrs={"transform.target_tag": StringAttr.get("payload")}
                )
                def payload():
                    matmul_i16_i16.emit(force=True)

                @builtin.module(attrs={"transform.with_named_sequence": UnitAttr.get()})
                def mod_transform():
                    @named_sequence("affine_unroll", [any_op_t()], [])
                    def affine_unroll(target: any_op_t()):
                        func = structured_match(any_op_t(), target, ops=["func.func"])
                        new_func = apply_registered_pass(
                            any_op_t(), func, "convert-linalg-to-affine-loops"
                        )
                        m = structured_match(any_op_t(), new_func, ops=["arith.addi"])
                        # unroll inner loop
                        loop = get_parent_op(pdl.op_t(), m, op_name="affine.for")
                        # loop_unroll(loop, 32 // scale)

                    @named_sequence("affine_super_vectorize", [any_op_t()], [])
                    def super_vectorize(target: any_op_t()):
                        func = structured_match(any_op_t(), target, ops=["func.func"])
                        func = apply_registered_pass(
                            any_op_t(),
                            func,
                            "affine-super-vectorize",
                            # has to match unroll factor (TODO: figure out where error is getting lost if not true)
                            options="virtual-vector-size=32",
                        )
                        mod = apply_registered_pass(
                            any_op_t(),
                            target,
                            "convert-vector-to-aievec",
                            options="aie-target=aieml",
                        )

        mod_aievecc = find_ops(
            mod,
            lambda x: isinstance(x.opview, builtin.ModuleOp)
            and x.opview.sym_name.value == "mod_aievecc",
            single=True,
        )
        affine_loops = run_pipeline(
            mod_aievecc,
            "builtin.module(transform-interpreter{ entry-point=affine_unroll debug-payload-root-tag=payload }, canonicalize, cse)",
        )

        super_vec = run_pipeline(
            affine_loops,
            "builtin.module(transform-interpreter{ entry-point=affine_super_vectorize debug-payload-root-tag=payload }, lower-affine)",
        )

        mod_aievecc = find_ops(
            super_vec.operation,
            lambda x: "transform.target_tag" in x.attributes,
            single=True,
        )
        aievec_cpp = translate_aie_vec_to_cpp(mod_aievecc.operation, aieml=True)
        print(aievec_cpp)
        aievec_cpp = aievec_cpp.replace("void", 'extern "C" void')

        # print(" ".join([f"{k}={v}" for k, v in ENV.items()]))
        with tempfile.NamedTemporaryFile(
            delete=False, mode="w", prefix=f"{WORKDIR}/aievec", suffix=".cpp"
        ) as temp_xchess_input:
            temp_xchess_input.write(aievec_cpp)
            temp_xchess_input.flush()
            cmd = [
                *XCHESS_ARGS,
                "-c",
                "-f",
                "+f",
                "+P",
                "4",
                temp_xchess_input.name,
                "-o",
                temp_xchess_input.name + ".ll",
            ]
            r = subprocess.run(cmd, capture_output=True, cwd=WORKDIR, env=ENV)
            # print(" ".join(r.args), r.stderr, r.stdout)
            maybe_errs = re.findall(
                r"(\d+) errors", r.stdout.decode(), flags=re.MULTILINE
            )
            assert (
                len(maybe_errs) == 1
            ), "couldn't find 'Compilation finished successfully' string"
            if maybe_errs[0] != "0":
                print(r.stdout.decode(), file=sys.stderr)
                print(r.stderr.decode(), file=sys.stderr)
                raise Exception(r.stderr.decode())

        with open(temp_xchess_input.name + ".ll", "r") as temp_xchess_output:
            aievec_ll = temp_xchess_output.read()

        mod_aie = find_ops(
            mod,
            lambda x: isinstance(x.opview, builtin.ModuleOp)
            and x.opview.sym_name.value == "mod_aie",
            single=True,
        )
        input_with_addresses_and_routed = run_pipeline(
            mod_aie,
            "builtin.module(" + ",".join(lower_to_cf_with_addresses_and_routed) + ")",
        )
        # print(input_with_addresses_and_routed)

        generated_ipu_insts = run_pipeline(
            input_with_addresses_and_routed,
            "builtin.module(aie.device(aie-dma-to-ipu))",
        )
        # print(generated_ipu_insts)

        input_opt_with_addresses = run_pipeline(
            input_with_addresses_and_routed,
            "builtin.module(" + ",".join(lower_to_llvm) + ")",
        )
        # print(input_opt_with_addresses)

        input_ll = translate_mlir_to_llvmir(input_opt_with_addresses.operation)
        # print(input_ll)

        # TODO(max): aie_llvm_link does something that then chess-llvm-link isn't happy about?

        with (
            tempfile.NamedTemporaryFile(
                delete=False, mode="w", prefix=f"{WORKDIR}/aie_input", suffix=".ll"
            ) as temp_xchess_llvm_link_aie_input,
            tempfile.NamedTemporaryFile(
                delete=False, mode="w", prefix=f"{WORKDIR}/aievec_input", suffix=".ll"
            ) as temp_xchess_llvm_link_aievec_input,
        ):
            temp_xchess_llvm_link_aie_input.write(chesshack(input_ll))
            temp_xchess_llvm_link_aie_input.flush()
            temp_xchess_llvm_link_aievec_input.write(aievec_ll)
            temp_xchess_llvm_link_aievec_input.flush()

            cmd = [
                f"{AIETOOLS_DIR}/tps/lnx64/target_aie_ml/bin/LNa64bin/chess-llvm-link",
                str(Path(__file__).parent / "chess_intrinsic_wrapper.ll"),
                temp_xchess_llvm_link_aie_input.name,
                temp_xchess_llvm_link_aievec_input.name,
                "--opaque-pointers=1",
                "-S",
                "-o",
                temp_xchess_llvm_link_aie_input.name + "fullylinked.ll",
            ]
            r = subprocess.run(cmd, capture_output=True, cwd=WORKDIR, env=ENV)
            if len(r.stderr) and "error" in r.stderr.decode().lower():
                print(
                    " ".join(r.args), "\n", r.stdout.decode(), "\n", r.stderr.decode()
                )
                raise Exception("failed compile.")
            # print(" ".join(r.args), "\n", r.stdout.decode(), "\n", r.stderr.decode())

        with open(
            temp_xchess_llvm_link_aie_input.name + "fullylinked.ll", "r"
        ) as temp_chess_llvm_link_output:
            aie_ll = temp_chess_llvm_link_output.read()
        # print(aie_ll)

        cmd = [
            *XCHESS_ARGS,
            "-c",  # compile/assemble only, do not link
            temp_xchess_llvm_link_aie_input.name + "fullylinked.ll",
            "-o",
            "input.o",
        ]
        r = subprocess.run(cmd, capture_output=True, cwd=WORKDIR, env=ENV)
        maybe_errs = re.findall(r"(\d+) errors", r.stdout.decode(), flags=re.MULTILINE)
        assert (
            len(maybe_errs) == 1
        ), "couldn't find 'Compilation finished successfully' string"
        if maybe_errs[0] != "0":
            print(r.stdout.decode(), file=sys.stderr)
            print(r.stderr.decode(), file=sys.stderr)
            raise Exception(r.stderr.decode())

        cores = generate_cores_list(str(input_with_addresses_and_routed))
        # print(cores)
        col, row, _ = cores[0]
        core_bcf = generate_bcf(input_with_addresses_and_routed.operation, col, row)
        # print(core_bcf)

        core_name = f"core_{col}_{row}"
        with open(WORKDIR / f"{core_name}.bcf", "w") as f:
            f.write(core_bcf)

        cmd = [
            *XCHESS_ARGS,
            "input.o",
            *extract_input_files(core_bcf),
            "+l",  # linker configuration file
            f"{core_name}.bcf",
            "-o",
            f"{core_name}.elf",
        ]
        # print(f"LD_LIBRARY_PATH={ld_path} PATH={path} {RDI_DATADIR=} {' '.join(cmd)}")
        r = subprocess.run(cmd, capture_output=True, cwd=WORKDIR, env=ENV)
        maybe_errs = re.findall(r"(\d+) errors", r.stdout.decode(), flags=re.MULTILINE)
        assert (
            len(maybe_errs) == 1
        ), "couldn't find 'Compilation finished successfully' string"
        if maybe_errs[0] != "0":
            print(r.stdout.decode(), file=sys.stderr)
            print(r.stderr.decode(), file=sys.stderr)
            raise Exception(r.stderr.decode())

        generate_cdo_direct(input_with_addresses_and_routed.operation, str(WORKDIR))

        with open(WORKDIR / "mem_topology.json", "w") as f:
            json.dump(mem_topology, f, indent=2)
        with open(WORKDIR / "aie_partition.json", "w") as f:
            json.dump(emit_partition(str(mod)), f, indent=2)
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
        subprocess.run(cmd, check=True, cwd=WORKDIR, env=ENV)

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
            WORKDIR / "final.xclbin",
        ]
        subprocess.run(cmd, check=True, cwd=WORKDIR, env=ENV)

        ipu_insts = ipu_instgen(generated_ipu_insts.operation)
        # print("\n".join(ipu_insts))

        if os.getenv("HAS_RYZEN_AI", None) != "echo":
            handle = subprocess.run(
                [
                    "flock",
                    "/tmp/ipu.lock",
                    "/opt/xilinx/xrt/amdaie/setup_xclbin_firmware.sh",
                    "-dev",
                    "Phoenix",
                    "-xclbin",
                    WORKDIR / "final.xclbin",
                ],
                capture_output=True,
                cwd=WORKDIR,
                env=ENV,
            )
            stderr = handle.stderr.decode("utf-8").strip()
            if len(stderr):
                print(f"{stderr=}", file=sys.stderr)
                assert False

            xclbin = XCLBin(f"{WORKDIR / 'final.xclbin'}", "MLIR_AIE")
            ipu_insts = [int(inst, 16) for inst in ipu_insts]
            xclbin.load_ipu_instructions(ipu_insts)

            A_VOLUME = M * K
            B_VOLUME = N * K
            C_VOLUME = M * N

            inps, outps = xclbin.mmap_buffers(
                [(A_VOLUME,), (B_VOLUME,)], [(C_VOLUME,)], np.int16
            )

            wrap_A = np.asarray(inps[0])
            wrap_B = np.asarray(inps[1])
            wrap_C = np.asarray(outps[0])

            # A = np.random.randint(0, 10, A_VOLUME, dtype=np.int16)
            A = np.identity(M, dtype=np.int16).flatten()
            # B = np.random.randint(0, 10, B_VOLUME, dtype=np.int16)
            B = np.identity(N, dtype=np.int16).flatten()
            # B = np.identity(N, dtype=np.int16).flatten()
            C = np.zeros(C_VOLUME, dtype=np.int16)

            np.copyto(wrap_A, A, casting="no")
            np.copyto(wrap_B, B, casting="no")
            np.copyto(wrap_C, C, casting="no")

            xclbin.sync_buffers_to_device()
            xclbin.run()
            xclbin.wait()
            xclbin.sync_buffers_from_device()

            C = A.reshape(M, K) @ B.reshape(N, K)
            out_C = wrap_C.reshape(M, N)
            with np.printoptions(threshold=sys.maxsize, linewidth=2000):
                print(C)
                print(out_C)
            # assert np.allclose(C, out_C)
