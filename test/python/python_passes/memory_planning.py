# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s
import re
from pathlib import Path

from aie.extras.dialects.ext.arith import addi, constant
from aie.extras.dialects.ext.memref import load, store, alloc
from aie.extras.dialects.ext.func import func
from aie.extras.meta import bb
from aie.extras.runtime.passes import run_pipeline
from aie.extras.util import find_ops

# noinspection PyUnresolvedReferences
import aie.dialects.aie
from aie.dialects.aie import (
    AIEDevice,
    CoreOp,
    DMAChannelDir,
    LockAction,
    ObjectFifoType,
    WireBundle,
    buffer,
    core,
    device,
    dma_bd,
    dma_start,
    end as end_,
    external_buffer,
    flow,
    generate_bcf,
    generate_cdo,
    generate_xaie,
    ipu_instgen,
    lock,
    mem,
    next_bd,
    objectfifo,
    objectfifo_link,
    shim_dma,
    tile,
    translate_mlir_to_llvmir,
    use_lock,
)
from aie.dialects.aiex import ipu_dma_memcpy_nd, ipu_sync
from aie.dialects.scf import for_, yield_
from aie.extras import types as T
from aie.ir import TypeAttr
from util import construct_and_print_module

range_ = for_

DMA = WireBundle.DMA
S2MM = DMAChannelDir.S2MM
MM2S = DMAChannelDir.MM2S
Acquire = LockAction.Acquire
Release = LockAction.Release


def prepare_for_chesshack(chess_intrinsic_wrapper):
    chess_intrinsic_wrapper = re.sub(
        r"^target.*", "", chess_intrinsic_wrapper, flags=re.MULTILINE
    )
    chess_intrinsic_wrapper = re.sub(
        r"noalias_sidechannel[^,]*,", "", chess_intrinsic_wrapper
    )
    chess_intrinsic_wrapper = re.sub(r"nocallback[^,]*,", "", chess_intrinsic_wrapper)
    return chess_intrinsic_wrapper


def chesshack(llvmir, chess_intrinsic_wrapper):
    llvmir = llvmir.replace("noundef", "")
    llvmir = re.sub(r"noalias_sidechannel[^,],", "", llvmir)

    # fmt: off
    # await self.do_call(task, ["llvm-link", llvmir_chesshack, self.chess_intrinsic_wrapper, "-S", "-o", llvmir_chesslinked])
    # fmt: on

    llvmir_chesslinked_ = llvmir
    llvmir_chesslinked_ = llvmir_chesslinked_.replace("noundef", "")
    # Formal function argument names not used in older LLVM
    llvmir_chesslinked_ = re.sub(
        r"^define .*@.*",
        lambda m: re.sub(r"%[0-9]*", "", m.group(0)),
        llvmir_chesslinked_,
        flags=re.MULTILINE,
    )
    llvmir_chesslinked_ = (
        llvmir_chesslinked_.replace("mustprogress", "")
        .replace("poison", "undef")
        .replace("nocallback", "")
        .replace("memory(none)", "readnone")
        .replace("memory(read)", "readonly")
        .replace("memory(write)", "writeonly")
        .replace("memory(argmem: readwrite)", "argmemonly")
        .replace("memory(argmem: read)", "argmemonly readonly")
        .replace("memory(argmem: write)", "argmemonly writeonly")
        .replace("memory(inaccessiblemem: readwrite)", "inaccessiblememonly")
        .replace("memory(inaccessiblemem: read)", "inaccessiblememonly readonly")
        .replace("memory(inaccessiblemem: write)", "inaccessiblememonly writeonly")
        .replace(
            "memory(argmem: readwrite, inaccessiblemem: readwrite)",
            "inaccessiblemem_or_argmemonly",
        )
        .replace(
            "memory(argmem: read, inaccessiblemem: read)",
            "inaccessiblemem_or_argmemonly readonly",
        )
        .replace(
            "memory(argmem: write, inaccessiblemem: write)",
            "inaccessiblemem_or_argmemonly writeonly",
        )
    )
    llvmir_chesslinked_ = re.sub(
        r'target triple = "aie.*"',
        'target triple = "pdarch-unknown-unknown-elf"',
        llvmir_chesslinked_,
    )

    return llvmir_chesslinked_


def extract_input_files(core_bcf):
    return re.findall(r"^_include _file (.*)", core_bcf, re.MULTILINE)


# CHECK-LABEL: test_29_mb_matrix_add
# CHECK: module {
# CHECK:   %tile_7_0 = aie.tile(7, 0)
# CHECK:   %tile_7_2 = aie.tile(7, 2)
# CHECK:   aie.flow(%tile_7_0, DMA : 0, %tile_7_2, DMA : 0)
# CHECK:   aie.flow(%tile_7_0, DMA : 1, %tile_7_2, DMA : 1)
# CHECK:   aie.flow(%tile_7_2, DMA : 0, %tile_7_0, DMA : 0)
# CHECK:   aie.flow(%tile_7_2, DMA : 1, %tile_7_0, DMA : 1)
# CHECK:   %buffer_7_2 = aie.buffer(%tile_7_2) {sym_name = "ping_a"} : memref<128xi32>
# CHECK:   %buffer_7_2_0 = aie.buffer(%tile_7_2) {sym_name = "ping_b"} : memref<128xi32>
# CHECK:   %buffer_7_2_1 = aie.buffer(%tile_7_2) {sym_name = "ping_c"} : memref<128xi32>
# CHECK:   %buffer_7_2_2 = aie.buffer(%tile_7_2) {sym_name = "pong_a"} : memref<128xi32>
# CHECK:   %buffer_7_2_3 = aie.buffer(%tile_7_2) {sym_name = "pong_b"} : memref<128xi32>
# CHECK:   %buffer_7_2_4 = aie.buffer(%tile_7_2) {sym_name = "pong_c"} : memref<128xi32>
# CHECK:   %lock_7_2 = aie.lock(%tile_7_2, 0)
# CHECK:   %lock_7_2_5 = aie.lock(%tile_7_2, 1)
# CHECK:   %lock_7_2_6 = aie.lock(%tile_7_2, 2)
# CHECK:   %lock_7_2_7 = aie.lock(%tile_7_2, 3)
# CHECK:   %lock_7_2_8 = aie.lock(%tile_7_2, 4)
# CHECK:   %lock_7_2_9 = aie.lock(%tile_7_2, 5)
# CHECK:   %mem_7_2 = aie.mem(%tile_7_2) {
# CHECK:     %[[VAL_0:.*]] = aie.dma_start(S2MM, 0, ^bb3, ^bb1)
# CHECK:   ^bb1:  // pred: ^bb0
# CHECK:     %[[VAL_1:.*]] = aie.dma_start(S2MM, 1, ^bb5, ^bb2)
# CHECK:   ^bb2:  // pred: ^bb1
# CHECK:     %[[VAL_2:.*]] = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
# CHECK:   ^bb3:  // 2 preds: ^bb0, ^bb4
# CHECK:     aie.use_lock(%lock_7_2, Acquire, 0)
# CHECK:     aie.dma_bd(%buffer_7_2 : memref<128xi32>, 0, 128)
# CHECK:     aie.use_lock(%lock_7_2, Release, 1)
# CHECK:     aie.next_bd ^bb4
# CHECK:   ^bb4:  // pred: ^bb3
# CHECK:     aie.use_lock(%lock_7_2_5, Acquire, 0)
# CHECK:     aie.dma_bd(%buffer_7_2_2 : memref<128xi32>, 0, 128)
# CHECK:     aie.use_lock(%lock_7_2_5, Release, 1)
# CHECK:     aie.next_bd ^bb3
# CHECK:   ^bb5:  // 2 preds: ^bb1, ^bb6
# CHECK:     aie.use_lock(%lock_7_2_8, Acquire, 0)
# CHECK:     aie.dma_bd(%buffer_7_2_0 : memref<128xi32>, 0, 128)
# CHECK:     aie.use_lock(%lock_7_2_8, Release, 1)
# CHECK:     aie.next_bd ^bb6
# CHECK:   ^bb6:  // pred: ^bb5
# CHECK:     aie.use_lock(%lock_7_2_9, Acquire, 0)
# CHECK:     aie.dma_bd(%buffer_7_2_3 : memref<128xi32>, 0, 128)
# CHECK:     aie.use_lock(%lock_7_2_9, Release, 1)
# CHECK:     aie.next_bd ^bb5
# CHECK:   ^bb7:  // 2 preds: ^bb2, ^bb8
# CHECK:     aie.use_lock(%lock_7_2_6, Acquire, 1)
# CHECK:     aie.dma_bd(%buffer_7_2_1 : memref<128xi32>, 0, 128)
# CHECK:     aie.use_lock(%lock_7_2_6, Release, 0)
# CHECK:     aie.next_bd ^bb8
# CHECK:   ^bb8:  // pred: ^bb7
# CHECK:     aie.use_lock(%lock_7_2_7, Acquire, 1)
# CHECK:     aie.dma_bd(%buffer_7_2_4 : memref<128xi32>, 0, 128)
# CHECK:     aie.use_lock(%lock_7_2_7, Release, 0)
# CHECK:     aie.next_bd ^bb7
# CHECK:    ^bb9:  // pred: ^bb2
# CHECK:     aie.end
# CHECK:   }
# CHECK:   %core_7_2 = aie.core(%tile_7_2) {
# CHECK:     %c0 = arith.constant 0 : index
# CHECK:     %c16 = arith.constant 16 : index
# CHECK:     %c1 = arith.constant 1 : index
# CHECK:     scf.for %arg0 = %c0 to %c16 step %c1 {
# CHECK:       aie.use_lock(%lock_7_2, Acquire, 1)
# CHECK:       aie.use_lock(%lock_7_2_8, Acquire, 1)
# CHECK:       aie.use_lock(%lock_7_2_6, Acquire, 0)
# CHECK:       %c0_10 = arith.constant 0 : index
# CHECK:       %c128 = arith.constant 128 : index
# CHECK:       %c1_11 = arith.constant 1 : index
# CHECK:       scf.for %arg1 = %c0_10 to %c128 step %c1_11 {
# CHECK:         %[[VAL_0:.*]] = memref.load %buffer_7_2[%arg1] : memref<128xi32>
# CHECK:         %[[VAL_1:.*]] = memref.load %buffer_7_2_0[%arg1] : memref<128xi32>
# CHECK:         %[[VAL_2:.*]] = arith.addi %[[VAL_0]], %[[VAL_1]] : i32
# CHECK:         memref.store %[[VAL_2]], %buffer_7_2_1[%arg1] : memref<128xi32>
# CHECK:       }
# CHECK:       aie.use_lock(%lock_7_2, Release, 0)
# CHECK:       aie.use_lock(%lock_7_2_8, Release, 0)
# CHECK:       aie.use_lock(%lock_7_2_6, Release, 1)
# CHECK:       aie.use_lock(%lock_7_2_5, Acquire, 1)
# CHECK:       aie.use_lock(%lock_7_2_9, Acquire, 1)
# CHECK:       aie.use_lock(%lock_7_2_7, Acquire, 0)
# CHECK:       %c0_12 = arith.constant 0 : index
# CHECK:       %c128_13 = arith.constant 128 : index
# CHECK:       %c1_14 = arith.constant 1 : index
# CHECK:       scf.for %arg1 = %c0_12 to %c128_13 step %c1_14 {
# CHECK:         %[[VAL_0:.*]] = memref.load %buffer_7_2_2[%arg1] : memref<128xi32>
# CHECK:         %[[VAL_1:.*]] = memref.load %buffer_7_2_3[%arg1] : memref<128xi32>
# CHECK:         %[[VAL_2:.*]] = arith.addi %[[VAL_0]], %[[VAL_1]] : i32
# CHECK:         memref.store %[[VAL_2]], %buffer_7_2_4[%arg1] : memref<128xi32>
# CHECK:       }
# CHECK:       aie.use_lock(%lock_7_2_5, Release, 0)
# CHECK:       aie.use_lock(%lock_7_2_9, Release, 0)
# CHECK:       aie.use_lock(%lock_7_2_7, Release, 1)
# CHECK:     }
# CHECK:     aie.end
# CHECK:   }
# CHECK: }
@construct_and_print_module
def test_29_mb_matrix_add(module):
    tile_7_0 = tile(7, 0)
    tile_7_2 = tile(7, 2)

    flow(tile_7_0, DMA, 0, tile_7_2, DMA, 0)
    flow(tile_7_0, DMA, 1, tile_7_2, DMA, 1)
    flow(tile_7_2, DMA, 0, tile_7_0, DMA, 0)
    flow(tile_7_2, DMA, 1, tile_7_0, DMA, 1)

    buf72_0 = buffer(T.memref(128, T.i32()), tile_7_2, sym_name="ping_a")
    buf72_4 = buffer(T.memref(128, T.i32()), tile_7_2, sym_name="ping_b")
    buf72_1 = buffer(T.memref(128, T.i32()), tile_7_2, sym_name="ping_c")
    buf72_2 = buffer(T.memref(128, T.i32()), tile_7_2, sym_name="pong_a")
    buf72_5 = buffer(T.memref(128, T.i32()), tile_7_2, sym_name="pong_b")
    buf72_3 = buffer(T.memref(128, T.i32()), tile_7_2, sym_name="pong_c")

    l72_0 = lock(tile_7_2, lock_id=0)
    l72_1 = lock(tile_7_2, lock_id=1)
    l72_2 = lock(tile_7_2, lock_id=2)
    l72_3 = lock(tile_7_2, lock_id=3)
    l72_4 = lock(tile_7_2, lock_id=4)
    l72_5 = lock(tile_7_2, lock_id=5)

    @mem(tile_7_2)
    def m72():
        bd0, src1 = dma_start(S2MM, 0)
        with bb(src1):
            bd4, dma0 = dma_start(S2MM, 1)
        with bb(dma0):
            bd2, end = dma_start(MM2S, 0)
        with bb(bd0):
            use_lock(l72_0, 0, Acquire)
            dma_bd(buf72_0, 0, 128)
            use_lock(l72_0, 1, Release)
            bd1 = next_bd()
        with bb(bd1):
            use_lock(l72_1, 0, Acquire)
            dma_bd(buf72_2, 0, 128)
            use_lock(l72_1, 1, Release)
            next_bd(bd0)
        with bb(bd4):
            use_lock(l72_4, 0, Acquire)
            dma_bd(buf72_4, 0, 128)
            use_lock(l72_4, 1, Release)
            bd5 = next_bd()
        with bb(bd5):
            use_lock(l72_5, 0, Acquire)
            dma_bd(buf72_5, 0, 128)
            use_lock(l72_5, 1, Release)
            next_bd(bd4)
        with bb(bd2):
            use_lock(l72_2, 1, Acquire)
            dma_bd(buf72_1, 0, 128)
            use_lock(l72_2, 0, Release)
            bd3 = next_bd()
        with bb(bd3):
            use_lock(l72_3, 1, Acquire)
            dma_bd(buf72_3, 0, 128)
            use_lock(l72_3, 0, Release)
            next_bd(bd2)
        with bb(end):
            end_()

    @core(tile_7_2)
    def payload():
        for arg5 in range_(0, 16):
            use_lock(l72_0, 1, Acquire)
            use_lock(l72_4, 1, Acquire)
            use_lock(l72_2, 0, Acquire)

            for arg3 in range_(0, 128):
                v0 = load(buf72_0, [arg3])
                v1 = load(buf72_4, [arg3])
                v2 = addi(v0, v1)
                store(v2, buf72_1, [arg3])
                yield_([])

            use_lock(l72_0, 0, Release)
            use_lock(l72_4, 0, Release)
            use_lock(l72_2, 1, Release)

            use_lock(l72_1, 1, Acquire)
            use_lock(l72_5, 1, Acquire)
            use_lock(l72_3, 0, Acquire)

            for arg4 in range_(0, 128):
                v3 = load(buf72_2, [arg4])
                v4 = load(buf72_5, [arg4])
                v5 = addi(v3, v4)
                store(v5, buf72_3, [arg4])
                yield_([])

            use_lock(l72_1, 0, Release)
            use_lock(l72_5, 0, Release)
            use_lock(l72_3, 1, Release)

            yield_([])

    print(module)


# CHECK-LABEL: test_tutorial_5
# CHECK: module {
# CHECK:   %tile_3_4 = aie.tile(3, 4)
# CHECK:   %tile_7_0 = aie.tile(7, 0)
# CHECK:   %buffer_3_4 = aie.buffer(%tile_3_4) {sym_name = "a34"} : memref<256xi32>
# CHECK:   %0 = aie.external_buffer {sym_name = "ddr_test_buffer_in"} : memref<256xi32>
# CHECK:   %1 = aie.external_buffer {sym_name = "ddr_test_buffer_out"} : memref<256xi32>
# CHECK:   %lock_3_4 = aie.lock(%tile_3_4, 7)
# CHECK:   %lock_3_4_0 = aie.lock(%tile_3_4, 8)
# CHECK:   %lock_7_0 = aie.lock(%tile_7_0, 3)
# CHECK:   %lock_7_0_1 = aie.lock(%tile_7_0, 4)
# CHECK:   aie.flow(%tile_7_0, DMA : 0, %tile_3_4, DMA : 1)
# CHECK:   aie.flow(%tile_3_4, DMA : 0, %tile_7_0, DMA : 0)
# CHECK:   %shim_dma_7_0 = aie.shim_dma(%tile_7_0) {
# CHECK:     %2 = aie.dma_start(MM2S, 0, ^bb2, ^bb1)
# CHECK:   ^bb1:  // pred: ^bb0
# CHECK:     %3 = aie.dma_start(S2MM, 0, ^bb3, ^bb4)
# CHECK:   ^bb2:  // pred: ^bb0
# CHECK:     aie.use_lock(%lock_7_0, Acquire, 1)
# CHECK:     aie.dma_bd(%0 : memref<256xi32>, 0, 256)
# CHECK:     aie.use_lock(%lock_7_0, Release, 0)
# CHECK:     aie.next_bd ^bb4
# CHECK:   ^bb3:  // pred: ^bb1
# CHECK:     aie.use_lock(%lock_7_0_1, Acquire, 1)
# CHECK:     aie.dma_bd(%1 : memref<256xi32>, 0, 256)
# CHECK:     aie.use_lock(%lock_7_0_1, Release, 0)
# CHECK:     aie.next_bd ^bb4
# CHECK:   ^bb4:  // 3 preds: ^bb1, ^bb2, ^bb3
# CHECK:     aie.end
# CHECK:   }
# CHECK:   %core_3_4 = aie.core(%tile_3_4) {
# CHECK:     aie.use_lock(%lock_3_4_0, Acquire, 0)
# CHECK:     aie.use_lock(%lock_3_4, Acquire, 1)
# CHECK:     %c3 = arith.constant 3 : index
# CHECK:     %2 = memref.load %buffer_3_4[%c3] : memref<256xi32>
# CHECK:     %c100_i32 = arith.constant 100 : i32
# CHECK:     %3 = arith.addi %2, %c100_i32 : i32
# CHECK:     %c5 = arith.constant 5 : index
# CHECK:     memref.store %3, %buffer_3_4[%c5] : memref<256xi32>
# CHECK:     aie.use_lock(%lock_3_4, Release, 0)
# CHECK:     aie.use_lock(%lock_3_4_0, Release, 1)
# CHECK:     aie.end
# CHECK:   }
# CHECK:   %mem_3_4 = aie.mem(%tile_3_4) {
# CHECK:     %2 = aie.dma_start(S2MM, 1, ^bb2, ^bb1)
# CHECK:   ^bb1:  // pred: ^bb0
# CHECK:     %3 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
# CHECK:   ^bb2:  // pred: ^bb0
# CHECK:     aie.use_lock(%lock_3_4, Acquire, 0)
# CHECK:     aie.dma_bd(%buffer_3_4 : memref<256xi32>, 0, 256)
# CHECK:     aie.use_lock(%lock_3_4, Release, 1)
# CHECK:     aie.next_bd ^bb4
# CHECK:   ^bb3:  // pred: ^bb1
# CHECK:     aie.use_lock(%lock_3_4_0, Acquire, 1)
# CHECK:     aie.dma_bd(%buffer_3_4 : memref<256xi32>, 0, 256)
# CHECK:     aie.use_lock(%lock_3_4_0, Release, 0)
# CHECK:     aie.next_bd ^bb4
# CHECK:   ^bb4:  // 3 preds: ^bb1, ^bb2, ^bb3
# CHECK:     aie.end
# CHECK:   }
# CHECK: }
@construct_and_print_module
def test_tutorial_5(module):
    # 1 tile in row 4 (col 3)
    # even rows have local memory to its left
    tile34 = tile(3, 4)

    # 1 tile in row 0 (col 7)
    # col 7, row 0 has access to a shim_dma
    tile70 = tile(7, 0)

    # Declare local memory of tile(1,4) and tile (3,4) which are not shared
    buf34 = buffer(T.memref(256, T.i32()), tile34, sym_name="a34")

    # Declare external buffers, which represent pointers to external memory locations.
    ext_buf70_in = external_buffer(
        T.memref(256, T.i32()), sym_name="ddr_test_buffer_in"
    )
    ext_buf70_out = external_buffer(
        T.memref(256, T.i32()), sym_name="ddr_test_buffer_out"
    )

    # Declare local locks for tile(3,4) and shim tile (7,0) giving new
    # unique lock ID values 7 and 8
    lock34_in = lock(tile34, lock_id=7)
    lock34_out = lock(tile34, lock_id=8)
    lock70_in = lock(tile70, lock_id=3)
    lock70_out = lock(tile70, lock_id=4)

    # Connect DMA channel 0 on tile(7,0) to DMA channel 1 in tile(3,4)
    # with automatic shortest distance routing
    flow(tile70, DMA, 0, tile34, DMA, 1)
    flow(tile34, DMA, 0, tile70, DMA, 0)

    # shim DMA programming is nearly identical to tile DMA programming
    # shim_dma are blocking on release 1 (user intervention)
    @shim_dma(tile70)
    def shimdma70():
        bd1, ch2 = dma_start(MM2S, 0)
        with bb(ch2):
            bd2, end1 = dma_start(S2MM, 0)
        with bb(bd1):
            # Lock used to allow host to start transfer
            use_lock(lock70_in, 1, Acquire)
            dma_bd(ext_buf70_in, 0, 256)
            use_lock(lock70_in, 0, Release)
            end2 = next_bd()
        with bb(bd2):
            use_lock(lock70_out, 1, Acquire)
            dma_bd(ext_buf70_out, 0, 256)
            use_lock(lock70_out, 0, Release)
            end3 = next_bd()
        with bb(end1, end2, end3):
            end_()

    # Define core algorithm for tile(3,4) which reads value set by tile(1,4)
    # buf[5] = buf[3] + 100
    @core(tile34)
    def core34():
        # This acquire will stall since locks are initialized to Release, 0
        use_lock(lock34_out, 0, Acquire)  # Acquire out lock
        use_lock(lock34_in, 1, Acquire)  # Acquire in lock
        # This will block while tileDMA moves data so we want to acquire this 2nd
        idx1 = constant(3, index=True)
        d1 = load(buf34, [idx1])
        c1 = constant(100, T.i32())
        d2 = addi(d1, c1)
        idx2 = constant(5, index=True)
        store(d2, buf34, [idx2])

        # This release doesn't do much in our example but mimics ping-pong
        use_lock(lock34_in, 0, Release)  # Release in lock
        use_lock(lock34_out, 1, Release)  # Release out lock

    # Define local tile memory behavior (i.e. tileDMA)
    @mem(tile34)
    def mem34():
        # sequence of DMAs declaration and buffer descriptors (bd)
        # ^bd0 - first label/ bd definition to set
        # ^end - next label/ bd definition to set
        # (here, that is aie.end to indicate no more)
        bd0, ch2 = dma_start(S2MM, 1)
        with bb(ch2):
            bd1, end1 = dma_start(MM2S, 0)
        with bb(bd0):
            # Add locks behvaior around bd definition
            use_lock(lock34_in, 0, Acquire)
            # bd definition
            # buf34 - local buffer
            # 0   - offset of transfer
            # 256 - length of transfer
            dma_bd(buf34, 0, 256)
            use_lock(lock34_in, 1, Release)
            end2 = next_bd()
        with bb(bd1):
            use_lock(lock34_out, 1, Acquire)
            dma_bd(buf34, 0, 256)
            use_lock(lock34_out, 0, Release)
            end3 = next_bd()
        with bb(end1, end2, end3):
            end_()

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
    module = run_pipeline(module, "builtin.module(" + pass_pipeline + ")")
    # aie-generate-corelist
    cores = [
        (c.tile.owner.opview.col.value, c.tile.owner.opview.row.value, None)
        for c in find_ops(
            module.operation, lambda o: isinstance(o.operation.opview, CoreOp)
        )
    ]
    # aie-generate-target-arch
    target_arch = "aie2"

    pass_pipeline = ",".join(
        [
            "aie.device(aie-localize-locks",
            "aie-normalize-address-spaces)",
            "aie-standard-lowering{ tilecol=%d tilerow=%d }" % cores[0][0:2],
            "aiex-standard-lowering",
        ]
    )
    module = run_pipeline(module, "builtin.module(" + pass_pipeline + ")")
    # CHECK: module attributes {llvm.target_triple = "aie"} {
    # CHECK:   memref.global "public" @a34 : memref<256xi32>
    # CHECK:   func.func private @debug_i32(i32)
    # CHECK:   func.func private @llvm.aie.event0()
    # CHECK:   func.func private @llvm.aie.event1()
    # CHECK:   func.func private @llvm.aie.put.ms(i32, i32)
    # CHECK:   func.func private @llvm.aie.put.wms(i32, i128)
    # CHECK:   func.func private @llvm.aie.put.fms(i32, f32)
    # CHECK:   func.func private @llvm.aie.get.ss(i32) -> i32
    # CHECK:   func.func private @llvm.aie.get.wss(i32) -> i128
    # CHECK:   func.func private @llvm.aie.get.fss(i32) -> f32
    # CHECK:   func.func private @llvm.aie.put.mcd(i384)
    # CHECK:   func.func private @llvm.aie.get.scd() -> i384
    # CHECK:   func.func private @llvm.aie.lock.acquire.reg(i32, i32)
    # CHECK:   func.func private @llvm.aie.lock.release.reg(i32, i32)
    # CHECK:   func.func @core_3_4() {
    # CHECK:     %c23 = arith.constant 23 : index
    # CHECK:     %c24 = arith.constant 24 : index
    # CHECK:     %0 = arith.index_cast %c24 : index to i32
    # CHECK:     %c0_i32 = arith.constant 0 : i32
    # CHECK:     call @llvm.aie.lock.acquire.reg(%0, %c0_i32) : (i32, i32) -> ()
    # CHECK:     %1 = arith.index_cast %c23 : index to i32
    # CHECK:     %c1_i32 = arith.constant 1 : i32
    # CHECK:     call @llvm.aie.lock.acquire.reg(%1, %c1_i32) : (i32, i32) -> ()
    # CHECK:     %c3 = arith.constant 3 : index
    # CHECK:     %2 = memref.get_global @a34 : memref<256xi32>
    # CHECK:     memref.assume_alignment %2, 32 : memref<256xi32>
    # CHECK:     %3 = memref.load %2[%c3] : memref<256xi32>
    # CHECK:     %c100_i32 = arith.constant 100 : i32
    # CHECK:     %4 = arith.addi %3, %c100_i32 : i32
    # CHECK:     %c5 = arith.constant 5 : index
    # CHECK:     %5 = memref.get_global @a34 : memref<256xi32>
    # CHECK:     memref.assume_alignment %5, 32 : memref<256xi32>
    # CHECK:     memref.store %4, %5[%c5] : memref<256xi32>
    # CHECK:     %6 = arith.index_cast %c23 : index to i32
    # CHECK:     %c0_i32_0 = arith.constant 0 : i32
    # CHECK:     call @llvm.aie.lock.release.reg(%6, %c0_i32_0) : (i32, i32) -> ()
    # CHECK:     %7 = arith.index_cast %c24 : index to i32
    # CHECK:     %c1_i32_1 = arith.constant 1 : i32
    # CHECK:     call @llvm.aie.lock.release.reg(%7, %c1_i32_1) : (i32, i32) -> ()
    # CHECK:     return
    # CHECK:   }
    # CHECK: }
    print(module)

    pass_pipeline = ",".join(
        [
            "canonicalize",
            "cse",
            "convert-vector-to-llvm",
            "expand-strided-metadata",
            "lower-affine",
            "convert-math-to-llvm",
            "convert-arith-to-llvm",
            "finalize-memref-to-llvm",
            "convert-func-to-llvm{use-bare-ptr-memref-call-conv}",
            "convert-cf-to-llvm",
            "canonicalize",
            "cse",
        ]
    )
    module = run_pipeline(module, "builtin.module(" + pass_pipeline + ")")
    # CHECK: module attributes {llvm.target_triple = "aie"} {
    # CHECK:   llvm.mlir.global external @a34() {addr_space = 0 : i32} : !llvm.array<256 x i32>
    # CHECK:   llvm.func @debug_i32(i32) attributes {sym_visibility = "private"}
    # CHECK:   llvm.func @llvm.aie.event0() attributes {sym_visibility = "private"}
    # CHECK:   llvm.func @llvm.aie.event1() attributes {sym_visibility = "private"}
    # CHECK:   llvm.func @llvm.aie.put.ms(i32, i32) attributes {sym_visibility = "private"}
    # CHECK:   llvm.func @llvm.aie.put.wms(i32, i128) attributes {sym_visibility = "private"}
    # CHECK:   llvm.func @llvm.aie.put.fms(i32, f32) attributes {sym_visibility = "private"}
    # CHECK:   llvm.func @llvm.aie.get.ss(i32) -> i32 attributes {sym_visibility = "private"}
    # CHECK:   llvm.func @llvm.aie.get.wss(i32) -> i128 attributes {sym_visibility = "private"}
    # CHECK:   llvm.func @llvm.aie.get.fss(i32) -> f32 attributes {sym_visibility = "private"}
    # CHECK:   llvm.func @llvm.aie.put.mcd(i384) attributes {sym_visibility = "private"}
    # CHECK:   llvm.func @llvm.aie.get.scd() -> i384 attributes {sym_visibility = "private"}
    # CHECK:   llvm.func @llvm.aie.lock.acquire.reg(i32, i32) attributes {sym_visibility = "private"}
    # CHECK:   llvm.func @llvm.aie.lock.release.reg(i32, i32) attributes {sym_visibility = "private"}
    # CHECK:   llvm.func @core_3_4() {
    # CHECK:     %0 = llvm.mlir.constant(31 : index) : i64
    # CHECK:     %1 = llvm.mlir.constant(0 : index) : i64
    # CHECK:     %2 = llvm.mlir.constant(0 : i32) : i32
    # CHECK:     %3 = llvm.mlir.constant(1 : i32) : i32
    # CHECK:     %4 = llvm.mlir.constant(100 : i32) : i32
    # CHECK:     %5 = llvm.mlir.constant(24 : i32) : i32
    # CHECK:     %6 = llvm.mlir.constant(23 : i32) : i32
    # CHECK:     llvm.call @llvm.aie.lock.acquire.reg(%5, %2) : (i32, i32) -> ()
    # CHECK:     llvm.call @llvm.aie.lock.acquire.reg(%6, %3) : (i32, i32) -> ()
    # CHECK:     %7 = llvm.mlir.addressof @a34 : !llvm.ptr
    # CHECK:     %8 = llvm.getelementptr %7[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<256 x i32>
    # CHECK:     %9 = llvm.ptrtoint %8 : !llvm.ptr to i64
    # CHECK:     %10 = llvm.and %9, %0  : i64
    # CHECK:     %11 = llvm.icmp "eq" %10, %1 : i64
    # CHECK:     "llvm.intr.assume"(%11) : (i1) -> ()
    # CHECK:     %12 = llvm.getelementptr %8[3] : (!llvm.ptr) -> !llvm.ptr, i32
    # CHECK:     %13 = llvm.load %12 : !llvm.ptr -> i32
    # CHECK:     %14 = llvm.add %13, %4  : i32
    # CHECK:     "llvm.intr.assume"(%11) : (i1) -> ()
    # CHECK:     %15 = llvm.getelementptr %8[5] : (!llvm.ptr) -> !llvm.ptr, i32
    # CHECK:     llvm.store %14, %15 : i32, !llvm.ptr
    # CHECK:     llvm.call @llvm.aie.lock.release.reg(%6, %2) : (i32, i32) -> ()
    # CHECK:     llvm.call @llvm.aie.lock.release.reg(%5, %3) : (i32, i32) -> ()
    # CHECK:     llvm.return
    # CHECK:   }
    # CHECK: }
    print(module)

    llvmir = translate_mlir_to_llvmir(module.operation)
    # CHECK: ; ModuleID = 'LLVMDialectModule'
    # CHECK: source_filename = "LLVMDialectModule"
    # CHECK: target triple = "aie"
    # CHECK: @a34 = external global [256 x i32]
    # CHECK: declare void @debug_i32(i32)
    # CHECK: declare void @llvm.aie.event0()
    # CHECK: declare void @llvm.aie.event1()
    # CHECK: declare void @llvm.aie.put.ms(i32, i32)
    # CHECK: declare void @llvm.aie.put.wms(i32, i128)
    # CHECK: declare void @llvm.aie.put.fms(i32, float)
    # CHECK: declare i32 @llvm.aie.get.ss(i32)
    # CHECK: declare i128 @llvm.aie.get.wss(i32)
    # CHECK: declare float @llvm.aie.get.fss(i32)
    # CHECK: declare void @llvm.aie.put.mcd(i384)
    # CHECK: declare i384 @llvm.aie.get.scd()
    # CHECK: declare void @llvm.aie.lock.acquire.reg(i32, i32)
    # CHECK: declare void @llvm.aie.lock.release.reg(i32, i32)
    # CHECK: define void @core_3_4() {
    # CHECK:   call void @llvm.aie.lock.acquire.reg(i32 24, i32 0)
    # CHECK:   call void @llvm.aie.lock.acquire.reg(i32 23, i32 1)
    # CHECK:   %1 = and i64 ptrtoint (ptr @a34 to i64), 31
    # CHECK:   %2 = icmp eq i64 %1, 0
    # CHECK:   call void @llvm.assume(i1 %2)
    # CHECK:   %3 = load i32, ptr getelementptr (i32, ptr @a34, i32 3), align 4
    # CHECK:   %4 = add i32 %3, 100
    # CHECK:   call void @llvm.assume(i1 %2)
    # CHECK:   store i32 %4, ptr getelementptr (i32, ptr @a34, i32 5), align 4
    # CHECK:   call void @llvm.aie.lock.release.reg(i32 23, i32 0)
    # CHECK:   call void @llvm.aie.lock.release.reg(i32 24, i32 1)
    # CHECK:   ret void
    # CHECK: }
    # CHECK: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
    # CHECK: declare void @llvm.assume(i1 noundef) #0
    # CHECK: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
    # CHECK: !llvm.module.flags = !{!0}
    # CHECK: !0 = !{i32 2, !"Debug Info Version", i32 3}
    print(llvmir)


@construct_and_print_module
def my_passthrough(module):
    N = 4096
    ofifo_mem_ref_ty = TypeAttr.get(ObjectFifoType.get(T.memref(1024, T.i32())))
    tensor_ty = T.memref(N, T.i32())

    @device(AIEDevice.ipu)
    def device_body():
        # Tile declarations
        shim_tile = tile(0, 0)
        compute_tile2 = tile(0, 2)

        # AIE-array data movement with object fifos
        objectfifo("in", shim_tile, [compute_tile2], 2, ofifo_mem_ref_ty, [], [])
        objectfifo("out", compute_tile2, [shim_tile], 2, ofifo_mem_ref_ty, [], [])
        objectfifo_link(["in"], ["out"])

        @core(compute_tile2)
        def core_body():
            tmp = alloc([1], T.i32())
            v0 = constant(0, T.i32())
            store(v0, tmp, [0])

        @func(emit=True)
        def sequence(A: tensor_ty, B: tensor_ty, C: tensor_ty):
            ipu_dma_memcpy_nd(metadata="out", bd_id=0, mem=C, lengths=[1, 1, 1, N])
            ipu_dma_memcpy_nd(metadata="in", bd_id=1, mem=A, lengths=[1, 1, 1, N])
            ipu_sync(column=0, row=0, direction=0, channel=0)

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
    input_with_addresses = run_pipeline(module, "builtin.module(" + pass_pipeline + ")")
    # print(module)
    cores = [
        (c.tile.owner.opview.col.value, c.tile.owner.opview.row.value, None)
        for c in find_ops(
            input_with_addresses.operation,
            lambda o: isinstance(o.operation.opview, CoreOp),
        )
    ]
    target_arch = "aie2"

    generated_ipu_insts = run_pipeline(
        input_with_addresses, "builtin.module(aie.device(aie-dma-to-ipu))"
    )
    ipu_insts = ipu_instgen(generated_ipu_insts.operation)

    pass_pipeline = ",".join(
        [
            "aie.device(aie-localize-locks",
            "aie-normalize-address-spaces)",
            "aie-standard-lowering{ tilecol=%d tilerow=%d }" % cores[0][0:2],
            "aiex-standard-lowering",
        ]
    )
    input_opt_with_addresses = run_pipeline(
        input_with_addresses, "builtin.module(" + pass_pipeline + ")"
    )

    pass_pipeline = ",".join(
        [
            "canonicalize",
            "cse",
            "convert-vector-to-llvm",
            "expand-strided-metadata",
            "lower-affine",
            "convert-math-to-llvm",
            "convert-arith-to-llvm",
            "finalize-memref-to-llvm",
            "convert-func-to-llvm{use-bare-ptr-memref-call-conv}",
            "convert-cf-to-llvm",
            "canonicalize",
            "cse",
        ]
    )
    llvmlir = run_pipeline(
        input_opt_with_addresses, "builtin.module(" + pass_pipeline + ")"
    )
    llvmir = translate_mlir_to_llvmir(llvmlir.operation)

    with open(Path(__file__).parent / "chess_intrinsic_wrapper.ll") as f:
        chess_intrinsic_wrapper = prepare_for_chesshack(f.read())
    input_llchesslinked = chesshack(llvmir, chess_intrinsic_wrapper)

    pass_pipeline = ",".join(
        [
            "aie-create-pathfinder-flows",
            "aie-lower-broadcast-packet",
            "aie-create-packet-flows",
            "aie-lower-multicast",
        ]
    )
    input_physical = run_pipeline(
        input_with_addresses, "builtin.module(aie.device(" + pass_pipeline + "))"
    )

    aie_inc = generate_xaie(input_physical.operation)
    aie_control = generate_cdo(input_physical.operation)
    core_0_2 = generate_bcf(input_with_addresses.operation, 0, 2)
    extract_input_files(core_0_2)

    # xchesscc_wrapper aie2 \
    #   +w \
    #   /home/mlevental/dev_projects/mlir-aie/cmake-build-debug/test.mlir.prj/work \
    #   -d \
    #   -f \
    #   /home/mlevental/dev_projects/mlir-aie/cmake-build-debug/test.mlir.prj/input.o \
    #   +l \
    #   /home/mlevental/dev_projects/mlir-aie/cmake-build-debug/test.mlir.prj/core_0_2.bcf \
    #   -o \
    #   ./core_0_2.elf
    #
    # clang++ -fPIC -c -std=c++17 -D__AIEARCH__=20 -D__AIESIM__ -D__CDO__ -D__PS_INIT_AIE__ -D__LOCK_FENCE_MODE__=2 -DAIE_OPTION_SCALAR_FLOAT_ON_VECTOR -DAIE2_FP32_EMULATION_ACCURACY_FAST -Wno-deprecated-declarations -I/home/mlevental/dev_projects/mlir-aie/cmake-build-debug/test.mlir.prj -I/home/mlevental/dev_projects/mlir-aie/cmake-build-debug/runtime_lib/x86_64/xaiengine/cdo/include -Iinclude -o /home/mlevental/dev_projects/mlir-aie/cmake-build-debug/test.mlir.prj/gen_cdo.o /home/mlevental/dev_projects/mlir-aie/cmake-build-debug/data/generated-source/gen_cdo.cpp
    #
    # clang++ -fPIC -c -std=c++17 -I/home/mlevental/dev_projects/mlir-aie/cmake-build-debug/test.mlir.prj -I/home/mlevental/dev_projects/mlir-aie/cmake-build-debug/runtime_lib/x86_64/xaiengine/cdo/include -Iinclude -o /home/mlevental/dev_projects/mlir-aie/cmake-build-debug/test.mlir.prj/cdo_main.o /home/mlevental/dev_projects/mlir-aie/cmake-build-debug/data/generated-source/cdo_main.cpp
    #
    # clang++ -L/home/mlevental/dev_projects/mlir-aie/cmake-build-debug/runtime_lib/x86_64/xaiengine/cdo -Llib/lnx64.o -lxaienginecdo -lcdo_driver -o /home/mlevental/dev_projects/mlir-aie/cmake-build-debug/test.mlir.prj/cdo_main.out /home/mlevental/dev_projects/mlir-aie/cmake-build-debug/test.mlir.prj/gen_cdo.o /home/mlevental/dev_projects/mlir-aie/cmake-build-debug/test.mlir.prj/cdo_main.o
    #
    # /home/mlevental/dev_projects/mlir-aie/cmake-build-debug/test.mlir.prj/cdo_main.out --work-dir-path /home/mlevental/dev_projects/mlir-aie/cmake-build-debug/test.mlir.prj/ /home/mlevental/dev_projects/mlir-aie/cmake-build-debug/test.mlir.prj/kernels.json /home/mlevental/dev_projects/mlir-aie/cmake-build-debug/test.mlir.prj/design.bif

    # bootgen -arch versal -image /home/mlevental/dev_projects/mlir-aie/cmake-build-debug/test.mlir.prj/design.bif -o /home/mlevental/dev_projects/mlir-aie/cmake-build-debug/test.mlir.prj/design.pdi -w
    #
    # xclbinutil --add-replace-section MEM_TOPOLOGY:JSON:/home/mlevental/dev_projects/mlir-aie/cmake-build-debug/test.mlir.prj/mem_topology.json --add-kernel /home/mlevental/dev_projects/mlir-aie/cmake-build-debug/test.mlir.prj/kernels.json --add-replace-section AIE_PARTITION:JSON:/home/mlevental/dev_projects/mlir-aie/cmake-build-debug/test.mlir.prj/aie_partition.json --force --output aie.xclbin
