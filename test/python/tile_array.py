# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

# RUN: %python %s | FileCheck %s

import numpy as np

from aie.dialects import aie
from aie.extras import types as T
from aie.dialects.aie import (
    AIEDevice,
    DMAChannelDir,
    LockAction,
    WireBundle,
    find_neighbors,
)
from aie.dialects.aiex import TileArray, Channel
from util import construct_and_print_module

DMA = WireBundle.DMA
S2MM = DMAChannelDir.S2MM
MM2S = DMAChannelDir.MM2S
Acquire = LockAction.Acquire
AcquireGreaterEqual = LockAction.AcquireGreaterEqual
Release = LockAction.Release


# CHECK-LABEL: broadcast
@construct_and_print_module
def broadcast(module):
    @aie.device(AIEDevice.npu)
    def npu():
        df = TileArray()
        assert df[[0, 1], 0].shape == (2, 1)
        assert df[[0, 1], 3:].shape == (2, 3)

        fls = df[0, 0] >> df[0, 1]
        # CHECK: aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
        print(fls)

        print()

        fls = df[[0, 1], 0] >> df[[0, 1], 3:]
        # CHECK: aie.flow(%tile_0_0, DMA : 1, %tile_0_3, DMA : 0)
        # CHECK: aie.flow(%tile_0_0, DMA : 1, %tile_0_4, DMA : 0)
        # CHECK: aie.flow(%tile_0_0, DMA : 1, %tile_0_5, DMA : 0)
        # CHECK: aie.flow(%tile_1_0, DMA : 0, %tile_1_3, DMA : 0)
        # CHECK: aie.flow(%tile_1_0, DMA : 0, %tile_1_4, DMA : 0)
        # CHECK: aie.flow(%tile_1_0, DMA : 0, %tile_1_5, DMA : 0)

        for f in fls:
            print(f)

        print()

        fls = df[0, 0] >> df[1, 0:3]
        # CHECK: aie.flow(%tile_0_0, DMA : 2, %tile_1_0, DMA : 0)
        # CHECK: aie.flow(%tile_0_0, DMA : 2, %tile_1_1, DMA : 0)
        # CHECK: aie.flow(%tile_0_0, DMA : 2, %tile_1_2, DMA : 0)
        for f in fls:
            print(f)

        print()

        fls = df[0, 0] >> df[[2, 3], 1:]
        # CHECK: aie.flow(%tile_0_0, DMA : 3, %tile_2_1, DMA : 0)
        # CHECK: aie.flow(%tile_0_0, DMA : 3, %tile_2_2, DMA : 0)
        # CHECK: aie.flow(%tile_0_0, DMA : 3, %tile_2_3, DMA : 0)
        # CHECK: aie.flow(%tile_0_0, DMA : 3, %tile_2_4, DMA : 0)
        # CHECK: aie.flow(%tile_0_0, DMA : 3, %tile_2_5, DMA : 0)
        # CHECK: aie.flow(%tile_0_0, DMA : 3, %tile_3_1, DMA : 0)
        # CHECK: aie.flow(%tile_0_0, DMA : 3, %tile_3_2, DMA : 0)
        # CHECK: aie.flow(%tile_0_0, DMA : 3, %tile_3_3, DMA : 0)
        # CHECK: aie.flow(%tile_0_0, DMA : 3, %tile_3_4, DMA : 0)
        # CHECK: aie.flow(%tile_0_0, DMA : 3, %tile_3_5, DMA : 0)
        for f in fls:
            print(f)

        print()

        fls = df[0, 0].flow(df[[2, 3], 1:], source_annot="bob", dest_annot="alice")
        # CHECK: aie.flow(%tile_0_0, DMA : 4, %tile_2_1, DMA : 1) {dest_annot = {alice}, source_annot = {bob}}
        # CHECK: aie.flow(%tile_0_0, DMA : 4, %tile_2_2, DMA : 1) {dest_annot = {alice}, source_annot = {bob}}
        # CHECK: aie.flow(%tile_0_0, DMA : 4, %tile_2_3, DMA : 1) {dest_annot = {alice}, source_annot = {bob}}
        # CHECK: aie.flow(%tile_0_0, DMA : 4, %tile_2_4, DMA : 1) {dest_annot = {alice}, source_annot = {bob}}
        # CHECK: aie.flow(%tile_0_0, DMA : 4, %tile_2_5, DMA : 1) {dest_annot = {alice}, source_annot = {bob}}
        # CHECK: aie.flow(%tile_0_0, DMA : 4, %tile_3_1, DMA : 1) {dest_annot = {alice}, source_annot = {bob}}
        # CHECK: aie.flow(%tile_0_0, DMA : 4, %tile_3_2, DMA : 1) {dest_annot = {alice}, source_annot = {bob}}
        # CHECK: aie.flow(%tile_0_0, DMA : 4, %tile_3_3, DMA : 1) {dest_annot = {alice}, source_annot = {bob}}
        # CHECK: aie.flow(%tile_0_0, DMA : 4, %tile_3_4, DMA : 1) {dest_annot = {alice}, source_annot = {bob}}
        # CHECK: aie.flow(%tile_0_0, DMA : 4, %tile_3_5, DMA : 1) {dest_annot = {alice}, source_annot = {bob}}
        for f in fls:
            print(f)

        print()

        # CHECK: aie.flow(%tile_0_0, DMA : 3, %tile_2_2, DMA : 0)
        # CHECK: aie.flow(%tile_0_0, DMA : 4, %tile_2_2, DMA : 1) {dest_annot = {alice}, source_annot = {bob}}
        for f in df[2, 2].flows():
            print(f)

        # CHECK: aie.flow(%tile_0_0, DMA : 4, %tile_2_2, DMA : 1) {dest_annot = {alice}, source_annot = {bob}}
        for f in df[2, 2].flows(source_annot="bob"):
            print(f)

        # CHECK: aie.flow(%tile_0_0, DMA : 4, %tile_2_2, DMA : 1) {dest_annot = {alice}, source_annot = {bob}}
        for f in df[2, 2].flows(dest_annot="alice"):
            print(f)

        # CHECK: aie.flow(%tile_0_0, DMA : 4, %tile_2_2, DMA : 1) {dest_annot = {alice}, source_annot = {bob}}
        for f in df[2, 2].flows(source_annot="bob", dest_annot="alice"):
            print(f)

        assert len(df[0, 3].flows(source_annot="bob", dest_annot="alice")) == 0

        # CHECK: aie.flow(%tile_0_0, DMA : 1, %tile_0_3, DMA : 0)
        for f in df[0, 3].flows():
            print(f)

        # CHECK: aie.flow(%tile_1_0, DMA : 0, %tile_1_3, DMA : 0)
        for f in df[1, 3].flows(filter_dest=True):
            print(f)

        # CHECK: module {
        # CHECK:   aie.device(npu) {
        # CHECK:     %tile_0_0 = aie.tile(0, 0)
        # CHECK:     %tile_0_1 = aie.tile(0, 1)
        # CHECK:     %tile_0_2 = aie.tile(0, 2)
        # CHECK:     %tile_0_3 = aie.tile(0, 3)
        # CHECK:     %tile_0_4 = aie.tile(0, 4)
        # CHECK:     %tile_0_5 = aie.tile(0, 5)
        # CHECK:     %tile_1_0 = aie.tile(1, 0)
        # CHECK:     %tile_1_1 = aie.tile(1, 1)
        # CHECK:     %tile_1_2 = aie.tile(1, 2)
        # CHECK:     %tile_1_3 = aie.tile(1, 3)
        # CHECK:     %tile_1_4 = aie.tile(1, 4)
        # CHECK:     %tile_1_5 = aie.tile(1, 5)
        # CHECK:     %tile_2_0 = aie.tile(2, 0)
        # CHECK:     %tile_2_1 = aie.tile(2, 1)
        # CHECK:     %tile_2_2 = aie.tile(2, 2)
        # CHECK:     %tile_2_3 = aie.tile(2, 3)
        # CHECK:     %tile_2_4 = aie.tile(2, 4)
        # CHECK:     %tile_2_5 = aie.tile(2, 5)
        # CHECK:     %tile_3_0 = aie.tile(3, 0)
        # CHECK:     %tile_3_1 = aie.tile(3, 1)
        # CHECK:     %tile_3_2 = aie.tile(3, 2)
        # CHECK:     %tile_3_3 = aie.tile(3, 3)
        # CHECK:     %tile_3_4 = aie.tile(3, 4)
        # CHECK:     %tile_3_5 = aie.tile(3, 5)
        # CHECK:     %tile_4_0 = aie.tile(4, 0)
        # CHECK:     %tile_4_1 = aie.tile(4, 1)
        # CHECK:     %tile_4_2 = aie.tile(4, 2)
        # CHECK:     %tile_4_3 = aie.tile(4, 3)
        # CHECK:     %tile_4_4 = aie.tile(4, 4)
        # CHECK:     %tile_4_5 = aie.tile(4, 5)
        # CHECK:     aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
        # CHECK:     aie.flow(%tile_0_0, DMA : 1, %tile_0_3, DMA : 0)
        # CHECK:     aie.flow(%tile_0_0, DMA : 1, %tile_0_4, DMA : 0)
        # CHECK:     aie.flow(%tile_0_0, DMA : 1, %tile_0_5, DMA : 0)
        # CHECK:     aie.flow(%tile_1_0, DMA : 0, %tile_1_3, DMA : 0)
        # CHECK:     aie.flow(%tile_1_0, DMA : 0, %tile_1_4, DMA : 0)
        # CHECK:     aie.flow(%tile_1_0, DMA : 0, %tile_1_5, DMA : 0)
        # CHECK:     aie.flow(%tile_0_0, DMA : 2, %tile_1_0, DMA : 0)
        # CHECK:     aie.flow(%tile_0_0, DMA : 2, %tile_1_1, DMA : 0)
        # CHECK:     aie.flow(%tile_0_0, DMA : 2, %tile_1_2, DMA : 0)
        # CHECK:     aie.flow(%tile_0_0, DMA : 3, %tile_2_1, DMA : 0)
        # CHECK:     aie.flow(%tile_0_0, DMA : 3, %tile_2_2, DMA : 0)
        # CHECK:     aie.flow(%tile_0_0, DMA : 3, %tile_2_3, DMA : 0)
        # CHECK:     aie.flow(%tile_0_0, DMA : 3, %tile_2_4, DMA : 0)
        # CHECK:     aie.flow(%tile_0_0, DMA : 3, %tile_2_5, DMA : 0)
        # CHECK:     aie.flow(%tile_0_0, DMA : 3, %tile_3_1, DMA : 0)
        # CHECK:     aie.flow(%tile_0_0, DMA : 3, %tile_3_2, DMA : 0)
        # CHECK:     aie.flow(%tile_0_0, DMA : 3, %tile_3_3, DMA : 0)
        # CHECK:     aie.flow(%tile_0_0, DMA : 3, %tile_3_4, DMA : 0)
        # CHECK:     aie.flow(%tile_0_0, DMA : 3, %tile_3_5, DMA : 0)
        # CHECK:     aie.flow(%tile_0_0, DMA : 4, %tile_2_1, DMA : 1) {dest_annot = {alice}, source_annot = {bob}}
        # CHECK:     aie.flow(%tile_0_0, DMA : 4, %tile_2_2, DMA : 1) {dest_annot = {alice}, source_annot = {bob}}
        # CHECK:     aie.flow(%tile_0_0, DMA : 4, %tile_2_3, DMA : 1) {dest_annot = {alice}, source_annot = {bob}}
        # CHECK:     aie.flow(%tile_0_0, DMA : 4, %tile_2_4, DMA : 1) {dest_annot = {alice}, source_annot = {bob}}
        # CHECK:     aie.flow(%tile_0_0, DMA : 4, %tile_2_5, DMA : 1) {dest_annot = {alice}, source_annot = {bob}}
        # CHECK:     aie.flow(%tile_0_0, DMA : 4, %tile_3_1, DMA : 1) {dest_annot = {alice}, source_annot = {bob}}
        # CHECK:     aie.flow(%tile_0_0, DMA : 4, %tile_3_2, DMA : 1) {dest_annot = {alice}, source_annot = {bob}}
        # CHECK:     aie.flow(%tile_0_0, DMA : 4, %tile_3_3, DMA : 1) {dest_annot = {alice}, source_annot = {bob}}
        # CHECK:     aie.flow(%tile_0_0, DMA : 4, %tile_3_4, DMA : 1) {dest_annot = {alice}, source_annot = {bob}}
        # CHECK:     aie.flow(%tile_0_0, DMA : 4, %tile_3_5, DMA : 1) {dest_annot = {alice}, source_annot = {bob}}
        # CHECK:   }
        # CHECK: }
        print(module)


# CHECK-LABEL: lshift
@construct_and_print_module
def lshift(module):
    @aie.device(AIEDevice.npu)
    def npu():
        tiles = TileArray()

        fls = tiles[2, 1] << tiles[0, [2, 3]]
        # CHECK: aie.flow(%tile_0_2, DMA : 0, %tile_2_1, DMA : 0)
        # CHECK: aie.flow(%tile_0_3, DMA : 0, %tile_2_1, DMA : 1)
        for f in fls:
            print(f)

        fls = tiles[2, 1] << tiles[0, [2, 3]]
        # CHECK: aie.flow(%tile_0_2, DMA : 1, %tile_2_1, DMA : 2)
        # CHECK: aie.flow(%tile_0_3, DMA : 1, %tile_2_1, DMA : 3)
        for f in fls:
            print(f)


# CHECK-LABEL: locks
@construct_and_print_module
def locks(module):
    @aie.device(AIEDevice.npu)
    def npu():
        tiles = TileArray()

        aie.lock(tiles[0, 1].tile)
        # CHECK: %lock_0_1 = aie.lock(%tile_0_1)
        for l in tiles[0, 1].locks():
            print(l.owner)

        aie.lock(tiles[0, 2].tile)
        aie.lock(tiles[0, 2].tile, annot="bob")
        aie.lock(tiles[0, 3].tile)
        aie.lock(tiles[0, 3].tile, annot="alice")

        # CHECK: %lock_0_2 = aie.lock(%tile_0_2)
        # CHECK: %lock_0_2_0 = aie.lock(%tile_0_2) {annot = {bob}}
        for l in tiles[0, 2].locks():
            print(l.owner)

        # CHECK: %lock_0_2_0 = aie.lock(%tile_0_2) {annot = {bob}}
        assert len(tiles[0, 2].locks(annot="bob"))
        for l in tiles[0, 2].locks(annot="bob"):
            print(l.owner)

        assert len(tiles[0, 2].locks(annot="alice")) == 0

        assert len(tiles[0, 3].locks(annot="alice")) == 1
        # CHECK: %lock_0_3_1 = aie.lock(%tile_0_3) {annot = {alice}}
        for l in tiles[0, 3].locks(annot="alice"):
            print(l.owner)


# CHECK-LABEL: neighbors
@construct_and_print_module
def neighbors(module):
    @aie.device(AIEDevice.npu)
    def npu():
        tiles = TileArray()

        # CHECK: Neighbors(north=%tile_2_3 = aie.tile(2, 3), west=%tile_1_2 = aie.tile(1, 2), south=None)
        print(find_neighbors(tiles[2, 2].tile))

        assert tiles[1:3, 1:3].neighbors().shape == (2, 2)
        # CHECK: tile(col=1, row=1) : Neighbors(north=None, west=None, south=None)
        # CHECK: tile(col=1, row=2) : Neighbors(north=<TileArray: [%tile_1_3 = aie.tile(1, 3)]>, west=<TileArray: [%tile_0_2 = aie.tile(0, 2)]>, south=None)
        # CHECK: tile(col=2, row=1) : Neighbors(north=None, west=<TileArray: [%tile_1_1 = aie.tile(1, 1)]>, south=None)
        # CHECK: tile(col=2, row=2) : Neighbors(north=<TileArray: [%tile_2_3 = aie.tile(2, 3)]>, west=<TileArray: [%tile_1_2 = aie.tile(1, 2)]>, south=None)
        for idx, n in np.ndenumerate(tiles[1:3, 1:3].neighbors()):
            print(tiles[1:3, 1:3][idx].tile, ":", n)

        # CHECK: tile(col=1, row=1) : Neighbors(north=<TileArray: [%tile_1_2 = aie.tile(1, 2)]>, west=None, south=<TileArray: [%tile_1_0 = aie.tile(1, 0)]>)
        # CHECK: tile(col=1, row=2) : Neighbors(north=<TileArray: [%tile_1_3 = aie.tile(1, 3)]>, west=<TileArray: [%tile_0_2 = aie.tile(0, 2)]>, south=<TileArray: [%tile_1_1 = aie.tile(1, 1)]>)
        # CHECK: tile(col=2, row=1) : Neighbors(north=<TileArray: [%tile_2_2 = aie.tile(2, 2)]>, west=<TileArray: [%tile_1_1 = aie.tile(1, 1)]>, south=<TileArray: [%tile_2_0 = aie.tile(2, 0)]>)
        # CHECK: tile(col=2, row=2) : Neighbors(north=<TileArray: [%tile_2_3 = aie.tile(2, 3)]>, west=<TileArray: [%tile_1_2 = aie.tile(1, 2)]>, south=<TileArray: [%tile_2_1 = aie.tile(2, 1)]>)
        for idx, n in np.ndenumerate(tiles[1:3, 1:3].neighbors(logical=False)):
            print(tiles[1:3, 1:3][idx].tile, ":", n)


# CHECK-LABEL: channels_basic
@construct_and_print_module
def channels_basic(module):

    # CHECK-LABEL: test-basic
    print("test-basic")

    @aie.device(AIEDevice.npu)
    def npu():
        tiles = TileArray()

        b = aie.buffer(tiles[2, 2].tile, (10, 10), T.i32(), name="bob")
        c = Channel(tiles[2, 2].tile, b)
        c = Channel(
            tiles[2, 2].tile, shape=(10, 10), dtype=T.i32(), buffer_name="alice"
        )

    # CHECK: %bob = aie.buffer(%tile_2_2) {sym_name = "bob"} : memref<10x10xi32>
    # CHECK: %bob_producer_lock = aie.lock(%tile_2_2) {sym_name = "bob_producer_lock"}
    # CHECK: %bob_consumer_lock = aie.lock(%tile_2_2) {sym_name = "bob_consumer_lock"}
    # CHECK: %alice = aie.buffer(%tile_2_2) {sym_name = "alice"} : memref<10x10xi32>
    # CHECK: %alice_producer_lock = aie.lock(%tile_2_2) {sym_name = "alice_producer_lock"}
    # CHECK: %alice_consumer_lock = aie.lock(%tile_2_2) {sym_name = "alice_consumer_lock"}
    print(npu)

    # CHECK-LABEL: test-context-manager
    print("test-context-manager")

    @aie.device(AIEDevice.npu)
    def npu():
        tiles = TileArray()

        c = Channel(
            tiles[2, 2].tile, shape=(10, 10), dtype=T.i32(), buffer_name="alice"
        )

        @aie.mem(tiles[2, 2].tile)
        def mem():
            with c.put() as buffer:
                # CHECK: %30 = "aie.buffer"(%14) <{sym_name = "alice"}> : (index) -> memref<10x10xi32>
                print(buffer.owner)
            aie.end()

        @aie.core(tiles[2, 2].tile)
        def core():
            with c.get() as buffer:
                # CHECK: %30 = "aie.buffer"(%14) <{sym_name = "alice"}> : (index) -> memref<10x10xi32>
                print(buffer.owner)

    # CHECK: %alice = aie.buffer(%tile_2_2) {sym_name = "alice"} : memref<10x10xi32>
    # CHECK: %alice_producer_lock = aie.lock(%tile_2_2) {sym_name = "alice_producer_lock"}
    # CHECK: %alice_consumer_lock = aie.lock(%tile_2_2) {sym_name = "alice_consumer_lock"}
    # CHECK: %mem_2_2 = aie.mem(%tile_2_2) {
    # CHECK:   aie.use_lock(%alice_producer_lock, AcquireGreaterEqual)
    # CHECK:   aie.use_lock(%alice_consumer_lock, Release)
    # CHECK:   aie.end
    # CHECK: }
    # CHECK: %core_2_2 = aie.core(%tile_2_2) {
    # CHECK:   aie.use_lock(%alice_consumer_lock, AcquireGreaterEqual)
    # CHECK:   aie.use_lock(%alice_producer_lock, Release)
    # CHECK:   aie.end
    # CHECK: }
    print(npu)


# CHECK-LABEL: nd_channels
@construct_and_print_module
def nd_channels(module):
    @aie.device(AIEDevice.npu)
    def npu():
        tiles = TileArray()

        shapes = np.array([(10, 10)], dtype="i,i").astype(object)
        c = tiles[2, 2].channel(shape=shapes, dtype=[T.i32()])
        # CHECK: <Channel: buffer=MemRef(%buffer_2_2, memref<10x10xi32>) producer_lock=Scalar(%buffer_2_2_producer_lock = aie.lock(%tile_2_2) {sym_name = "buffer_2_2_producer_lock"}) consumer_lock=Scalar(%buffer_2_2_consumer_lock = aie.lock(%tile_2_2) {sym_name = "buffer_2_2_consumer_lock"})>
        print(c)
        cs = tiles[2:4, 2:4].channel(shape=shapes, dtype=[T.i32()])
        assert cs.shape == (2, 2)

        # CHECK: (0, 0) <Channel: buffer=MemRef(%buffer_2_2_0, memref<10x10xi32>) producer_lock=Scalar(%buffer_2_2_0_producer_lock = aie.lock(%tile_2_2) {sym_name = "buffer_2_2_0_producer_lock"}) consumer_lock=Scalar(%buffer_2_2_0_consumer_lock = aie.lock(%tile_2_2) {sym_name = "buffer_2_2_0_consumer_lock"})>
        # CHECK: (0, 1) <Channel: buffer=MemRef(%buffer_2_3, memref<10x10xi32>) producer_lock=Scalar(%buffer_2_3_producer_lock = aie.lock(%tile_2_3) {sym_name = "buffer_2_3_producer_lock"}) consumer_lock=Scalar(%buffer_2_3_consumer_lock = aie.lock(%tile_2_3) {sym_name = "buffer_2_3_consumer_lock"})>
        # CHECK: (1, 0) <Channel: buffer=MemRef(%buffer_3_2, memref<10x10xi32>) producer_lock=Scalar(%buffer_3_2_producer_lock = aie.lock(%tile_3_2) {sym_name = "buffer_3_2_producer_lock"}) consumer_lock=Scalar(%buffer_3_2_consumer_lock = aie.lock(%tile_3_2) {sym_name = "buffer_3_2_consumer_lock"})>
        # CHECK: (1, 1) <Channel: buffer=MemRef(%buffer_3_3, memref<10x10xi32>) producer_lock=Scalar(%buffer_3_3_producer_lock = aie.lock(%tile_3_3) {sym_name = "buffer_3_3_producer_lock"}) consumer_lock=Scalar(%buffer_3_3_consumer_lock = aie.lock(%tile_3_3) {sym_name = "buffer_3_3_consumer_lock"})>
        for idx, c in np.ndenumerate(cs):
            print(idx, c)

        shapes = np.array([[(1, 2), (3, 4)], [(5, 6), (7, 8)]], dtype="i,i").astype(
            object
        )
        cs = tiles[2:4, 2:4].channel(shape=shapes, dtype=[T.i32()])
        assert cs.shape == (2, 2)

        # CHECK: (0, 0) <Channel: buffer=MemRef(%buffer_2_2_1, memref<1x2xi32>) producer_lock=Scalar(%buffer_2_2_1_producer_lock = aie.lock(%tile_2_2) {sym_name = "buffer_2_2_1_producer_lock"}) consumer_lock=Scalar(%buffer_2_2_1_consumer_lock = aie.lock(%tile_2_2) {sym_name = "buffer_2_2_1_consumer_lock"})>
        # CHECK: (0, 1) <Channel: buffer=MemRef(%buffer_2_3_2, memref<3x4xi32>) producer_lock=Scalar(%buffer_2_3_2_producer_lock = aie.lock(%tile_2_3) {sym_name = "buffer_2_3_2_producer_lock"}) consumer_lock=Scalar(%buffer_2_3_2_consumer_lock = aie.lock(%tile_2_3) {sym_name = "buffer_2_3_2_consumer_lock"})>
        # CHECK: (1, 0) <Channel: buffer=MemRef(%buffer_3_2_3, memref<5x6xi32>) producer_lock=Scalar(%buffer_3_2_3_producer_lock = aie.lock(%tile_3_2) {sym_name = "buffer_3_2_3_producer_lock"}) consumer_lock=Scalar(%buffer_3_2_3_consumer_lock = aie.lock(%tile_3_2) {sym_name = "buffer_3_2_3_consumer_lock"})>
        # CHECK: (1, 1) <Channel: buffer=MemRef(%buffer_3_3_4, memref<7x8xi32>) producer_lock=Scalar(%buffer_3_3_4_producer_lock = aie.lock(%tile_3_3) {sym_name = "buffer_3_3_4_producer_lock"}) consumer_lock=Scalar(%buffer_3_3_4_consumer_lock = aie.lock(%tile_3_3) {sym_name = "buffer_3_3_4_consumer_lock"})>
        for idx, c in np.ndenumerate(cs):
            print(idx, c)


# CHECK-LABEL: buffer_test_this_needs_to_distinct_from_all_other_mentions_of_buffer_in_this_file
@construct_and_print_module
def buffer_test_this_needs_to_distinct_from_all_other_mentions_of_buffer_in_this_file(
    module,
):
    @aie.device(AIEDevice.npu)
    def npu():
        tiles = TileArray()

        shapes = [(10, 10)]
        c = tiles[2, 2].buffer(shape=shapes, dtype=[T.i32()])
        # CHECK: MemRef(%buffer_2_2, memref<10x10xi32>)
        print(c)
        cs = tiles[2:4, 2:4].buffer(shape=shapes, dtype=[T.i32()])
        assert cs.shape == (2, 2)

        # CHECK: (0, 0) MemRef(%buffer_2_2_0, memref<10x10xi32>)
        # CHECK: (0, 1) MemRef(%buffer_2_3, memref<10x10xi32>)
        # CHECK: (1, 0) MemRef(%buffer_3_2, memref<10x10xi32>)
        # CHECK: (1, 1) MemRef(%buffer_3_3, memref<10x10xi32>)
        for idx, c in np.ndenumerate(cs):
            print(idx, c)

        shapes = [[(1, 2), (3, 4)], [(5, 6), (7, 8)]]
        cs = tiles[2:4, 2:4].buffer(shape=shapes, dtype=[T.i32()])
        assert cs.shape == (2, 2)

        # CHECK: (0, 0) MemRef(%buffer_2_2_1, memref<1x2xi32>)
        # CHECK: (0, 1) MemRef(%buffer_2_3_2, memref<3x4xi32>)
        # CHECK: (1, 0) MemRef(%buffer_3_2_3, memref<5x6xi32>)
        # CHECK: (1, 1) MemRef(%buffer_3_3_4, memref<7x8xi32>)
        for idx, c in np.ndenumerate(cs):
            print(idx, c)
