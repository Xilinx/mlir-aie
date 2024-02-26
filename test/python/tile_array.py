# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.
import numpy as np

# RUN: %python %s | FileCheck %s
# REQUIRES: py310

from aie.dialects.aie import (
    AIEDevice,
    DMAChannelDir,
    LockAction,
    WireBundle,
    device,
    lock,
    find_neighbors,
)
from aie.dialects.aiex import TileArray
from util import construct_and_print_module

DMA = WireBundle.DMA
S2MM = DMAChannelDir.S2MM
MM2S = DMAChannelDir.MM2S
Acquire = LockAction.Acquire
AcquireGreaterEqual = LockAction.AcquireGreaterEqual
Release = LockAction.Release


@construct_and_print_module
def broadcast(module):
    @device(AIEDevice.ipu)
    def ipu():
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
        # CHECK:   aie.device(ipu) {
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


@construct_and_print_module
def lshift(module):
    @device(AIEDevice.ipu)
    def ipu():
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


@construct_and_print_module
def locks(module):
    @device(AIEDevice.ipu)
    def ipu():
        tiles = TileArray()

        lock(tiles[0, 1].df)
        # CHECK: %lock_0_1 = aie.lock(%tile_0_1)
        for l in tiles[0, 1].locks():
            print(l.owner)

        lock(tiles[0, 2].df)
        lock(tiles[0, 2].df, annot="bob")
        lock(tiles[0, 3].df)
        lock(tiles[0, 3].df, annot="alice")

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


@construct_and_print_module
def neighbors(module):
    @device(AIEDevice.ipu)
    def ipu():
        tiles = TileArray()

        # CHECK: Neighbors(north=%tile_2_3 = aie.tile(2, 3), west=%tile_1_2 = aie.tile(1, 2), south=None)
        print(find_neighbors(tiles[2, 2].df))

        assert tiles[1:3, 1:3].neighbors().shape == (2, 2)
        # CHECK: tile(col=1, row=1) : Neighbors(north=None, west=None, south=None)
        # CHECK: tile(col=1, row=2) : Neighbors(north=<TileArray: [%tile_1_3 = aie.tile(1, 3)]>, west=<TileArray: [%tile_0_2 = aie.tile(0, 2)]>, south=None)
        # CHECK: tile(col=2, row=1) : Neighbors(north=None, west=<TileArray: [%tile_1_1 = aie.tile(1, 1)]>, south=None)
        # CHECK: tile(col=2, row=2) : Neighbors(north=<TileArray: [%tile_2_3 = aie.tile(2, 3)]>, west=<TileArray: [%tile_1_2 = aie.tile(1, 2)]>, south=None)
        for idx, n in np.ndenumerate(tiles[1:3, 1:3].neighbors()):
            print(tiles[1:3, 1:3][idx].df, ":", n)

        # CHECK: tile(col=1, row=1) : Neighbors(north=<TileArray: [%tile_1_2 = aie.tile(1, 2)]>, west=None, south=<TileArray: [%tile_1_0 = aie.tile(1, 0)]>)
        # CHECK: tile(col=1, row=2) : Neighbors(north=<TileArray: [%tile_1_3 = aie.tile(1, 3)]>, west=<TileArray: [%tile_0_2 = aie.tile(0, 2)]>, south=<TileArray: [%tile_1_1 = aie.tile(1, 1)]>)
        # CHECK: tile(col=2, row=1) : Neighbors(north=<TileArray: [%tile_2_2 = aie.tile(2, 2)]>, west=<TileArray: [%tile_1_1 = aie.tile(1, 1)]>, south=<TileArray: [%tile_2_0 = aie.tile(2, 0)]>)
        # CHECK: tile(col=2, row=2) : Neighbors(north=<TileArray: [%tile_2_3 = aie.tile(2, 3)]>, west=<TileArray: [%tile_1_2 = aie.tile(1, 2)]>, south=<TileArray: [%tile_2_1 = aie.tile(2, 1)]>)
        for idx, n in np.ndenumerate(tiles[1:3, 1:3].neighbors(logical=False)):
            print(tiles[1:3, 1:3][idx].df, ":", n)
