# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.
from pprint import pprint

# RUN: %python %s | FileCheck %s

from aie.dialects import aie
from aie.dialects.aie import (
    AIEDevice,
    DMAChannelDir,
    LockAction,
    WireBundle,
    FlowEndPoint,
)
from util import construct_and_print_module

DMA = WireBundle.DMA
S2MM = DMAChannelDir.S2MM
MM2S = DMAChannelDir.MM2S
Acquire = LockAction.Acquire
AcquireGreaterEqual = LockAction.AcquireGreaterEqual
Release = LockAction.Release


# CHECK-LABEL: flow_dsl_1
@construct_and_print_module
def flow_dsl_1(module):
    FlowEndPoint._reset_used_channels()

    @aie.device(AIEDevice.ipu)
    def ipu():
        tile_0_0 = aie.tile(0, 0)
        tile_0_1 = aie.tile(0, 1)
        tile_0_2 = aie.tile(0, 2)

        # fmt: off
        tile_0_0.ep(DMA, 0) >> tile_0_1.ep(DMA, 0) >> tile_0_2.ep(DMA, 0) >> tile_0_1.ep(DMA, 1) >> tile_0_0.ep(DMA, 0)
        # fmt: on

    # CHECK:  aie.device(ipu) {
    # CHECK:    %tile_0_0 = aie.tile(0, 0)
    # CHECK:    %tile_0_1 = aie.tile(0, 1)
    # CHECK:    %tile_0_2 = aie.tile(0, 2)
    # CHECK:    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    # CHECK:    aie.flow(%tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    # CHECK:    aie.flow(%tile_0_2, DMA : 0, %tile_0_1, DMA : 1)
    # CHECK:    aie.flow(%tile_0_1, DMA : 1, %tile_0_0, DMA : 0)
    # CHECK:  }
    print(module)


@construct_and_print_module
def flow_dsl_2(module):
    FlowEndPoint._reset_used_channels()

    @aie.device(AIEDevice.ipu)
    def ipu():
        tile_0_0 = aie.tile(0, 0)
        tile_0_1 = aie.tile(0, 1)
        tile_0_2 = aie.tile(0, 2)

        # fmt: off
        tile_0_0.ep(DMA) >> tile_0_1.ep(DMA) >> tile_0_2.ep(DMA) >> tile_0_1.ep(DMA) >> tile_0_0.ep(DMA)
        # fmt: on

    # CHECK:  aie.device(ipu) {
    # CHECK:    %tile_0_0 = aie.tile(0, 0)
    # CHECK:    %tile_0_1 = aie.tile(0, 1)
    # CHECK:    %tile_0_2 = aie.tile(0, 2)
    # CHECK:    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    # CHECK:    aie.flow(%tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    # CHECK:    aie.flow(%tile_0_2, DMA : 0, %tile_0_1, DMA : 1)
    # CHECK:    aie.flow(%tile_0_1, DMA : 1, %tile_0_0, DMA : 0)
    # CHECK:  }
    print(module)


@construct_and_print_module
def flow_dsl_3(module):
    FlowEndPoint._reset_used_channels()

    @aie.device(AIEDevice.ipu)
    def ipu():
        tile_0_0 = aie.tile(0, 0)
        tile_0_1 = aie.tile(0, 1)
        tile_0_2 = aie.tile(0, 2)

        tile_0_0 >> tile_0_1 >> tile_0_2 >> tile_0_1 >> tile_0_0

        # CHECK: tile(col=0, row=1)
        # CHECK: aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
        # CHECK: aie.flow(%tile_0_1, DMA : 1, %tile_0_0, DMA : 0)
        for k, fs in tile_0_0.flows.items():
            print(k)
            for f in fs:
                print(f)

        # CHECK: tile(col=0, row=0)
        # CHECK: aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
        # CHECK: aie.flow(%tile_0_1, DMA : 1, %tile_0_0, DMA : 0)
        # CHECK: tile(col=0, row=2)
        # CHECK: aie.flow(%tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
        # CHECK: aie.flow(%tile_0_2, DMA : 0, %tile_0_1, DMA : 1)
        for k, fs in tile_0_1.flows.items():
            print(k)
            for f in fs:
                print(f)

        # CHECK: tile(col=0, row=1)
        # CHECK: aie.flow(%tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
        # CHECK: aie.flow(%tile_0_2, DMA : 0, %tile_0_1, DMA : 1)
        for k, fs in tile_0_2.flows.items():
            print(k)
            for f in fs:
                print(f)

    # CHECK:  aie.device(ipu) {
    # CHECK:    %tile_0_0 = aie.tile(0, 0)
    # CHECK:    %tile_0_1 = aie.tile(0, 1)
    # CHECK:    %tile_0_2 = aie.tile(0, 2)
    # CHECK:    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    # CHECK:    aie.flow(%tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    # CHECK:    aie.flow(%tile_0_2, DMA : 0, %tile_0_1, DMA : 1)
    # CHECK:    aie.flow(%tile_0_1, DMA : 1, %tile_0_0, DMA : 0)
    # CHECK:  }
    print(module)


@construct_and_print_module
def flow_dsl_4(module):
    FlowEndPoint._reset_used_channels()

    @aie.device(AIEDevice.ipu)
    def ipu():
        tile_0_0 = aie.tile(0, 0)
        tile_0_1 = aie.tile(0, 1)
        tile_0_2 = aie.tile(0, 2)
        tile_0_3 = aie.tile(0, 3)
        tile_0_4 = aie.tile(0, 4)

        tile_0_0 << tile_0_1 << tile_0_2 << tile_0_3 << tile_0_4

    # CHECK:  aie.device(ipu) {
    # CHECK:    %tile_0_0 = aie.tile(0, 0)
    # CHECK:    %tile_0_1 = aie.tile(0, 1)
    # CHECK:    %tile_0_2 = aie.tile(0, 2)
    # CHECK:    %tile_0_3 = aie.tile(0, 3)
    # CHECK:    %tile_0_4 = aie.tile(0, 4)
    # CHECK:    aie.flow(%tile_0_1, DMA : 0, %tile_0_0, DMA : 0)
    # CHECK:    aie.flow(%tile_0_2, DMA : 0, %tile_0_1, DMA : 0)
    # CHECK:    aie.flow(%tile_0_3, DMA : 0, %tile_0_2, DMA : 0)
    # CHECK:    aie.flow(%tile_0_4, DMA : 0, %tile_0_3, DMA : 0)
    # CHECK:  }
    print(module)


@construct_and_print_module
def flow_dsl_5(module):
    FlowEndPoint._reset_used_channels()

    @aie.device(AIEDevice.ipu)
    def ipu():
        tile_0_0 = aie.tile(0, 0)
        tile_0_1 = aie.tile(0, 1)
        tile_0_2 = aie.tile(0, 2)

        tile_0_0 << tile_0_1 >> tile_0_2

        # CHECK: tile(col=0, row=1)
        # CHECK: aie.flow(%tile_0_1, DMA : 0, %tile_0_0, DMA : 0)
        for k, fs in tile_0_0.flows.items():
            print(k)
            for f in fs:
                print(f)

        # CHECK: tile(col=0, row=0)
        # CHECK: aie.flow(%tile_0_1, DMA : 0, %tile_0_0, DMA : 0)
        # CHECK: tile(col=0, row=2)
        # CHECK: aie.flow(%tile_0_1, DMA : 1, %tile_0_2, DMA : 0)
        for k, fs in tile_0_1.flows.items():
            print(k)
            for f in fs:
                print(f)

        # CHECK: tile(col=0, row=1)
        # CHECK: aie.flow(%tile_0_1, DMA : 1, %tile_0_2, DMA : 0)
        for k, fs in tile_0_2.flows.items():
            print(k)
            for f in fs:
                print(f)

    # CHECK:  aie.device(ipu) {
    # CHECK:    %tile_0_0 = aie.tile(0, 0)
    # CHECK:    %tile_0_1 = aie.tile(0, 1)
    # CHECK:    %tile_0_2 = aie.tile(0, 2)
    # CHECK:    aie.flow(%tile_0_1, DMA : 0, %tile_0_0, DMA : 0)
    # CHECK:    aie.flow(%tile_0_1, DMA : 1, %tile_0_2, DMA : 0)
    # CHECK:  }
    print(module)


@construct_and_print_module
def flow_dsl_6(module):
    FlowEndPoint._reset_used_channels()

    @aie.device(AIEDevice.ipu)
    def ipu():
        tile_0_0 = aie.tile(0, 0)
        tile_0_1 = aie.tile(0, 1)
        tile_0_2 = aie.tile(0, 2)
        tile_0_3 = aie.tile(0, 3)

        tile_0_0 << tile_0_1 >> tile_0_2 >> tile_0_3

        # CHECK: tile(col=0, row=1)
        # CHECK: aie.flow(%tile_0_1, DMA : 0, %tile_0_0, DMA : 0)
        for k, fs in tile_0_0.flows.items():
            print(k)
            for f in fs:
                print(f)

        # CHECK: tile(col=0, row=0)
        # CHECK: aie.flow(%tile_0_1, DMA : 0, %tile_0_0, DMA : 0)
        # CHECK: tile(col=0, row=2)
        # CHECK: aie.flow(%tile_0_1, DMA : 1, %tile_0_2, DMA : 0)
        for k, fs in tile_0_1.flows.items():
            print(k)
            for f in fs:
                print(f)

        # CHECK: tile(col=0, row=1)
        # CHECK: aie.flow(%tile_0_1, DMA : 1, %tile_0_2, DMA : 0)
        # CHECK: tile(col=0, row=3)
        # CHECK: aie.flow(%tile_0_2, DMA : 0, %tile_0_3, DMA : 0)
        for k, fs in tile_0_2.flows.items():
            print(k)
            for f in fs:
                print(f)

    # CHECK:  aie.device(ipu) {
    # CHECK:    %tile_0_0 = aie.tile(0, 0)
    # CHECK:    %tile_0_1 = aie.tile(0, 1)
    # CHECK:    %tile_0_2 = aie.tile(0, 2)
    # CHECK:    aie.flow(%tile_0_1, DMA : 0, %tile_0_0, DMA : 0)
    # CHECK:    aie.flow(%tile_0_1, DMA : 1, %tile_0_2, DMA : 0)
    # CHECK:    aie.flow(%tile_0_2, DMA : 0, %tile_0_3, DMA : 0)
    # CHECK:  }
    print(module)
