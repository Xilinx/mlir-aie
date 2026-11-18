# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s
# RUN: %python %s | aie-opt | FileCheck %s

# Every `aiex.cert` op built from Python on the multi-controller CERT target:
# registers and wait handles as plain integers, barriers by their assembler
# names, and the region ops (section, page, job, group, uC DMA chain) as
# decorators that add the terminator. Each CHECK sits above the statement that
# prints it; the module is printed once at the end.

import numpy as np

from aie.dialects.aie import device, AIEDevice, tile, buffer
from aie.dialects.aiex import *
from aie.ir import Context, Location, Module, InsertionPoint

with Context(), Location.unknown():
    module = Module.create()
    with InsertionPoint(module.body):

        # CHECK: aie.device(xcve3858) {
        @device(AIEDevice.xcve3858)
        def device_body():
            t = tile(0, 2)
            # CHECK: aie.buffer(%{{.*}}) {sym_name = "bd_data"}
            buffer(t, np.ndarray[(9,), np.dtype[np.int32]], name="bd_data")

            # The target of load_pdi and preempt below.
            # CHECK: aiex.cert.section @sec {
            @cert_section("sec")
            def _():
                # CHECK: aiex.cert.page {
                @cert_page
                def _():
                    # CHECK: aiex.cert.job(20) {
                    @cert_job(20)
                    def _():
                        # CHECK: aiex.cert.nop
                        cert_nop()

            # CHECK: aiex.cert.uc_dma_chain @chain {
            @cert_uc_dma_chain("chain")
            def _():
                # CHECK: aiex.cert.uc_dma_bd @bd_data, 36864, 9, true
                cert_uc_dma_bd("bd_data", 0x9000, 9, True)
                # CHECK: aiex.cert.uc_dma_bd @bd_data, 35389440, 8, false
                cert_uc_dma_bd("bd_data", 0x21C0000, 8, False)

            # One microcontroller's stream. Two jobs in a page are
            # cooperatively scheduled.
            # CHECK: aiex.cert.attach_to_group(2) {
            @cert_attach_to_group(2)
            def _():
                # CHECK: aiex.cert.page {
                @cert_page()
                def _():
                    # CHECK: aiex.cert.job(0) {
                    @cert_job(0)
                    def _():
                        # CHECK: aiex.cert.write32(4096, 2779096485)
                        cert_write32(0x1000, 0xA5A5A5A5)
                        # CHECK: aiex.cert.maskwrite32(4096, 255, 16)
                        cert_maskwrite32(0x1000, 0xFF, 0x10)
                        # CHECK: aiex.cert.write32_d(4096, 42)
                        cert_write32_d(0x1000, 42)
                        # CHECK: aiex.cert.write32_d(3, 7) {address_is_reg, value_is_reg}
                        cert_write32_d(3, 7, address_is_reg=True, value_is_reg=True)
                        # CHECK: aiex.cert.read32(3, 4096)
                        cert_read32(3, 0x1000)
                        # CHECK: aiex.cert.read32_d(3, 4)
                        cert_read32_d(3, 4)
                        # CHECK: aiex.cert.maskpoll32(4096, 255, 16)
                        cert_maskpoll32(0x1000, 0xFF, 0x10)
                        # CHECK: aiex.cert.nop
                        cert_nop()

                    # CHECK: aiex.cert.job(1) {
                    @cert_job(1)
                    def _():
                        # CHECK: aiex.cert.local_barrier(lb1, 2)
                        cert_local_barrier("lb1", 2)
                        # CHECK: aiex.cert.local_barrier(lb15, 2)
                        cert_local_barrier(15, 2)
                        # CHECK: aiex.cert.remote_barrier(rb1, 5)
                        cert_remote_barrier("rb1", 0x5)
                        # CHECK: aiex.cert.remote_barrier(rb6, 3)
                        cert_remote_barrier(6, 0x3)
                        # CHECK: aiex.cert.rel_acq_sync(90116, 90632)
                        cert_rel_acq_sync(0x16004, 0x16208)
                        # CHECK: aiex.cert.wait_tcts(0, 0, 1)
                        cert_wait_tcts(0, 0, 1)
                        # CHECK: aiex.cert.save_register(4096, 7)
                        cert_save_register(0x1000, 7)
                        # CHECK: aiex.cert.save_timestamps(101)
                        cert_save_timestamps(101)

            # The page's `placement` is a resolved uC group id.
            # CHECK: aiex.cert.page {
            @cert_page(placement=4)
            def _():
                # CHECK: aiex.cert.job(2) {
                @cert_job(2)
                def _():
                    # CHECK: aiex.cert.uc_dma_write_des_sync(@chain)
                    cert_uc_dma_write_des_sync("chain")
                    # CHECK: aiex.cert.uc_dma_write_des(5, @chain)
                    cert_uc_dma_write_des(5, "chain")
                    # CHECK: aiex.cert.wait_uc_dma(5)
                    cert_wait_uc_dma(5)
                    # CHECK: aiex.cert.uc_dma_write_des(23, @chain)
                    cert_uc_dma_write_des(23, "chain")
                    # CHECK: aiex.cert.wait_uc_dma(23)
                    cert_wait_uc_dma(23)
                    # CHECK: aiex.cert.apply_offset_57(@bd_data, 1, 0)
                    cert_apply_offset_57("bd_data", 1, 0)

            # CHECK: } {placement = 4 : i32}

            # load_pdi and preempt each take a whole job in a whole page.
            # CHECK: aiex.cert.page {
            @cert_page
            def _():
                # CHECK: aiex.cert.job(3) {
                @cert_job(3)
                def _():
                    # CHECK: aiex.cert.load_pdi(1, @sec)
                    cert_load_pdi(1, "sec")

            # CHECK: aiex.cert.page {
            @cert_page
            def _():
                # CHECK: aiex.cert.job(4) {
                @cert_job(4)
                def _():
                    # CHECK: aiex.cert.preempt(0, @sec, @sec)
                    cert_preempt(0, "sec", "sec")

    assert module.operation.verify()
    print(module)
